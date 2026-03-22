"""
LangGraph-powered worker-step runner for KestrelAgentWorker.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any, TypedDict

from KestrelAI.graphs.persistence import get_langgraph_runtime

if TYPE_CHECKING:
    from KestrelAI.model_loop import KestrelAgentWorker
    from KestrelAI.shared.models import Task
else:  # Runtime aliases so LangGraph can resolve TypedDict annotations
    KestrelAgentWorker = Any
    Task = Any

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - dependency-gated path
    END = START = StateGraph = None

from KestrelAI.shared.models import TaskMetrics, TaskStatus

logger = logging.getLogger(__name__)


class WorkerStepGraphState(TypedDict):
    task_id: str
    task: Task
    elapsed: float
    progress: float
    plan_ready: bool
    step_timed_out: bool
    notes: str
    current_subtask: str
    progress_info: dict[str, Any]
    has_meaningful_activity: bool
    activity_type: str
    should_stop: bool
    stop_reason: str


class LangGraphWorkerStepRunner:
    """Executes one worker processing step using LangGraph."""

    def __init__(self, worker: KestrelAgentWorker):
        if StateGraph is None:
            raise ImportError("langgraph is not installed")
        self.worker = worker
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(WorkerStepGraphState)
        workflow.add_node("preflight", self._preflight)
        workflow.add_node("execute_step", self._execute_step)
        workflow.add_node("sync_state", self._sync_state)
        workflow.add_node("aggregate_activity", self._aggregate_activity)
        workflow.add_node("emit_outputs", self._emit_outputs)
        workflow.add_node("stop_task", self._stop_task)

        workflow.add_edge(START, "preflight")
        workflow.add_conditional_edges(
            "preflight",
            self._route_after_preflight,
            {
                "run": "execute_step",
                "stop": "stop_task",
                "wait": END,
            },
        )
        workflow.add_edge("execute_step", "sync_state")
        workflow.add_edge("sync_state", "aggregate_activity")
        workflow.add_edge("aggregate_activity", "emit_outputs")
        workflow.add_conditional_edges(
            "emit_outputs",
            self._route_after_emit,
            {
                "stop": "stop_task",
                "done": END,
            },
        )
        workflow.add_edge("stop_task", END)
        checkpointer, store = get_langgraph_runtime()
        compile_kwargs = {}
        if checkpointer is not None:
            compile_kwargs["checkpointer"] = checkpointer
        if store is not None:
            compile_kwargs["store"] = store
        return workflow.compile(**compile_kwargs)

    def run(self, task_id: str, task: Task) -> None:
        initial_state: WorkerStepGraphState = {
            "task_id": task_id,
            "task": task,
            "elapsed": 0.0,
            "progress": 0.0,
            "plan_ready": False,
            "step_timed_out": False,
            "notes": "",
            "current_subtask": "",
            "progress_info": {},
            "has_meaningful_activity": False,
            "activity_type": "analysis",
            "should_stop": False,
            "stop_reason": "",
        }
        run_config = {
            "configurable": {
                "thread_id": f"worker:{task_id}",
                "checkpoint_ns": "worker_step_runner",
            }
        }
        self.graph.invoke(initial_state, config=run_config)

    @staticmethod
    def _budget_policy() -> str:
        return os.getenv("WORKER_BUDGET_POLICY", "soft").strip().lower()

    def _hard_budget_enforced(self) -> bool:
        return self._budget_policy() == "hard"

    @staticmethod
    def _all_subtasks_completed(progress_info: dict[str, Any]) -> bool:
        total = int(progress_info.get("total") or 0)
        completed = int(progress_info.get("completed") or 0)
        return total > 0 and completed >= total

    @staticmethod
    def _compact_activity_message(message: str, *, fallback: str) -> str:
        compact = " ".join(str(message or "").split()).strip()
        if not compact:
            return fallback
        replacements = {
            "[THOUGHT]": "Thought:",
            "[SEARCH]": "Search:",
            "[SUMMARY]": "Summary:",
            "[CHECKPOINT]": "Checkpoint:",
            "[COMPLETE]": "Complete:",
            "[NO RESULTS]": "No results:",
            "[SKIP]": "Skipped:",
        }
        for old, new in replacements.items():
            compact = compact.replace(old, new)
        return compact[:240]

    def _preflight(self, state: WorkerStepGraphState) -> dict[str, Any]:
        task_id = state["task_id"]
        task = state["task"]
        self.worker.task_metrics.setdefault(
            task_id,
            {
                "search_count": 0,
                "think_count": 0,
                "summary_count": 0,
                "checkpoint_count": 0,
                "action_count": 0,
                "searches": [],
                "start_time": time.time(),
                "last_research_plan_state": {
                    "subtask_index": -1,
                    "completed_subtasks": set(),
                },
            },
        )

        self.worker._send_progress_heartbeat(task_id, task)
        task_state = self.worker.orchestrator.task_states.get(task.name)
        if not task_state or not task_state.research_plan:
            total_elapsed = max(
                0.0,
                time.time()
                - self.worker.task_metrics.get(task_id, {}).get(
                    "start_time", time.time()
                ),
            )
            logger.debug("Waiting for planning phase to complete for task %s", task_id)
            return {
                "elapsed": total_elapsed,
                "progress": 0.0,
                "plan_ready": False,
            }

        if not isinstance(
            self.worker.task_metrics.get(task_id, {}).get("execution_start_time"),
            (int, float),
        ):
            self.worker.task_metrics[task_id]["execution_start_time"] = time.time()

        elapsed, progress = self.worker._compute_elapsed_and_progress(task_id, task)
        if self._hard_budget_enforced() and progress >= 100.0:
            logger.warning(
                "Hard budget reached before next step for task %s (elapsed=%.1fs); stopping task",
                task_id,
                elapsed,
            )
            return {
                "elapsed": elapsed,
                "progress": progress,
                "should_stop": True,
                "stop_reason": "budget_exhausted",
            }

        return {
            "elapsed": elapsed,
            "progress": progress,
            "plan_ready": True,
        }

    def _route_after_preflight(self, state: WorkerStepGraphState) -> str:
        if state.get("should_stop"):
            return "stop"
        if not state.get("plan_ready"):
            return "wait"
        return "run"

    def _execute_step(self, state: WorkerStepGraphState) -> dict[str, Any]:
        task_id = state["task_id"]
        task = state["task"]
        elapsed = state.get("elapsed", 0.0)

        logger.info("Processing research step for task %s", task_id)

        budget_minutes = getattr(task, "budgetMinutes", None)
        if not isinstance(budget_minutes, (int, float)) or budget_minutes <= 0:
            budget_minutes = self.worker.current_task_config.get("budgetMinutes", 1)
        budget_seconds = max(float(budget_minutes) * 60.0, 1.0)
        disable_timeouts = os.getenv(
            "GLOBAL_DISABLE_TIMEOUTS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        raw_step_timeout_cap_seconds = float(
            os.getenv("WORKER_STEP_TIMEOUT_SECONDS", "60")
        )
        step_timeout_cap_seconds: float | None
        if disable_timeouts or raw_step_timeout_cap_seconds <= 0:
            step_timeout_cap_seconds = None
        else:
            step_timeout_cap_seconds = max(1.0, raw_step_timeout_cap_seconds)
        task_state = self.worker.orchestrator.task_states.get(task.name)
        unbounded_finalization_step = os.getenv(
            "WORKER_UNBOUNDED_FINALIZATION_STEP", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        in_finalization = (
            task_state is not None
            and bool(task_state.research_plan)
            and len(task_state.research_plan.subtasks) > 0
            and len(task_state.completed_subtasks)
            >= len(task_state.research_plan.subtasks)
            and task.status != TaskStatus.COMPLETE
        )
        if in_finalization and unbounded_finalization_step:
            step_timeout_seconds: float | None = None
        elif step_timeout_cap_seconds is None:
            step_timeout_seconds = None
        elif self._hard_budget_enforced():
            remaining_budget_seconds = max(1.0, budget_seconds - elapsed)
            step_timeout_seconds = min(
                remaining_budget_seconds, step_timeout_cap_seconds
            )
        else:
            # In soft-budget mode we keep per-step timeout stable so the agent
            # can finish subtasks even after budget guidance is exceeded.
            step_timeout_seconds = step_timeout_cap_seconds

        step_timed_out = False
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            try:

                async def _run_orchestrator_step() -> str:
                    return await self.worker.orchestrator.next_action(task)

                if step_timeout_seconds is None:
                    logger.info(
                        "Running unbounded finalization step for task %s (no worker step timeout)",
                        task_id,
                    )
                    research_result = loop.run_until_complete(_run_orchestrator_step())
                else:
                    research_result = loop.run_until_complete(
                        asyncio.wait_for(
                            _run_orchestrator_step(),
                            timeout=step_timeout_seconds,
                        )
                    )
                self.worker.latest_feedback = research_result
                logger.info("Research step completed: %s...", research_result[:100])
            except asyncio.TimeoutError:
                step_timed_out = True
                logger.warning(
                    "Research step timed out at budget boundary for task %s (timeout=%.1fs)",
                    task_id,
                    step_timeout_seconds,
                )
                research_result = (
                    "Research step stopped after reaching the configured time budget."
                )
                self.worker.latest_feedback = research_result
            except Exception as e:
                logger.error("Error executing orchestrator step: %s", e)
                research_result = f"Processing research step - {str(e)[:100]}"
                self.worker.latest_feedback = research_result
        except Exception as e:
            logger.error("Error executing orchestrator step: %s", e)
            research_result = f"Processing research step - {str(e)[:100]}"
            self.worker.latest_feedback = research_result

        return {
            "step_timed_out": step_timed_out,
            "notes": research_result,
        }

    def _sync_state(self, state: WorkerStepGraphState) -> dict[str, Any]:
        task_id = state["task_id"]
        task = state["task"]
        notes = state.get("notes", "")

        current_subtask = self.worker.orchestrator.get_current_subtask(task.name)
        if current_subtask:
            self.worker.latest_subtask = current_subtask

        progress_info = self.worker.orchestrator.get_task_progress(task.name)

        task_state = self.worker.orchestrator.task_states.get(task.name)
        if task_state and task_state.research_plan:
            last_state = self.worker.task_metrics[task_id].get(
                "last_research_plan_state",
                {
                    "subtask_index": -1,
                    "completed_subtasks": set(),
                },
            )

            current_subtask_index = task_state.subtask_index
            current_completed = task_state.completed_subtasks.copy()
            last_completed = last_state.get("completed_subtasks", set())
            last_completed_set = (
                set(last_completed)
                if isinstance(last_completed, (set, frozenset))
                else set()
            )

            subtask_changed = (
                current_subtask_index != last_state.get("subtask_index", -1)
                or current_completed != last_completed_set
            )

            if subtask_changed:
                logger.info(
                    "Subtask state changed for task %s: index=%s, completed=%s",
                    task_id,
                    current_subtask_index,
                    current_completed,
                )
                self.worker.send_research_plan_update(task_id, task.name)
                self.worker.task_metrics[task_id]["last_research_plan_state"] = {
                    "subtask_index": current_subtask_index,
                    "completed_subtasks": current_completed.copy(),
                }

        return {
            "notes": notes,
            "current_subtask": current_subtask or "",
            "progress_info": progress_info,
        }

    def _aggregate_activity(self, state: WorkerStepGraphState) -> dict[str, Any]:
        task_id = state["task_id"]
        task = state["task"]
        current_subtask = state.get("current_subtask", "")
        progress_info = state.get("progress_info", {})

        total_searches = 0
        total_thinks = 0
        total_summaries = 0
        total_checkpoints = 0
        total_actions = 0

        for subtask_info in progress_info.get("subtasks", []):
            if "agent_metrics" in subtask_info:
                metrics = subtask_info["agent_metrics"]
                total_searches += metrics.get("total_searches", 0)
                total_thinks += metrics.get("total_thoughts", 0)
                total_summaries += metrics.get("total_summaries", 0)
                total_checkpoints += metrics.get("total_checkpoints", 0)
                total_actions += metrics.get("action_count", 0)

        self.worker.task_metrics[task_id].update(
            {
                "search_count": total_searches,
                "think_count": total_thinks,
                "summary_count": total_summaries,
                "checkpoint_count": total_checkpoints,
                "action_count": total_actions,
                "searches": [],
                "search_history": [],
                "current_focus": current_subtask or "Preparing research",
            }
        )

        has_meaningful_activity = False
        activity_type = "analysis"
        message = ""

        if (
            hasattr(self.worker.orchestrator, "task_states")
            and task.name in self.worker.orchestrator.task_states
        ):
            task_state = self.worker.orchestrator.task_states[task.name]
            if task_state.subtask_index in task_state.subtask_agents:
                current_agent = task_state.subtask_agents[task_state.subtask_index]
                if task.name in current_agent._state:
                    agent_state = current_agent._state[task.name]
                    last_action_count = self.worker.task_metrics[task_id].get(
                        "last_agent_action_count", 0
                    )
                    if agent_state.action_count > last_action_count:
                        has_meaningful_activity = True

                        if agent_state.search_count > self.worker.task_metrics[
                            task_id
                        ].get("last_search_count", 0):
                            activity_type = "search"
                        elif agent_state.think_count > self.worker.task_metrics[
                            task_id
                        ].get("last_think_count", 0):
                            activity_type = "thinking"
                        elif agent_state.summary_count > self.worker.task_metrics[
                            task_id
                        ].get("last_summary_count", 0):
                            activity_type = "summary"
                        elif agent_state.checkpoint_count > self.worker.task_metrics[
                            task_id
                        ].get("last_checkpoint_count", 0):
                            activity_type = "checkpoint"
                        else:
                            activity_type = (
                                getattr(agent_state, "last_step_activity", "")
                                or "analysis"
                            )

                        message = self._compact_activity_message(
                            getattr(agent_state, "last_step_feedback", ""),
                            fallback=f"{activity_type.title()} update",
                        )

                        self.worker.task_metrics[task_id][
                            "last_agent_action_count"
                        ] = agent_state.action_count
                        self.worker.task_metrics[task_id][
                            "last_search_count"
                        ] = agent_state.search_count
                        self.worker.task_metrics[task_id][
                            "last_think_count"
                        ] = agent_state.think_count
                        self.worker.task_metrics[task_id][
                            "last_summary_count"
                        ] = agent_state.summary_count
                        self.worker.task_metrics[task_id][
                            "last_checkpoint_count"
                        ] = agent_state.checkpoint_count

        if has_meaningful_activity:
            self.worker.redis_client.send_activity(task_id, activity_type, message)

        return {
            "has_meaningful_activity": has_meaningful_activity,
            "activity_type": activity_type,
        }

    def _emit_outputs(self, state: WorkerStepGraphState) -> dict[str, Any]:
        task_id = state["task_id"]
        task = state["task"]
        notes = state.get("notes", "")
        progress_info = state.get("progress_info", {})
        has_meaningful_activity = state.get("has_meaningful_activity", False)
        activity_type = state.get("activity_type", "analysis")
        step_timed_out = state.get("step_timed_out", False)

        elapsed, progress = self.worker._compute_elapsed_and_progress(task_id, task)
        plan_progress = float(progress_info.get("progress", 0.0) or 0.0)
        display_progress = (
            plan_progress if int(progress_info.get("total") or 0) > 0 else progress
        )
        if task.status == TaskStatus.COMPLETE:
            display_progress = 100.0

        metric_getter = getattr(self.worker.agent, "get_global_metrics", None)
        if not callable(metric_getter):
            metric_getter = getattr(self.worker.agent, "get_metrics", None)
        if callable(metric_getter):
            agent_metrics = metric_getter()
            self.worker.global_metrics.update(agent_metrics)

        metrics_model = TaskMetrics(
            searchCount=self.worker.task_metrics[task_id]["search_count"],
            thinkCount=self.worker.task_metrics[task_id]["think_count"],
            summaryCount=self.worker.task_metrics[task_id]["summary_count"],
            checkpointCount=self.worker.task_metrics[task_id]["checkpoint_count"],
            webFetchCount=self.worker.global_metrics.get("total_web_fetches", 0),
            llmTokensUsed=self.worker.global_metrics.get("total_llm_calls", 0) * 1000,
            errorCount=0,
        )
        if hasattr(metrics_model, "model_dump"):
            metrics_payload = metrics_model.model_dump()
        else:
            metrics_payload = metrics_model.dict()

        self.worker.redis_client.send_update(
            task_id,
            status=(
                task.status.value
                if hasattr(task, "status")
                else TaskStatus.ACTIVE.value
            ),
            progress=display_progress,
            elapsed=int(elapsed),
            metrics=metrics_payload,
        )

        search_queries = []
        if (
            hasattr(self.worker.orchestrator, "task_states")
            and task.name in self.worker.orchestrator.task_states
        ):
            task_state = self.worker.orchestrator.task_states[task.name]
            if task_state.subtask_index in task_state.subtask_agents:
                current_agent = task_state.subtask_agents[task_state.subtask_index]
                agent_task_metrics = current_agent.get_task_metrics(task.name)
                search_queries = agent_task_metrics.get("searches", [])

        if "sent_searches" not in self.worker.task_metrics[task_id]:
            self.worker.task_metrics[task_id]["sent_searches"] = set()

        for query in search_queries:
            if (
                query
                and query not in self.worker.task_metrics[task_id]["sent_searches"]
            ):
                self.worker.redis_client.send_search(
                    task_id, query, results=4, sources=["orchestrator_search"]
                )
                self.worker.task_metrics[task_id]["sent_searches"].add(query)
                logger.info("Sent search update: %s", query)

        if has_meaningful_activity and (
            activity_type in ["summary", "checkpoint"] or (notes and len(notes) > 200)
        ):
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task.name
            ).strip()
            notes_path = f"notes/{safe_name.upper()}.txt"
            with open(notes_path, "w", encoding="utf-8") as fh:
                fh.write(notes)

            self.worker.redis_client.send_report(
                task_id,
                f"Research Update - {task.name}",
                notes,
                metadata={
                    "progress": progress,
                    "action_count": self.worker.task_metrics[task_id].get(
                        "action_count", 0
                    ),
                    "activity_type": activity_type,
                },
            )

        all_subtasks_completed = self._all_subtasks_completed(progress_info)
        should_stop = False
        stop_reason = ""
        max_consecutive_timeouts = max(
            1,
            int(os.getenv("WORKER_MAX_CONSECUTIVE_TIMEOUTS", "3")),
        )
        timeout_policy = os.getenv("WORKER_TIMEOUT_POLICY", "recover").strip().lower()
        if step_timed_out:
            consecutive_timeouts = (
                self.worker.task_metrics[task_id].get("consecutive_step_timeouts", 0)
                + 1
            )
            self.worker.task_metrics[task_id][
                "consecutive_step_timeouts"
            ] = consecutive_timeouts
            self.worker.redis_client.send_activity(
                task_id,
                "timeout",
                (
                    "⏱️ Research step timed out; continuing. "
                    f"({consecutive_timeouts}/{max_consecutive_timeouts})"
                ),
            )
            if consecutive_timeouts >= max_consecutive_timeouts:
                logger.warning(
                    "Consecutive step timeout threshold reached for task %s (%s)",
                    task_id,
                    consecutive_timeouts,
                )
                if timeout_policy == "complete" and all_subtasks_completed:
                    logger.warning(
                        "Applying timeout completion policy for task %s", task_id
                    )
                    task.status = TaskStatus.COMPLETE
                else:
                    if timeout_policy == "complete":
                        logger.warning(
                            "Timeout completion policy requested for task %s but subtasks are incomplete; attempting recovery instead",
                            task_id,
                        )
                    recovery_note = self._recover_after_timeout_threshold(task)
                    recovery_cycles = (
                        self.worker.task_metrics[task_id].get(
                            "timeout_recovery_cycles", 0
                        )
                        + 1
                    )
                    self.worker.task_metrics[task_id][
                        "timeout_recovery_cycles"
                    ] = recovery_cycles
                    self.worker.redis_client.send_activity(
                        task_id,
                        "timeout_recovery",
                        (
                            "🔁 Timeout threshold reached; attempting recovery and continuing. "
                            f"{recovery_note}"
                        ),
                    )
                    max_recovery_cycles = max(
                        1,
                        int(os.getenv("WORKER_MAX_TIMEOUT_RECOVERY_CYCLES", "2")),
                    )
                    if recovery_cycles >= max_recovery_cycles:
                        escalation_note = self._force_progress_after_timeout_exhaustion(
                            task,
                            task_id,
                        )
                        self.worker.redis_client.send_activity(
                            task_id,
                            "timeout_escalation",
                            (
                                "⚠️ Repeated timeout recoveries exceeded threshold; "
                                f"escalating progression. {escalation_note}"
                            ),
                        )
                        self.worker.task_metrics[task_id]["timeout_recovery_cycles"] = 0
                    self.worker.task_metrics[task_id]["consecutive_step_timeouts"] = 0
        else:
            self.worker.task_metrics[task_id]["consecutive_step_timeouts"] = 0
            self.worker.task_metrics[task_id]["timeout_recovery_cycles"] = 0

        if task.status == TaskStatus.COMPLETE:
            should_stop = True
            stop_reason = "task_complete"
        elif all_subtasks_completed:
            # Subtasks may complete before orchestrator final report synthesis
            # finishes. Give the orchestrator a short grace window to finalize.
            wait_cycles = (
                self.worker.task_metrics[task_id].get("finalization_wait_cycles", 0) + 1
            )
            self.worker.task_metrics[task_id]["finalization_wait_cycles"] = wait_cycles
            raw_max_wait_cycles = int(
                os.getenv("WORKER_FINALIZATION_GRACE_CYCLES", "3")
            )
            unlimited_finalization_wait = raw_max_wait_cycles <= 0
            max_wait_cycles = max(1, raw_max_wait_cycles)
            if (not unlimited_finalization_wait) and wait_cycles >= max_wait_cycles:
                logger.warning(
                    "Task %s subtasks are complete but orchestrator finalization did not settle after %s cycles; forcing completion",
                    task_id,
                    wait_cycles,
                )
                task.status = TaskStatus.COMPLETE
                should_stop = True
                stop_reason = "task_complete"
            else:
                logger.info(
                    "Task %s subtasks complete; waiting for orchestrator finalization (%s/%s)",
                    task_id,
                    wait_cycles,
                    "unbounded" if unlimited_finalization_wait else max_wait_cycles,
                )
                self.worker.redis_client.send_activity(
                    task_id,
                    "finalizing",
                    "🧾 Finalizing consolidated report before task completion.",
                )
        elif self._hard_budget_enforced() and progress >= 100.0:
            should_stop = True
            stop_reason = "budget_exhausted"
            self.worker.task_metrics[task_id]["finalization_wait_cycles"] = 0
        else:
            self.worker.task_metrics[task_id]["finalization_wait_cycles"] = 0

        return {
            "elapsed": elapsed,
            "progress": display_progress,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
        }

    def _recover_after_timeout_threshold(self, task: Task) -> str:
        """
        Best-effort recovery when repeated step timeouts occur.
        Default strategy recreates current subtask agent to clear stale state.
        """
        try:
            task_state = self.worker.orchestrator.task_states.get(task.name)
            if task_state is None or not task_state.research_plan:
                return "No active task state to recover."

            subtask_index = int(getattr(task_state, "subtask_index", 0))
            if subtask_index >= len(task_state.research_plan.subtasks):
                return "No active subtask to recover."

            # Drop and recreate the current subtask agent.
            task_state.subtask_agents.pop(subtask_index, None)
            task_state.create_subtask_agent(
                subtask_index,
                self.worker.llm,
                self.worker.mem,
                self.worker.orchestrator.mcp_manager
                if getattr(self.worker.orchestrator, "use_mcp", False)
                else None,
            )
            return f"Reset subtask agent #{subtask_index + 1}."
        except Exception as e:
            logger.warning("Timeout recovery failed for task %s: %s", task.name, e)
            return f"Recovery failed: {e}"

    def _force_progress_after_timeout_exhaustion(self, task: Task, task_id: str) -> str:
        """
        Escalation path when timeout recovery repeatedly fails to restore progress.
        Advances to the next subtask, or marks final subtask complete.
        """
        try:
            task_state = self.worker.orchestrator.task_states.get(task.name)
            if task_state is None or not task_state.research_plan:
                return "No active task state available for escalation."

            total_subtasks = len(task_state.research_plan.subtasks)
            if total_subtasks <= 0:
                return "No subtasks available for escalation."

            current_index = int(getattr(task_state, "subtask_index", 0))
            if current_index >= total_subtasks:
                return "Task already beyond final subtask."

            task_state.subtask_reports[
                current_index
            ] = "Auto-advanced after repeated timeout recoveries exceeded threshold."
            task_state.mark_subtask_complete(current_index)

            if task_state.subtask_index < total_subtasks:
                task_state.create_subtask_agent(
                    task_state.subtask_index,
                    self.worker.llm,
                    self.worker.mem,
                    self.worker.orchestrator.mcp_manager
                    if getattr(self.worker.orchestrator, "use_mcp", False)
                    else None,
                )
                self.worker.send_research_plan_update(task_id, task.name)
                return (
                    f"Moved to subtask {task_state.subtask_index + 1}/{total_subtasks}."
                )

            task.status = TaskStatus.COMPLETE
            self.worker.send_research_plan_update(task_id, task.name)
            return "Marked final subtask complete after repeated timeout recoveries."
        except Exception as e:
            logger.warning("Timeout escalation failed for task %s: %s", task.name, e)
            return f"Escalation failed: {e}"

    def _route_after_emit(self, state: WorkerStepGraphState) -> str:
        return "stop" if state.get("should_stop") else "done"

    def _stop_task(self, state: WorkerStepGraphState) -> dict[str, Any]:
        stop_reason = (state.get("stop_reason") or "").strip().lower()
        completed = stop_reason == "task_complete"
        self.worker.stop_task(completed=completed, reason=stop_reason)
        return {}
