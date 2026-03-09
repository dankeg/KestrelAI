"""
LangGraph-powered orchestrator runner for task-level control flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from KestrelAI.graphs.persistence import get_langgraph_runtime

if TYPE_CHECKING:
    from KestrelAI.agents.research_orchestrator import ResearchOrchestrator
    from KestrelAI.shared.models import Task
else:  # Runtime aliases so LangGraph can resolve TypedDict annotations
    ResearchOrchestrator = Any
    Task = Any

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - dependency-gated path
    END = START = StateGraph = None


class OrchestratorGraphState(TypedDict):
    task: Task
    status: str
    latest_notes: str
    decision: Any
    result: str


class LangGraphOrchestratorRunner:
    """Executes orchestrator control flow using LangGraph."""

    def __init__(self, orchestrator: ResearchOrchestrator):
        if StateGraph is None:
            raise ImportError("langgraph is not installed")
        self.orchestrator = orchestrator
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(OrchestratorGraphState)
        workflow.add_node("run_subtask_research", self._run_subtask_research)
        workflow.add_node("review_decision", self._review_decision)
        workflow.add_node("apply_decision", self._apply_decision)
        workflow.add_node("finalize_no_more_subtasks", self._finalize_no_more_subtasks)

        workflow.add_edge(START, "run_subtask_research")
        workflow.add_conditional_edges(
            "run_subtask_research",
            self._route_after_subtask,
            {
                "finalize": "finalize_no_more_subtasks",
                "review": "review_decision",
            },
        )
        workflow.add_edge("review_decision", "apply_decision")
        workflow.add_edge("apply_decision", END)
        workflow.add_edge("finalize_no_more_subtasks", END)
        checkpointer, store = get_langgraph_runtime()
        compile_kwargs = {}
        if checkpointer is not None:
            compile_kwargs["checkpointer"] = checkpointer
        if store is not None:
            compile_kwargs["store"] = store
        return workflow.compile(**compile_kwargs)

    async def run(self, task: Task) -> str:
        initial_state: OrchestratorGraphState = {
            "task": task,
            "status": "in_progress",
            "latest_notes": "",
            "decision": None,
            "result": "",
        }
        run_config = {
            "configurable": {
                "thread_id": f"orchestrator:{task.name}",
                "checkpoint_ns": "orchestrator_runner",
            }
        }
        output = await self.graph.ainvoke(initial_state, config=run_config)
        return output.get("result", "")

    async def _run_subtask_research(
        self, state: OrchestratorGraphState
    ) -> dict[str, Any]:
        task = state["task"]
        research_result = await self.orchestrator.run_subtask_research(task)
        status = research_result.get("status", "in_progress")
        latest_notes = research_result.get("result", "")
        return {"status": status, "latest_notes": latest_notes}

    def _route_after_subtask(self, state: OrchestratorGraphState) -> str:
        if state.get("status") == "no_more_subtasks":
            return "finalize"
        return "review"

    async def _review_decision(self, state: OrchestratorGraphState) -> dict[str, Any]:
        task = state["task"]
        latest_notes = state.get("latest_notes", "")
        decision = await self.orchestrator._review(task, latest_notes)
        task_state = self.orchestrator.task_states[task.name]
        task_state.record_decision(decision.decision, decision.feedback)
        return {"decision": decision}

    async def _apply_decision(self, state: OrchestratorGraphState) -> dict[str, Any]:
        task = state["task"]
        latest_notes = state.get("latest_notes", "")
        decision = state.get("decision")
        result = await self.orchestrator._apply_review_decision(
            task, latest_notes, decision
        )
        return {"result": result}

    async def _finalize_no_more_subtasks(
        self, state: OrchestratorGraphState
    ) -> dict[str, Any]:
        task = state["task"]
        result = await self.orchestrator._finalize_when_no_more_subtasks(task)
        return {"result": result}
