# model_loop.py
"""
KestrelAI Research Agent with Redis Integration — v2 (Ollama-first, provider-agnostic)

- Rewired to LangGraph orchestrator (single-pass per step) + ResearchAgentRunner (burst actions).
- Synthesizes **Markdown** research memos (human-readable) and deduplicates to avoid spam.
- Redis wire formats preserved.
"""

from __future__ import annotations

import os
import json
import time
import pathlib
from datetime import datetime
from typing import Dict, List, Optional, Any
import redis
import logging

from shared.models import Task, TaskStatus
from memory.vector_store import MemoryStore
from agents.base import LlmWrapper

from agents.orchestrator import LangGraphOrchestrator, StepBudget
from agents.research_agents import ResearchAgentRunner

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Env
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

DEFAULT_SLICE_MIN = int(os.getenv("KESTREL_SLICE_MIN", 15))
DEFAULT_BURST_ACTIONS = int(os.getenv("KESTREL_BURST_ACTIONS", 6))

# -----------------------------------------------------------------------------
# LC-compatible adapter for LlmWrapper (structured output)
# -----------------------------------------------------------------------------
class _LCCompatStructured:
    def __init__(self, llm: LlmWrapper, schema):
        self.llm = llm
        self.schema = schema
        self.parser = PydanticOutputParser(pydantic_object=schema)

    def invoke(self, messages: List[Any]):
        wire: List[Dict[str, str]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                wire.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                wire.append({"role": "user", "content": m.content})
            elif isinstance(m, dict) and "role" in m and "content" in m:
                wire.append(m)
        fmt = self.parser.get_format_instructions()
        wire.append({"role": "system", "content": f"Follow these FORMAT INSTRUCTIONS exactly:\n{fmt}"})
        text = self.llm.chat(wire)
        try:
            return self.parser.parse(text)
        except Exception:
            import re as _re, json as _json
            m = _re.search(r"\{.*\}", text, _re.S)
            data = _json.loads(m.group(0)) if m else {}
            return self.schema(**data)

class LCCompatModel:
    def __init__(self, llm: LlmWrapper):
        self.llm = llm
    def with_structured_output(self, schema):
        return _LCCompatStructured(self.llm, schema)

# -----------------------------------------------------------------------------
# Redis client
# -----------------------------------------------------------------------------
class KestrelRedisClient:
    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT, db: int = REDIS_DB):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.task_id: Optional[str] = None
        self.task_config: Dict[str, Any] = {}

    def get_next_command(self, timeout: int = 1) -> Optional[Dict[str, Any]]:
        if self.task_id:
            queue = f"kestrel:queue:commands:{self.task_id}"
            result = self.redis.brpop(queue, timeout=timeout)
            if result:
                _, data = result
                return json.loads(data)

        result = self.redis.brpop("kestrel:queue:commands", timeout=timeout)
        if result:
            _, data = result
            command = json.loads(data)
            if not self.task_id or command.get("taskId") == self.task_id:
                return command
        return None

    def send_update(self, task_id: str, **kwargs):
        update = {"taskId": task_id, "timestamp": int(time.time() * 1000), **kwargs}
        self.redis.lpush("kestrel:queue:updates", json.dumps(update))
        self._update_task_state(task_id, update)

    def _update_task_state(self, task_id: str, updates: Dict[str, Any]):
        key = f"kestrel:task:{task_id}:state"
        current = self.redis.get(key)
        task = json.loads(current) if current else {}
        task.update(updates)
        task["updatedAt"] = int(time.time() * 1000)
        self.redis.set(key, json.dumps(task))

    def send_activity(self, task_id: str, activity_type: str, message: str):
        activity = {
            "taskId": task_id,
            "type": activity_type,
            "message": message,
            "timestamp": int(time.time() * 1000),
            "time": str(int(time.time() * 1000)),
        }
        self.redis.lpush("kestrel:queue:activities", json.dumps(activity))

    def checkpoint(self, task_id: str, state: Dict[str, Any]):
        self.redis.set(f"kestrel:task:{task_id}:checkpoint", json.dumps(state))

    def restore_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        data = self.redis.get(f"kestrel:task:{task_id}:checkpoint")
        return json.loads(data) if data else None

    def send_report(self, task_id: str, title: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        report = {
            "taskId": task_id,
            "title": title,
            "content": content,
            "metadata": metadata or {},
            "timestamp": int(time.time() * 1000),
        }
        self.redis.lpush("kestrel:queue:reports", json.dumps(report))

    def send_search(self, task_id: str, query: str, results: int, sources: List[str]):
        payload = {
            "taskId": task_id,
            "query": query,
            "results": results,
            "sources": sources,
            "timestamp": int(time.time() * 1000),
            "time": str(time.time() * 1000),
        }
        self.redis.lpush("kestrel:queue:searches", json.dumps(payload))


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------
class KestrelAgentWorker:
    def __init__(self):
        self.redis_client = KestrelRedisClient()

        self.mem = MemoryStore()
        self.llm = LlmWrapper(model="gemma3:12b")  # Ollama-backed

        # LC-compatible adapters over our LlmWrapper
        self.policy_model = LCCompatModel(self.llm)
        self.planner_model = LCCompatModel(self.llm)
        self.evaluator_model = LCCompatModel(self.llm)  # Added for orchestrator v3

        self.runner: Optional[ResearchAgentRunner] = None
        self.orch: Optional[LangGraphOrchestrator] = None

        self.running = False
        self.paused = False
        self.current_task_id: Optional[str] = None
        self.current_task_config: Dict[str, Any] = {}

        self._orch_task_id: Optional[str] = None
        self._orch_subtask_id: Optional[str] = None
        self._orch_agent_id: Optional[str] = None

        self.task_metrics: Dict[str, Dict[str, Any]] = {}
        self.global_metrics = {
            "total_llm_calls": 0,
            "total_searches": 0,
            "total_summaries": 0,
            "total_checkpoints": 0,
            "total_web_fetches": 0,
        }

        self._last_update_fingerprint: Dict[str, str] = {}  # dedupe reports
        pathlib.Path("notes").mkdir(exist_ok=True)

    # Loop
    def run(self):
        logger.info("KestrelAI Agent Worker started")
        while True:
            command = self.redis_client.get_next_command(timeout=1)
            if command:
                self.handle_command(command)

            if self.running and not self.paused and self.current_task_id:
                self.process_task_step()

            time.sleep(0.1)

    # Commands
    def handle_command(self, command: Dict[str, Any]):
        cmd_type = command.get("type")
        task_id = command.get("taskId")
        payload = command.get("payload", {})

        logger.info(f"Handling command: {cmd_type} for task {task_id}")

        if cmd_type == "start":
            self.start_task(task_id, payload)
        elif cmd_type == "pause":
            self.pause_task()
        elif cmd_type == "resume":
            self.resume_task()
        elif cmd_type == "stop":
            self.stop_task()
        elif cmd_type == "update_config":
            self.update_config(task_id, payload)

    def start_task(self, task_id: str, config: Dict[str, Any]):
        self.current_task_id = task_id
        self.current_task_config = config or {}

        task = Task(
            name=config.get("name", "Research Task"),
            description=config.get("description", ""),
            budgetMinutes=config.get("budgetMinutes", 180),
            status=TaskStatus.ACTIVE,
        )
        self.tasks: Dict[str, Task] = {task_id: task}

        # Initialize research agent runner with evaluator model for v4
        self.runner = ResearchAgentRunner(
            memory=self.mem,
            llm=self.llm,
            agent_id="agent-1",
            planner_model=self.planner_model,
            evaluator_model=self.evaluator_model,  # Added for v4
        )
        
        # Initialize orchestrator with all required models and llm_wrapper
        self.orch = LangGraphOrchestrator(
            agent_runner=self.runner,
            policy_model=self.policy_model,
            planner_model=self.planner_model,
            evaluator_model=self.evaluator_model,
            llm_wrapper=self.llm,  # Now passing llm_wrapper for analysis/synthesis
        )

        self._orch_task_id = self.orch.create_task(
            task.name, 
            task.description,
            objectives=config.get("objectives", []),
            constraints=config.get("constraints", []),
            priority=int(config.get("priority", 3))
        )
        self._orch_agent_id = self.orch.add_agent(role="researcher", model=os.getenv("KESTREL_LC_MODEL"))
        
        # For v3 orchestrator, subtasks are created automatically by the planner
        # but we can add a manual one if needed
        if config.get("subtaskTitle"):
            self._orch_subtask_id = self.orch.add_subtask(
                task_id=self._orch_task_id,
                title=config.get("subtaskTitle", "Main Research"),
                acceptance_criteria=config.get("acceptanceCriteria", []),
                owner_agent_id=self._orch_agent_id,
            )

        self.orch.set_budget(
            max_actions=int(config.get("maxActions", DEFAULT_BURST_ACTIONS)),
            max_searches=int(config.get("maxSearches", 10)),
            max_summaries=int(config.get("maxSummaries", 3))
        )

        self.task_metrics[task_id] = {
            "search_count": 0,
            "think_count": 0,
            "summary_count": 0,
            "checkpoint_count": 0,
            "action_count": 0,
            "searches": [],
            "start_time": time.time(),
        }

        self.running = True
        self.paused = False

        self.redis_client.send_update(task_id, status=TaskStatus.ACTIVE.name, progress=0.0)
        self.redis_client.send_activity(task_id, "task_start", f"🦅 Starting research: {task.name}")
        logger.info(f"Started task {task_id}: {task.name}")

    def pause_task(self):
        if self.running and not self.paused and self.current_task_id:
            self.paused = True
            task = self.tasks.get(self.current_task_id)
            if task:
                task.status = TaskStatus.PAUSED

            checkpoint_state = {
                "config": self.current_task_config,
                "metrics": self.task_metrics.get(self.current_task_id, {}),
                "global_metrics": self.global_metrics,
            }
            self.redis_client.checkpoint(self.current_task_id, checkpoint_state)

            self.redis_client.send_update(self.current_task_id, status=TaskStatus.PAUSED.name)
            self.redis_client.send_activity(self.current_task_id, "task_pause", "⏸️ Task paused")
            logger.info(f"Paused task {self.current_task_id}")

    def resume_task(self):
        if self.running and self.paused and self.current_task_id:
            checkpoint = self.redis_client.restore_checkpoint(self.current_task_id)
            if checkpoint:
                self.task_metrics[self.current_task_id] = checkpoint.get("metrics", self.task_metrics.get(self.current_task_id, {}))
                self.global_metrics = checkpoint.get("global_metrics", self.global_metrics)

            self.paused = False
            task = self.tasks.get(self.current_task_id)
            if task:
                task.status = TaskStatus.ACTIVE

            self.redis_client.send_update(self.current_task_id, status=TaskStatus.ACTIVE.name)
            self.redis_client.send_activity(self.current_task_id, "task_resume", "▶️ Task resumed")
            logger.info(f"Resumed task {self.current_task_id}")

    def stop_task(self):
        if self.running and self.current_task_id:
            ext_id = self.current_task_id
            task = self.tasks.get(ext_id)
            if task:
                task.status = TaskStatus.COMPLETE

            self.generate_final_report()

            self.redis_client.send_update(
                ext_id,
                status=TaskStatus.COMPLETE.name,
                progress=100.0,
                elapsed=int(time.time() - self.task_metrics.get(ext_id, {}).get("start_time", time.time())),
            )
            self.redis_client.send_activity(ext_id, "task_complete", "✅ Task completed")

            self.running = False
            self.paused = False
            self.current_task_id = None
            self.current_task_config = {}
            logger.info(f"Stopped task {ext_id}")

    def update_config(self, task_id: str, config: Dict[str, Any]):
        if task_id == self.current_task_id:
            self.current_task_config.update(config)
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.name = config.get("name", task.name)
                task.description = config.get("description", task.description)
                task.budgetMinutes = config.get("budgetMinutes", task.budgetMinutes)
            logger.info(f"Updated config for task {task_id}")

    # Processing
    def process_task_step(self):
        if not self.orch or not self.current_task_id:
            return

        ext_id = self.current_task_id
        task = self.tasks.get(ext_id)
        if not task:
            return

        self.orch.step()
    
        # logger.error(f"Orchestrator step failed: {e}")
        # self.redis_client.send_activity(ext_id, "error", f"⚠️ Error: {str(e)[:100]}")
        # return

        updates = self.orch.get_updates()
        decisions = self.orch.get_decisions()
        last_update = updates[-1] if updates else None
        last_decision = decisions[-1] if decisions else None

        # Enhanced activity detection for v3 orchestrator
        activity_type = "analysis"
        message = "⚙️ Processing"
        
        if last_decision:
            decision_type = last_decision.decision_type if hasattr(last_decision, 'decision_type') else last_decision.get('decision_type')
            if decision_type == "assign":
                message = "📋 Assigning subtask"
            elif decision_type == "plan_created":
                message = "📝 Creating research plan"
            elif decision_type in ["continue", "refine"]:
                message = "🔄 Refining approach"
            elif decision_type == "evaluate":
                message = "✓ Evaluating quality"
                
        if last_update:
            if last_update.sources:
                activity_type = "search"
                message = f"🔍 Found {len(last_update.sources)} sources"
            elif hasattr(last_update, 'key_insights') and last_update.key_insights:
                activity_type = "insight"
                message = f"💡 {last_update.key_insights[0][:80]}..."
            elif last_update.blockers:
                activity_type = "thinking"
                message = f"🤔 {last_update.blockers[0]}"

        self.task_metrics[ext_id]["action_count"] += 1
        self.redis_client.send_activity(ext_id, activity_type, message)

        # Calculate progress based on orchestrator state
        elapsed = time.time() - self.task_metrics[ext_id]["start_time"]
        
        # Get task status for better progress tracking
        if self._orch_task_id:
            task_status = self.orch.get_task_status(self._orch_task_id)
            if task_status:
                # Use orchestrator's progress if available
                completion_info = task_status.get("progress", "0/1")
                completed, total = 0, 1
                if "/" in completion_info:
                    parts = completion_info.split("/")
                    completed = int(parts[0]) if parts[0].isdigit() else 0
                    total = int(parts[1].split()[0]) if parts[1].split()[0].isdigit() else 1
                
                if total > 0:
                    progress = (completed / total) * 100
                else:
                    progress = min(100.0, (elapsed / (task.budgetMinutes * 60.0)) * 100.0)
            else:
                progress = min(100.0, (elapsed / (task.budgetMinutes * 60.0)) * 100.0)
        else:
            progress = min(100.0, (elapsed / (task.budgetMinutes * 60.0)) * 100.0)

        if self.runner and hasattr(self.runner, "get_metrics"):
            self.global_metrics.update(self.runner.get_metrics())

        metrics_update = {
            "searchCount": self.global_metrics.get("total_searches", 0),
            "thinkCount": self.global_metrics.get("total_thoughts", 0),
            "summaryCount": self.global_metrics.get("total_summaries", 0),
            "checkpointCount": self.global_metrics.get("total_checkpoints", 0),
            "llmTokensUsed": self.global_metrics.get("total_llm_calls", 0) * 1000,
            "analysisCount": self.global_metrics.get("total_analyses", 0),
            "evidenceCount": self.global_metrics.get("total_evidence", 0),
        }

        logger.info(f"Metrics snapshot: {metrics_update}")
        self.redis_client.send_update(
            ext_id,
            status=(task.status.name if hasattr(task, "status") else TaskStatus.ACTIVE.name),
            progress=progress,
            elapsed=int(elapsed),
            metrics=metrics_update,
        )

        # Handle search tracking with v4 agent improvements
        if last_update and last_update.proposed_next:
            for proposal in last_update.proposed_next[:3]:  # Limit to 3 proposals
                if proposal.startswith("Search for evidence on:"):
                    query = proposal[len("Search for evidence on:"):].strip()
                    self.redis_client.send_search(ext_id, query, results=0, sources=["proposed"])
                elif proposal.startswith("Next query:"):
                    query = proposal[len("Next query:"):].strip()
                    self.redis_client.send_search(ext_id, query, results=0, sources=["planned"])

        # Synthesized Markdown report
        report_md = ""
        if last_update:
            try:
                task_spec = self.orch.tasks.get(last_update.task_id)
                sub_spec = self.orch.plans.subtasks.get(last_update.subtask_id)
                if task_spec and sub_spec:
                    report_md = self.runner.build_markdown_report(task_spec, sub_spec)
            except Exception as e:
                logger.error(f"Report synthesis failed: {e}")

        if report_md:
            fp = str(hash(report_md))
            if self._last_update_fingerprint.get(ext_id) != fp:
                self._last_update_fingerprint[ext_id] = fp

                notes_path = self._notes_path(task.name)
                with open(notes_path, "a", encoding="utf-8") as fh:
                    fh.write(report_md + "\n\n")

                # Extract confidence if available
                confidence = None
                if last_update and hasattr(last_update, 'confidence'):
                    confidence = last_update.confidence

                self.redis_client.send_report(
                    ext_id,
                    f"Research Update - {task.name}",
                    report_md,
                    metadata={
                        "progress": progress,
                        "action_count": self.task_metrics[ext_id]["action_count"],
                        "confidence": confidence,
                        "phase": self.orch._base_state.get("current_phase", "unknown") if self.orch else "unknown"
                    },
                )

        # Check if task is complete based on orchestrator state
        if self._orch_task_id and self.orch:
            orch_task = self.orch.tasks.get(self._orch_task_id)
            if orch_task and orch_task.status.value == "complete":
                self.stop_task()
        elif progress >= 100.0:
            self.stop_task()

    # Reporting
    def _notes_path(self, task_name: str) -> str:
        safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task_name).strip().upper()
        return f"notes/{safe}.txt"

    def generate_final_report(self):
        if not self.current_task_id:
            return
        ext_id = self.current_task_id
        task = self.tasks.get(ext_id)
        if not task:
            return

        notes_file = self._notes_path(task.name)
        if pathlib.Path(notes_file).exists():
            with open(notes_file, "r", encoding="utf-8") as fh:
                content = fh.read()
        else:
            content = "No research notes available."

        metrics = self.task_metrics.get(ext_id, {})
        
        # Get final task summary from orchestrator if available
        task_summary = ""
        if self._orch_task_id and self.orch:
            try:
                task_summary = self.orch.get_research_summary(self._orch_task_id)
            except Exception as e:
                logger.error(f"Failed to get task summary: {e}")

        report_content = f"""# Research Report: {task.name}

## Summary
**Task:** {task.description}
**Duration:** {task.budgetMinutes} minutes budget
**Status:** Complete

## Metrics
- Total Searches: {self.global_metrics.get('total_searches', 0)}
- Analysis Steps: {self.global_metrics.get('total_thoughts', 0)}
- Summaries Created: {self.global_metrics.get('total_summaries', 0)}
- Checkpoints: {self.global_metrics.get('total_checkpoints', 0)}
- Evidence Evaluated: {self.global_metrics.get('total_evidence', 0)}
- Total Actions: {metrics.get('action_count', 0)}

## Executive Summary
{task_summary if task_summary else "See research findings below."}

## Research Findings
{content}

---
*Report generated at {datetime.now().isoformat()}*
"""
        self.redis_client.send_report(
            ext_id,
            f"Final Report - {task.name}",
            report_content,
            metadata={"final": True, "metrics": metrics},
        )

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    worker = KestrelAgentWorker()
    worker.run()

if __name__ == "__main__":
    main()