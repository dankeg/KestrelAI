"""
KestrelAI Research Agent with Redis Integration
Extracted from Panel UI to work with Redis queues
"""

import os
import json
import time
import pathlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
try:
    from KestrelAI.shared.models import Task, TaskStatus
    from KestrelAI.memory.vector_store import MemoryStore
    from KestrelAI.agents.base import LlmWrapper
    from KestrelAI.agents.web_research_agent import WebResearchAgent, ResearchConfig
    from KestrelAI.agents.research_orchestrator import ResearchOrchestrator
    from KestrelAI.shared.redis_utils import (
        RedisConfig, get_sync_redis_client
    )
except ImportError:
    # Fallback for different import contexts (Docker, local, etc.)
    from shared.models import Task, TaskStatus
    from memory.vector_store import MemoryStore
    from agents.base import LlmWrapper
    from agents.web_research_agent import WebResearchAgent, ResearchConfig
    from agents.research_orchestrator import ResearchOrchestrator
    from shared.redis_utils import (
        RedisConfig, get_sync_redis_client
    )

from copy import deepcopy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Redis configuration
REDIS_CONFIG = RedisConfig(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


# -----------------------------------------------------------------------------
# KestrelAI Research Agent Worker
# -----------------------------------------------------------------------------
class KestrelAgentWorker:
    """Main agent worker that processes research tasks"""

    def __init__(self):
        # Initialize Redis client
        self.redis_client = get_sync_redis_client(REDIS_CONFIG)

        # Initialize AI components
        self.mem = MemoryStore()
        # Use environment variable for Ollama host, default to localhost for local development
        # CRITICAL: On macOS, use local Ollama for performance, not Docker Ollama
        ollama_host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm = LlmWrapper(model="gemma3:27b", host=ollama_host)
        self.agent = WebResearchAgent("main-agent", self.llm, self.mem)

        # State management
        self.running = False
        self.paused = False
        self.current_task_id = None
        self.current_task_config = {}
        self.tasks: Dict[str, Task] = {}
        self.orchestrator = None

        # Settings management
        self.app_settings = {
            "ollamaMode": "local",
            "orchestrator": "kestrel"
        }

        self.latest_feedback = "No Feedback Yet!"
        self.latest_subtask = ""

        # Metrics tracking
        self.task_metrics: Dict[str, Dict[str, Any]] = {}
        self.global_metrics = {
            "total_llm_calls": 0,
            "total_searches": 0,
            "total_summaries": 0,
            "total_checkpoints": 0,
            "total_web_fetches": 0,
        }

        # Ensure notes directory exists
        pathlib.Path("notes").mkdir(exist_ok=True)

    def run(self):
        """Main agent loop"""
        logger.info("KestrelAI Agent Worker started")

        while True:
            # try:
            # Check for commands
            command = self.redis_client.get_next_command(timeout=1)

            if command:
                self.handle_command(command)

            # Process active task
            if self.running and not self.paused and self.current_task_id:
                self.process_task_step()
            else:
                logger.debug(f"Not processing task: running={self.running}, paused={self.paused}, current_task_id={self.current_task_id}")

            time.sleep(0.5)  # Reasonable frequency for research processing

    def handle_command(self, command: Dict[str, Any]):
        """Handle command from backend"""
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
        elif cmd_type == "update_settings":
            self.update_settings(payload)

    def start_task(self, task_id: str, config: Dict[str, Any]):
        """Start a new research task"""
        self.current_task_id = task_id
        self.task_id = task_id  # Set for Redis client
        self.current_task_config = config

        # Create Task object aligned to the actual model (budgetMinutes + enum status)
        task = Task(
            name=config.get("name", "Research Task"),
            description=config.get("description", ""),
            budgetMinutes=config.get("budgetMinutes", 180),
            status=TaskStatus.ACTIVE,
        )
        self.tasks[task_id] = task

        # Initialize orchestrator with task and settings
        logger.info(f"Initializing orchestrator for task {task_id}")
        orchestrator_profile = self.app_settings.get("orchestrator", "kestrel")
        self.orchestrator = ResearchOrchestrator([task], self.llm, profile=orchestrator_profile)
        logger.info(f"Orchestrator initialized for task {task_id}")
        
        # Initialize planning phase in background thread to avoid blocking main loop
        logger.info(f"Starting planning phase for task {task_id}")
        def run_planning():
            import asyncio
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.orchestrator._planning_phase(task))
                loop.close()
                logger.info(f"Planning phase completed for task {task_id}")
                # Send research plan update after completion
                self.send_research_plan_update(task_id, task.name)
            except Exception as e:
                logger.error(f"Error during planning phase: {e}", exc_info=True)
        
        planning_thread = threading.Thread(target=run_planning, daemon=True)
        planning_thread.start()
        logger.info(f"Planning phase started in background for task {task_id}")
        
        # Get initial subtask
        self.latest_subtask = self.orchestrator.get_current_subtask(task.name) or "Initial research"

        # Initialize task metrics
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

        # Send initial updates (status as enum value)
        self.redis_client.send_update(
            task_id, status=TaskStatus.ACTIVE.value, progress=0.0
        )
        self.redis_client.send_activity(
            task_id, "task_start", f"🦅 Starting research: {task.name}"
        )

        logger.info(f"Started task {task_id}: {task.name}")

    def pause_task(self):
        """Pause current task"""
        if self.running and not self.paused and self.current_task_id:
            self.paused = True

            # Update in-memory task status
            task = self.tasks.get(self.current_task_id)
            if task:
                task.status = TaskStatus.PAUSED

            # Save checkpoint
            checkpoint_state = {
                "config": self.current_task_config,
                "metrics": self.task_metrics.get(self.current_task_id, {}),
                "global_metrics": self.global_metrics,
            }
            self.redis_client.checkpoint(self.current_task_id, checkpoint_state)

            self.redis_client.send_update(
                self.current_task_id, status=TaskStatus.PAUSED.value
            )
            self.redis_client.send_activity(
                self.current_task_id, "task_pause", "⏸️ Task paused"
            )

            logger.info(f"Paused task {self.current_task_id}")

    def resume_task(self):
        """Resume paused task"""
        if self.running and self.paused and self.current_task_id:
            # Restore from checkpoint
            checkpoint = self.redis_client.restore_checkpoint(self.current_task_id)
            if checkpoint:
                self.task_metrics[self.current_task_id] = checkpoint.get("metrics", {})
                self.global_metrics = checkpoint.get(
                    "global_metrics", self.global_metrics
                )

            self.paused = False

            # Update in-memory task status
            task = self.tasks.get(self.current_task_id)
            if task:
                task.status = TaskStatus.ACTIVE

            self.redis_client.send_update(
                self.current_task_id, status=TaskStatus.ACTIVE.value
            )
            self.redis_client.send_activity(
                self.current_task_id, "task_resume", "▶️ Task resumed"
            )

            logger.info(f"Resumed task {self.current_task_id}")

    def stop_task(self):
        """Stop current task"""
        if self.running and self.current_task_id:
            task_id = self.current_task_id

            # Update in-memory task status
            task = self.tasks.get(task_id)
            if task:
                task.status = TaskStatus.COMPLETE

            # Generate final report
            self.generate_final_report()

            # Update status
            self.redis_client.send_update(
                task_id,
                status=TaskStatus.COMPLETE.value,
                progress=100.0,
                elapsed=int(
                    time.time()
                    - self.task_metrics.get(task_id, {}).get("start_time", time.time())
                ),
            )
            self.redis_client.send_activity(
                task_id, "task_complete", "✅ Task completed"
            )

            # Clean up
            self.running = False
            self.paused = False
            self.current_task_id = None
            self.current_task_config = {}

            logger.info(f"Stopped task {task_id}")

    def update_config(self, task_id: str, config: Dict[str, Any]):
        """Update task configuration"""
        if task_id == self.current_task_id:
            self.current_task_config.update(config)

            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.name = config.get("name", task.name)
                task.description = config.get("description", task.description)
                task.budgetMinutes = config.get("budgetMinutes", task.budgetMinutes)

            logger.info(f"Updated config for task {task_id}")

    def update_settings(self, settings: Dict[str, Any]):
        """Update application settings"""
        self.app_settings.update(settings)
        
        # Apply settings to components
        if "ollamaMode" in settings:
            # Update LLM wrapper based on Ollama mode
            # CRITICAL: On macOS, always prefer local Ollama for performance
            if settings["ollamaMode"] == "docker":
                # Use Docker Ollama endpoint (slower on macOS)
                ollama_host = "http://ollama:11434"
            else:
                # Use local Ollama endpoint (faster on macOS)
                ollama_host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            self.llm = LlmWrapper(model="gemma3:27b", host=ollama_host)
            
            # Update the agent with new LLM
            self.agent = WebResearchAgent("main-agent", self.llm, self.mem)
        
        if "orchestrator" in settings:
            # Update orchestrator behavior based on setting
            if self.orchestrator:
                orchestrator_profile = settings["orchestrator"]
                # Reinitialize orchestrator with new profile
                if self.current_task_id and self.current_task_id in self.tasks:
                    task = self.tasks[self.current_task_id]
                    self.orchestrator = ResearchOrchestrator([task], self.llm, profile=orchestrator_profile)
                    # Re-run planning phase with new configuration (async method)
                    try:
                        import asyncio
                        # Use existing event loop or create new one
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        try:
                            loop.run_until_complete(self.orchestrator._planning_phase(task))
                            self.latest_subtask = self.orchestrator.get_current_subtask(task.name) or "Initial research"
                            logger.info(f"Orchestrator reinitialized with profile: {orchestrator_profile}")
                        except Exception as e:
                            logger.error(f"Error reinitializing orchestrator: {e}")
                    except Exception as e:
                        logger.error(f"Error reinitializing orchestrator: {e}")
            logger.info(f"Orchestrator setting updated to: {settings['orchestrator']}")
        
        logger.info(f"Updated app settings: {settings}")

    def process_task_step(self):
        """Process one step of the research task"""
        if not self.orchestrator or not self.current_task_id:
            return

        task_id = self.current_task_id
        task = self.tasks.get(task_id)
        if not task:
            return

        # Wait for planning phase to complete before processing
        task_state = self.orchestrator.task_states.get(task.name)
        if not task_state or not task_state.research_plan:
            # Planning phase not complete yet, skip this iteration
            logger.debug(f"Waiting for planning phase to complete for task {task_id}")
            return
        
        logger.info(f"Processing research step for task {task_id}")

        # Execute research step using orchestrator's subtask agents
        # The orchestrator now handles the research internally with dedicated subtask agents
        try:
            # Run the async method properly
            import asyncio
            # Use existing event loop or create new one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            try:
                research_result = loop.run_until_complete(self.orchestrator.next_action(task))
                self.latest_feedback = research_result
                logger.info(f"Research step completed: {research_result[:100]}...")
            except Exception as e:
                logger.error(f"Error executing orchestrator step: {e}")
                research_result = f"Processing research step - {str(e)[:100]}"
                self.latest_feedback = research_result
        except Exception as e:
            logger.error(f"Error executing orchestrator step: {e}")
            research_result = f"Processing research step - {str(e)[:100]}"
            self.latest_feedback = research_result
        
        # Get the latest notes from the orchestrator's current subtask
        current_subtask = self.orchestrator.get_current_subtask(task.name)
        if current_subtask:
            self.latest_subtask = current_subtask
        
        # The research_result contains the actual research output with [SEARCH], [THOUGHT], etc.
        notes = research_result

        # Update local metrics with orchestrator's subtask agent metrics
        progress_info = self.orchestrator.get_task_progress(task.name)
        
        # Send research plan update if subtask progress has changed
        self.send_research_plan_update(task_id, task.name)
        
        # Aggregate metrics from all subtask agents
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
                total_actions += metrics.get("total_llm_calls", 0)
        
        self.task_metrics[task_id].update({
            "search_count": total_searches,
            "think_count": total_thinks,
            "summary_count": total_summaries,
            "checkpoint_count": total_checkpoints,
            "action_count": total_actions,
            "searches": [],  # Will be populated from subtask agents if needed
            "search_history": [],
            "current_focus": current_subtask or "Preparing research",
        })

        # Check for meaningful activity by examining the orchestrator's current subtask agent
        has_meaningful_activity = False
        activity_type = "analysis"
        message = f"⚙️ Working on research"
        
        if hasattr(self.orchestrator, 'task_states') and task.name in self.orchestrator.task_states:
            task_state = self.orchestrator.task_states[task.name]
            if task_state.subtask_index in task_state.subtask_agents:
                current_agent = task_state.subtask_agents[task_state.subtask_index]
                if task.name in current_agent._state:
                    agent_state = current_agent._state[task.name]
                    
                    # Check if there's new activity since last check
                    last_action_count = self.task_metrics[task_id].get("last_agent_action_count", 0)
                    if agent_state.action_count > last_action_count:
                        has_meaningful_activity = True
                        
                        # Determine activity type based on recent actions with concise messages
                        if agent_state.search_count > self.task_metrics[task_id].get("last_search_count", 0):
                            activity_type = "search"
                            message = f"🔍 Searching for opportunities"
                        elif agent_state.think_count > self.task_metrics[task_id].get("last_think_count", 0):
                            activity_type = "thinking"
                            message = f"🤔 Analyzing findings"
                        elif agent_state.summary_count > self.task_metrics[task_id].get("last_summary_count", 0):
                            activity_type = "summary"
                            message = f"📝 Summarizing findings"
                        elif agent_state.checkpoint_count > self.task_metrics[task_id].get("last_checkpoint_count", 0):
                            activity_type = "checkpoint"
                            message = f"💾 Saving progress"
                        
                        # Update tracking
                        self.task_metrics[task_id]["last_agent_action_count"] = agent_state.action_count
                        self.task_metrics[task_id]["last_search_count"] = agent_state.search_count
                        self.task_metrics[task_id]["last_think_count"] = agent_state.think_count
                        self.task_metrics[task_id]["last_summary_count"] = agent_state.summary_count
                        self.task_metrics[task_id]["last_checkpoint_count"] = agent_state.checkpoint_count
        
        # Send activity updates - either meaningful activity or periodic status updates
        if has_meaningful_activity:
            self.task_metrics[task_id]["action_count"] += 1
            self.redis_client.send_activity(task_id, activity_type, message)
        else:
            # Send periodic status updates to keep frontend informed
            # Only send every 10 seconds to avoid spam
            last_status_update = self.task_metrics[task_id].get("last_status_update", 0)
            if time.time() - last_status_update > 10:
                self.redis_client.send_activity(task_id, "status", message)
                self.task_metrics[task_id]["last_status_update"] = time.time()

        # Calculate progress (0..100) using budgetMinutes
        elapsed = time.time() - self.task_metrics[task_id]["start_time"]
        progress = min(100.0, (elapsed / (task.budgetMinutes * 60.0)) * 100.0)

        # Update metrics from agent if available
        if hasattr(self.agent, "get_global_metrics"):
            agent_metrics = self.agent.get_global_metrics()
            self.global_metrics.update(agent_metrics)

        # Send updates with consistent metrics structure
        metrics_update = {
            "searchCount": self.task_metrics[task_id]["search_count"],
            "thinkCount": self.task_metrics[task_id]["think_count"],
            "summaryCount": self.task_metrics[task_id]["summary_count"],
            "checkpointCount": self.task_metrics[task_id]["checkpoint_count"],
            "webFetchCount": self.global_metrics.get("total_web_fetches", 0),
            "llmTokensUsed": self.global_metrics.get("total_llm_calls", 0) * 1000,  # Estimate
            "errorCount": 0,  # Add error count tracking
        }
        # logger.error(f"Current task Metrics{ self.task_metrics}")

        print(f"Task Information: {task}, {task.status}")

        self.redis_client.send_update(
            task_id,
            status=(
                task.status.value
                if hasattr(task, "status")
                else TaskStatus.ACTIVE.value
            ),
            progress=progress,
            elapsed=int(elapsed),
            metrics=metrics_update,
        )

        # Collect search queries from orchestrator's current subtask agent
        search_queries = []
        if hasattr(self.orchestrator, 'task_states') and task.name in self.orchestrator.task_states:
            task_state = self.orchestrator.task_states[task.name]
            if task_state.subtask_index in task_state.subtask_agents:
                current_agent = task_state.subtask_agents[task_state.subtask_index]
                # Get task-specific metrics from the current agent
                agent_task_metrics = current_agent.get_task_metrics(task.name)
                search_queries = agent_task_metrics.get("searches", [])
        
        # Send search history to Redis (only new searches)
        # Initialize sent_searches set if it doesn't exist
        if "sent_searches" not in self.task_metrics[task_id]:
            self.task_metrics[task_id]["sent_searches"] = set()
        
        for query in search_queries:
            if query and query not in self.task_metrics[task_id]["sent_searches"]:
                self.redis_client.send_search(
                    task_id, query, results=4, sources=["orchestrator_search"]
                )
                self.task_metrics[task_id]["sent_searches"].add(query)
                logger.info(f"Sent search update: {query}")

        # Only send reports for actual research findings (not every loop iteration)
        # Send reports when there's meaningful research output (summaries, checkpoints, or substantial content)
        if has_meaningful_activity and (activity_type in ["summary", "checkpoint"] or 
                                      (notes and len(notes) > 200)):
            # Save notes to file
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task.name
            ).strip()
            notes_path = f"notes/{safe_name.upper()}.txt"
            with open(notes_path, "w", encoding="utf-8") as fh:
                fh.write(notes)

            # Send report update only for meaningful research findings
            self.redis_client.send_report(
                task_id,
                f"Research Update - {task.name}",
                notes,
                metadata={
                    "progress": progress,
                    "action_count": self.task_metrics[task_id]["action_count"],
                    "activity_type": activity_type,
                },
            )

        # Orchestrator already handled the research step above
        # No need to call next_action again here
        
        # Check if task is complete based on orchestrator state or time budget
        if (task.status == TaskStatus.COMPLETE or 
            progress >= 100.0 or 
            progress_info.get("progress", 0) >= 100.0):
            self.stop_task()

    def send_research_plan_update(self, task_id: str, task_name: str):
        """Send research plan update to Redis"""
        if not self.orchestrator or not hasattr(self.orchestrator, 'task_states'):
            logger.warning(f"Cannot send research plan update: orchestrator not initialized for task {task_id}")
            return
            
        task_state = self.orchestrator.task_states.get(task_name)
        if not task_state or not task_state.research_plan:
            logger.warning(f"Cannot send research plan update: no research plan for task {task_id}")
            return
            
        # Convert research plan to dict format
        try:
            research_plan_data = {
                "restated_task": task_state.research_plan.restated_task,
                "subtasks": [
                    {
                        "order": subtask.order,
                        "description": subtask.description,
                        "success_criteria": subtask.success_criteria,
                        "status": "completed" if i in task_state.completed_subtasks else 
                                 "in_progress" if i == task_state.subtask_index else "pending"
                    }
                    for i, subtask in enumerate(task_state.research_plan.subtasks)
                ],
                "current_subtask_index": task_state.subtask_index,
                "created_at": int(time.time() * 1000)
            }
            
            # Send research plan update
            self.redis_client.send_update(
                task_id, 
                research_plan=research_plan_data
            )
            
            logger.info(f"Sent research plan update for task {task_id} with {len(research_plan_data['subtasks'])} subtasks")
        except Exception as e:
            logger.error(f"Failed to send research plan update for task {task_id}: {e}")

    def generate_final_report(self):
        """Generate and send final report"""
        if not self.current_task_id:
            return

        task_id = self.current_task_id
        task = self.tasks.get(task_id)
        if not task:
            return

        metrics = self.task_metrics.get(task_id, {})

        # Read the latest notes
        safe_name = "".join(
            c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task.name
        ).strip()
        notes_file = f"notes/{safe_name.upper()}.txt"
        if pathlib.Path(notes_file).exists():
            with open(notes_file, "r", encoding="utf-8") as fh:
                content = fh.read()
        else:
            content = "No research notes available."

        # Create final report
        report_content = f"""# Research Report: {task.name}

## Task Summary
**Description:** {task.description}
**Duration:** {task.budgetMinutes} minutes budget
**Status:** Complete

## Research Metrics
- Total Searches: {metrics.get('search_count', 0)}
- Analysis Steps: {metrics.get('think_count', 0)}
- Summaries Created: {metrics.get('summary_count', 0)}
- Checkpoints: {metrics.get('checkpoint_count', 0)}

## Key Findings
{content}

---
*Report generated at {datetime.now().isoformat()}*"""

        self.redis_client.send_report(
            task_id,
            f"Final Report - {task.name}",
            report_content,
            metadata={"final": True, "metrics": metrics},
        )


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main():
    """Main entry point for the agent worker"""
    worker = KestrelAgentWorker()

    worker.run()


if __name__ == "__main__":
    main()
