"""
KestrelAI Research Agent with Redis Integration
Extracted from Panel UI to work with Redis queues
"""

import os
import json
import time
import pathlib
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
import redis
import logging
from shared.models import Task, TaskStatus
from memory.vector_store import MemoryStore
from agents.base import LlmWrapper
from agents.research_agents import SEARCH_RESULTS, ResearchAgent
from agents.orchestrator import Orchestrator
from shared.redis_utils import get_task, save_task, send_command, init_redis, close_redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# -----------------------------------------------------------------------------
# Redis Client for Agent Communication
# -----------------------------------------------------------------------------
class KestrelRedisClient:
    """Redis client for KestrelAI agent to communicate with backend"""

    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT, db: int = REDIS_DB):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.task_id: Optional[str] = None
        self.task_config: Dict[str, Any] = {}

    def get_next_command(self, timeout: int = 1) -> Optional[Dict[str, Any]]:
        """Get next command from queue"""
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
        """Send status update to backend"""
        update = {"taskId": task_id, "timestamp": int(time.time() * 1000), **kwargs}
        self.redis.lpush("kestrel:queue:updates", json.dumps(update))
        self._update_task_state(task_id, update)

    def _update_task_state(self, task_id: str, updates: Dict[str, Any]):
        """Update task state in Redis"""
        key = f"kestrel:task:{task_id}:state"
        current = self.redis.get(key)
        task = json.loads(current) if current else {}
        task.update(updates)
        task["updatedAt"] = int(time.time() * 1000)
        self.redis.set(key, json.dumps(task))

    def send_activity(self, task_id: str, activity_type: str, message: str):
        """Send activity log to backend"""
        activity = {
            "taskId": task_id,
            "type": activity_type,
            "message": message,
            "timestamp": int(time.time() * 1000),
            "time": str(int(time.time() * 1000))
        }
        self.redis.lpush("kestrel:queue:activities", json.dumps(activity))

    # The following helpers are referenced elsewhere in the worker; keep stubs if your backend handles them.
    def checkpoint(self, task_id: str, state: Dict[str, Any]):
        self.redis.set(f"kestrel:task:{task_id}:checkpoint", json.dumps(state))

    def restore_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        data = self.redis.get(f"kestrel:task:{task_id}:checkpoint")
        return json.loads(data) if data else None

    def send_report(self, task_id: str, title: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        report = {"taskId": task_id, "title": title, "content": content, "metadata": metadata or {},
                  "timestamp": int(time.time() * 1000)}
        self.redis.lpush("kestrel:queue:reports", json.dumps(report))

    def send_search(self, task_id: str, query: str, results: int, sources: List[str]):
        payload = {"taskId": task_id, "query": query, "results": results, "sources": sources,
                   "timestamp": int(time.time() * 1000), "time": str(time.time() * 1000)}
        self.redis.lpush("kestrel:queue:searches", json.dumps(payload))

# -----------------------------------------------------------------------------
# KestrelAI Research Agent Worker
# -----------------------------------------------------------------------------
class KestrelAgentWorker:
    """Main agent worker that processes research tasks"""
    
    def __init__(self):
        # Initialize Redis client
        self.redis_client = KestrelRedisClient()
        
        # Initialize AI components
        self.mem = MemoryStore()
        self.llm = LlmWrapper(model="gemma3:12b")  # Configure your model here
        self.agent = ResearchAgent(self.mem, self.llm)
        
        # State management
        self.running = False
        self.paused = False
        self.current_task_id = None
        self.current_task_config = {}
        self.tasks: Dict[str, Task] = {}
        self.orchestrator = None
        
        # Metrics tracking
        self.task_metrics: Dict[str, Dict[str, Any]] = {}
        self.global_metrics = {
            "total_llm_calls": 0,
            "total_searches": 0,
            "total_summaries": 0,
            "total_checkpoints": 0,
            "total_web_fetches": 0
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
            
            time.sleep(0.1)
                
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
    
    def start_task(self, task_id: str, config: Dict[str, Any]):
        """Start a new research task"""
        self.current_task_id = task_id
        self.current_task_config = config
        
        # Create Task object aligned to the actual model (budgetMinutes + enum status)
        task = Task(
            name=config.get("name", "Research Task"),
            description=config.get("description", ""),
            budgetMinutes=config.get("budgetMinutes", 180),
            status=TaskStatus.ACTIVE
        )
        self.tasks[task_id] = task
        
        # Initialize orchestrator with task
        self.orchestrator = Orchestrator([task], self.llm)
        
        # Initialize task metrics
        self.task_metrics[task_id] = {
            "search_count": 0,
            "think_count": 0,
            "summary_count": 0,
            "checkpoint_count": 0,
            "action_count": 0,
            "searches": [],
            "start_time": time.time()
        }
        
        self.running = True
        self.paused = False
        
        # Send initial updates (status as enum name)
        self.redis_client.send_update(task_id, status=TaskStatus.ACTIVE.name, progress=0.0)
        self.redis_client.send_activity(
            task_id,
            "task_start",
            f"🦅 Starting research: {task.name}"
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
                "global_metrics": self.global_metrics
            }
            self.redis_client.checkpoint(self.current_task_id, checkpoint_state)
            
            self.redis_client.send_update(self.current_task_id, status=TaskStatus.PAUSED.name)
            self.redis_client.send_activity(
                self.current_task_id,
                "task_pause",
                "⏸️ Task paused"
            )
            
            logger.info(f"Paused task {self.current_task_id}")
    
    def resume_task(self):
        """Resume paused task"""
        if self.running and self.paused and self.current_task_id:
            # Restore from checkpoint
            checkpoint = self.redis_client.restore_checkpoint(self.current_task_id)
            if checkpoint:
                self.task_metrics[self.current_task_id] = checkpoint.get("metrics", {})
                self.global_metrics = checkpoint.get("global_metrics", self.global_metrics)
            
            self.paused = False

            # Update in-memory task status
            task = self.tasks.get(self.current_task_id)
            if task:
                task.status = TaskStatus.ACTIVE
            
            self.redis_client.send_update(self.current_task_id, status=TaskStatus.ACTIVE.name)
            self.redis_client.send_activity(
                self.current_task_id,
                "task_resume",
                "▶️ Task resumed"
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
            self.redis_client.send_update(task_id, status=TaskStatus.COMPLETE.name, progress=100.0, elapsed=int(
                time.time() - self.task_metrics.get(task_id, {}).get("start_time", time.time())
            ))
            self.redis_client.send_activity(
                task_id,
                "task_complete",
                "✅ Task completed"
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
    
    def process_task_step(self):
        """Process one step of the research task"""
        if not self.orchestrator or not self.current_task_id:
            return
        
        task_id = self.current_task_id
        task = self.tasks.get(task_id)
        if not task:
            return
        
        # Execute research step
        notes = self.agent.run_step(task)

        # Update local metrics with agent's actual values
        agent_task_metrics = self.agent.get_task_metrics(task.name)
        self.task_metrics[task_id].update({
            "search_count": agent_task_metrics["search_count"],
            "think_count": agent_task_metrics["think_count"],
            "summary_count": agent_task_metrics["summary_count"],
            "checkpoint_count": agent_task_metrics["checkpoint_count"],
            "action_count": agent_task_metrics["action_count"],
            "searches": agent_task_metrics["searches"],
            "search_history": agent_task_metrics["search_history"],
            "current_focus": agent_task_metrics["current_focus"]
        })
        
        if notes and len(notes) > 10:
            # Parse notes for activity type
            if "[SEARCH]" in notes:
                activity_type = "search"
                message = "🔍 Searching for information"
            elif "[THOUGHT]" in notes or "[THINKING]" in notes:
                activity_type = "thinking"
                message = "🤔 Analyzing findings"
            elif "[SUMMARY]" in notes:
                activity_type = "summary"
                message = "📝 Creating summary"
            elif "[CHECKPOINT" in notes:
                activity_type = "checkpoint"
                message = "💾 Saving checkpoint"
            else:
                activity_type = "analysis"
                message = "⚙️ Processing"
            
            self.task_metrics[task_id]["action_count"] += 1
            
            # Send activity
            self.redis_client.send_activity(task_id, activity_type, message)
            
            # Calculate progress (0..100) using budgetMinutes
            elapsed = time.time() - self.task_metrics[task_id]["start_time"]
            progress = min(100.0, (elapsed / (task.budgetMinutes * 60.0)) * 100.0)
            
            # Update metrics from agent if available
            if hasattr(self.agent, 'get_global_metrics'):
                agent_metrics = self.agent.get_global_metrics()
                self.global_metrics.update(agent_metrics)
            
            # Send updates
            metrics_update = {
                "searchCount": self.task_metrics[task_id]["search_count"],
                "thinkCount": self.task_metrics[task_id]["think_count"],
                "summaryCount": self.task_metrics[task_id]["summary_count"],
                "checkpointCount": self.task_metrics[task_id]["checkpoint_count"],
                "llmTokensUsed": self.global_metrics.get("total_llm_calls", 0) * 1000  # Estimate
            }
            logger.error(f"Current task Metrics{ self.task_metrics}")
            
            self.redis_client.send_update(
                task_id,
                status=task.status.name if hasattr(task, "status") else TaskStatus.ACTIVE.name,
                progress=progress,
                elapsed=int(elapsed),
                metrics=metrics_update
            )

            logger.info(f"Searches {agent_task_metrics['searches']}")
            # Send search history to Redis
            for query in agent_task_metrics["searches"]:
                self.redis_client.send_search(
                    task_id,
                    query,
                    results=SEARCH_RESULTS,
                    sources=["agent_search"]
                )
            
            # Save notes to file and send as report
            safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task.name).strip()
            notes_path = f"notes/{safe_name.upper()}.txt"
            with open(notes_path, "w", encoding="utf-8") as fh:
                fh.write(notes)
            
            # Send report update
            self.redis_client.send_report(
                task_id,
                f"Research Update - {task.name}",
                notes,
                metadata={
                    "progress": progress,
                    "action_count": self.task_metrics[task_id]["action_count"]
                }
            )
            
            # Let orchestrator decide next action
            self.orchestrator.next_action(task, notes)
            
            # Check if task is complete
            if progress >= 100.0:
                self.stop_task()
        
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
        safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task.name).strip()
        notes_file = f"notes/{safe_name.upper()}.txt"
        if pathlib.Path(notes_file).exists():
            with open(notes_file, "r", encoding="utf-8") as fh:
                content = fh.read()
        else:
            content = "No research notes available."
        
        # Create final report
        report_content = f"""# Research Report: {task.name}

## Summary
**Task:** {task.description}
**Duration:** {task.budgetMinutes} minutes budget
**Status:** Complete

## Metrics
- Total Searches: {metrics.get('search_count', 0)}
- Analysis Steps: {metrics.get('think_count', 0)}
- Summaries Created: {metrics.get('summary_count', 0)}
- Checkpoints: {metrics.get('checkpoint_count', 0)}

## Research Findings
{content}

---
*Report generated at {datetime.now().isoformat()}*
"""
        
        self.redis_client.send_report(
            task_id,
            f"Final Report - {task.name}",
            report_content,
            metadata={
                "final": True,
                "metrics": metrics
            }
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
