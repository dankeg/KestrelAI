"""
Base Agent Classes for KestrelAI
Provides clean abstractions and interfaces for all agent types
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import json
import re
import logging

try:
    from .base import LlmWrapper
    from memory.vector_store import MemoryStore
    from shared.models import Task
except ImportError:
    from KestrelAI.agents.base import LlmWrapper
    from KestrelAI.memory.vector_store import MemoryStore
    from KestrelAI.shared.models import Task

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Base state container for all agents"""
    task_id: str
    queries: Set[str] = field(default_factory=set)
    history: deque = field(default_factory=lambda: deque(maxlen=20))
    action_count: int = 0
    think_count: int = 0
    search_count: int = 0
    summary_count: int = 0
    checkpoint_count: int = 0
    last_checkpoint: str = ""
    checkpoints: List[str] = field(default_factory=list)
    current_focus: str = ""
    search_history: List[Dict] = field(default_factory=list)
    
    # Loop prevention
    repeated_queries: Dict[str, int] = field(default_factory=dict)
    consecutive_thinks: int = 0
    consecutive_searches: int = 0
    last_action: str = ""
    action_pattern: List[str] = field(default_factory=lambda: deque(maxlen=10))
    
    def is_in_loop(self, max_repeats: int = 3) -> bool:
        """Check if agent is stuck in a repetitive loop"""
        if self.consecutive_thinks >= max_repeats:
            return True
        if self.consecutive_searches >= max_repeats:
            return True
        for query, count in self.repeated_queries.items():
            if count >= max_repeats:
                return True
        if len(self.action_pattern) >= 6:
            recent_actions = list(self.action_pattern)[-6:]
            if len(set(recent_actions)) <= 2:
                return True
        return False
    
    def record_action(self, action: str, query: str = ""):
        """Record an action for loop detection"""
        self.last_action = action
        self.action_pattern.append(action)
        
        if action == "think":
            self.consecutive_thinks += 1
            self.consecutive_searches = 0
        elif action == "search":
            self.consecutive_searches += 1
            self.consecutive_thinks = 0
            if query:
                self.repeated_queries[query] = self.repeated_queries.get(query, 0) + 1
        else:
            self.consecutive_thinks = 0
            self.consecutive_searches = 0


class BaseAgent(ABC):
    """Abstract base class for all KestrelAI agents"""
    
    def __init__(self, agent_id: str, llm: LlmWrapper, memory: MemoryStore):
        self.agent_id = agent_id
        self.llm = llm
        self.memory = memory
        self.state: Optional[AgentState] = None
        
        # Base metrics
        self.metrics = {
            "total_llm_calls": 0,
            "total_searches": 0,
            "total_summaries": 0,
            "total_checkpoints": 0,
            "total_thoughts": 0,
            "total_web_fetches": 0,
            "total_search_results": 0,
        }
    
    @abstractmethod
    async def run_step(self, task: Task) -> str:
        """Run one step of the agent's workflow"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        pass
    
    def _chat(self, messages: List[Dict]) -> str:
        """Send chat request to LLM"""
        self.metrics["total_llm_calls"] += 1
        return self.llm.chat(messages)
    
    def _json_from(self, text: str) -> Union[Dict, None]:
        """Parse JSON from text with fallback patterns"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract from markdown code blocks
            m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try first brace pattern
            m = re.search(r"\{.*?\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        
        logger.warning("No valid JSON found in response")
        return None
    
    def _add_to_rag(self, task: Task, text: str, doc_type: str) -> None:
        """Add document to RAG with metadata"""
        from uuid import uuid4
        doc_id = f"{task.name}-{doc_type}-{uuid4().hex[:8]}"
        metadata = {
            "task": task.name,
            "type": doc_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "length": len(text),
        }
        self.memory.add(doc_id, text, metadata)


class ResearchAgent(BaseAgent):
    """Base research agent with common research functionality"""
    
    def __init__(self, agent_id: str, llm: LlmWrapper, memory: MemoryStore):
        super().__init__(agent_id, llm, memory)
        self._state: Dict[str, AgentState] = {}
    
    def get_task_metrics(self, task_name: str) -> Dict:
        """Get detailed metrics for a specific task"""
        if task_name not in self._state:
            return {
                "searches": [],
                "search_history": [],
                "action_count": 0,
                "search_count": 0,
                "think_count": 0,
                "summary_count": 0,
                "checkpoint_count": 0,
                "current_focus": "",
            }
        
        state = self._state[task_name]
        return {
            "searches": list(state.queries),
            "search_history": state.search_history,
            "action_count": state.action_count,
            "search_count": state.search_count,
            "think_count": state.think_count,
            "summary_count": state.summary_count,
            "checkpoint_count": state.checkpoint_count,
            "current_focus": state.current_focus,
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all global metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = {
            "total_llm_calls": 0,
            "total_searches": 0,
            "total_summaries": 0,
            "total_checkpoints": 0,
            "total_thoughts": 0,
            "total_web_fetches": 0,
            "total_search_results": 0,
        }
        for state in self._state.values():
            state.action_count = 0
            state.think_count = 0
            state.search_count = 0
            state.summary_count = 0
            state.checkpoint_count = 0
            state.search_history.clear()


class OrchestratorAgent(BaseAgent):
    """Base orchestrator agent with common orchestration functionality"""
    
    def __init__(self, agent_id: str, llm: LlmWrapper, memory: MemoryStore):
        super().__init__(agent_id, llm, memory)
        self.tasks: Dict[str, Task] = {}
        self.current_task: Optional[str] = None
        self.task_states: Dict[str, Any] = {}
    
    @abstractmethod
    async def next_action(self, task: Task, notes: str = "") -> str:
        """Get next action for a task"""
        pass
    
    @abstractmethod
    def get_task_progress(self, task_name: str) -> Dict[str, Any]:
        """Get progress information for a task"""
        pass
    
    def get_current_subtask(self, task_name: str) -> Optional[str]:
        """Get current subtask description for a task"""
        return None  # Override in subclasses
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        return {
            **self.metrics,
            "total_tasks": len(self.tasks),
            "active_tasks": len([t for t in self.tasks.values() if t.status.value == "active"]),
        }
