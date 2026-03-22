"""
Consolidated Orchestrator for KestrelAI
Replaces all duplicate orchestrator implementations with a single, configurable orchestrator
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from KestrelAI.agents.base import LlmWrapper
from KestrelAI.agents.config import get_orchestrator_config
from KestrelAI.agents.context_manager import ContextManager, TokenBudget, TokenCounter
from KestrelAI.agents.multi_level_summarizer import MultiLevelSummarizer
from KestrelAI.memory.vector_store import MemoryStore
from KestrelAI.shared.models import SubtaskType, Task, TaskStatus
from KestrelAI.shared.research_utils import (
    RESEARCH_AUDIENCE_TERMS,
    RESEARCH_EVIDENCE_TERMS,
    RESEARCH_SOURCE_CLASS_TERMS,
    build_research_task_profile,
    extract_domains_from_text,
    infer_research_task_family,
    subtask_looks_low_signal,
    task_targets_discrete_opportunities,
    timeouts_disabled,
)
from KestrelAI.shared.runtime_settings import normalize_max_context_tokens

from .base_agent import OrchestratorAgent
from .langchain_adapter import LangChainChatAdapter
from .langchain_orchestrator_chains import OrchestratorLangChainControlChains
from .langchain_report_chains import OrchestratorLangChainChains
from .searxng_service import SearXNGService
from .url_utils import URLFlagManager, clean_url
from .web_research_agent import ResearchConfig, WebResearchAgent

try:
    from KestrelAI.graphs.orchestrator_runner import LangGraphOrchestratorRunner
except Exception:  # pragma: no cover - dependency-gated path
    LangGraphOrchestratorRunner = None

logger = logging.getLogger(__name__)

# Context management constants
MAX_FEEDBACK_HISTORY = 20  # Maximum feedback entries to keep for loop detection
META_CRITIQUE_PATTERNS = (
    r"\bwould you like me\b",
    r"\boverall assessment\b",
    r"\bminor suggestions? for improvement\b",
    r"\bthis report(?:'s| is)\b",
    r"\bremarkably thorough\b",
    r"\bwell-structured report\b",
    r"\bstrengths:\b",
)

META_CRITIQUE_ALLOWED_TASK_PATTERNS = (
    r"\bcritique\b",
    r"\breview\b",
    r"\bevaluate\b",
    r"\bassessment\b",
    r"\bgrade\b",
    r"\bfeedback\b",
)

META_CRITIQUE_HARD_BAN_PATTERNS = (
    r"(?im)^\s*\*{0,2}strengths\*{0,2}\s*:\s*$",
    r"(?im)^\s*\*{0,2}minor suggestions?(?: for (?:refinement|improvement))?\*{0,2}\s*:\s*$",
    r"(?im)^\s*\*{0,2}overall(?: assessment)?\*{0,2}\s*:\s*$",
    r"\bthis is an outstanding final research report\b",
    r"\bhere(?:'|’)s a breakdown of why this report is strong\b",
)

FINAL_REPORT_CORE_SECTIONS = (
    "## Executive Summary",
    "## Scope and Method",
    "## Findings",
    "## Comparative Analysis",
    "## Limitations and Open Questions",
    "## Recommended Next Steps",
)

FINAL_REPORT_REQUIRED_SECTIONS = FINAL_REPORT_CORE_SECTIONS + (
    "## Evidence Status Appendix",
)

REPORT_CONTROL_LINE_PREFIXES = (
    "[ORCHESTRATOR FEEDBACK]",
    "[ORCHESTRATOR GUARD]",
    "[SEARCH]",
    "[NO RESULTS]",
    "[SKIP]",
    "[THOUGHT]",
    "[SUMMARY]",
    "[COMPLETE]",
    "[CHECKPOINT]",
    "[PROGRESS]",
    "[MCP_TOOL]",
)

CLAIM_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "by",
    "from",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "this",
    "that",
    "these",
    "those",
    "it",
    "as",
    "into",
    "their",
    "its",
    "than",
    "then",
    "also",
}


class OrchestratorDecision(BaseModel):
    reasoning: str
    decision: Literal["continue", "switch", "done"]
    feedback: str
    subtask: Literal["stay", "proceed"]
    next_task: str


class Subtask(BaseModel):
    order: int
    description: str
    success_criteria: str
    subtask_type: SubtaskType = SubtaskType.GENERAL


class PlanningPlan(BaseModel):
    restated_task: str
    subtasks: list[Subtask]


class PrePlanningAction(BaseModel):
    reasoning: str = ""
    action: Literal["think", "search", "mcp_tool", "done"] = "done"
    query: str = ""
    thought: str = ""
    tool_name: str = ""
    tool_parameters: dict[str, Any] = Field(default_factory=dict)


class VerifiedEvidenceItem(BaseModel):
    statement: str
    title: str
    domain: str
    url: str = ""
    query: str = ""
    source_excerpt: str = ""
    source_tier: str = ""
    authority_score: int = 0
    official: bool = False


class SourceEvidenceRecord(BaseModel):
    subtask: int
    title: str
    domain: str
    url: str = ""
    official: bool = False
    authority_score: int = 0
    source_tier: str = ""
    fetched: bool = False
    query: str = ""
    summary: str = ""
    content_excerpt: str = ""
    snippet: str = ""
    tokens: set[str] = Field(default_factory=set)
    title_tokens: set[str] = Field(default_factory=set)
    domain_tokens: set[str] = Field(default_factory=set)
    query_tokens: set[str] = Field(default_factory=set)
    evidence_tokens: set[str] = Field(default_factory=set)


class EvidenceLead(BaseModel):
    title: str
    domain: str
    url: str = ""
    official: bool = False
    source_tier: str = ""
    authority_score: int = 0
    query: str = ""
    evidence_excerpt: str = ""
    supported_summary: str = ""
    supported_details: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


CLAIM_RELATION_HINTS = (
    " is ",
    " are ",
    " was ",
    " were ",
    " has ",
    " have ",
    " includes ",
    " include ",
    " requires ",
    " require ",
    " accepts ",
    " accept ",
    " offers ",
    " offer ",
    " supports ",
    " support ",
    " provides ",
    " provide ",
    " lists ",
    " list ",
    " shows ",
    " show ",
    " notes ",
    " note ",
    " states ",
    " state ",
    " describes ",
    " describe ",
    " covers ",
    " cover ",
    " includes ",
    " include ",
    " contains ",
    " contain ",
    " uses ",
    " use ",
    " applies ",
    " apply ",
    " available ",
    " unavailable ",
    " open ",
    " closed ",
    " opens ",
    " closes ",
    " starts ",
    " ends ",
    " begins ",
    " runs ",
    " costs ",
    " cost ",
    " funds ",
    " funded ",
    " funding ",
    " contact ",
    " contacts ",
    " email ",
    " phone ",
    " link ",
    " links ",
    " date ",
    " dates ",
    " deadline ",
    " deadlines ",
    " eligibility ",
    " application ",
    " applications ",
)


class TaskState:
    """Enhanced task state tracking with subtask-specific agents"""

    def __init__(self, task: Task, max_context_tokens: int = 32768):
        self.task = task
        self.max_context_tokens = max_context_tokens
        self.orchestrator = None
        self.subtask_index = 0
        self.completed_subtasks: set[int] = set()
        self.notes_history: list[str] = []
        self.last_decision = None
        self.decision_count = 0
        self.stuck_count = 0
        self.last_progress_time = datetime.now()
        self.research_plan = None
        self.feedback_history: list[str] = []
        self.search_history: set[str] = set()
        self.repeated_actions: dict[str, int] = {}

        # Subtask-specific tracking
        self.subtask_agents: dict[int, WebResearchAgent] = {}
        self.subtask_findings: dict[int, list[str]] = {}
        self.subtask_reports: dict[int, str] = {}
        self.subtask_guidance: dict[int, str] = {}
        self.subtask_control_hints: dict[int, dict[str, Any]] = {}
        self.subtask_metric_snapshots: dict[int, dict[str, int]] = {}
        self.subtask_stagnation_rounds: dict[int, int] = {}
        self.subtask_last_review_action_count: dict[int, int] = {}
        self.all_findings: list[str] = []
        self.all_reports: list[str] = []  # Track all reports for accumulation

    def is_stuck(self, max_stuck_count: int = 3) -> bool:
        """Check if task is stuck in a loop"""
        return self.stuck_count >= max_stuck_count

    def record_decision(self, decision: str, feedback: str):
        """Record orchestrator decision for loop detection"""
        self.decision_count += 1
        self.last_decision = decision
        self.feedback_history.append(feedback)

        # Keep only last MAX_FEEDBACK_HISTORY feedback entries for loop detection
        if len(self.feedback_history) > MAX_FEEDBACK_HISTORY:
            self.feedback_history = self.feedback_history[-MAX_FEEDBACK_HISTORY:]

        # Check for repeated decisions
        if len(self.feedback_history) >= 3:
            recent_feedback = self.feedback_history[-3:]
            if len(set(recent_feedback)) == 1:  # All same feedback
                self.stuck_count += 1
            else:
                self.stuck_count = max(0, self.stuck_count - 1)

    def mark_subtask_complete(self, subtask_index: int):
        """Mark a subtask as completed"""
        self.completed_subtasks.add(subtask_index)
        self.subtask_index = max(self.completed_subtasks) + 1
        self.stuck_count = 0  # Reset stuck count on progress
        self.subtask_guidance.pop(subtask_index, None)
        self.subtask_control_hints.pop(subtask_index, None)
        self.subtask_metric_snapshots.pop(subtask_index, None)
        self.subtask_stagnation_rounds.pop(subtask_index, None)
        self.subtask_last_review_action_count.pop(subtask_index, None)
        # Collect findings and reports from completed subtask
        if subtask_index in self.subtask_reports:
            report = self.subtask_reports[subtask_index]
            self.all_findings.append(f"Subtask {subtask_index + 1} Report:\n{report}")
            self.all_reports.append(report)  # Track report for accumulation

    def set_subtask_guidance(self, subtask_index: int, guidance: str) -> None:
        """Store normalized orchestrator guidance for a subtask."""
        normalized = (guidance or "").strip()
        if normalized:
            self.subtask_guidance[subtask_index] = normalized
        else:
            self.subtask_guidance.pop(subtask_index, None)

    def get_subtask_guidance(self, subtask_index: int) -> str:
        """Get active orchestrator guidance for a subtask."""
        return self.subtask_guidance.get(subtask_index, "")

    def set_subtask_control_hints(
        self, subtask_index: int, hints: dict[str, Any] | None
    ) -> None:
        normalized = dict(hints or {})
        if normalized:
            self.subtask_control_hints[subtask_index] = normalized
        else:
            self.subtask_control_hints.pop(subtask_index, None)

    def get_subtask_control_hints(self, subtask_index: int) -> dict[str, Any]:
        return dict(self.subtask_control_hints.get(subtask_index, {}))

    def create_subtask_agent(
        self,
        subtask_index: int,
        llm: LlmWrapper,
        memory: MemoryStore,
        mcp_manager=None,
    ) -> WebResearchAgent:
        """Create a new subtask-specific research agent"""
        if not self.research_plan or subtask_index >= len(self.research_plan.subtasks):
            raise ValueError(f"Invalid subtask index: {subtask_index}")

        subtask = self.research_plan.subtasks[subtask_index]
        subtask_id = f"{self.task.name}-subtask-{subtask_index}"

        # Collect previous findings from completed subtasks
        previous_findings = "\n\n".join(self.all_findings) if self.all_findings else ""

        # Collect previous reports for accumulation (pass actual report content, not summaries)
        previous_reports = self.all_reports.copy() if self.all_reports else []

        # Create research config for subtask agent
        config = ResearchConfig(
            is_subtask_agent=True,
            subtask_description=subtask.description,
            success_criteria=subtask.success_criteria,
            previous_findings=previous_findings,
            previous_reports=previous_reports,  # Pass previous reports for accumulation
            orchestrator_guidance=self.get_subtask_guidance(subtask_index),
            orchestrator_control_hints=self.get_subtask_control_hints(subtask_index),
            max_context_tokens=self.max_context_tokens,
            use_mcp=mcp_manager is not None,
            mcp_manager=mcp_manager,
        )

        agent = WebResearchAgent(
            agent_id=subtask_id, llm=llm, memory=memory, config=config
        )
        agent.orchestrator = getattr(self, "orchestrator", None)
        agent.parent_task_name = self.task.name

        self.subtask_agents[subtask_index] = agent
        self.subtask_findings[subtask_index] = []
        return agent

    def get_current_subtask_agent(self) -> WebResearchAgent | None:
        """Get the current subtask agent"""
        if self.subtask_index in self.subtask_agents:
            return self.subtask_agents[self.subtask_index]
        return None

    def get_progress_percentage(self) -> float:
        """Calculate task progress based on completed subtasks"""
        if not self.research_plan or not self.research_plan.subtasks:
            return 0.0
        return (len(self.completed_subtasks) / len(self.research_plan.subtasks)) * 100.0


class ResearchOrchestrator(OrchestratorAgent):
    """Research orchestrator that manages research tasks and subtasks"""

    def __init__(
        self,
        tasks: list[Task],
        llm: LlmWrapper,
        profile: str = "kestrel",
        mcp_manager=None,
        use_mcp: bool = False,
        max_context_tokens: int | None = None,
    ):
        super().__init__("research-orchestrator", llm, MemoryStore())

        self.tasks = {t.name: t for t in tasks}
        self.current = tasks[0].name if tasks else None
        self.task_states: dict[str, TaskState] = {}
        self.langgraph_runner = None
        self.max_context_tokens = normalize_max_context_tokens(max_context_tokens)

        # Initialize memory store for subtask agents
        self.memory = MemoryStore()

        # Load configuration
        self.config = get_orchestrator_config(profile)

        # Initialize task states
        for task in tasks:
            task_state = TaskState(task, max_context_tokens=self.max_context_tokens)
            task_state.orchestrator = self
            self.task_states[task.name] = task_state

        # Loop prevention settings from config
        self.max_total_iterations = self.config.max_total_iterations
        self.max_iterations_per_subtask = max(
            1,
            int(
                os.getenv(
                    "ORCHESTRATOR_MAX_ITERATIONS_PER_SUBTASK",
                    self.config.max_iterations_per_subtask,
                )
            ),
        )
        self.total_iterations = 0

        # MCP configuration
        self.use_mcp = use_mcp
        self.mcp_manager = mcp_manager
        self.mcp_connected = False

        # LangChain adapter (optional)
        self.langchain_adapter: LangChainChatAdapter | None = None
        self.report_chains: OrchestratorLangChainChains | None = None
        self.control_chains: OrchestratorLangChainControlChains | None = None
        try:
            self.langchain_adapter = LangChainChatAdapter.from_llm(llm)
        except Exception as e:
            raise RuntimeError("Orchestrator LangChain adapter unavailable") from e
        try:
            self.report_chains = OrchestratorLangChainChains(
                model=self.langchain_adapter.client
            )
        except Exception as e:
            raise RuntimeError("Orchestrator report chains unavailable") from e
        try:
            self.control_chains = OrchestratorLangChainControlChains(
                model=self.langchain_adapter.client,
                review_schema=OrchestratorDecision,
                planning_schema=PlanningPlan,
                preplanning_schema=PrePlanningAction,
            )
        except Exception as e:
            raise RuntimeError("Orchestrator control chains unavailable") from e

        self.preplanning_search = SearXNGService(search_results=3, debug=False)

        # Initialize context management and summarization for orchestrator
        try:
            # Get model name from LLM wrapper if available (for TokenCounter)
            # Note: We pass llm object (not model_name string) to MultiLevelSummarizer
            model_name = getattr(llm, "model", "gemma3:27b")
            self.token_counter = TokenCounter(model_name=model_name)
            self.token_budget = TokenBudget(max_context=self.max_context_tokens)
            self.summarizer = MultiLevelSummarizer(
                llm=llm,  # Pass the actual llm object, not model_name string
                token_counter=self.token_counter,
                extract_facts=True,
            )
            self.context_manager = ContextManager(
                self.token_counter,
                self.token_budget,
                llm=llm,
                summarizer=self.summarizer,  # Pass summarizer to context manager
            )
            self.context_management_enabled = True
            logger.info(
                f"Orchestrator context management enabled with model: {model_name}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize orchestrator context management: {e}. Continuing without it."
            )
            self.token_counter = None
            self.token_budget = None
            self.context_manager = None
            self.summarizer = None
            self.context_management_enabled = False

        logger.info(
            f"Initialized research orchestrator with profile '{profile}': {self.config}"
        )

    async def initialize_mcp(self) -> bool:
        """Initialize MCP manager if configured"""
        if not self.use_mcp or not self.mcp_manager:
            return False

        try:
            if not self.mcp_manager.is_initialized:
                self.mcp_connected = await self.mcp_manager.initialize()
            else:
                self.mcp_connected = self.mcp_manager.is_initialized

            if self.mcp_connected:
                logger.info("MCP manager initialized successfully for orchestrator")
                tools = self.mcp_manager.get_available_tools()
                logger.info(f"Available MCP tools: {tools}")
            else:
                logger.error(
                    "MCP manager failed to initialize - no MCP capabilities available"
                )
            return self.mcp_connected
        except Exception as e:
            logger.error(f"Failed to initialize MCP manager: {e}")
            self.mcp_connected = False
            return False

    async def cleanup_mcp(self):
        """Cleanup MCP manager"""
        if self.mcp_manager:
            await self.mcp_manager.cleanup()
            self.mcp_connected = False

    def _fallback_review_decision(
        self, task: Task, task_state: TaskState, reason: str
    ) -> OrchestratorDecision:
        """Fallback decision when structured review fails."""
        metrics = self._get_current_subtask_agent_metrics(task, task_state)
        stagnation_rounds = task_state.subtask_stagnation_rounds.get(
            task_state.subtask_index, 0
        )
        readiness_ok, readiness_reason = self._evaluate_subtask_completion_readiness(
            task, task_state
        )
        plateau_ok, plateau_reason = self._evaluate_plateau_progression_readiness(
            task, task_state
        )
        if plateau_ok or (
            readiness_ok
            and metrics["action_count"] >= self.max_iterations_per_subtask
            and stagnation_rounds >= 2
        ):
            return OrchestratorDecision(
                reasoning=(
                    f"Structured review unavailable; using deterministic progression fallback. "
                    f"{plateau_reason if plateau_ok else readiness_reason}"
                ).strip(),
                decision="switch",
                feedback=(
                    "Advance to the next subtask, carry forward the strongest evidence, "
                    "and preserve unresolved gaps as explicit uncertainty."
                ),
                subtask="proceed",
                next_task=task.name,
            )

        planner_failures = int(metrics.get("consecutive_planner_failures", 0) or 0)
        feedback = (
            "Planner instability detected. Pivot to simpler, high-signal searches focused on official sources, "
            "primary organizations, and directory/listing pages tied directly to the current success criteria."
            if planner_failures > 0
            else "Continue the current subtask, but pivot away from low-yield queries toward official sources, "
            "primary organizations, and direct verification searches."
        )
        return OrchestratorDecision(
            reasoning=reason,
            decision="continue",
            feedback=feedback,
            subtask="stay",
            next_task=task.name,
        )

    def _build_current_subtask_info(self, task_state: TaskState) -> str:
        """Get human-readable status for the current subtask pointer."""
        if not task_state.research_plan or not task_state.research_plan.subtasks:
            return ""
        if task_state.subtask_index >= len(task_state.research_plan.subtasks):
            return "All subtasks completed"

        current_subtask = task_state.research_plan.subtasks[task_state.subtask_index]
        return (
            f"Current subtask: {current_subtask.description}\n"
            f"Success criteria: {current_subtask.success_criteria}\n"
            f"Subtask {task_state.subtask_index + 1} of "
            f"{len(task_state.research_plan.subtasks)}"
        )

    def _get_current_subtask_agent_metrics(
        self, task: Task, task_state: TaskState
    ) -> dict[str, int]:
        """Best-effort pull of current subtask agent activity metrics."""
        empty = {
            "action_count": 0,
            "search_attempt_count": 0,
            "search_count": 0,
            "zero_result_search_count": 0,
            "summary_count": 0,
            "checkpoint_count": 0,
            "query_count": 0,
            "planner_fallback_count": 0,
            "consecutive_planner_failures": 0,
            "pathway_count": 0,
            "pathway_attempted_count": 0,
            "pathway_hit_count": 0,
            "pathway_uncovered_count": 0,
        }
        agent = task_state.get_current_subtask_agent()
        if (
            agent is None
            or not hasattr(agent, "_state")
            or task.name not in agent._state
        ):
            return empty

        agent_state = agent._state[task.name]
        search_pathways = list(getattr(agent_state, "search_pathways", []) or [])
        pathway_attempted_count = sum(
            1
            for pathway in search_pathways
            if int(pathway.get("attempt_count", 0) or 0) > 0
        )
        pathway_hit_count = sum(
            1
            for pathway in search_pathways
            if int(pathway.get("hit_count", 0) or 0) > 0
        )
        return {
            "action_count": int(getattr(agent_state, "action_count", 0)),
            "search_attempt_count": int(
                getattr(agent_state, "search_attempt_count", 0)
            ),
            "search_count": int(getattr(agent_state, "search_count", 0)),
            "zero_result_search_count": int(
                getattr(agent_state, "zero_result_search_count", 0)
            ),
            "summary_count": int(getattr(agent_state, "summary_count", 0)),
            "checkpoint_count": int(getattr(agent_state, "checkpoint_count", 0)),
            "query_count": int(len(getattr(agent_state, "queries", set()) or set())),
            "planner_fallback_count": int(
                getattr(agent_state, "planner_fallback_count", 0)
            ),
            "consecutive_planner_failures": int(
                getattr(agent_state, "consecutive_planner_failures", 0)
            ),
            "pathway_count": len(search_pathways),
            "pathway_attempted_count": pathway_attempted_count,
            "pathway_hit_count": pathway_hit_count,
            "pathway_uncovered_count": max(
                0,
                len(search_pathways) - pathway_attempted_count,
            ),
        }

    def _get_current_subtask_agent_state(self, task: Task, task_state: TaskState):
        """Return the current subtask agent state when available."""
        agent = task_state.get_current_subtask_agent()
        if agent is None or not hasattr(agent, "_state"):
            return None
        return agent._state.get(task.name)

    def _build_subtask_control_hints(
        self,
        task: Task,
        task_state: TaskState,
        subtask_index: int,
    ) -> dict[str, Any]:
        hints: dict[str, Any] = {}
        if subtask_index != task_state.subtask_index:
            return hints

        current_subtask = None
        if (
            task_state.research_plan
            and task_state.research_plan.subtasks
            and 0 <= subtask_index < len(task_state.research_plan.subtasks)
        ):
            current_subtask = task_state.research_plan.subtasks[subtask_index]

        metrics = self._get_current_subtask_agent_metrics(task, task_state)
        if (
            current_subtask is None
            or not self._subtask_requires_broad_discovery(current_subtask)
            or metrics["pathway_count"] <= 0
        ):
            return hints

        agent_state = self._get_current_subtask_agent_state(task, task_state)
        if agent_state is None:
            return hints

        search_pathways = list(getattr(agent_state, "search_pathways", []) or [])
        ranked_pathways = sorted(
            search_pathways,
            key=lambda pathway: (
                int(pathway.get("hit_count", 0) or 0) > 0,
                int(pathway.get("attempt_count", 0) or 0),
                str(pathway.get("id", "")),
            ),
        )
        preferred_pathway_ids = [
            str(pathway.get("id", "") or "")
            for pathway in ranked_pathways
            if str(pathway.get("id", "") or "")
        ]
        if not preferred_pathway_ids:
            return hints

        hints["discovery_mode"] = "pathway_first"
        hints["preferred_pathway_ids"] = preferred_pathway_ids
        hints["pathway_count"] = metrics["pathway_count"]
        hints["pathway_attempted_count"] = metrics["pathway_attempted_count"]
        hints["pathway_hit_count"] = metrics["pathway_hit_count"]
        hints["pathway_uncovered_count"] = metrics["pathway_uncovered_count"]
        hints["stagnating"] = (
            task_state.subtask_stagnation_rounds.get(subtask_index, 0) >= 2
        )
        return hints

    def _set_subtask_guidance(
        self, task: Task, task_state: TaskState, subtask_index: int, guidance: str
    ) -> None:
        """Persist guidance and structured control hints onto the existing subtask agent config."""
        task_state.set_subtask_guidance(subtask_index, guidance)
        control_hints = self._build_subtask_control_hints(
            task, task_state, subtask_index
        )
        task_state.set_subtask_control_hints(subtask_index, control_hints)
        existing_agent = task_state.subtask_agents.get(subtask_index)
        if (
            existing_agent is not None
            and hasattr(existing_agent, "config")
            and hasattr(existing_agent.config, "orchestrator_guidance")
        ):
            existing_agent.config.orchestrator_guidance = (
                task_state.get_subtask_guidance(subtask_index)
            )
            if hasattr(existing_agent.config, "orchestrator_control_hints"):
                existing_agent.config.orchestrator_control_hints = (
                    task_state.get_subtask_control_hints(subtask_index)
                )

    def _update_subtask_stagnation(self, task: Task, task_state: TaskState) -> int:
        """
        Track stagnant review rounds when evidence metrics stop changing for the
        current subtask. Do not include action_count, summaries, or checkpoints;
        those can increase even when the evidence frontier is flat.
        """
        current_index = task_state.subtask_index
        metrics = self._get_current_subtask_agent_metrics(task, task_state)
        evidence_stats = self._get_current_subtask_evidence_stats(task, task_state)
        tracked = {
            "search_count": metrics["search_count"],
            "query_count": metrics["query_count"],
            "pathway_attempted_count": metrics["pathway_attempted_count"],
            "pathway_hit_count": metrics["pathway_hit_count"],
            "authoritative_results": evidence_stats["authoritative_results"],
            "official_results": evidence_stats["official_results"],
            "fetched_results": evidence_stats["fetched_results"],
            "distinct_authoritative_titles": evidence_stats[
                "distinct_authoritative_titles"
            ],
            "distinct_official_titles": evidence_stats["distinct_official_titles"],
            "unique_authoritative_domains": evidence_stats[
                "unique_authoritative_domains"
            ],
        }
        previous = task_state.subtask_metric_snapshots.get(current_index)
        if previous and previous == tracked and metrics["action_count"] > 0:
            task_state.subtask_stagnation_rounds[current_index] = (
                task_state.subtask_stagnation_rounds.get(current_index, 0) + 1
            )
        else:
            task_state.subtask_stagnation_rounds[current_index] = 0
        task_state.subtask_metric_snapshots[current_index] = tracked
        return task_state.subtask_stagnation_rounds.get(current_index, 0)

    def _truncate_text_by_token_budget(
        self,
        text: str,
        *,
        max_tokens: int,
        fallback_label: str = "truncated",
    ) -> str:
        """Truncate text to a token budget; approximate only if tokenizer unavailable."""
        if not text:
            return text
        safe_max_tokens = max(1, int(max_tokens))
        if self.token_counter is not None:
            return self.token_counter.truncate_to_tokens(text, safe_max_tokens)

        # Fallback for tokenizer-unavailable paths (rare): 1 token ~= 4 chars.
        approx_chars = safe_max_tokens * 4
        if len(text) <= approx_chars:
            return text
        return text[:approx_chars] + f"\n\n...[{fallback_label}]"

    def _is_meta_critique_output(self, content: str, task_description: str) -> bool:
        """
        Detect invalid final synthesis outputs that evaluate a report instead of
        providing task-focused findings.
        """
        text = (content or "").strip()
        if not text:
            return True

        task_desc = (task_description or "").lower()
        if any(
            re.search(pat, task_desc) for pat in META_CRITIQUE_ALLOWED_TASK_PATTERNS
        ):
            return False

        hits = sum(
            1
            for pat in META_CRITIQUE_PATTERNS
            if re.search(pat, text, flags=re.IGNORECASE)
        )
        if hits >= 2:
            return True

        lowered = text.lower()
        if lowered.startswith("okay, this is") and "report" in lowered[:220]:
            return True
        return False

    @staticmethod
    def _strip_control_channel_annotations(text: str) -> str:
        """Remove orchestration/progress control markers from evidence text."""
        if not text:
            return ""
        cleaned_lines: list[str] = []
        for raw_line in re.sub(r"\r\n?", "\n", str(text)).splitlines():
            stripped = raw_line.strip()
            if any(
                stripped.startswith(prefix) for prefix in REPORT_CONTROL_LINE_PREFIXES
            ):
                continue
            cleaned_lines.append(raw_line.rstrip())
        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def _has_disallowed_report_review_language(
        self, content: str, task_description: str
    ) -> bool:
        if not content:
            return True
        task_desc = (task_description or "").lower()
        if any(
            re.search(pat, task_desc) for pat in META_CRITIQUE_ALLOWED_TASK_PATTERNS
        ):
            return False
        return any(
            re.search(pat, content, flags=re.IGNORECASE)
            for pat in META_CRITIQUE_HARD_BAN_PATTERNS
        )

    @staticmethod
    def _report_body_without_appendix(report_text: str) -> str:
        text = (report_text or "").strip()
        appendix_idx = text.find("## Evidence Status Appendix")
        if appendix_idx >= 0:
            return text[:appendix_idx].strip()
        return text

    @staticmethod
    def _dedupe_top_level_sections(report_text: str) -> str:
        text = (report_text or "").strip()
        if not text:
            return text
        pattern = re.compile(r"(?ms)^##\s+[^\n]+.*?(?=^##\s+[^\n]+|\Z)")
        matches = list(pattern.finditer(text))
        if not matches:
            return text

        kept_order: list[str] = []
        kept_blocks: dict[str, str] = {}
        for match in matches:
            block = match.group(0).strip()
            heading = block.splitlines()[0].strip()
            if heading not in kept_blocks:
                kept_order.append(heading)
                kept_blocks[heading] = block
            elif len(block) > len(kept_blocks[heading]):
                kept_blocks[heading] = block
        return "\n\n".join(kept_blocks[heading] for heading in kept_order).strip()

    def _has_required_final_report_structure(self, report_text: str) -> bool:
        text = (report_text or "").strip()
        if not text:
            return False
        missing = [
            section for section in FINAL_REPORT_REQUIRED_SECTIONS if section not in text
        ]
        if missing:
            return False
        body = self._report_body_without_appendix(text)
        if len(re.findall(r"[A-Za-z]{3,}", body)) < 80:
            return False
        return True

    def _final_report_is_invalid(self, content: str, task_description: str) -> bool:
        text = (content or "").strip()
        if not text:
            return True
        if self._is_meta_critique_output(text, task_description):
            return True
        if self._has_disallowed_report_review_language(text, task_description):
            return True
        if not self._has_required_final_report_structure(text):
            return True
        return False

    def _build_guidance_from_decision(
        self,
        task: Task,
        task_state: TaskState,
        decision: OrchestratorDecision,
        readiness_reason: str = "",
    ) -> str:
        """Generate concrete guidance for subtask agents from decision + metrics."""
        guidance_parts: list[str] = []

        metrics = self._get_current_subtask_agent_metrics(task, task_state)
        evidence_stats = self._get_current_subtask_evidence_stats(task, task_state)
        feedback = self._sanitize_feedback_for_agent(
            task,
            task_state,
            decision.feedback or "",
            metrics=metrics,
            evidence_stats=evidence_stats,
        )
        if feedback and not self._should_suppress_ungrounded_feedback(
            task,
            task_state,
            feedback,
            metrics=metrics,
            evidence_stats=evidence_stats,
        ):
            guidance_parts.append(feedback)
        current_subtask = None
        if (
            task_state.research_plan
            and task_state.research_plan.subtasks
            and task_state.subtask_index < len(task_state.research_plan.subtasks)
        ):
            current_subtask = task_state.research_plan.subtasks[
                task_state.subtask_index
            ]
        task_family = self._guidance_task_family(task, task_state)

        min_searches = max(0, int(os.getenv("ORCHESTRATOR_MIN_SUBTASK_SEARCHES", "2")))
        min_unique_queries = max(
            0, int(os.getenv("ORCHESTRATOR_MIN_SUBTASK_UNIQUE_QUERIES", "2"))
        )
        min_authoritative_results = max(
            1, int(os.getenv("ORCHESTRATOR_MIN_AUTHORITATIVE_RESULTS", "2"))
        )
        min_official_results = max(
            0, int(os.getenv("ORCHESTRATOR_MIN_OFFICIAL_RESULTS", "1"))
        )
        missing_searches = max(0, min_searches - metrics["search_count"])
        missing_queries = max(0, min_unique_queries - metrics["query_count"])
        missing_authoritative = max(
            0, min_authoritative_results - evidence_stats["authoritative_results"]
        )
        missing_official = max(
            0, min_official_results - evidence_stats["official_results"]
        )
        pathway_guidance_applies = (
            current_subtask is not None
            and self._subtask_requires_broad_discovery(current_subtask)
            and metrics["pathway_count"] > 0
        )
        min_pathways = max(
            1,
            int(
                os.getenv(
                    "ORCHESTRATOR_MIN_DISCOVERY_PATHWAYS_ATTEMPTED",
                    "2",
                )
            ),
        )
        effective_min_pathways = min(min_pathways, metrics["pathway_count"])
        missing_pathways = max(
            0,
            effective_min_pathways - metrics["pathway_attempted_count"],
        )
        if missing_searches > 0:
            guidance_parts.append(
                f"Run at least {missing_searches} more targeted searches tied directly to the current success criteria."
            )
        if pathway_guidance_applies and missing_pathways > 0:
            guidance_parts.append(
                f"Cover at least {missing_pathways} more discovery pathway or source-class routes before advancing."
            )
        elif missing_queries > 0:
            guidance_parts.append(
                f"Increase search diversity with at least {missing_queries} additional distinct query variants."
            )
        if missing_authoritative > 0:
            guidance_parts.append(
                self._authoritative_guidance_text(task_family, missing_authoritative)
            )
        if missing_official > 0:
            guidance_parts.append(self._primary_verification_guidance_text(task_family))
        if metrics["checkpoint_count"] == 0 and metrics["action_count"] >= 3:
            guidance_parts.append(
                "Create a checkpoint summary to lock in evidence before deciding to transition."
            )

        stagnation_rounds = task_state.subtask_stagnation_rounds.get(
            task_state.subtask_index, 0
        )
        if stagnation_rounds >= 2:
            if pathway_guidance_applies and metrics["pathway_uncovered_count"] > 0:
                guidance_parts.append(
                    "Current line of inquiry is stagnating; pivot to an uncovered pathway or source class instead of paraphrasing prior queries."
                )
            else:
                guidance_parts.append(
                    "Current line of inquiry is stagnating; pivot to a different angle, source type, or constraint."
                )

        agent_state = self._get_current_subtask_agent_state(task, task_state)
        if agent_state is not None and getattr(agent_state, "search_history", None):
            recent_queries: list[str] = []
            seen_recent_queries: set[str] = set()
            for item in list(agent_state.search_history)[-3:]:
                query = str(item.get("query", "")).strip()
                normalized_query = self._normalize_recent_query_for_guidance(query)
                if normalized_query and normalized_query not in seen_recent_queries:
                    seen_recent_queries.add(normalized_query)
                    recent_queries.append(normalized_query)
            if recent_queries:
                guidance_parts.append(
                    "Avoid repeating recent queries: " + "; ".join(recent_queries)
                )

        if current_subtask is not None and not guidance_parts:
            guidance_parts.append(
                "Continue focused exploration and gather concrete evidence for: "
                f"{current_subtask.success_criteria}"
            )

        if readiness_reason:
            guidance_parts.append(f"Transition blocked reason: {readiness_reason}")

        deduped_parts: list[str] = []
        for part in guidance_parts:
            normalized = " ".join(part.split())
            if normalized and normalized not in deduped_parts:
                deduped_parts.append(normalized)
        return " ".join(deduped_parts).strip()

    def _guidance_task_family(self, task: Task, task_state: TaskState) -> str:
        texts = [
            str(getattr(task, "description", "") or ""),
            str(getattr(task, "name", "") or ""),
        ]
        if task_state.research_plan and task_state.research_plan.subtasks:
            if 0 <= task_state.subtask_index < len(task_state.research_plan.subtasks):
                subtask = task_state.research_plan.subtasks[task_state.subtask_index]
                texts.extend(
                    [
                        str(getattr(subtask, "description", "") or ""),
                        str(getattr(subtask, "success_criteria", "") or ""),
                    ]
                )
        return infer_research_task_family(*texts)

    @staticmethod
    def _authoritative_guidance_text(task_family: str, missing_count: int) -> str:
        if task_family == "papers":
            return (
                f"Find at least {missing_count} more authoritative sources such as proceedings pages, "
                "publisher pages, lab pages, or benchmark repositories."
            )
        if task_family == "ecosystem":
            return (
                f"Find at least {missing_count} more authoritative sources such as official repositories, "
                "maintainer documentation, organization pages, or ecosystem indexes."
            )
        return (
            f"Find at least {missing_count} more authoritative sources (.gov, .edu, "
            "official program pages, or primary organizations)."
        )

    @staticmethod
    def _primary_verification_guidance_text(task_family: str) -> str:
        if task_family == "papers":
            return "Verify key claims directly on a proceedings page, publisher page, lab page, or primary paper source before advancing."
        if task_family == "ecosystem":
            return "Verify key claims directly on a maintainer-controlled repository, documentation page, or primary organization source before advancing."
        return "Verify key claims directly on an official source before advancing."

    def _sanitize_feedback_for_agent(
        self,
        task: Task,
        task_state: TaskState,
        feedback: str,
        *,
        metrics: dict[str, int],
        evidence_stats: dict[str, int],
    ) -> str:
        text = " ".join((feedback or "").split()).strip()
        if not text:
            return ""

        text = re.sub(
            r"(?i)\buse queries like:\s*.+$",
            "",
            text,
        ).strip(" ;,")
        text = re.sub(
            r"(?i)\brun searches including\s+[^.]+\.?",
            "",
            text,
        ).strip(" ;,")

        grounded_text = " ".join(
            [
                str(getattr(task, "description", "") or ""),
                str(getattr(task, "name", "") or ""),
            ]
        ).lower()
        if task_state.research_plan and task_state.research_plan.subtasks:
            if 0 <= task_state.subtask_index < len(task_state.research_plan.subtasks):
                subtask = task_state.research_plan.subtasks[task_state.subtask_index]
                grounded_text += (
                    " " + str(getattr(subtask, "description", "") or "").lower()
                )
                grounded_text += (
                    " " + str(getattr(subtask, "success_criteria", "") or "").lower()
                )

        agent_state = self._get_current_subtask_agent_state(task, task_state)
        if agent_state is not None:
            for entry in list(getattr(agent_state, "search_history", []) or [])[-6:]:
                grounded_text += " " + str(entry.get("query", "") or "").lower()
                for hit in list(entry.get("results", []) or [])[:6]:
                    grounded_text += " " + str(hit.get("title", "") or "").lower()
                    grounded_text += " " + str(hit.get("domain", "") or "").lower()

        grounded_tokens = set(re.findall(r"[a-z][a-z0-9]{1,}", grounded_text))
        low_evidence = (
            int(metrics.get("search_count", 0) or 0) <= 1
            and int(evidence_stats.get("authoritative_results", 0) or 0) <= 1
            and int(evidence_stats.get("official_results", 0) or 0) == 0
        )
        profile = build_research_task_profile(grounded_text)
        allowed_abstract_tokens = (
            set(RESEARCH_AUDIENCE_TERMS)
            | set(RESEARCH_SOURCE_CLASS_TERMS)
            | set(RESEARCH_EVIDENCE_TERMS)
            | set(profile.topic_terms)
            | set(profile.target_terms)
            | set(profile.source_terms)
            | set(profile.evidence_terms)
            | set(profile.audience_terms)
            | {
                "adjacent",
                "alongside",
                "authoritative",
                "broad",
                "broaden",
                "broader",
                "host",
                "include",
                "primary",
                "related",
                "verification",
                "verify",
            }
        )
        drop_markers = (
            "use queries like",
            "run searches including",
            "site:.",
            "`",
            '"',
        )
        sentence_candidates = re.split(r"(?<=[.!?])\s+", text)
        kept: list[str] = []
        for sentence in sentence_candidates:
            normalized = sentence.strip(" ;,")
            if not normalized:
                continue
            lowered = normalized.lower()
            if any(marker in lowered for marker in drop_markers):
                continue
            sentence_tokens = set(re.findall(r"[a-z][a-z0-9]{1,}", lowered))
            expansion_markers = (
                "broaden",
                "broader",
                "adjacent",
                "alongside",
                "related to",
                "include",
            )
            if any(marker in lowered for marker in expansion_markers):
                scope_sentence = re.sub(r"(?i)\bsuch as\b.+$", "", normalized).strip(
                    " ;,"
                )
                scope_sentence = re.sub(
                    r"(?i)^\s*(?:also|specifically)\s*,\s*", "", scope_sentence
                ).strip(" ;,")
                if scope_sentence:
                    kept.append(scope_sentence)
                    continue
            expansion_terms = set()
            if (
                any(marker in lowered for marker in expansion_markers)
                and "such as" not in lowered
            ):
                expansion_terms = set(build_research_task_profile(lowered).topic_terms)
            suspicious_tokens = {
                token
                for token in sentence_tokens
                if len(token) >= 4
                and token not in grounded_tokens
                and token not in allowed_abstract_tokens
                and token not in expansion_terms
            }
            if low_evidence and suspicious_tokens:
                continue
            if "such as" in lowered and suspicious_tokens:
                continue
            if len(suspicious_tokens) >= 3:
                continue
            kept.append(normalized)

        sanitized = " ".join(dict.fromkeys(kept)).strip()
        if sanitized:
            return sanitized

        planner_failures = int(metrics.get("consecutive_planner_failures", 0) or 0)
        if planner_failures > 0:
            return (
                "Planner instability detected. Pivot to simpler searches centered on official pages, "
                "primary organizations, and directories directly tied to the current success criteria."
            )
        if int(evidence_stats.get("authoritative_results", 0) or 0) <= 1:
            task_family = self._guidance_task_family(task, task_state)
            if task_family == "papers":
                return (
                    "Broaden the search to adjacent in-scope paper source classes and artifact types. "
                    "Prioritize proceedings, publishers, lab pages, and benchmark repositories instead of repeating narrow lead-specific queries."
                )
            if task_family == "ecosystem":
                return (
                    "Broaden the search to adjacent in-scope ecosystem source classes and artifact types. "
                    "Prioritize repositories, maintainer documentation, organization pages, and ecosystem indexes instead of repeating narrow lead-specific queries."
                )
            return (
                "Broaden the search to adjacent in-scope opportunity types and prioritize official pages, "
                "primary organizations, and directories instead of repeating narrow lead-specific queries."
            )
        return ""

    @staticmethod
    def _should_suppress_ungrounded_feedback(
        task: Task,
        task_state: TaskState,
        feedback: str,
        *,
        metrics: dict[str, int],
        evidence_stats: dict[str, int],
    ) -> bool:
        text = (feedback or "").strip().lower()
        if not text:
            return False

        low_evidence = (
            int(metrics.get("search_count", 0) or 0) <= 1
            and int(evidence_stats.get("authoritative_results", 0) or 0) == 0
            and int(evidence_stats.get("official_results", 0) or 0) == 0
        )
        if not low_evidence:
            return False

        prescriptive_markers = (
            "try the query",
            "switch to a new query",
            "execute the following searches",
            "focus on the query",
            "specifically,",
            "search for opportunities on the websites",
            "site:.",
            "`",
            '"',
        )
        if not any(marker in text for marker in prescriptive_markers):
            return False

        grounded_text = " ".join(
            [
                str(getattr(task, "description", "") or ""),
                str(getattr(task, "name", "") or ""),
            ]
        ).lower()
        if task_state.research_plan and task_state.research_plan.subtasks:
            if 0 <= task_state.subtask_index < len(task_state.research_plan.subtasks):
                subtask = task_state.research_plan.subtasks[task_state.subtask_index]
                grounded_text += (
                    " " + str(getattr(subtask, "description", "") or "").lower()
                )
                grounded_text += (
                    " " + str(getattr(subtask, "success_criteria", "") or "").lower()
                )

        grounded_tokens = set(re.findall(r"[a-z][a-z0-9]{1,}", grounded_text))
        feedback_tokens = set(re.findall(r"[a-z][a-z0-9]{1,}", text))
        novel_tokens = {
            token
            for token in feedback_tokens
            if len(token) >= 3 and token not in grounded_tokens
        }

        suspicious_tokens = {
            token
            for token in novel_tokens
            if token
            not in {
                "query",
                "search",
                "official",
                "program",
                "programs",
                "research",
                "undergraduate",
                "students",
                "university",
                "universities",
                "organization",
                "organizations",
                "authoritative",
                "source",
                "sources",
                "opportunities",
                "fellowship",
                "fellowships",
                "grant",
                "grants",
                "scholarship",
                "scholarships",
            }
        }
        if suspicious_tokens and any(
            marker in text
            for marker in (
                "broaden",
                "broader",
                "adjacent",
                "alongside",
                "related to",
                "include",
            )
        ):
            suspicious_tokens = {
                token
                for token in suspicious_tokens
                if token not in set(build_research_task_profile(text).topic_terms)
            }
        return bool(suspicious_tokens)

    @staticmethod
    def _normalize_recent_query_for_guidance(query: str) -> str:
        """Remove low-signal synthetic suffixes and suppress degenerate query echoes."""
        normalized = " ".join((query or "").split()).strip()
        if not normalized:
            return ""
        normalized = re.sub(r"(?i)\s+angle\s+\d+\b", "", normalized).strip()
        normalized = re.sub(r"\s{2,}", " ", normalized).strip()
        if not normalized:
            return ""
        token_count = len(normalized.split())
        if token_count < 2:
            return ""
        return normalized

    def _evaluate_subtask_completion_readiness(
        self, task: Task, task_state: TaskState
    ) -> tuple[bool, str]:
        """
        Determine whether we have enough exploration signal to advance/complete.
        This is a deterministic safety net against premature orchestrator transitions.
        """
        metrics = self._get_current_subtask_agent_metrics(task, task_state)
        min_actions = max(1, int(os.getenv("ORCHESTRATOR_MIN_SUBTASK_ACTIONS", "4")))
        min_searches = max(0, int(os.getenv("ORCHESTRATOR_MIN_SUBTASK_SEARCHES", "2")))
        min_unique_queries = max(
            0, int(os.getenv("ORCHESTRATOR_MIN_SUBTASK_UNIQUE_QUERIES", "2"))
        )
        min_checkpoints = max(
            0, int(os.getenv("ORCHESTRATOR_MIN_SUBTASK_CHECKPOINTS", "0"))
        )
        min_authoritative_results = max(
            1, int(os.getenv("ORCHESTRATOR_MIN_AUTHORITATIVE_RESULTS", "2"))
        )
        min_unique_domains = max(
            1, int(os.getenv("ORCHESTRATOR_MIN_UNIQUE_SOURCE_DOMAINS", "2"))
        )
        min_official_results = max(
            0, int(os.getenv("ORCHESTRATOR_MIN_OFFICIAL_RESULTS", "1"))
        )
        min_fetched_results = max(
            0, int(os.getenv("ORCHESTRATOR_MIN_FETCHED_RESULTS", "1"))
        )
        min_pathways = max(
            1,
            int(
                os.getenv(
                    "ORCHESTRATOR_MIN_DISCOVERY_PATHWAYS_ATTEMPTED",
                    "2",
                )
            ),
        )

        evidence_stats = self._get_current_subtask_evidence_stats(task, task_state)
        current_subtask = None
        if task_state.research_plan and task_state.research_plan.subtasks:
            if 0 <= task_state.subtask_index < len(task_state.research_plan.subtasks):
                current_subtask = task_state.research_plan.subtasks[
                    task_state.subtask_index
                ]

        actions_ok = metrics["action_count"] >= min_actions
        coverage_ok = metrics["query_count"] >= min_unique_queries
        evidence_ok = (
            metrics["search_count"] >= min_searches
            and metrics["checkpoint_count"] >= min_checkpoints
        ) or (
            metrics["summary_count"] > 0
            and metrics["search_count"] >= min_searches
            and coverage_ok
        )
        authority_ok = (
            evidence_stats["authoritative_results"] >= min_authoritative_results
            and evidence_stats["unique_domains"] >= min_unique_domains
            and evidence_stats["official_results"] >= min_official_results
            and evidence_stats["fetched_results"] >= min_fetched_results
        )

        breadth_ok = True
        breadth_reason = ""
        if current_subtask is not None and self._subtask_requires_broad_discovery(
            current_subtask
        ):
            target_count = self._subtask_target_count(current_subtask)
            distinct_target = max(
                3,
                int(
                    os.getenv(
                        "ORCHESTRATOR_MIN_DISTINCT_DISCOVERY_RESULTS",
                        str(target_count),
                    )
                ),
            )
            breadth_ok = evidence_stats[
                "distinct_authoritative_titles"
            ] >= distinct_target and evidence_stats[
                "unique_authoritative_domains"
            ] >= min(
                3, distinct_target
            )
            if not breadth_ok:
                breadth_reason = (
                    f"distinct_authoritative_titles {evidence_stats['distinct_authoritative_titles']}/{distinct_target}, "
                    f"unique_authoritative_domains {evidence_stats['unique_authoritative_domains']}/{min(3, distinct_target)}"
                )
            if metrics["pathway_count"] > 0:
                required_pathways = min(min_pathways, metrics["pathway_count"])
                pathway_ok = metrics["pathway_attempted_count"] >= required_pathways
                breadth_ok = breadth_ok and pathway_ok
                if not pathway_ok:
                    pathway_reason = (
                        f"pathways_attempted {metrics['pathway_attempted_count']}/{required_pathways}, "
                        f"pathways_productive {metrics['pathway_hit_count']}, "
                        f"pathways_uncovered {metrics['pathway_uncovered_count']}"
                    )
                    breadth_reason = (
                        f"{breadth_reason}, {pathway_reason}"
                        if breadth_reason
                        else pathway_reason
                    )

        if actions_ok and evidence_ok and coverage_ok and authority_ok and breadth_ok:
            return True, "Exploration depth appears sufficient."

        reason = (
            "Insufficient exploration depth: "
            f"actions {metrics['action_count']}/{min_actions}, "
            f"searches {metrics['search_count']}/{min_searches}, "
            f"unique_queries {metrics['query_count']}/{min_unique_queries}, "
            f"checkpoints {metrics['checkpoint_count']}/{min_checkpoints}, "
            f"summaries {metrics['summary_count']}, "
            f"authoritative_results {evidence_stats['authoritative_results']}/{min_authoritative_results}, "
            f"official_results {evidence_stats['official_results']}/{min_official_results}, "
            f"fetched_results {evidence_stats['fetched_results']}/{min_fetched_results}, "
            f"unique_domains {evidence_stats['unique_domains']}/{min_unique_domains}"
            + (f", breadth {breadth_reason}" if breadth_reason else "")
            + "."
        )
        return False, reason

    def _evaluate_ceiling_progression_readiness(
        self, task: Task, task_state: TaskState
    ) -> tuple[bool, str]:
        """
        Apply a stricter readiness bar before using the iteration ceiling to force
        progression. Ceiling-based advancement should be a last resort, not normal
        task flow.
        """
        base_ready, base_reason = self._evaluate_subtask_completion_readiness(
            task, task_state
        )
        if not base_ready:
            return False, base_reason

        metrics = self._get_current_subtask_agent_metrics(task, task_state)
        evidence_stats = self._get_current_subtask_evidence_stats(task, task_state)
        current_subtask = None
        if task_state.research_plan and task_state.research_plan.subtasks:
            if 0 <= task_state.subtask_index < len(task_state.research_plan.subtasks):
                current_subtask = task_state.research_plan.subtasks[
                    task_state.subtask_index
                ]

        min_ceiling_searches = max(
            4, int(os.getenv("ORCHESTRATOR_CEILING_MIN_SEARCHES", "4"))
        )
        min_ceiling_queries = max(
            3, int(os.getenv("ORCHESTRATOR_CEILING_MIN_UNIQUE_QUERIES", "3"))
        )
        min_ceiling_checkpoints = max(
            1, int(os.getenv("ORCHESTRATOR_CEILING_MIN_CHECKPOINTS", "1"))
        )
        min_ceiling_fetched_results = max(
            2, int(os.getenv("ORCHESTRATOR_CEILING_MIN_FETCHED_RESULTS", "2"))
        )
        min_ceiling_official_results = max(
            2, int(os.getenv("ORCHESTRATOR_CEILING_MIN_OFFICIAL_RESULTS", "2"))
        )
        min_ceiling_official_titles = max(
            2, int(os.getenv("ORCHESTRATOR_CEILING_MIN_DISTINCT_OFFICIAL_TITLES", "2"))
        )

        if (
            metrics["search_count"] < min_ceiling_searches
            or metrics["query_count"] < min_ceiling_queries
            or metrics["checkpoint_count"] < min_ceiling_checkpoints
            or evidence_stats["fetched_results"] < min_ceiling_fetched_results
            or evidence_stats["official_results"] < min_ceiling_official_results
            or evidence_stats["distinct_official_titles"] < min_ceiling_official_titles
        ):
            reason = (
                "Ceiling progression blocked: stronger evidence required. "
                f"searches {metrics['search_count']}/{min_ceiling_searches}, "
                f"unique_queries {metrics['query_count']}/{min_ceiling_queries}, "
                f"checkpoints {metrics['checkpoint_count']}/{min_ceiling_checkpoints}, "
                f"fetched_results {evidence_stats['fetched_results']}/{min_ceiling_fetched_results}, "
                f"official_results {evidence_stats['official_results']}/{min_ceiling_official_results}, "
                f"distinct_official_titles {evidence_stats['distinct_official_titles']}/{min_ceiling_official_titles}."
            )
            return False, reason

        if current_subtask is not None and self._subtask_requires_broad_discovery(
            current_subtask
        ):
            target_count = max(4, self._subtask_target_count(current_subtask))
            if evidence_stats[
                "distinct_authoritative_titles"
            ] < target_count or evidence_stats["unique_authoritative_domains"] < min(
                4, target_count
            ):
                reason = (
                    "Ceiling progression blocked: discovery breadth is still too narrow. "
                    f"distinct_authoritative_titles {evidence_stats['distinct_authoritative_titles']}/{target_count}, "
                    f"unique_authoritative_domains {evidence_stats['unique_authoritative_domains']}/{min(4, target_count)}."
                )
                return False, reason

        return True, "Ceiling progression thresholds satisfied."

    def _hard_subtask_iteration_limit(self) -> int:
        default_limit = self.max_iterations_per_subtask + max(
            5,
            self.max_iterations_per_subtask // 2,
        )
        raw_value = os.getenv(
            "ORCHESTRATOR_HARD_MAX_ITERATIONS_PER_SUBTASK",
            str(default_limit),
        )
        try:
            configured = int(raw_value)
        except (TypeError, ValueError):
            configured = default_limit
        return max(self.max_iterations_per_subtask + 1, configured)

    def _hard_ceiling_progression_required(
        self, task: Task, task_state: TaskState
    ) -> tuple[bool, str]:
        metrics = self._get_current_subtask_agent_metrics(task, task_state)
        hard_limit = self._hard_subtask_iteration_limit()
        action_count = int(metrics.get("action_count", 0) or 0)
        if action_count < hard_limit:
            return False, ""

        evidence_stats = self._get_current_subtask_evidence_stats(task, task_state)
        reason = (
            "Hard per-subtask iteration limit reached; forcing progression with explicit uncertainty. "
            f"actions {action_count}/{hard_limit}, "
            f"search_attempts {metrics['search_attempt_count']}, "
            f"successful_searches {metrics['search_count']}, "
            f"unique_queries {metrics['query_count']}, "
            f"authoritative_results {evidence_stats['authoritative_results']}, "
            f"official_results {evidence_stats['official_results']}."
        )
        return True, reason

    def _evaluate_plateau_progression_readiness(
        self, task: Task, task_state: TaskState
    ) -> tuple[bool, str]:
        """
        Allow discovery subtasks to conclude when breadth has clearly plateaued
        near target and current evidence is already strong enough to support a
        constrained shortlist with uncertainty carried forward.
        """
        current_subtask = None
        if task_state.research_plan and task_state.research_plan.subtasks:
            if 0 <= task_state.subtask_index < len(task_state.research_plan.subtasks):
                current_subtask = task_state.research_plan.subtasks[
                    task_state.subtask_index
                ]
        if current_subtask is None or not self._subtask_requires_broad_discovery(
            current_subtask
        ):
            return (
                False,
                "Plateau progression is only applicable to discovery subtasks.",
            )

        plateau_rounds = task_state.subtask_stagnation_rounds.get(
            task_state.subtask_index, 0
        )
        min_plateau_rounds = max(
            2, int(os.getenv("ORCHESTRATOR_DISCOVERY_PLATEAU_ROUNDS", "3"))
        )
        if plateau_rounds < min_plateau_rounds:
            return False, (
                "Discovery plateau not yet established: "
                f"stagnation_rounds {plateau_rounds}/{min_plateau_rounds}."
            )

        metrics = self._get_current_subtask_agent_metrics(task, task_state)
        evidence_stats = self._get_current_subtask_evidence_stats(task, task_state)
        target_count = max(4, self._subtask_target_count(current_subtask))
        near_target_buffer = max(
            1, int(os.getenv("ORCHESTRATOR_DISCOVERY_NEAR_TARGET_BUFFER", "1"))
        )
        effective_target = max(3, target_count - near_target_buffer)

        practical_searches = max(
            3, int(os.getenv("ORCHESTRATOR_DISCOVERY_PLATEAU_MIN_SEARCHES", "3"))
        )
        practical_queries = max(
            3, int(os.getenv("ORCHESTRATOR_DISCOVERY_PLATEAU_MIN_UNIQUE_QUERIES", "3"))
        )
        practical_checkpoints = max(
            1, int(os.getenv("ORCHESTRATOR_DISCOVERY_PLATEAU_MIN_CHECKPOINTS", "1"))
        )
        practical_authoritative = max(
            4,
            int(
                os.getenv(
                    "ORCHESTRATOR_DISCOVERY_PLATEAU_MIN_AUTHORITATIVE_RESULTS", "4"
                )
            ),
        )
        practical_official = max(
            2,
            int(os.getenv("ORCHESTRATOR_DISCOVERY_PLATEAU_MIN_OFFICIAL_RESULTS", "2")),
        )
        practical_domains = max(
            3, int(os.getenv("ORCHESTRATOR_DISCOVERY_PLATEAU_MIN_UNIQUE_DOMAINS", "3"))
        )

        practical_ok = (
            metrics["search_count"] >= practical_searches
            and metrics["query_count"] >= practical_queries
            and metrics["checkpoint_count"] >= practical_checkpoints
            and evidence_stats["authoritative_results"] >= practical_authoritative
            and evidence_stats["official_results"] >= practical_official
            and evidence_stats["unique_authoritative_domains"] >= practical_domains
            and evidence_stats["distinct_authoritative_titles"] >= effective_target
        )
        if not practical_ok:
            return False, (
                "Discovery plateau detected but practical sufficiency is not yet met. "
                f"searches {metrics['search_count']}/{practical_searches}, "
                f"unique_queries {metrics['query_count']}/{practical_queries}, "
                f"checkpoints {metrics['checkpoint_count']}/{practical_checkpoints}, "
                f"authoritative_results {evidence_stats['authoritative_results']}/{practical_authoritative}, "
                f"official_results {evidence_stats['official_results']}/{practical_official}, "
                f"unique_authoritative_domains {evidence_stats['unique_authoritative_domains']}/{practical_domains}, "
                f"distinct_authoritative_titles {evidence_stats['distinct_authoritative_titles']}/{effective_target}."
            )

        return True, (
            "Discovery appears saturated near target; advancing with explicit uncertainty. "
            f"stagnation_rounds={plateau_rounds}, "
            f"distinct_authoritative_titles={evidence_stats['distinct_authoritative_titles']}/{target_count}."
        )

    def _get_current_subtask_evidence_stats(
        self, task: Task, task_state: TaskState
    ) -> dict[str, int]:
        agent_state = self._get_current_subtask_agent_state(task, task_state)
        if agent_state is None:
            return {
                "authoritative_results": 0,
                "official_results": 0,
                "unique_domains": 0,
                "fetched_results": 0,
            }

        authoritative_results = 0
        official_results = 0
        fetched_results = 0
        unique_domains: set[str] = set()
        unique_authoritative_domains: set[str] = set()
        unique_official_domains: set[str] = set()
        distinct_authoritative_titles: set[str] = set()
        distinct_official_titles: set[str] = set()
        for entry in list(getattr(agent_state, "search_history", []) or []):
            for hit in list(entry.get("results", []) or []):
                if hit.get("task_aligned") is False:
                    continue
                domain = str(hit.get("domain", "")).strip().lower()
                title = self._normalize_hit_title(hit.get("title", ""))
                if int(hit.get("authority_score", 0) or 0) >= 3:
                    authoritative_results += 1
                    if domain:
                        unique_authoritative_domains.add(domain)
                    if title:
                        distinct_authoritative_titles.add(title)
                if bool(hit.get("official_source", False)):
                    official_results += 1
                    if domain:
                        unique_official_domains.add(domain)
                    if title:
                        distinct_official_titles.add(title)
                if bool(hit.get("fetched", False)):
                    fetched_results += 1
                if domain:
                    unique_domains.add(domain)

        return {
            "authoritative_results": authoritative_results,
            "official_results": official_results,
            "unique_domains": len(unique_domains),
            "unique_authoritative_domains": len(unique_authoritative_domains),
            "unique_official_domains": len(unique_official_domains),
            "distinct_authoritative_titles": len(distinct_authoritative_titles),
            "distinct_official_titles": len(distinct_official_titles),
            "fetched_results": fetched_results,
        }

    @staticmethod
    def _normalize_hit_title(title: object) -> str:
        normalized = re.sub(r"\s+", " ", str(title or "").strip().lower())
        normalized = re.sub(r"^[0-9]{4}\s+", "", normalized)
        return normalized

    @staticmethod
    def _subtask_requires_broad_discovery(subtask: Subtask) -> bool:
        text = " ".join(
            [
                str(getattr(subtask, "description", "") or ""),
                str(getattr(subtask, "success_criteria", "") or ""),
            ]
        ).lower()
        trigger_terms = (
            "discover",
            "identify",
            "list",
            "shortlist",
            "all publicly available",
            "distinct",
            "opportunities",
        )
        return any(term in text for term in trigger_terms)

    @staticmethod
    def _subtask_target_count(subtask: Subtask) -> int:
        text = " ".join(
            [
                str(getattr(subtask, "description", "") or ""),
                str(getattr(subtask, "success_criteria", "") or ""),
            ]
        )
        patterns = (
            r"at least\s+(\d+)",
            r"list of\s+(\d+)",
            r"(\d+)\s+distinct",
            r"(\d+)\s+opportunit",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    raw_target = max(1, int(match.group(1)))
                    break
                except (TypeError, ValueError):
                    continue
        else:
            raw_target = 3

        raw_cap = os.getenv("ORCHESTRATOR_MAX_DISCOVERY_TARGET_COUNT", "6")
        try:
            cap = max(3, int(raw_cap))
        except (TypeError, ValueError):
            cap = 6
        return min(raw_target, cap)

    def _should_defer_llm_review(
        self, task: Task, task_state: TaskState
    ) -> tuple[bool, str]:
        """
        Decide whether to skip an expensive LLM review this round.
        This keeps worker steps bounded and avoids timeout-heavy review spam.
        """
        current_index = task_state.subtask_index
        metrics = self._get_current_subtask_agent_metrics(task, task_state)
        action_count = metrics["action_count"]
        search_attempt_count = metrics["search_attempt_count"]
        min_actions_before_review = max(
            1, int(os.getenv("ORCHESTRATOR_MIN_ACTIONS_BEFORE_REVIEW", "4"))
        )
        review_every_actions = max(
            1, int(os.getenv("ORCHESTRATOR_REVIEW_EVERY_N_ACTIONS", "3"))
        )
        last_review_action_count = task_state.subtask_last_review_action_count.get(
            current_index, 0
        )

        agent_state = self._get_current_subtask_agent_state(task, task_state)
        last_action = (
            str(getattr(agent_state, "last_action", "")).strip().lower()
            if agent_state is not None
            else ""
        )
        force_review_actions = {"summarize", "complete"}
        stagnation_rounds = task_state.subtask_stagnation_rounds.get(current_index, 0)
        evidence_stats = self._get_current_subtask_evidence_stats(task, task_state)
        low_progress_min_attempts = max(
            2, int(os.getenv("ORCHESTRATOR_LOW_PROGRESS_MIN_SEARCH_ATTEMPTS", "2"))
        )

        # Never defer review when evidence has stagnated; we need orchestration decisions.
        if stagnation_rounds >= 2:
            return False, ""
        if int(metrics.get("consecutive_planner_failures", 0) or 0) > 0:
            return False, ""
        if (
            action_count >= 2
            and search_attempt_count >= low_progress_min_attempts
            and metrics["search_count"] == 0
        ):
            return False, ""
        if (
            action_count >= min_actions_before_review
            and search_attempt_count >= low_progress_min_attempts
            and evidence_stats["authoritative_results"] == 0
            and evidence_stats["official_results"] == 0
            and evidence_stats["fetched_results"] == 0
        ):
            return False, ""

        if (
            action_count < min_actions_before_review
            and last_action not in force_review_actions
        ):
            return (
                True,
                (
                    "Deferring orchestrator review until deeper subtask evidence exists "
                    f"(actions {action_count}/{min_actions_before_review})."
                ),
            )

        if (
            action_count - last_review_action_count < review_every_actions
            and last_action not in force_review_actions
        ):
            return (
                True,
                (
                    "Deferring orchestrator review to reduce overhead "
                    f"(delta_actions {action_count - last_review_action_count}/{review_every_actions})."
                ),
            )

        return False, ""

    def _build_basic_review_context(
        self,
        task: Task,
        task_state: TaskState,
        latest_notes: str,
        current_subtask_info: str,
    ) -> str:
        """Build compact non-token-aware review context."""
        subtask_metrics = self._get_current_subtask_agent_metrics(task, task_state)
        active_guidance = task_state.get_subtask_guidance(task_state.subtask_index)
        stagnation_rounds = task_state.subtask_stagnation_rounds.get(
            task_state.subtask_index, 0
        )
        max_notes_tokens = max(
            256,
            int(os.getenv("ORCHESTRATOR_BASIC_REVIEW_MAX_NOTES_TOKENS", "6000")),
        )
        processed_notes = self._truncate_text_by_token_budget(
            latest_notes,
            max_tokens=max_notes_tokens,
            fallback_label="approx token truncation",
        )
        context_parts = [
            f"Current time: {datetime.now()}",
            f"Task: {task.name} - {task.description}",
            f"Progress: {task_state.get_progress_percentage():.1f}%",
            f"Completed subtasks: {len(task_state.completed_subtasks)}/{len(task_state.research_plan.subtasks) if task_state.research_plan else 0}",
            f"Decision count: {task_state.decision_count}",
            f"Stuck count: {task_state.stuck_count}",
            f"Subtask stagnation rounds: {stagnation_rounds}",
            current_subtask_info,
            (
                "Current subtask metrics: "
                f"actions={subtask_metrics['action_count']}, "
                f"searches={subtask_metrics['search_count']}, "
                f"summaries={subtask_metrics['summary_count']}, "
                f"checkpoints={subtask_metrics['checkpoint_count']}, "
                f"unique_queries={subtask_metrics['query_count']}"
            ),
            f"Active orchestrator guidance: {active_guidance or 'None'}",
            f"Recent notes: {processed_notes}",
            f"Previous feedback: {task_state.feedback_history[-1] if task_state.feedback_history else 'None'}",
        ]
        return "\n".join(context_parts)

    def _build_token_aware_review_context(
        self,
        task: Task,
        task_state: TaskState,
        latest_notes: str,
        current_subtask_info: str,
        system_prompt: str,
    ) -> str:
        """Build token-budgeted review context."""
        metadata_parts = [
            f"Current time: {datetime.now()}",
            f"Progress: {task_state.get_progress_percentage():.1f}%",
            f"Completed subtasks: {len(task_state.completed_subtasks)}/{len(task_state.research_plan.subtasks) if task_state.research_plan else 0}",
            f"Decision count: {task_state.decision_count}",
            f"Stuck count: {task_state.stuck_count}",
            (
                "Subtask stagnation rounds: "
                f"{task_state.subtask_stagnation_rounds.get(task_state.subtask_index, 0)}"
            ),
            current_subtask_info,
            f"Previous feedback: {task_state.feedback_history[-1] if task_state.feedback_history else 'None'}",
            (
                "Active orchestrator guidance: "
                f"{task_state.get_subtask_guidance(task_state.subtask_index) or 'None'}"
            ),
        ]
        subtask_metrics = self._get_current_subtask_agent_metrics(task, task_state)
        metadata_parts.append(
            "Current subtask metrics: "
            f"actions={subtask_metrics['action_count']}, "
            f"searches={subtask_metrics['search_count']}, "
            f"summaries={subtask_metrics['summary_count']}, "
            f"checkpoints={subtask_metrics['checkpoint_count']}, "
            f"unique_queries={subtask_metrics['query_count']}"
        )
        full_task = f"{task.name} - {task.description}\n\n" + "\n".join(metadata_parts)
        task_tokens = self.token_counter.count_tokens(full_task)
        system_tokens = self.token_counter.count_tokens(system_prompt)

        available_for_user = (
            self.token_budget.max_context
            - system_tokens
            - self.token_budget.response_reserve
        )
        max_notes_tokens = max(100, available_for_user - task_tokens - 100)
        notes_tokens = self.token_counter.count_tokens(latest_notes)

        if notes_tokens > max_notes_tokens and self.summarizer:
            summary, level, _facts = self.summarizer.create_summary_on_demand(
                latest_notes,
                max_tokens=max_notes_tokens,
                preserve_facts=True,
            )
            logger.debug(
                "Summarized orchestrator notes: %s -> %s tokens (level: %s)",
                notes_tokens,
                self.token_counter.count_tokens(summary),
                level,
            )
            processed_notes = summary
        elif notes_tokens > max_notes_tokens:
            processed_notes = self.token_counter.truncate_to_tokens(
                latest_notes, max_notes_tokens
            )
            logger.debug(
                "Truncated orchestrator notes: %s -> %s tokens",
                notes_tokens,
                self.token_counter.count_tokens(processed_notes),
            )
        else:
            processed_notes = latest_notes

        user_content = f"Task: {full_task}\nRecent notes:\n{processed_notes}"
        total_user_tokens = self.token_counter.count_tokens(user_content)
        max_allowed = self.token_budget.max_context - self.token_budget.response_reserve
        total_with_system = system_tokens + total_user_tokens

        if total_with_system > max_allowed:
            excess = total_with_system - max_allowed
            processed_notes = self.token_counter.truncate_to_tokens(
                processed_notes,
                max(1, self.token_counter.count_tokens(processed_notes) - excess),
            )
            user_content = f"Task: {full_task}\nRecent notes:\n{processed_notes}"
            logger.warning(
                "Emergency truncation applied: reduced notes to fit context limit"
            )

        final_tokens = self.token_counter.count_tokens(user_content)
        logger.debug(
            "Orchestrator context: %s tokens (system: %s, task: %s, notes: %s, total: %s)",
            final_tokens,
            system_tokens,
            task_tokens,
            self.token_counter.count_tokens(processed_notes),
            system_tokens + final_tokens,
        )
        return user_content

    def _build_review_context(
        self,
        task: Task,
        task_state: TaskState,
        latest_notes: str,
        current_subtask_info: str,
        system_prompt: str,
    ) -> str:
        """Build review context with token-aware path and safe fallback."""
        if (
            not self.context_management_enabled
            or not self.token_counter
            or not self.token_budget
        ):
            return self._build_basic_review_context(
                task, task_state, latest_notes, current_subtask_info
            )

        try:
            return self._build_token_aware_review_context(
                task,
                task_state,
                latest_notes,
                current_subtask_info,
                system_prompt,
            )
        except Exception as e:
            logger.warning(
                "Error in orchestrator context management, using basic mode: %s",
                e,
                exc_info=True,
            )
            return self._build_basic_review_context(
                task, task_state, latest_notes, current_subtask_info
            )

    async def _invoke_chain_with_policy(
        self,
        invoke_fn,
        *,
        operation: str,
        timeout_seconds: float | None,
        retries: int,
        fallback_factory,
        async_mode: bool = False,
    ):
        """Invoke a sync chain function with retry/timeout and deterministic fallback."""
        attempts = max(1, retries + 1)
        last_error: Exception | None = None
        timeout_disabled = (
            timeouts_disabled()
            or timeout_seconds is None
            or float(timeout_seconds) <= 0
        )
        if timeout_disabled:
            for attempt in range(1, attempts + 1):
                try:
                    started_at = datetime.now()
                    if async_mode:
                        result = await invoke_fn()
                    else:
                        result = await asyncio.to_thread(invoke_fn)
                    elapsed = (datetime.now() - started_at).total_seconds()
                    logger.debug(
                        "Chain operation '%s' succeeded on attempt %s in %.2fs (unbounded timeout mode)",
                        operation,
                        attempt,
                        elapsed,
                    )
                    return result
                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Chain operation '%s' failed on attempt %s/%s: %s",
                        operation,
                        attempt,
                        attempts,
                        e,
                    )
            logger.warning(
                "Chain operation '%s' exhausted retries in unbounded timeout mode, using fallback: %s",
                operation,
                last_error,
            )
            return fallback_factory(last_error)

        total_budget_seconds = max(0.1, float(timeout_seconds))
        deadline = time.monotonic() + total_budget_seconds

        for attempt in range(1, attempts + 1):
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                last_error = TimeoutError(
                    f"chain invocation exceeded total timeout budget of {total_budget_seconds:.1f}s"
                )
                logger.warning(
                    "Chain operation '%s' exceeded total timeout budget on attempt %s/%s",
                    operation,
                    attempt,
                    attempts,
                )
                break
            try:
                started_at = datetime.now()
                if async_mode:
                    result = await asyncio.wait_for(
                        invoke_fn(),
                        timeout=remaining_seconds,
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(invoke_fn),
                        timeout=remaining_seconds,
                    )
                elapsed = (datetime.now() - started_at).total_seconds()
                logger.debug(
                    "Chain operation '%s' succeeded on attempt %s in %.2fs",
                    operation,
                    attempt,
                    elapsed,
                )
                return result
            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"chain invocation timed out after {remaining_seconds:.1f}s remaining budget"
                )
                logger.warning(
                    "Chain operation '%s' timed out on attempt %s/%s after %.1fs",
                    operation,
                    attempt,
                    attempts,
                    remaining_seconds,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "Chain operation '%s' failed on attempt %s/%s: %s",
                    operation,
                    attempt,
                    attempts,
                    e,
                )

        logger.warning(
            "Chain operation '%s' exhausted retries, using fallback: %s",
            operation,
            last_error,
        )
        return fallback_factory(last_error)

    async def _review(self, task: Task, latest_notes: str) -> OrchestratorDecision:
        """Enhanced review with loop prevention and better state tracking"""
        task_state = self.task_states[task.name]
        self.total_iterations += 1
        stagnation_rounds = self._update_subtask_stagnation(task, task_state)

        # Check for stuck state
        if task_state.is_stuck():
            logger.warning(f"Task {task.name} appears stuck, forcing progression")
            return OrchestratorDecision(
                reasoning="Task appears stuck in a loop, forcing progression to next subtask",
                decision="switch",
                feedback="Move to next subtask to break the loop",
                subtask="proceed",
                next_task=task.name,
            )

        # Check total iteration limit
        if self.total_iterations >= self.max_total_iterations:
            logger.warning("Maximum iterations reached, completing task")
            return OrchestratorDecision(
                reasoning="Maximum iterations reached, task should be completed",
                decision="done",
                feedback="Research completed within iteration limits",
                subtask="proceed",
                next_task="Not Applicable",
            )

        hard_ceiling_hit, hard_ceiling_reason = self._hard_ceiling_progression_required(
            task,
            task_state,
        )
        if hard_ceiling_hit:
            logger.warning(
                "Forcing subtask progression for task %s at hard per-subtask limit",
                task.name,
            )
            return OrchestratorDecision(
                reasoning=hard_ceiling_reason,
                decision="switch",
                feedback=(
                    "Stop the current subtask now, carry forward only the strongest supported findings, "
                    "and record unresolved gaps as explicit uncertainty before advancing."
                ),
                subtask="proceed",
                next_task=task.name,
            )

        # Enforce per-subtask iteration ceiling to prevent endless looping.
        subtask_metrics = self._get_current_subtask_agent_metrics(task, task_state)
        subtask_action_count = int(subtask_metrics.get("action_count", 0))
        if subtask_action_count >= self.max_iterations_per_subtask:
            current_subtask = None
            if task_state.research_plan and task_state.research_plan.subtasks:
                if (
                    0
                    <= task_state.subtask_index
                    < len(task_state.research_plan.subtasks)
                ):
                    current_subtask = task_state.research_plan.subtasks[
                        task_state.subtask_index
                    ]
            (
                readiness_ok,
                readiness_reason,
            ) = self._evaluate_ceiling_progression_readiness(
                task,
                task_state,
            )
            min_ceiling_stagnation = max(
                2,
                int(os.getenv("ORCHESTRATOR_CEILING_MIN_STAGNATION_ROUNDS", "2")),
            )
            plateau_ok, plateau_reason = self._evaluate_plateau_progression_readiness(
                task,
                task_state,
            )
            if current_subtask is not None and self._subtask_requires_broad_discovery(
                current_subtask
            ):
                if plateau_ok:
                    logger.warning(
                        "Advancing task %s at per-subtask ceiling due to discovery plateau (%s actions)",
                        task.name,
                        subtask_action_count,
                    )
                    return OrchestratorDecision(
                        reasoning=plateau_reason,
                        decision="switch",
                        feedback=(
                            "Discovery appears saturated. Advance to the next subtask, carry forward the strongest opportunities, "
                            "and explicitly preserve any remaining uncertainty from missing or tentative items."
                        ),
                        subtask="proceed",
                        next_task=task.name,
                    )
            elif readiness_ok and stagnation_rounds >= min_ceiling_stagnation:
                logger.warning(
                    "Forcing subtask progression for task %s at per-subtask iteration ceiling after stagnation (%s actions, stagnation_rounds=%s)",
                    task.name,
                    subtask_action_count,
                    stagnation_rounds,
                )
                return OrchestratorDecision(
                    reasoning=(
                        "Per-subtask iteration ceiling reached with sufficient evidence and stagnation; "
                        f"advancing. {readiness_reason}"
                    ).strip(),
                    decision="switch",
                    feedback=(
                        "Advance to the next subtask and carry forward concrete findings; "
                        "avoid repeating prior query patterns."
                    ),
                    subtask="proceed",
                    next_task=task.name,
                )
            logger.warning(
                "Per-subtask iteration ceiling reached for task %s but advancement conditions are not met; keeping subtask active. %s%s",
                task.name,
                readiness_reason,
                (
                    f" Stagnation rounds {stagnation_rounds}/{min_ceiling_stagnation}."
                    if readiness_ok and not plateau_ok
                    else ""
                ),
            )

        current_subtask_info = self._build_current_subtask_info(task_state)
        defer_review, defer_reason = self._should_defer_llm_review(task, task_state)
        if defer_review:
            return OrchestratorDecision(
                reasoning=defer_reason,
                decision="continue",
                feedback=(
                    "Continue gathering evidence for the current subtask using diversified, "
                    "success-criteria-aligned queries."
                ),
                subtask="stay",
                next_task=task.name,
            )

        system_prompt = self._get_system_prompt()
        user_content = self._build_review_context(
            task,
            task_state,
            latest_notes,
            current_subtask_info,
            system_prompt,
        )

        if self.control_chains is None:
            return self._fallback_review_decision(
                task,
                task_state,
                "Control chains unavailable for review; using deterministic fallback",
            )

        review_timeout_seconds = float(
            os.getenv(
                "ORCHESTRATOR_REVIEW_TIMEOUT_SECONDS",
                self.config.review_timeout_seconds,
            )
        )

        decision = await self._invoke_chain_with_policy(
            lambda: self.control_chains.review_decision(
                review_system_prompt=system_prompt,
                review_user_content=user_content,
            ),
            operation="review_decision",
            timeout_seconds=review_timeout_seconds,
            retries=3,
            fallback_factory=lambda err: self._fallback_review_decision(
                task,
                task_state,
                f"Failed to get valid response from LLM ({err}), using deterministic fallback",
            ),
            async_mode=False,
        )
        metrics = self._get_current_subtask_agent_metrics(task, task_state)
        task_state.subtask_last_review_action_count[task_state.subtask_index] = metrics[
            "action_count"
        ]
        return decision

    def _get_system_prompt(self) -> str:
        """Get system prompt based on configuration"""
        return OrchestratorLangChainControlChains.build_review_system_prompt(
            use_mcp=self.use_mcp,
            mcp_connected=self.mcp_connected,
        )

    def _fallback_planning_plan(self, task: Task, reason: str) -> PlanningPlan:
        """Create deterministic fallback plan when planning LLM fails."""
        logger.warning("Using fallback plan for task %s: %s", task.name, reason)
        raw_description = (task.description or task.name or "").strip()
        numbered_sections: list[str] = []
        if raw_description:
            parts = re.split(r"\(\d+\)\s*", raw_description)
            if len(parts) > 1:
                for part in parts[1:]:
                    section = re.sub(r"\s+", " ", part).strip(" .;,-")
                    if section:
                        numbered_sections.append(section)

        if numbered_sections:
            subtasks = [
                Subtask(
                    order=idx,
                    description=f"Research requirement: {section}",
                    success_criteria=(
                        "Collect concrete findings, named entities/programs, and "
                        "source-backed notes for this requirement."
                    ),
                )
                for idx, section in enumerate(numbered_sections[:6], start=1)
            ]
        else:
            subtasks = [
                Subtask(
                    order=1,
                    description=f"Scope and frame the research problem for {task.name}",
                    subtask_type="general",
                    success_criteria=(
                        "Define scope, assumptions, and information dimensions needed "
                        "for a strong final report."
                    ),
                ),
                Subtask(
                    order=2,
                    description="Gather high-signal evidence with targeted searches",
                    subtask_type="discovery",
                    success_criteria=(
                        "Collect source-backed evidence, examples, and factual details "
                        "that directly support the research objective."
                    ),
                ),
                Subtask(
                    order=3,
                    description="Synthesize findings into a structured final report",
                    subtask_type="synthesis",
                    success_criteria=(
                        "Produce a coherent report with clear conclusions, tradeoffs, "
                        "and actionable recommendations."
                    ),
                ),
            ]
        return PlanningPlan(
            restated_task=f"Research task: {task.description}",
            subtasks=subtasks,
        )

    def _fallback_preplanning_action(self, reason: str) -> PrePlanningAction:
        """Fallback pre-planning action when the control chain fails."""
        return PrePlanningAction(
            reasoning=reason,
            action="done",
            thought="Proceeding directly to planning due to pre-planning fallback.",
        )

    def _format_preplanning_search_results(
        self, query: str, hits: list[dict[str, Any]]
    ) -> str:
        if not hits:
            return f"[SEARCH] {query}\nNo results returned."

        lines = [f"[SEARCH] {query}", f"Top hits ({min(len(hits), 3)}):"]
        for idx, hit in enumerate(hits[:3], start=1):
            title = str(hit.get("title", "")).strip() or "Untitled"
            href = str(hit.get("href", "")).strip() or "No URL"
            snippet = str(hit.get("body", "")).strip()
            snippet = snippet[:220] + ("..." if len(snippet) > 220 else "")
            lines.append(f"{idx}. {title}")
            lines.append(f"   URL: {href}")
            if snippet:
                lines.append(f"   Snippet: {snippet}")
        return "\n".join(lines)

    async def _run_preplanning_mcp_tool(
        self,
        tool_name: str,
        tool_parameters: dict[str, Any],
        *,
        timeout_seconds: float | None,
    ) -> str:
        if not self.use_mcp or self.mcp_manager is None:
            return "[MCP] Skipped: MCP manager is not configured."
        if not self.mcp_connected:
            await self.initialize_mcp()
        if not self.mcp_connected:
            return "[MCP] Skipped: MCP is unavailable."
        if not tool_name:
            return "[MCP] Skipped: tool name was empty."

        try:
            mcp_call = self.mcp_manager.call_tool(tool_name, tool_parameters)
            if timeout_seconds is None or timeout_seconds <= 0:
                result = await mcp_call
            else:
                result = await asyncio.wait_for(
                    mcp_call,
                    timeout=timeout_seconds,
                )
            if not getattr(result, "success", False):
                error = getattr(result, "error", "Unknown MCP error")
                return f"[MCP] {tool_name} failed: {error}"

            data_preview = str(getattr(result, "data", ""))[:400]
            if len(data_preview) == 400:
                data_preview += "..."
            return f"[MCP] {tool_name} succeeded.\nResult: {data_preview}"
        except asyncio.TimeoutError:
            safe_timeout = timeout_seconds if timeout_seconds is not None else 0.0
            return f"[MCP] {tool_name} timed out after {safe_timeout:.1f}s."
        except Exception as e:
            return f"[MCP] {tool_name} failed: {e}"

    async def _run_preplanning_exploration(self, task: Task) -> str:
        """
        Run a bounded think/search/tool loop before final plan generation.
        Returns exploration notes to inject into planning context.
        """
        max_steps = max(
            0,
            int(
                os.getenv(
                    "ORCHESTRATOR_PREPLANNING_MAX_STEPS",
                    self.config.preplanning_max_steps,
                )
            ),
        )
        if max_steps <= 0 or self.control_chains is None:
            return ""

        step_timeout_seconds = float(
            os.getenv(
                "ORCHESTRATOR_PREPLANNING_STEP_TIMEOUT_SECONDS",
                self.config.preplanning_step_timeout_seconds,
            )
        )
        if timeouts_disabled() or step_timeout_seconds <= 0:
            step_timeout_seconds = 0.0
        preplanning_total_timeout_seconds = float(
            os.getenv(
                "ORCHESTRATOR_PREPLANNING_TOTAL_TIMEOUT_SECONDS",
                min(step_timeout_seconds * max_steps, 60.0),
            )
        )
        if timeouts_disabled() or preplanning_total_timeout_seconds <= 0:
            preplanning_total_timeout_seconds = 0.0
        mcp_enabled = bool(self.use_mcp and self.mcp_manager is not None)
        preplanning_deadline = (
            None
            if preplanning_total_timeout_seconds <= 0
            else time.monotonic() + max(1.0, preplanning_total_timeout_seconds)
        )

        notes: list[str] = []
        seen_queries: set[str] = set()

        for step_idx in range(max_steps):
            remaining_preplanning_seconds = (
                (preplanning_deadline - time.monotonic())
                if preplanning_deadline is not None
                else None
            )
            if (
                remaining_preplanning_seconds is not None
                and remaining_preplanning_seconds <= 0
            ):
                notes.append(
                    "[PREPLANNING] Total exploration budget reached; proceeding to planning."
                )
                break
            if step_timeout_seconds <= 0:
                effective_step_timeout = None
            elif remaining_preplanning_seconds is None:
                effective_step_timeout = step_timeout_seconds
            else:
                effective_step_timeout = min(
                    step_timeout_seconds, remaining_preplanning_seconds
                )
            exploration_log = "\n\n".join(notes[-8:]) if notes else "(none yet)"
            action = await self._invoke_chain_with_policy(
                lambda: self.control_chains.preplanning_action(
                    task_name=task.name,
                    task_description=task.description,
                    task_budget_minutes=task.budgetMinutes,
                    mcp_enabled=mcp_enabled,
                    exploration_log=exploration_log,
                ),
                operation="preplanning_action",
                timeout_seconds=effective_step_timeout,
                retries=1,
                fallback_factory=lambda err: self._fallback_preplanning_action(
                    f"pre-planning step failed ({err})"
                ),
                async_mode=False,
            )

            if not isinstance(action, PrePlanningAction):
                action = self._fallback_preplanning_action(
                    "pre-planning returned invalid schema"
                )

            if action.action == "done":
                stop_reason = action.reasoning.strip() or action.thought.strip()
                notes.append(
                    f"[STEP {step_idx + 1}] done: {stop_reason or 'enough context gathered'}"
                )
                break

            if action.action == "think":
                thought = action.thought.strip() or action.reasoning.strip()
                notes.append(
                    f"[STEP {step_idx + 1}] [THINK] {thought or 'Refining plan strategy.'}"
                )
                continue

            if action.action == "search":
                query = action.query.strip()
                if not query:
                    notes.append(
                        f"[STEP {step_idx + 1}] [SEARCH] Skipped: empty query from planner."
                    )
                    continue
                if query in seen_queries:
                    notes.append(
                        f"[STEP {step_idx + 1}] [SEARCH] Skipped duplicate query: {query}"
                    )
                    continue
                seen_queries.add(query)
                try:
                    search_call = asyncio.to_thread(
                        self.preplanning_search.search, query
                    )
                    if effective_step_timeout is None:
                        hits = await search_call
                    else:
                        hits = await asyncio.wait_for(
                            search_call,
                            timeout=effective_step_timeout,
                        )
                    notes.append(self._format_preplanning_search_results(query, hits))
                except asyncio.TimeoutError:
                    timeout_value = (
                        effective_step_timeout
                        if effective_step_timeout is not None
                        else 0.0
                    )
                    notes.append(
                        f"[STEP {step_idx + 1}] [SEARCH] Timeout after {timeout_value:.1f}s for query: {query}"
                    )
                continue

            if action.action == "mcp_tool":
                tool_name = action.tool_name.strip()
                tool_parameters = action.tool_parameters or {}
                mcp_note = await self._run_preplanning_mcp_tool(
                    tool_name,
                    tool_parameters,
                    timeout_seconds=effective_step_timeout,
                )
                notes.append(f"[STEP {step_idx + 1}] {mcp_note}")
                continue

            notes.append(
                f"[STEP {step_idx + 1}] Unsupported action '{action.action}', continuing."
            )

        exploration_notes = "\n\n".join(notes).strip()
        preplanning_notes_tokens = max(
            256,
            int(os.getenv("ORCHESTRATOR_PREPLANNING_MAX_NOTES_TOKENS", "2500")),
        )
        exploration_notes = self._truncate_text_by_token_budget(
            exploration_notes,
            max_tokens=preplanning_notes_tokens,
            fallback_label="preplanning approx token truncation",
        )

        logger.info(
            "Pre-planning exploration complete for task %s (steps=%s, tokens=%s)",
            task.name,
            max_steps,
            (
                self.token_counter.count_tokens(exploration_notes)
                if self.token_counter is not None
                else max(1, len(exploration_notes) // 4)
            ),
        )
        return exploration_notes

    async def _planning_phase(self, task: Task) -> None:
        """Planning phase using adapter-managed retries/timeout/fallback."""
        task_state = self.task_states[task.name]
        preplanning_context = await self._run_preplanning_exploration(task)
        planning_timeout_seconds = float(
            os.getenv(
                "ORCHESTRATOR_PLANNING_TIMEOUT_SECONDS",
                self.config.planning_timeout_seconds,
            )
        )
        if self.control_chains is None:
            research_plan = self._fallback_planning_plan(
                task,
                "planning chains unavailable",
            )
        else:
            research_plan = await self._invoke_chain_with_policy(
                lambda: self.control_chains.planning_plan(
                    task_name=task.name,
                    task_description=task.description,
                    task_budget_minutes=task.budgetMinutes,
                    preplanning_context=preplanning_context,
                ),
                operation="planning_plan",
                timeout_seconds=planning_timeout_seconds,
                retries=3,
                fallback_factory=lambda err: self._fallback_planning_plan(
                    task,
                    f"planning failed ({err})",
                ),
                async_mode=False,
            )

        if not isinstance(research_plan, PlanningPlan):
            research_plan = self._fallback_planning_plan(
                task,
                "planning returned invalid schema",
            )

        if not research_plan.subtasks:
            research_plan = self._fallback_planning_plan(
                task, "planning returned no subtasks"
            )

        research_plan = self._normalize_research_plan(task, research_plan)

        max_subtasks = max(
            1,
            int(os.getenv("ORCHESTRATOR_MAX_SUBTASKS", "5")),
        )
        if len(research_plan.subtasks) > max_subtasks:
            logger.warning(
                "Truncating plan subtasks for task %s from %s to %s (ORCHESTRATOR_MAX_SUBTASKS)",
                task.name,
                len(research_plan.subtasks),
                max_subtasks,
            )
            research_plan = self._truncate_research_plan_preserving_phases(
                task,
                research_plan,
                max_subtasks=max_subtasks,
            )

        for i, subtask in enumerate(research_plan.subtasks):
            subtask.order = i + 1

        logger.info(
            "Generated plan with %s subtasks for task %s",
            len(research_plan.subtasks),
            task.name,
        )
        task_state.research_plan = research_plan

        # Create initial subtask agent
        if research_plan.subtasks:
            task_state.create_subtask_agent(
                0,
                self.llm,
                self.memory,
                self.mcp_manager if self.use_mcp else None,
            )

    async def run_subtask_research(self, task: Task) -> dict[str, any]:
        """Run research for the current subtask using a dedicated agent"""
        task_state = self.task_states[task.name]

        # Get or create current subtask agent
        current_agent = task_state.get_current_subtask_agent()
        if not current_agent:
            # If we've already advanced past the last subtask, signal completion
            if not task_state.research_plan or task_state.subtask_index >= len(
                task_state.research_plan.subtasks
            ):
                return {
                    "status": "no_more_subtasks",
                    "message": "All subtasks completed",
                }

            current_agent = task_state.create_subtask_agent(
                task_state.subtask_index,
                self.llm,
                self.memory,
                self.mcp_manager if self.use_mcp else None,
            )

        # Run one research step for the current subtask agent
        result = await current_agent.run_step(task)

        # Always append the latest research output to notes history
        task_state.notes_history.append(result)

        # Best‑effort: pull structured findings from the agent's per‑task state
        try:
            if hasattr(current_agent, "_state") and task.name in current_agent._state:
                agent_state = current_agent._state[task.name]
                # Use checkpoints as a proxy for key findings for this subtask
                if hasattr(agent_state, "checkpoints"):
                    task_state.subtask_findings.setdefault(task_state.subtask_index, [])
                    task_state.subtask_findings[task_state.subtask_index].extend(
                        str(cp) for cp in agent_state.checkpoints
                    )
        except Exception as e:
            logger.warning(
                f"Failed to extract findings from subtask agent state: {e}",
                exc_info=True,
            )

        return {"status": "in_progress", "result": result}

    async def run_step(self, task: Task) -> str:
        """Run one step of the orchestrator workflow"""
        return await self.next_action(task)

    async def next_action(self, task: Task, notes: str = "") -> str:
        """
        Enhanced next action using subtask-specific research agents + LLM review loop.

        Flow:
        1. Run one research step for the current subtask via a dedicated WebResearchAgent.
        2. If all subtasks are complete, synthesize a final report and advance to the next task.
        3. Otherwise, call the orchestrator's review LLM (_review) to decide whether to:
           - continue on the current subtask,
           - proceed to the next subtask, or
           - mark the entire task as done.
        4. Update TaskState (subtask_index, completed_subtasks, feedback history).
        """
        if self.langgraph_runner is None:
            if LangGraphOrchestratorRunner is None:
                raise RuntimeError("LangGraph orchestrator runner unavailable")
            self.langgraph_runner = LangGraphOrchestratorRunner(self)

        return await self.langgraph_runner.run(task)

    async def _finalize_when_no_more_subtasks(self, task: Task) -> str:
        """Finalize task when all subtasks are already completed."""
        task_state = self.task_states[task.name]
        logger.info(f"All subtasks completed for {task.name}")
        evidence_appendix = self._build_task_evidence_appendix(task, task_state)
        evidence_brief = self._build_task_evidence_brief(task, task_state)
        verification_brief = self._build_claim_verification_brief(task, task_state)
        report_sources = self._collect_clean_subtask_reports(task_state)
        if evidence_brief:
            report_sources.append(evidence_brief)
        if verification_brief:
            report_sources.append(verification_brief)
        if evidence_appendix:
            report_sources.append(evidence_appendix)
        final_report = await self._synthesize_final_report_with_timeout(
            task,
            report_sources,
        )

        safe_name = "".join(
            c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task.name
        ).strip()
        report_path = f"notes/{safe_name.upper()}_FINAL_REPORT.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)

        task.status = TaskStatus.COMPLETE
        remaining = [t for t in self.tasks.values() if t.status != TaskStatus.COMPLETE]
        self.current = remaining[0].name if remaining else None
        return "Task completed - all subtasks finished"

    async def _finalize_task(self, task: Task, latest_notes: str = "") -> str:
        """Finalize task from current orchestrator state."""
        task_state = self.task_states[task.name]

        evidence_appendix = self._build_task_evidence_appendix(task, task_state)
        evidence_brief = self._build_task_evidence_brief(task, task_state)
        verification_brief = self._build_claim_verification_brief(task, task_state)
        cleaned_latest_notes = self._strip_control_channel_annotations(latest_notes)
        final_report_sources = self._collect_clean_subtask_reports(task_state)
        if not final_report_sources and cleaned_latest_notes:
            final_report_sources = [cleaned_latest_notes]
        if evidence_brief:
            final_report_sources.append(evidence_brief)
        if verification_brief:
            final_report_sources.append(verification_brief)
        if evidence_appendix:
            final_report_sources.append(evidence_appendix)
        final_report = await self._synthesize_final_report_with_timeout(
            task,
            final_report_sources,
        )
        safe_name = "".join(
            c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task.name
        ).strip()
        report_path = f"notes/{safe_name.upper()}_FINAL_REPORT.txt"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            logger.info(f"Wrote final report for task {task.name} to {report_path}")
        except Exception as e:
            logger.error(f"Failed to write final report for task {task.name}: {e}")

        task.status = TaskStatus.COMPLETE
        logger.info(f"Task {task.name} marked complete by orchestrator")
        remaining = [t for t in self.tasks.values() if t.status != TaskStatus.COMPLETE]
        self.current = remaining[0].name if remaining else None
        return "Task completed - orchestrator marked task as done"

    async def _synthesize_final_report_with_timeout(
        self,
        task: Task,
        report_sources: list[str],
    ) -> str:
        """Run final report synthesis with optional timeout (0/negative => unlimited)."""
        raw_timeout = os.getenv("AGENT_FINAL_REPORT_TIMEOUT_SECONDS", "0")
        try:
            timeout_seconds = float(raw_timeout)
        except (TypeError, ValueError):
            timeout_seconds = 0.0
        try:
            if timeout_seconds <= 0:
                return await self.synthesize_final_report(task, report_sources)
            return await asyncio.wait_for(
                self.synthesize_final_report(task, report_sources),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Final report synthesis timed out for task %s after %.1fs; using deterministic report builder",
                task.name,
                timeout_seconds,
            )
            return self._fallback_final_report(task, "timeout")
        except Exception as e:
            logger.warning(
                "Final report synthesis failed for task %s: %s; using deterministic report builder",
                task.name,
                e,
            )
            return self._fallback_final_report(task, str(e))

    def _fallback_final_report(self, task: Task, reason: str) -> str:
        """Deterministic fallback when LLM synthesis is unavailable."""
        return self._build_deterministic_final_report(
            task,
            self.task_states[task.name],
            reason=reason,
        )

    async def _apply_review_decision(
        self, task: Task, latest_notes: str, decision: OrchestratorDecision
    ) -> str:
        """Apply review decision to orchestrator state and return user-facing result."""
        task_state = self.task_states[task.name]
        logger.info(
            "Orchestrator decision for task %s: decision=%s, subtask=%s, next_task=%s",
            task.name,
            decision.decision,
            decision.subtask,
            decision.next_task,
        )

        total_subtasks = (
            len(task_state.research_plan.subtasks)
            if task_state.research_plan and task_state.research_plan.subtasks
            else 0
        )
        completed_count = len(task_state.completed_subtasks)
        remaining_after_current = max(0, total_subtasks - (completed_count + 1))
        should_advance = (
            decision.decision == "done"
            or decision.subtask == "proceed"
            or decision.decision == "switch"
        )
        if should_advance:
            (
                readiness_ok,
                readiness_reason,
            ) = self._evaluate_subtask_completion_readiness(task, task_state)
            plateau_ok, plateau_reason = self._evaluate_plateau_progression_readiness(
                task,
                task_state,
            )
            (
                hard_ceiling_hit,
                hard_ceiling_reason,
            ) = self._hard_ceiling_progression_required(task, task_state)
            if not readiness_ok and not plateau_ok and not hard_ceiling_hit:
                guidance = self._build_guidance_from_decision(
                    task,
                    task_state,
                    decision,
                    readiness_reason=readiness_reason,
                )
                self._set_subtask_guidance(
                    task, task_state, task_state.subtask_index, guidance
                )
                logger.info(
                    "Blocking orchestrator advance for task %s: %s",
                    task.name,
                    readiness_reason,
                )
                return (
                    latest_notes
                    + "\n\n[ORCHESTRATOR GUARD] Continue current subtask. "
                    + readiness_reason
                    + (f"\n[ORCHESTRATOR FEEDBACK] {guidance}" if guidance else "")
                ).strip()
            if plateau_ok and not readiness_ok:
                plateau_guidance = (
                    "Discovery appears saturated near target. Proceed using the strongest currently supported opportunities, "
                    "and carry unresolved breadth gaps forward as explicit uncertainty."
                )
                self._set_subtask_guidance(
                    task, task_state, task_state.subtask_index, plateau_guidance
                )
                logger.info(
                    "Allowing orchestrator advance for task %s via plateau readiness: %s",
                    task.name,
                    plateau_reason,
                )
            elif hard_ceiling_hit and not readiness_ok and not plateau_ok:
                logger.info(
                    "Allowing orchestrator advance for task %s via hard ceiling: %s",
                    task.name,
                    hard_ceiling_reason,
                )

        if decision.decision == "done":
            # Guardrail: avoid finalizing the full task when multiple subtasks still remain.
            if remaining_after_current > 0:
                logger.warning(
                    "Ignoring premature done decision for task %s: %s subtasks still remain after current",
                    task.name,
                    remaining_after_current,
                )
                decision = OrchestratorDecision(
                    reasoning=decision.reasoning,
                    decision="continue",
                    feedback=decision.feedback
                    or "Continue through remaining subtasks before finalizing.",
                    subtask="proceed",
                    next_task=task.name,
                )
            else:
                if task_state.subtask_index not in task_state.completed_subtasks:
                    task_state.subtask_reports[
                        task_state.subtask_index
                    ] = self._build_subtask_report_snapshot(
                        task,
                        task_state,
                        task_state.subtask_index,
                        latest_notes,
                    )
                    task_state.mark_subtask_complete(task_state.subtask_index)

                if task_state.research_plan and task_state.research_plan.subtasks:
                    for idx in range(len(task_state.research_plan.subtasks)):
                        task_state.completed_subtasks.add(idx)

                return await self._finalize_task(task, latest_notes)

        if decision.subtask == "proceed" or decision.decision == "switch":
            task_state.subtask_reports[
                task_state.subtask_index
            ] = self._build_subtask_report_snapshot(
                task,
                task_state,
                task_state.subtask_index,
                latest_notes,
            )
            task_state.mark_subtask_complete(task_state.subtask_index)

            if task_state.research_plan and task_state.subtask_index < len(
                task_state.research_plan.subtasks
            ):
                next_subtask = task_state.research_plan.subtasks[
                    task_state.subtask_index
                ]
                next_guidance = self._build_guidance_from_decision(
                    task,
                    task_state,
                    decision,
                )
                if not next_guidance:
                    next_guidance = (
                        "Start with targeted discovery and gather concrete evidence for: "
                        f"{next_subtask.success_criteria}"
                    )
                self._set_subtask_guidance(
                    task,
                    task_state,
                    task_state.subtask_index,
                    next_guidance,
                )
                logger.info(
                    "Proceeding to subtask %s for task %s",
                    task_state.subtask_index + 1,
                    task.name,
                )
                task_state.create_subtask_agent(
                    task_state.subtask_index,
                    self.llm,
                    self.memory,
                    self.mcp_manager if self.use_mcp else None,
                )
            else:
                logger.info(
                    "All subtasks completed for %s after orchestrator decision",
                    task.name,
                )
                return await self._finalize_task(task, latest_notes)

        guidance = self._build_guidance_from_decision(task, task_state, decision)
        self._set_subtask_guidance(task, task_state, task_state.subtask_index, guidance)
        if guidance:
            return (latest_notes + f"\n\n[ORCHESTRATOR FEEDBACK] {guidance}").strip()
        return latest_notes or "Continuing research"

    def _build_subtask_report_snapshot(
        self,
        task: Task,
        task_state: TaskState,
        subtask_index: int,
        latest_notes: str,
    ) -> str:
        """Capture structured evidence for a subtask at transition time."""
        subtask_desc = ""
        if task_state.research_plan and 0 <= subtask_index < len(
            task_state.research_plan.subtasks
        ):
            subtask_desc = task_state.research_plan.subtasks[subtask_index].description

        lines = [
            f"Subtask {subtask_index + 1}: {subtask_desc or 'N/A'}",
        ]
        cleaned_latest_notes = self._strip_control_channel_annotations(latest_notes)
        if cleaned_latest_notes.strip():
            lines.append("Latest Notes:")
            lines.append(cleaned_latest_notes.strip())

        agent = task_state.subtask_agents.get(subtask_index)
        if agent is None:
            return "\n".join(lines)

        try:
            agent_state = getattr(agent, "_state", {}).get(task.name)
            if agent_state is None:
                return "\n".join(lines)

            search_history = list(getattr(agent_state, "search_history", []) or [])
            if search_history:
                lines.append("")
                lines.append("Search Evidence:")
                for entry in search_history[-6:]:
                    query = str(entry.get("query", "")).strip()
                    results = list(entry.get("results", []) or [])
                    results_count = int(entry.get("results_count", len(results) or 0))
                    lines.append(f"- Query: {query} (results={results_count})")
                    for hit in results[:3]:
                        title = str(hit.get("title", "")).strip() or "Untitled"
                        url = str(hit.get("url", "")).strip()
                        domain = str(hit.get("domain", "")).strip()
                        tier = str(hit.get("source_tier", "")).strip()
                        authority_score = int(hit.get("authority_score", 0) or 0)
                        summary = self._normalize_evidence_excerpt(
                            str(hit.get("summary", "")).strip(),
                            limit=220,
                        )
                        excerpt = self._normalize_evidence_excerpt(
                            str(hit.get("content_excerpt", "")).strip()
                            or str(hit.get("snippet", "")).strip(),
                            limit=260,
                        )
                        if url:
                            lines.append(
                                f"  - {title} | {url} | domain={domain or 'unknown'} "
                                f"| tier={tier or 'unknown'} | authority={authority_score}"
                            )
                        else:
                            lines.append(f"  - {title}")
                        if summary:
                            lines.append(f"    Summary: {summary}")
                        if excerpt:
                            lines.append(f"    Evidence excerpt: {excerpt}")
            checkpoints = list(getattr(agent_state, "checkpoints", []) or [])
            if checkpoints:
                lines.append("")
                lines.append("Checkpoint Evidence:")
                lines.append(checkpoints[-1][:2000])

            evidence_stats = self._get_current_subtask_evidence_stats(task, task_state)
            lines.append("")
            lines.append("Evidence Quality Summary:")
            lines.append(
                "- authoritative_results={authoritative_results}, official_results={official_results}, "
                "unique_domains={unique_domains}, fetched_results={fetched_results}".format(
                    **evidence_stats
                )
            )
        except Exception as e:
            logger.debug(
                "Failed to build subtask report snapshot for task %s subtask %s: %s",
                task.name,
                subtask_index,
                e,
            )

        return "\n".join(lines)

    @staticmethod
    def _extract_domains_from_text(text: str) -> list[str]:
        return extract_domains_from_text(text)

    def _subtask_is_low_signal(self, task: Task, subtask: Subtask) -> bool:
        haystack = f"{subtask.description}\n{subtask.success_criteria}".lower()
        if subtask_looks_low_signal(
            haystack,
            task_text=" ".join(
                [
                    str(getattr(task, "name", "") or ""),
                    str(getattr(task, "description", "") or ""),
                ]
            ),
        ):
            return True
        return any(
            token in haystack
            for token in (
                "blog post",
                "blog article",
                "forum post",
                "reddit",
                "quora",
                "generic article",
                "commercial article",
            )
        )

    @staticmethod
    def _subtask_is_overanchored_lead(subtask: Subtask) -> bool:
        haystack = " ".join(
            [
                str(getattr(subtask, "description", "") or ""),
                str(getattr(subtask, "success_criteria", "") or ""),
            ]
        )
        lowered = haystack.lower()
        anchor_markers = (
            "webpage",
            "page at ",
            "article ",
            "dear colleague letter",
            "participants webpage",
            "investigate the ",
            "analyze the ",
        )
        quoted_title = bool(re.search(r"[\"'`].{12,140}?[\"'`]", haystack))
        long_titled_fragment = bool(re.search(r"[‘’“”].{12,160}?[‘’“”]", haystack))
        has_embedded_url = bool(re.search(r"https?://", haystack))
        is_discovery_or_verification = any(
            token in lowered
            for token in ("discover", "identify", "list", "verify", "compare", "rank")
        )
        if has_embedded_url:
            return True
        if is_discovery_or_verification and (quoted_title or long_titled_fragment):
            return True
        return any(marker in lowered for marker in anchor_markers) and (
            quoted_title or long_titled_fragment or " at " in lowered
        )

    @staticmethod
    def _replacement_subtask_for_overanchored_lead(order: int) -> Subtask:
        return Subtask(
            order=order,
            subtask_type="discovery",
            description=(
                "Identify distinct government, university, and nonprofit program opportunities relevant to the task from official sources."
            ),
            success_criteria=(
                "List multiple distinct opportunities supported by official program pages, .gov, .edu, or primary organization sources, with enough source diversity to support deeper verification."
            ),
        )

    @staticmethod
    def _subtask_introduces_out_of_scope_artifacts(
        task: Task, subtask: Subtask
    ) -> bool:
        task_text = " ".join(
            [
                str(getattr(task, "name", "") or ""),
                str(getattr(task, "description", "") or ""),
            ]
        ).lower()
        subtask_text = " ".join(
            [
                str(getattr(subtask, "description", "") or ""),
                str(getattr(subtask, "success_criteria", "") or ""),
            ]
        ).lower()
        gated_terms = (
            "arxiv",
            "abstract",
            "foa",
            "funding opportunity announcement",
            "dear colleague letter",
            "provided ",
        )
        for term in gated_terms:
            if term in subtask_text and term not in task_text:
                return True
        return False

    @staticmethod
    def _subtask_is_opportunity_task_misaligned(task: Task, subtask: Subtask) -> bool:
        if not task_targets_discrete_opportunities(task):
            return False
        text = " ".join(
            [
                str(getattr(subtask, "description", "") or ""),
                str(getattr(subtask, "success_criteria", "") or ""),
            ]
        )
        profile = build_research_task_profile(
            text,
            getattr(task, "description", "") or "",
            getattr(task, "name", "") or "",
        )
        if (
            profile.source_terms
            and not profile.target_terms
            and not profile.evidence_terms
        ):
            return True
        if (
            "organization" in profile.source_terms
            and not profile.target_terms
            and not {"status", "deadline", "application", "eligibility"}.intersection(
                profile.evidence_terms
            )
        ):
            return True
        return False

    @staticmethod
    def _replacement_subtask_for_opportunity_task(order: int) -> Subtask:
        return Subtask(
            order=order,
            subtask_type="discovery",
            description=(
                "Identify currently relevant candidate opportunities from primary or authoritative sources."
            ),
            success_criteria=(
                "List multiple distinct candidate opportunities from official program pages, primary organization sites, or authoritative directories with enough detail to support later verification."
            ),
        )

    @staticmethod
    def _replacement_subtask_for_out_of_scope_artifacts(order: int) -> Subtask:
        return Subtask(
            order=order,
            subtask_type="verification",
            description=(
                "Verify candidate opportunities on primary or authoritative program pages instead of tangential artifacts."
            ),
            success_criteria=(
                "Confirm current status, timing, requirements, and source fidelity while discarding announcement-only, abstract-only, or indirect references."
            ),
        )

    @staticmethod
    def _default_discovery_subtask(task: Task, order: int) -> Subtask:
        if task_targets_discrete_opportunities(task):
            return Subtask(
                order=order,
                subtask_type="discovery",
                description=(
                    "Identify the strongest candidate opportunities from primary or authoritative sources."
                ),
                success_criteria=(
                    "List multiple distinct candidate opportunities from official program pages, primary organization sites, or authoritative directories, including enough detail to support verification."
                ),
            )
        return Subtask(
            order=order,
            subtask_type="discovery",
            description="Discover the strongest source-backed leads that directly address the research task.",
            success_criteria=(
                "Collect concrete findings, named entities, and primary or authoritative sources that establish strong coverage of the task."
            ),
        )

    @staticmethod
    def _default_verification_subtask(task: Task, order: int) -> Subtask:
        if task_targets_discrete_opportunities(task):
            return Subtask(
                order=order,
                subtask_type="verification",
                description=(
                    "Verify the current status, requirements, timing, and application details of the strongest candidate opportunities on primary or authoritative pages."
                ),
                success_criteria=(
                    "Confirm current status, requirements, timing, and application method for the strongest candidate opportunities using primary or authoritative pages."
                ),
            )
        return Subtask(
            order=order,
            subtask_type="verification",
            description="Verify the strongest findings directly against primary or authoritative sources.",
            success_criteria=(
                "Confirm the highest-value claims, dates, requirements, and source details on primary or authoritative pages."
            ),
        )

    @staticmethod
    def _default_comparison_subtask(task: Task, order: int) -> Subtask:
        if task_targets_discrete_opportunities(task):
            return Subtask(
                order=order,
                subtask_type="comparison",
                description=(
                    "Compare the verified opportunities to identify the strongest options, tradeoffs, and remaining evidence gaps."
                ),
                success_criteria=(
                    "Produce a structured comparison of the strongest verified opportunities, including which details are confirmed and which still require checking."
                ),
            )
        return Subtask(
            order=order,
            subtask_type="comparison",
            description="Compare the strongest verified findings and highlight the most important tradeoffs.",
            success_criteria=(
                "Produce a source-backed comparison that distinguishes stronger and weaker options or explanations."
            ),
        )

    @staticmethod
    def _default_synthesis_subtask(task: Task, order: int) -> Subtask:
        return Subtask(
            order=order,
            subtask_type="synthesis",
            description="Synthesize the verified findings into a final report for the user.",
            success_criteria=(
                "Produce a structured final report that clearly separates supported findings, tentative findings, and open uncertainties."
            ),
        )

    @staticmethod
    def _explicit_subtask_phase(description: str) -> str | None:
        match = re.match(
            r"^\s*(discovery|verification|authoritative verification|comparison|compare|synthesis)\s*:",
            str(description or ""),
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        label = match.group(1).lower()
        if label in {"authoritative verification", "verification"}:
            return "verification"
        if label in {"comparison", "compare"}:
            return "comparison"
        return label

    def _infer_subtask_type(
        self,
        description: str,
        success_criteria: str,
    ) -> str:
        explicit = self._explicit_subtask_phase(description)
        if explicit:
            return explicit

        description_text = str(description or "").lower()
        text = " ".join(
            [
                str(description or ""),
                str(success_criteria or ""),
            ]
        ).lower()
        if any(
            token in text
            for token in (
                "synthes",
                "final report",
                "final deliverable",
                "final output",
            )
        ):
            return "synthesis"
        if any(
            token in description_text
            for token in (
                "discover",
                "identify",
                "list",
                "gather",
                "collect",
                "broad set",
                "candidate opportunities",
                "distinct opportunities",
            )
        ):
            return "discovery"
        if any(
            token in text
            for token in ("compare", "comparison", "rank", "shortlist", "tradeoff")
        ):
            return "comparison"
        if any(
            token in text
            for token in (
                "verify",
                "verification",
                "cross-check",
                "cross check",
                "current status",
                "official page",
                "official pages",
                "authoritative source",
                "authoritative sources",
            )
        ):
            return "verification"
        if any(
            token in text
            for token in (
                "discover",
                "identify",
                "list",
                "gather",
                "collect",
                "broad set",
                "candidate opportunities",
                "official sources",
                "authoritative sources",
            )
        ):
            return "discovery"
        return "general"

    def _task_aware_default_subtask(
        self, task: Task, phase: str, order: int
    ) -> Subtask:
        phase = str(phase or "general").lower()
        if phase == "discovery":
            return self._default_discovery_subtask(task, order)
        if phase == "verification":
            return self._default_verification_subtask(task, order)
        if phase == "comparison":
            return self._default_comparison_subtask(task, order)
        if phase == "synthesis":
            return self._default_synthesis_subtask(task, order)
        return self._default_discovery_subtask(task, order)

    @staticmethod
    def _phase_value(value: object) -> str:
        if hasattr(value, "value"):
            return str(getattr(value, "value")).lower()
        return str(value or "").lower()

    def _canonicalize_opportunity_plan(
        self,
        task: Task,
        subtasks: list[Subtask],
    ) -> list[Subtask]:
        phase_defaults = {
            "discovery": self._default_discovery_subtask(task, 1),
            "verification": self._default_verification_subtask(task, 2),
            "comparison": self._default_comparison_subtask(task, 3),
            "synthesis": self._default_synthesis_subtask(task, 4),
        }
        selected: dict[str, Subtask] = {}
        for phase in ("discovery", "verification", "comparison", "synthesis"):
            for subtask in subtasks:
                if self._phase_value(subtask.subtask_type) == phase:
                    selected[phase] = subtask
                    break
        # For opportunity tasks, standardized four-phase control is more reliable
        # than preserving extra discovery subtasks that usually degenerate into
        # directories, organization maps, or lexical query churn.
        ordered = [
            phase_defaults["discovery"],
            selected.get("verification", phase_defaults["verification"]),
            selected.get("comparison", phase_defaults["comparison"]),
            selected.get("synthesis", phase_defaults["synthesis"]),
        ]
        for idx, (subtask, phase) in enumerate(
            zip(ordered, ("discovery", "verification", "comparison", "synthesis")),
            start=1,
        ):
            subtask.order = idx
            subtask.subtask_type = phase
        return ordered

    def _truncate_research_plan_preserving_phases(
        self,
        task: Task,
        research_plan: PlanningPlan,
        *,
        max_subtasks: int,
    ) -> PlanningPlan:
        subtasks = list(research_plan.subtasks or [])
        if len(subtasks) <= max_subtasks:
            return research_plan

        selected: list[Subtask] = []
        phase_order = ("discovery", "verification", "comparison", "synthesis")
        for phase in phase_order:
            for subtask in subtasks:
                if self._phase_value(subtask.subtask_type) == phase:
                    selected.append(subtask)
                    break
            if len(selected) >= max_subtasks:
                break

        if len(selected) < max_subtasks:
            for subtask in subtasks:
                if subtask not in selected:
                    selected.append(subtask)
                if len(selected) >= max_subtasks:
                    break

        selected.sort(key=lambda item: item.order)
        if selected:
            if self._phase_value(selected[0].subtask_type) != "discovery":
                selected[0] = self._default_discovery_subtask(task, 1)
            if self._phase_value(selected[-1].subtask_type) != "synthesis":
                if len(selected) == max_subtasks:
                    selected[-1] = self._default_synthesis_subtask(task, max_subtasks)
                else:
                    selected.append(
                        self._default_synthesis_subtask(task, len(selected) + 1)
                    )
        selected = selected[:max_subtasks]
        for idx, subtask in enumerate(selected, start=1):
            subtask.order = idx
        research_plan.subtasks = selected
        return research_plan

    def _normalize_research_plan(
        self, task: Task, research_plan: PlanningPlan
    ) -> PlanningPlan:
        """Harden planning output against low-signal, over-anchored, and misaligned subtasks."""
        normalized_subtasks: list[Subtask] = []
        seen_descriptions: set[str] = set()

        for subtask in list(research_plan.subtasks or []):
            description = " ".join(str(subtask.description or "").split()).strip()
            success_criteria = " ".join(
                str(subtask.success_criteria or "").split()
            ).strip()
            candidate = Subtask(
                order=subtask.order,
                description=description,
                success_criteria=success_criteria,
                subtask_type=self._infer_subtask_type(description, success_criteria),
            )

            if self._subtask_is_low_signal(task, candidate):
                logger.warning(
                    "Replacing low-signal planning subtask for task %s: %s",
                    task.name,
                    description[:180],
                )
                candidate = self._default_verification_subtask(task, subtask.order)
            elif self._subtask_is_overanchored_lead(candidate):
                logger.warning(
                    "Replacing over-anchored planning subtask for task %s: %s",
                    task.name,
                    description[:180],
                )
                candidate = self._replacement_subtask_for_overanchored_lead(
                    subtask.order
                )
            elif self._subtask_introduces_out_of_scope_artifacts(task, candidate):
                logger.warning(
                    "Replacing out-of-scope planning subtask for task %s: %s",
                    task.name,
                    description[:180],
                )
                candidate = self._replacement_subtask_for_out_of_scope_artifacts(
                    subtask.order
                )
            elif self._subtask_is_opportunity_task_misaligned(task, candidate):
                logger.warning(
                    "Replacing opportunity-misaligned planning subtask for task %s: %s",
                    task.name,
                    description[:180],
                )
                candidate = self._replacement_subtask_for_opportunity_task(
                    subtask.order
                )

            candidate.subtask_type = self._infer_subtask_type(
                candidate.description, candidate.success_criteria
            )
            normalized_key = candidate.description.lower()
            if normalized_key in seen_descriptions:
                continue
            seen_descriptions.add(normalized_key)
            normalized_subtasks.append(candidate)

        if not normalized_subtasks:
            return self._fallback_planning_plan(
                task, "plan normalization removed all subtasks"
            )

        if task_targets_discrete_opportunities(task):
            research_plan.subtasks = self._canonicalize_opportunity_plan(
                task,
                normalized_subtasks,
            )
            return research_plan

        phase_map: dict[str, Subtask] = {}
        for subtask in normalized_subtasks:
            phase = self._phase_value(subtask.subtask_type)
            if phase in {"discovery", "verification", "comparison", "synthesis"}:
                phase_map.setdefault(phase, subtask)

        ordered: list[Subtask] = []
        ordered.append(
            phase_map.get("discovery") or self._default_discovery_subtask(task, 1)
        )
        ordered.append(
            phase_map.get("verification") or self._default_verification_subtask(task, 2)
        )
        if len(normalized_subtasks) >= 2 or "comparison" in phase_map:
            ordered.append(
                phase_map.get("comparison") or self._default_comparison_subtask(task, 3)
            )
        ordered.append(
            phase_map.get("synthesis")
            or self._default_synthesis_subtask(task, len(ordered) + 1)
        )

        deduped: list[Subtask] = []
        seen_phase_desc: set[str] = set()
        for idx, subtask in enumerate(ordered, start=1):
            key = f"{subtask.subtask_type}:{subtask.description.lower()}"
            if key in seen_phase_desc:
                continue
            seen_phase_desc.add(key)
            subtask.order = idx
            subtask.subtask_type = self._infer_subtask_type(
                subtask.description, subtask.success_criteria
            )
            deduped.append(subtask)

        research_plan.subtasks = deduped
        return research_plan

    def _build_task_evidence_appendix(self, task: Task, task_state: TaskState) -> str:
        urls: list[str] = []
        authoritative_links: list[str] = []
        query_lines: list[str] = []
        seen_urls: set[str] = set()
        for idx, agent in sorted(task_state.subtask_agents.items()):
            agent_state = getattr(agent, "_state", {}).get(task.name)
            if agent_state is None:
                continue
            for entry in list(getattr(agent_state, "search_history", []) or []):
                query = str(entry.get("query", "")).strip()
                results = list(entry.get("results", []) or [])
                if query:
                    query_lines.append(
                        f"- Subtask {idx + 1}: {query} (results={int(entry.get('results_count', len(results)) or 0)})"
                    )
                for hit in results:
                    url = clean_url(str(hit.get("url", "")).strip() or "")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    urls.append(url)
                    if int(hit.get("authority_score", 0) or 0) >= 3:
                        authoritative_links.append(
                            f"- Subtask {idx + 1}: {str(hit.get('domain', '')).strip() or 'unknown'} | {url}"
                        )
        if not urls and not query_lines:
            return ""
        lines = ["## Evidence Appendix"]
        if query_lines:
            lines.extend(["", "### Queries Attempted", *query_lines[:24]])
        if authoritative_links:
            lines.extend(["", "### Highest-Trust Sources", *authoritative_links[:20]])
        if urls:
            lines.extend(["", "### Source Links"])
            lines.extend(f"- {url}" for url in urls[:30])
        return "\n".join(lines)

    def _build_task_evidence_brief(self, task: Task, task_state: TaskState) -> str:
        lines = ["## Evidence Quality Brief"]
        for idx, agent in sorted(task_state.subtask_agents.items()):
            agent_state = getattr(agent, "_state", {}).get(task.name)
            if agent_state is None:
                continue
            authoritative = 0
            official = 0
            domains: set[str] = set()
            top_hits: list[str] = []
            for entry in list(getattr(agent_state, "search_history", []) or []):
                for hit in list(entry.get("results", []) or []):
                    if hit.get("task_aligned") is False:
                        continue
                    domain = str(hit.get("domain", "")).strip().lower()
                    if domain:
                        domains.add(domain)
                    if int(hit.get("authority_score", 0) or 0) >= 3:
                        authoritative += 1
                        if len(top_hits) < 3:
                            top_hits.append(
                                f"{str(hit.get('title', '')).strip() or 'Untitled'} | {domain or 'unknown'}"
                            )
                    if bool(hit.get("official_source", False)):
                        official += 1
            lines.append(
                f"- Subtask {idx + 1}: authoritative_results={authoritative}, official_results={official}, unique_domains={len(domains)}"
            )
            for hit in top_hits:
                lines.append(f"  - {hit}")
        return "\n".join(lines)

    @staticmethod
    def _tokenize_claim_text(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]{3,}", str(text or "").lower())
            if token not in CLAIM_STOPWORDS
        }

    def _normalize_evidence_excerpt(self, text: str, *, limit: int = 240) -> str:
        cleaned = self._strip_control_channel_annotations(text or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,")
        if len(cleaned) <= limit:
            return cleaned
        truncated = cleaned[:limit].rsplit(" ", 1)[0].strip()
        return (truncated or cleaned[:limit]).rstrip(" ,;:.") + "..."

    def _extract_claim_candidates(self, text: str) -> list[str]:
        claims: list[str] = []
        seen: set[str] = set()
        cleaned_text = self._strip_control_channel_annotations(text or "")
        for raw_line in cleaned_text.splitlines():
            line = raw_line.strip().lstrip("-* ").strip()
            if not line:
                continue
            if line.lower().startswith(
                (
                    "subtask ",
                    "latest notes:",
                    "search evidence:",
                    "checkpoint evidence:",
                    "evidence quality summary:",
                    "## ",
                    "### ",
                )
            ):
                continue
            for chunk in re.split(r"(?<=[.!?])\s+", line):
                candidate = self._sanitize_claim_candidate(chunk)
                if candidate is None:
                    continue
                normalized = candidate.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                claims.append(candidate)
        return claims[:18]

    @staticmethod
    def _sanitize_claim_candidate(candidate: str) -> str | None:
        cleaned = str(candidate or "").strip()
        if not cleaned:
            return None
        cleaned = re.sub(
            r"^\[(?:SUMMARY|SEARCH|THOUGHT|CHECKPOINT|COMPLETE|ORCHESTRATOR FEEDBACK)\]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^(?:Query|query)\s*:\s*", "", cleaned)
        cleaned = re.sub(r"\s*\[(?:primary|partial)\s+support:.*?\]\s*$", "", cleaned)
        cleaned = re.sub(r"https?://\S+", "", cleaned)
        cleaned = re.sub(r"[*_`#]+", " ", cleaned)
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.")
        lowered = cleaned.lower()
        if len(cleaned) < 40 or len(cleaned) > 280:
            return None
        if any(
            phrase in lowered
            for phrase in (
                "executive summary",
                "final report",
                "report generated at",
                "success-criteria-aligned queries",
                "avoid repeating recent queries",
                "run at least",
                "increase search diversity",
                "find at least",
            )
        ):
            return None
        if any(
            marker in lowered
            for marker in (
                "domain=",
                "tier=",
                "authority=",
                "results=",
                "query:",
                "| |",
            )
        ):
            return None
        if "|" in cleaned:
            return None
        if not any(hint in f" {lowered} " for hint in CLAIM_RELATION_HINTS):
            return None
        if len(re.findall(r"[A-Za-z]{3,}", cleaned)) < 6:
            return None
        return cleaned

    def _authoritative_evidence_records(
        self, task: Task, task_state: TaskState
    ) -> list[SourceEvidenceRecord]:
        records: list[SourceEvidenceRecord] = []
        for idx, agent in sorted(task_state.subtask_agents.items()):
            agent_state = getattr(agent, "_state", {}).get(task.name)
            if agent_state is None:
                continue
            for entry in list(getattr(agent_state, "search_history", []) or []):
                query = str(entry.get("query", "")).strip()
                for hit in list(entry.get("results", []) or []):
                    if hit.get("task_aligned") is False:
                        continue
                    authority_score = int(hit.get("authority_score", 0) or 0)
                    if authority_score < 3:
                        continue
                    title = str(hit.get("title", "")).strip()
                    domain = str(hit.get("domain", "")).strip().lower()
                    url = clean_url(str(hit.get("url", "")).strip() or "") or ""
                    summary = self._normalize_evidence_excerpt(
                        str(hit.get("summary", "")).strip(),
                        limit=220,
                    )
                    excerpt = self._normalize_evidence_excerpt(
                        str(hit.get("content_excerpt", "")).strip()
                        or str(hit.get("snippet", "")).strip(),
                        limit=320,
                    )
                    records.append(
                        SourceEvidenceRecord(
                            subtask=idx + 1,
                            title=title,
                            domain=domain,
                            url=url,
                            official=bool(hit.get("official_source", False)),
                            authority_score=authority_score,
                            source_tier=str(hit.get("source_tier", "")).strip(),
                            fetched=bool(hit.get("fetched", False)),
                            query=query,
                            summary=summary,
                            content_excerpt=excerpt,
                            snippet=self._normalize_evidence_excerpt(
                                str(hit.get("snippet", "")).strip(), limit=220
                            ),
                            tokens=self._tokenize_claim_text(
                                " ".join(
                                    part
                                    for part in (title, domain, query, summary, excerpt)
                                    if part
                                )
                            ),
                            title_tokens=self._tokenize_claim_text(title),
                            domain_tokens=self._tokenize_claim_text(domain),
                            query_tokens=self._tokenize_claim_text(query),
                            evidence_tokens=self._tokenize_claim_text(
                                " ".join(part for part in (summary, excerpt) if part)
                            ),
                        )
                    )
        return records

    def _build_primary_record_claims(
        self,
        authoritative_records: list[SourceEvidenceRecord],
    ) -> list[str]:
        claims: list[str] = []
        seen: set[tuple[str, str]] = set()
        for record in authoritative_records:
            if not record.official:
                continue
            key = (record.title.lower(), record.domain)
            if not record.title or not record.domain or key in seen:
                continue
            seen.add(key)
            excerpt = record.content_excerpt or record.summary or record.snippet
            if excerpt:
                claims.append(
                    f"- {record.title}: {excerpt} [primary support: {record.domain}]"
                )
            else:
                claims.append(
                    f"- {record.title} is an official source relevant to this task. [primary support: {record.domain}]"
                )
            if len(claims) >= 6:
                break
        return claims

    @staticmethod
    def _claim_is_specific(candidate: str) -> bool:
        text = str(candidate or "")
        lowered = text.lower()
        if any(
            token in lowered
            for token in (
                "deadline",
                "eligibility",
                "application",
                "applications",
                "current status",
                "currently open",
                "open to",
                "closed",
                "rolling applications",
            )
        ):
            return True
        return False

    def _match_claim_to_record(
        self,
        claim: str,
        records: list[SourceEvidenceRecord],
    ) -> SourceEvidenceRecord | None:
        claim_tokens = self._tokenize_claim_text(claim)
        if not claim_tokens:
            return None
        scored: list[tuple[int, int, int, SourceEvidenceRecord]] = []
        for record in records:
            title_overlap = len(claim_tokens & record.title_tokens)
            evidence_overlap = len(claim_tokens & record.evidence_tokens)
            query_overlap = len(claim_tokens & record.query_tokens)
            domain_overlap = len(claim_tokens & record.domain_tokens)
            score = (
                title_overlap * 5
                + evidence_overlap * 3
                + domain_overlap * 2
                + query_overlap
            )
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    int(record.official),
                    int(record.authority_score),
                    record,
                )
            )
        if not scored:
            return None
        scored.sort(reverse=True, key=lambda item: item[:3])
        best = scored[0][3]
        if scored[0][0] < 3:
            return None
        return best

    def _build_verified_evidence_items(
        self, task: Task, task_state: TaskState
    ) -> list[VerifiedEvidenceItem]:
        items: list[VerifiedEvidenceItem] = []
        seen_urls: set[str] = set()
        seen_statements: set[str] = set()
        records = self._authoritative_evidence_records(task, task_state)
        reports: list[str] = []
        for idx in sorted(task_state.subtask_reports.keys()):
            reports.append(str(task_state.subtask_reports[idx]))
        reports.extend(
            str(item) for item in task_state.all_findings if str(item).strip()
        )
        claims = self._extract_claim_candidates("\n\n".join(reports[-8:]))

        for claim in claims:
            record = self._match_claim_to_record(claim, records)
            if record is None or not record.official:
                continue
            statement = claim
            if record.content_excerpt and not self._claim_is_specific(claim):
                statement = f"{record.title}: {record.content_excerpt}"
            statement = self._sanitize_claim_candidate(statement) or statement
            normalized = claim.lower()
            if normalized in seen_statements:
                continue
            seen_statements.add(normalized)
            if record.url and record.url in seen_urls:
                continue
            if record.url:
                seen_urls.add(record.url)
            items.append(
                VerifiedEvidenceItem(
                    statement=statement,
                    title=record.title,
                    domain=record.domain,
                    url=record.url,
                    query=record.query,
                    source_excerpt=record.content_excerpt,
                    source_tier=record.source_tier,
                    authority_score=record.authority_score,
                    official=record.official,
                )
            )
            if len(items) >= 8:
                return items

        for record in records:
            if not record.official:
                continue
            statement = (
                self._build_primary_record_claims([record])[0]
                .split(" [primary support:", 1)[0]
                .lstrip("- ")
                .strip()
            )
            normalized = statement.lower()
            if normalized in seen_statements:
                continue
            if record.url and record.url in seen_urls:
                continue
            seen_statements.add(normalized)
            if record.url:
                seen_urls.add(record.url)
            items.append(
                VerifiedEvidenceItem(
                    statement=statement,
                    title=record.title,
                    domain=record.domain,
                    url=record.url,
                    query=record.query,
                    source_excerpt=record.content_excerpt,
                    source_tier=record.source_tier,
                    authority_score=record.authority_score,
                    official=record.official,
                )
            )
            if len(items) >= 8:
                break
        return items

    def _build_claim_verification_brief(self, task: Task, task_state: TaskState) -> str:
        records = self._authoritative_evidence_records(task, task_state)
        if not records:
            return "## Claim Verification Brief\n- No authoritative evidence records available yet."

        reports: list[str] = []
        for idx in sorted(task_state.subtask_reports.keys()):
            reports.append(str(task_state.subtask_reports[idx]))
        reports.extend(
            str(item) for item in task_state.all_findings if str(item).strip()
        )
        claims = self._extract_claim_candidates("\n\n".join(reports[-8:]))
        if not claims:
            fallback_verified = self._build_primary_record_claims(records)
            if not fallback_verified:
                return (
                    "## Claim Verification Brief\n- No claim candidates extracted yet."
                )
            return "\n".join(
                [
                    "## Claim Verification Brief",
                    "### Verified Claims",
                    *fallback_verified,
                ]
            )

        verified: list[str] = []
        tentative: list[str] = []
        unsupported: list[str] = []
        for claim in claims:
            claim_tokens = self._tokenize_claim_text(claim)
            supporting: list[SourceEvidenceRecord] = []
            for record in records:
                overlap = len(claim_tokens & record.title_tokens) * 2 + len(
                    claim_tokens & record.evidence_tokens
                )
                if overlap >= 2:
                    supporting.append(record)
            supporting.sort(
                key=lambda record: (int(record.official), int(record.authority_score)),
                reverse=True,
            )
            primary = [record for record in supporting if record.official]
            if primary:
                refs = ", ".join(item.domain or "unknown" for item in primary[:2])
                verified.append(f"- {claim} [primary support: {refs}]")
            elif supporting:
                refs = ", ".join(item.domain or "unknown" for item in supporting[:2])
                tentative.append(f"- {claim} [partial support: {refs}]")
            else:
                unsupported.append(f"- {claim}")

        lines = ["## Claim Verification Brief"]
        if verified:
            lines.extend(["### Verified Claims", *verified[:8]])
        else:
            fallback_verified = self._build_primary_record_claims(records)
            if fallback_verified:
                lines.extend(["### Verified Claims", *fallback_verified[:6]])
        if tentative:
            lines.extend(["", "### Tentative Claims", *tentative[:6]])
        if unsupported:
            lines.extend(
                ["", "### Unsupported or Weakly Supported Claims", *unsupported[:6]]
            )
        return "\n".join(lines)

    def _extract_verified_claim_lines(
        self,
        verification_brief: str,
        verified_items: list[VerifiedEvidenceItem] | None = None,
    ) -> list[str]:
        if verified_items:
            return [f"- {item.statement}" for item in verified_items]
        match = re.search(
            r"(?ims)^###\s+Verified Claims\s*(.*?)(?=^###\s+|\Z)",
            verification_brief or "",
        )
        if not match:
            return []
        lines: list[str] = []
        for raw_line in match.group(1).splitlines():
            line = raw_line.strip()
            if not line.startswith("- "):
                continue
            cleaned = re.sub(
                r"\s*\[(?:primary|partial)\s+support:.*?\]\s*$", "", line
            ).strip()
            sanitized = self._sanitize_claim_candidate(cleaned.lstrip("- ").strip())
            if sanitized:
                lines.append(f"- {sanitized}")
        return lines

    def _extract_tentative_claim_lines(self, verification_brief: str) -> list[str]:
        match = re.search(
            r"(?ims)^###\s+Tentative Claims\s*(.*?)(?=^###\s+|\Z)",
            verification_brief or "",
        )
        if not match:
            return []
        lines: list[str] = []
        for raw_line in match.group(1).splitlines():
            line = raw_line.strip()
            if not line.startswith("- "):
                continue
            cleaned = re.sub(r"\s*\[partial support:.*?\]\s*$", "", line).strip()
            sanitized = self._sanitize_claim_candidate(cleaned.lstrip("- ").strip())
            if sanitized:
                lines.append(f"- {sanitized}")
        return lines

    def _extract_open_uncertainty_lines(self, verification_brief: str) -> list[str]:
        match = re.search(
            r"(?ims)^###\s+Unsupported or Weakly Supported Claims\s*(.*?)(?=^###\s+|\Z)",
            verification_brief or "",
        )
        if not match:
            return []
        lines: list[str] = []
        for raw_line in match.group(1).splitlines():
            line = raw_line.strip()
            if line.startswith("- "):
                lines.append(line.rstrip("."))
        return lines

    def _enforce_evidence_status_appendix_policy(
        self,
        report_text: str,
        verification_brief: str,
        *,
        verified_items: list[VerifiedEvidenceItem] | None = None,
    ) -> str:
        verified_lines = self._extract_verified_claim_lines(
            verification_brief,
            verified_items=verified_items,
        ) or ["- No primary-source-supported claims met the verification bar."]
        tentative_lines = self._extract_tentative_claim_lines(verification_brief) or [
            "- No additional tentative claims were preserved."
        ]
        uncertainty_lines = self._extract_open_uncertainty_lines(
            verification_brief
        ) or ["- No open uncertainties were preserved."]
        appendix = "\n".join(
            [
                "## Evidence Status Appendix",
                "### Verified Findings",
                *verified_lines,
                "",
                "### Tentative Findings",
                *tentative_lines,
                "",
                "### Open Uncertainties",
                *uncertainty_lines,
            ]
        )
        base = re.sub(
            r"(?ims)\n*##\s+Evidence Status Appendix\b.*\Z",
            "",
            (report_text or "").strip(),
        ).strip()
        if base:
            return f"{base}\n\n{appendix}".strip()
        return appendix

    def _report_has_unknown_urls(
        self,
        report_text: str,
        *,
        allowed_urls: list[str],
    ) -> bool:
        allowed = {clean_url(url) for url in allowed_urls if clean_url(url)}
        found = [
            clean_url(raw) for raw in re.findall(r"https?://[^\s)]+", report_text or "")
        ]
        return any(url and url not in allowed for url in found)

    def _sanitize_final_report_output(
        self,
        report_text: str,
        verification_brief: str,
        verified_items: list[VerifiedEvidenceItem] | None = None,
    ) -> str:
        text = (report_text or "").strip()
        if not text:
            return text
        text = re.sub(
            r"(?is)^\s*(okay[,!].*?|here(?:'|’)s\b.*?|certainly[,!].*?)\n+",
            "",
            text,
            count=1,
        ).strip()
        text = re.sub(r"(?im)^\s*#\s+Final Research Report\s*\n?", "", text)
        text = re.sub(r"(?im)^\s*##\s+Final Report:.*\n?", "", text)
        text = re.sub(
            r"(?im)^\s*\*\*Executive Summary:\*\*\s*",
            "## Executive Summary\n",
            text,
        )
        for heading in FINAL_REPORT_CORE_SECTIONS + ("## Evidence Status Appendix",):
            text = re.sub(
                rf"\s*{re.escape(heading)}\s*",
                f"\n{heading}\n",
                text,
            )
        text = re.sub(r"\s+(###\s+)", r"\n\1", text)
        text = re.sub(r"(###\s+[^\n*]+?)\s+([*-])", r"\1\n\2", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        text = self._dedupe_top_level_sections(text)
        text = self._enforce_evidence_status_appendix_policy(
            text,
            verification_brief,
            verified_items=verified_items,
        )
        return text.strip()

    def _collect_clean_subtask_reports(self, task_state: TaskState) -> list[str]:
        reports: list[str] = []
        for idx in sorted(task_state.subtask_reports.keys()):
            cleaned = self._strip_control_channel_annotations(
                str(task_state.subtask_reports[idx] or "")
            )
            if cleaned.strip():
                reports.append(cleaned.strip())
        for report in list(task_state.all_reports or []):
            cleaned = self._strip_control_channel_annotations(str(report or ""))
            if cleaned.strip():
                reports.append(cleaned.strip())
        deduped: list[str] = []
        seen: set[str] = set()
        for report in reports:
            normalized = re.sub(r"\s+", " ", report).strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(report)
        return deduped

    def _build_structured_evidence_packet(
        self,
        task: Task,
        task_state: TaskState,
        *,
        verification_brief: str,
        verified_items: list[VerifiedEvidenceItem],
    ) -> str:
        records = self._authoritative_evidence_records(task, task_state)
        record_by_url = {record.url: record for record in records if record.url}
        lines = [
            "## Structured Evidence Packet",
            "### Verified Findings With Source Lineage",
        ]
        if verified_items:
            for item in verified_items[:10]:
                lines.append(f"- {item.statement}")
                lines.append(
                    f"  Source: {item.title} | {item.domain} | {item.url or 'No URL preserved'}"
                )
                record = record_by_url.get(item.url or "")
                excerpt = (
                    item.source_excerpt
                    or (record.content_excerpt if record else "")
                    or (record.summary if record else "")
                )
                if excerpt:
                    lines.append(f"  Evidence excerpt: {excerpt}")
        else:
            lines.append("- No verified findings with source lineage were retained.")
        uncertainties = self._extract_open_uncertainty_lines(verification_brief)
        lines.extend(["", "### Remaining Open Uncertainties"])
        if uncertainties:
            lines.extend(uncertainties[:10])
        else:
            lines.append("- No open uncertainties were preserved.")
        return "\n".join(lines)

    def _build_deterministic_final_report(
        self,
        task: Task,
        task_state: TaskState,
        *,
        combined_findings: str = "",
        verification_brief: str = "",
        verified_items: list[VerifiedEvidenceItem] | None = None,
        reason: str = "",
    ) -> str:
        verified_items = verified_items or self._build_verified_evidence_items(
            task, task_state
        )
        verification_brief = verification_brief or self._build_claim_verification_brief(
            task, task_state
        )
        structured_packet = self._build_structured_evidence_packet(
            task,
            task_state,
            verification_brief=verification_brief,
            verified_items=verified_items,
        )
        findings_lines = [
            "## Executive Summary",
            "This report preserves only the strongest retained evidence and distinguishes supported findings from unresolved gaps.",
            "",
            "## Scope and Method",
            "The report is synthesized from retained subtask notes, authoritative search results, and deterministic claim verification over official or primary sources.",
            "",
            "## Findings",
            "### What the retained evidence supports",
        ]
        if verified_items:
            for item in verified_items[:6]:
                source_suffix = (
                    f" (Source: {item.title} | {item.url})"
                    if item.url
                    else f" (Source: {item.title} | {item.domain})"
                )
                findings_lines.append(f"- {item.statement}{source_suffix}")
        else:
            findings_lines.append(
                "- No primary-source-supported findings were retained strongly enough to summarize here."
            )
        findings_lines.extend(
            [
                "",
                "## Comparative Analysis",
                "The strongest leads are the ones backed by official or primary program pages with concrete supporting excerpts. Weaker leads are retained only as tentative or unresolved.",
                "",
                "## Limitations and Open Questions",
                "### Still unclear from the retained evidence",
            ]
        )
        open_uncertainties = self._extract_open_uncertainty_lines(verification_brief)
        if open_uncertainties:
            findings_lines.extend(open_uncertainties[:8])
        else:
            findings_lines.append("- No additional open uncertainties were preserved.")
        findings_lines.extend(
            [
                "",
                "## Recommended Next Steps",
                "1. Verify the strongest official leads directly on their current program pages.",
                "2. Treat any missing current-status, eligibility, funding, or deadline detail as unresolved until directly confirmed.",
            ]
        )
        report = "\n".join(findings_lines)
        return self._enforce_evidence_status_appendix_policy(
            report,
            verification_brief,
            verified_items=verified_items,
        )

    def _fallback_final_report_from_findings(
        self,
        task: Task,
        combined_findings: str,
        *,
        reason: str,
    ) -> str:
        return self._build_deterministic_final_report(
            task,
            self.task_states[task.name],
            combined_findings=combined_findings,
            reason=reason,
        )

    def get_current_subtask(self, task_name: str) -> str | None:
        if task_name not in self.task_states:
            return None
        task_state = self.task_states[task_name]
        if not task_state.research_plan or not task_state.research_plan.subtasks:
            return None
        if task_state.subtask_index < len(task_state.research_plan.subtasks):
            subtask = task_state.research_plan.subtasks[task_state.subtask_index]
            return f"{subtask.description} (Success: {subtask.success_criteria})"
        return "All subtasks completed"

    def get_task_progress(self, task_name: str) -> dict[str, any]:
        if task_name not in self.task_states:
            return {"progress": 0.0, "subtasks": [], "completed": 0, "total": 0}
        task_state = self.task_states[task_name]
        if not task_state.research_plan:
            return {"progress": 0.0, "subtasks": [], "completed": 0, "total": 0}
        subtasks = []
        for i, subtask in enumerate(task_state.research_plan.subtasks):
            subtask_info = {
                "order": subtask.order,
                "description": subtask.description,
                "success_criteria": subtask.success_criteria,
                "completed": i in task_state.completed_subtasks,
                "current": i == task_state.subtask_index,
                "findings_count": len(task_state.subtask_findings.get(i, [])),
                "has_report": i in task_state.subtask_reports,
                "guidance": task_state.get_subtask_guidance(i),
            }
            if i in task_state.subtask_agents:
                subtask_info["agent_metrics"] = task_state.subtask_agents[
                    i
                ].get_metrics()
            subtasks.append(subtask_info)
        return {
            "progress": task_state.get_progress_percentage(),
            "subtasks": subtasks,
            "completed": len(task_state.completed_subtasks),
            "total": len(task_state.research_plan.subtasks),
            "current_subtask": task_state.subtask_index + 1,
            "decision_count": task_state.decision_count,
            "stuck_count": task_state.stuck_count,
            "total_findings": len(task_state.all_findings),
            "subtask_agents_count": len(task_state.subtask_agents),
        }

    async def synthesize_final_report(
        self, task: Task, research_reports: list[str]
    ) -> str:
        task_state = self.task_states[task.name]
        cleaned_reports = [
            self._strip_control_channel_annotations(str(report or "")).strip()
            for report in research_reports
            if str(report or "").strip()
        ]
        combined_findings = "\n\n---\n\n".join(cleaned_reports)
        evidence_brief = self._build_task_evidence_brief(task, task_state)
        verification_brief = self._build_claim_verification_brief(task, task_state)
        verified_items = self._build_verified_evidence_items(task, task_state)
        structured_packet = self._build_structured_evidence_packet(
            task,
            task_state,
            verification_brief=verification_brief,
            verified_items=verified_items,
        )
        allowed_urls = [item.url for item in verified_items if item.url]
        deterministic_report = self._build_deterministic_final_report(
            task,
            task_state,
            combined_findings=combined_findings,
            verification_brief=verification_brief,
            verified_items=verified_items,
            reason="deterministic-baseline",
        )

        if self.report_chains is None:
            return deterministic_report

        manager = URLFlagManager()
        flagged_findings, flag_map = manager.replace_urls_with_flags(combined_findings)
        flagged_evidence_brief, evidence_map = manager.replace_urls_with_flags(
            evidence_brief
        )
        flag_map.update(evidence_map)
        flagged_verification_brief, verification_map = manager.replace_urls_with_flags(
            verification_brief
        )
        flag_map.update(verification_map)
        flagged_packet, packet_map = manager.replace_urls_with_flags(structured_packet)
        flag_map.update(packet_map)
        url_reference_table = (
            "\n".join(
                ["URL reference table:"]
                + [f"- {flag}: {url}" for flag, url in sorted(flag_map.items())]
            )
            if flag_map
            else ""
        )

        async def _generate() -> str:
            self.metrics["total_llm_calls"] += 1
            draft = await self.report_chains.asynthesize(
                task_name=task.name,
                task_description=task.description,
                combined_findings_with_flags=flagged_findings,
                evidence_brief_with_flags=flagged_evidence_brief,
                verification_brief_with_flags=flagged_verification_brief,
                structured_evidence_packet_with_flags=flagged_packet,
                url_reference_table=url_reference_table,
            )
            return manager.replace_flags_with_urls(draft, flag_map)

        async def _repair(invalid_draft: str) -> str:
            self.metrics["total_llm_calls"] += 1
            repaired = await self.report_chains.arepair_synthesis(
                task_name=task.name,
                task_description=task.description,
                invalid_draft=invalid_draft,
                combined_findings_with_flags=flagged_findings,
                evidence_brief_with_flags=flagged_evidence_brief,
                verification_brief_with_flags=flagged_verification_brief,
                structured_evidence_packet_with_flags=flagged_packet,
                url_reference_table=url_reference_table,
            )
            if not isinstance(repaired, str):
                return ""
            return manager.replace_flags_with_urls(repaired, flag_map)

        model_report = None
        try:
            model_report = await _generate()
        except Exception as exc:
            logger.warning("Final synthesis failed for task %s: %s", task.name, exc)

        for candidate in [model_report]:
            if not candidate:
                continue
            sanitized = self._sanitize_final_report_output(
                candidate,
                verification_brief,
                verified_items=verified_items,
            )
            if self._final_report_is_invalid(sanitized, task.description):
                continue
            if self._report_has_unknown_urls(sanitized, allowed_urls=allowed_urls):
                continue
            if any(
                phrase in sanitized.lower()
                for phrase in ("currently open opportunities", "immediately actionable")
            ):
                continue
            return sanitized

        if model_report:
            try:
                repaired = await _repair(model_report)
                sanitized = self._sanitize_final_report_output(
                    repaired,
                    verification_brief,
                    verified_items=verified_items,
                )
                if (
                    not self._final_report_is_invalid(sanitized, task.description)
                    and not self._report_has_unknown_urls(
                        sanitized, allowed_urls=allowed_urls
                    )
                    and "currently open opportunities" not in sanitized.lower()
                    and "immediately actionable" not in sanitized.lower()
                ):
                    return sanitized
            except Exception as exc:
                logger.warning(
                    "Final synthesis repair failed for task %s: %s", task.name, exc
                )

        return deterministic_report

    async def _dedupe_report_with_timeout(
        self,
        *,
        report: str,
        index: int,
        total: int,
    ) -> str:
        if self.report_chains is None:
            return report
        self.metrics["total_llm_calls"] += 1
        raw_timeout = float(os.getenv("ORCHESTRATOR_DEDUPE_TIMEOUT_SECONDS", "120"))
        fallback_max_tokens = max(
            256,
            int(os.getenv("ORCHESTRATOR_DEDUPE_FALLBACK_MAX_TOKENS", "2500")),
        )
        try:
            if raw_timeout <= 0:
                return await self.report_chains.adedupe_report(
                    report=report,
                    index=index,
                    total=total,
                )
            return await asyncio.wait_for(
                self.report_chains.adedupe_report(
                    report=report,
                    index=index,
                    total=total,
                ),
                timeout=max(1.0, raw_timeout),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Report dedupe timed out for report %s/%s after %.1fs; using truncated fallback",
                index,
                total,
                max(0.0, raw_timeout),
            )
        except Exception as e:
            logger.warning(
                "Report dedupe failed for report %s/%s: %s; using truncated fallback",
                index,
                total,
                e,
            )
        return self._truncate_text_by_token_budget(
            report,
            max_tokens=fallback_max_tokens,
            fallback_label="report dedupe fallback",
        )

    async def _summarize_on_demand_with_timeout(
        self,
        content: str,
        *,
        max_tokens: int,
        preserve_facts: bool,
        label: str,
    ) -> tuple[str, str, Any]:
        if not self.summarizer:
            return content, "none", None
        safe_max_tokens = max(64, int(max_tokens))
        raw_timeout = float(os.getenv("ORCHESTRATOR_SUMMARIZER_TIMEOUT_SECONDS", "120"))
        try:
            if raw_timeout <= 0:
                return await asyncio.to_thread(
                    self.summarizer.create_summary_on_demand,
                    content,
                    safe_max_tokens,
                    preserve_facts,
                )
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self.summarizer.create_summary_on_demand,
                    content,
                    safe_max_tokens,
                    preserve_facts,
                ),
                timeout=max(1.0, raw_timeout),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Summarization timed out for %s after %.1fs; using token-truncation fallback",
                label,
                max(0.0, raw_timeout),
            )
        except Exception as e:
            logger.warning(
                "Summarization failed for %s: %s; using token-truncation fallback",
                label,
                e,
            )
        fallback = self._truncate_text_by_token_budget(
            content,
            max_tokens=safe_max_tokens,
            fallback_label="orchestrator summary fallback",
        )
        return fallback, "fallback", None

    async def __aenter__(self):
        if self.use_mcp:
            await self.initialize_mcp()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.use_mcp:
            await self.cleanup_mcp()
