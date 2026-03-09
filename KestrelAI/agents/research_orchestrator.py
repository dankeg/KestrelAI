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
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from KestrelAI.agents.base import LlmWrapper
from KestrelAI.agents.config import get_orchestrator_config
from KestrelAI.agents.context_manager import ContextManager, TokenBudget, TokenCounter
from KestrelAI.agents.multi_level_summarizer import MultiLevelSummarizer
from KestrelAI.memory.vector_store import MemoryStore
from KestrelAI.shared.models import Task, TaskStatus

from .base_agent import OrchestratorAgent
from .langchain_adapter import LangChainChatAdapter
from .langchain_orchestrator_chains import OrchestratorLangChainControlChains
from .langchain_report_chains import OrchestratorLangChainChains
from .searxng_service import SearXNGService
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

LOW_SIGNAL_PLAN_DOMAINS = (
    "reddit.com",
    "quora.com",
    "medium.com",
    "substack.com",
    "shopify.com",
    "linkedin.com",
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


def _timeouts_disabled() -> bool:
    return os.getenv("GLOBAL_DISABLE_TIMEOUTS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _resolve_max_context_tokens(raw_value: Any | None) -> int:
    """Resolve max context tokens from settings/env with sane bounds."""
    fallback = 32768
    if raw_value is None:
        raw_value = os.getenv("MAX_CONTEXT_TOKENS", fallback)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return fallback
    return max(2048, min(value, 262144))


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
            max_context_tokens=self.max_context_tokens,
            use_mcp=mcp_manager is not None,
            mcp_manager=mcp_manager,
        )

        agent = WebResearchAgent(
            agent_id=subtask_id, llm=llm, memory=memory, config=config
        )

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
        self.max_context_tokens = _resolve_max_context_tokens(max_context_tokens)

        # Initialize memory store for subtask agents
        self.memory = MemoryStore()

        # Load configuration
        self.config = get_orchestrator_config(profile)

        # Initialize task states
        for task in tasks:
            self.task_states[task.name] = TaskState(
                task, max_context_tokens=self.max_context_tokens
            )

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
        self, task_name: str, reason: str
    ) -> OrchestratorDecision:
        """Fallback decision when structured review fails."""
        return OrchestratorDecision(
            reasoning=reason,
            decision="continue",
            feedback="Continue with current research direction",
            subtask="stay",
            next_task=task_name,
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
            "search_count": 0,
            "summary_count": 0,
            "checkpoint_count": 0,
            "query_count": 0,
        }
        agent = task_state.get_current_subtask_agent()
        if (
            agent is None
            or not hasattr(agent, "_state")
            or task.name not in agent._state
        ):
            return empty

        agent_state = agent._state[task.name]
        return {
            "action_count": int(getattr(agent_state, "action_count", 0)),
            "search_count": int(getattr(agent_state, "search_count", 0)),
            "summary_count": int(getattr(agent_state, "summary_count", 0)),
            "checkpoint_count": int(getattr(agent_state, "checkpoint_count", 0)),
            "query_count": int(len(getattr(agent_state, "queries", set()) or set())),
        }

    def _get_current_subtask_agent_state(self, task: Task, task_state: TaskState):
        """Return the current subtask agent state when available."""
        agent = task_state.get_current_subtask_agent()
        if agent is None or not hasattr(agent, "_state"):
            return None
        return agent._state.get(task.name)

    def _set_subtask_guidance(
        self, task_state: TaskState, subtask_index: int, guidance: str
    ) -> None:
        """Persist guidance and sync it onto the existing subtask agent config."""
        task_state.set_subtask_guidance(subtask_index, guidance)
        existing_agent = task_state.subtask_agents.get(subtask_index)
        if (
            existing_agent is not None
            and hasattr(existing_agent, "config")
            and hasattr(existing_agent.config, "orchestrator_guidance")
        ):
            existing_agent.config.orchestrator_guidance = (
                task_state.get_subtask_guidance(subtask_index)
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

    def _fallback_final_report_from_findings(
        self,
        task: Task,
        combined_findings: str,
        *,
        reason: str,
    ) -> str:
        """
        Deterministic fallback for invalid synthesis output. This avoids returning
        meta-critique text when the model drifts.
        """
        sanitized = re.sub(
            r"(?is)\b(overall assessment|strengths:|minor suggestions? for improvement).*",
            "",
            combined_findings or "",
        ).strip()
        body = self._truncate_text_by_token_budget(
            sanitized or "No findings captured.",
            max_tokens=3200,
            fallback_label="final-report-fallback",
        )
        return (
            "# Final Research Report\n\n"
            f"## Task\n{task.description}\n\n"
            f"## Verified Findings\n{body}\n\n"
            "## Tentative Findings\n- Additional items may require verification.\n\n"
            "## Open Uncertainties\n- Evidence was incomplete, so some findings could not be fully verified.\n\n"
            "## Next Verification Steps\n- Re-run focused verification on official program, .gov, .edu, or primary organization pages.\n\n"
            f"_Fallback reason: {reason}_"
        )

    def _build_guidance_from_decision(
        self,
        task: Task,
        task_state: TaskState,
        decision: OrchestratorDecision,
        readiness_reason: str = "",
    ) -> str:
        """Generate concrete guidance for subtask agents from decision + metrics."""
        guidance_parts: list[str] = []
        if decision.feedback and decision.feedback.strip():
            guidance_parts.append(decision.feedback.strip())

        metrics = self._get_current_subtask_agent_metrics(task, task_state)
        current_subtask = None
        if (
            task_state.research_plan
            and task_state.research_plan.subtasks
            and task_state.subtask_index < len(task_state.research_plan.subtasks)
        ):
            current_subtask = task_state.research_plan.subtasks[
                task_state.subtask_index
            ]

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
        evidence_stats = self._get_current_subtask_evidence_stats(task, task_state)
        missing_authoritative = max(
            0, min_authoritative_results - evidence_stats["authoritative_results"]
        )
        missing_official = max(
            0, min_official_results - evidence_stats["official_results"]
        )
        if missing_searches > 0:
            guidance_parts.append(
                f"Run at least {missing_searches} more targeted searches tied directly to the current success criteria."
            )
        if missing_queries > 0:
            guidance_parts.append(
                f"Increase search diversity with at least {missing_queries} additional distinct query variants."
            )
        if missing_authoritative > 0:
            guidance_parts.append(
                f"Find at least {missing_authoritative} more authoritative sources (.gov, .edu, official program pages, or primary organizations)."
            )
        if missing_official > 0:
            guidance_parts.append(
                "Verify key claims directly on an official source before advancing."
            )
        if metrics["checkpoint_count"] == 0 and metrics["action_count"] >= 3:
            guidance_parts.append(
                "Create a checkpoint summary to lock in evidence before deciding to transition."
            )

        stagnation_rounds = task_state.subtask_stagnation_rounds.get(
            task_state.subtask_index, 0
        )
        if stagnation_rounds >= 2:
            guidance_parts.append(
                "Current line of inquiry is stagnating; pivot to a different angle, source type, or constraint."
            )

        agent_state = self._get_current_subtask_agent_state(task, task_state)
        if agent_state is not None and getattr(agent_state, "search_history", None):
            recent_queries = []
            for item in list(agent_state.search_history)[-3:]:
                query = str(item.get("query", "")).strip()
                if query:
                    recent_queries.append(query)
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
                    return max(1, int(match.group(1)))
                except (TypeError, ValueError):
                    continue
        return 3

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

        # Never defer review when evidence has stagnated; we need orchestration decisions.
        if stagnation_rounds >= 2:
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
            _timeouts_disabled()
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
                task.name,
                "Control chains unavailable for review; defaulting to continue",
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
                task.name,
                f"Failed to get valid response from LLM ({err}), defaulting to continue",
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
                    success_criteria=(
                        "Define scope, assumptions, and information dimensions needed "
                        "for a strong final report."
                    ),
                ),
                Subtask(
                    order=2,
                    description="Gather high-signal evidence with targeted searches",
                    success_criteria=(
                        "Collect source-backed evidence, examples, and factual details "
                        "that directly support the research objective."
                    ),
                ),
                Subtask(
                    order=3,
                    description="Synthesize findings into a structured final report",
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
        if _timeouts_disabled() or step_timeout_seconds <= 0:
            step_timeout_seconds = 0.0
        preplanning_total_timeout_seconds = float(
            os.getenv(
                "ORCHESTRATOR_PREPLANNING_TOTAL_TIMEOUT_SECONDS",
                min(step_timeout_seconds * max_steps, 60.0),
            )
        )
        if _timeouts_disabled() or preplanning_total_timeout_seconds <= 0:
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
            research_plan.subtasks = research_plan.subtasks[:max_subtasks]

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
        report_sources = list(task_state.all_findings)
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
        final_report_sources = (
            list(task_state.all_findings) if task_state.all_findings else [latest_notes]
        )
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
                "Final report synthesis timed out for task %s after %.1fs; using fallback report",
                task.name,
                timeout_seconds,
            )
            return self._fallback_final_report(task, report_sources, "timeout")
        except Exception as e:
            logger.warning(
                "Final report synthesis failed for task %s: %s; using fallback report",
                task.name,
                e,
            )
            return self._fallback_final_report(task, report_sources, str(e))

    @staticmethod
    def _fallback_final_report(
        task: Task, report_sources: list[str], reason: str
    ) -> str:
        """Deterministic fallback when LLM synthesis is unavailable."""
        non_empty = [str(s).strip() for s in report_sources if str(s).strip()]
        condensed = "\n\n".join(non_empty[:4]) if non_empty else "No findings captured."
        return (
            f"# Final Research Report (Fallback)\n\n"
            f"Task: {task.name}\n\n"
            f"Reason fallback used: {reason}\n\n"
            f"## Consolidated Findings\n\n{condensed}\n"
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
            if not readiness_ok and not plateau_ok:
                guidance = self._build_guidance_from_decision(
                    task,
                    task_state,
                    decision,
                    readiness_reason=readiness_reason,
                )
                self._set_subtask_guidance(
                    task_state, task_state.subtask_index, guidance
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
                    task_state, task_state.subtask_index, plateau_guidance
                )
                logger.info(
                    "Allowing orchestrator advance for task %s via plateau readiness: %s",
                    task.name,
                    plateau_reason,
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
        self._set_subtask_guidance(task_state, task_state.subtask_index, guidance)
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
        if latest_notes.strip():
            lines.append("Latest Notes:")
            lines.append(latest_notes.strip())

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
                        if url:
                            lines.append(
                                f"  - {title} | {url} | domain={domain or 'unknown'} "
                                f"| tier={tier or 'unknown'} | authority={authority_score}"
                            )
                        else:
                            lines.append(f"  - {title}")
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
        domains: list[str] = []
        for raw in re.findall(r"https?://[^\s)]+", text or ""):
            host = (urlparse(raw).netloc or "").lower().strip()
            if host.startswith("www."):
                host = host[4:]
            if host:
                domains.append(host)
        return domains

    def _subtask_is_low_signal(self, subtask: Subtask) -> bool:
        haystack = f"{subtask.description}\n{subtask.success_criteria}".lower()
        domains = self._extract_domains_from_text(haystack)
        if any(domain.endswith(LOW_SIGNAL_PLAN_DOMAINS) for domain in domains):
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
            description=(
                "Identify distinct NSF, university, and nonprofit program opportunities relevant to the task from official sources."
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
    def _replacement_subtask_for_out_of_scope_artifacts(order: int) -> Subtask:
        return Subtask(
            order=order,
            description=(
                "Verify candidate opportunities and program requirements directly from official NSF and university program sources."
            ),
            success_criteria=(
                "Confirm which opportunities are real, current, and relevant by checking official program pages for deadlines, eligibility, and research focus."
            ),
        )

    def _normalize_research_plan(
        self, task: Task, research_plan: PlanningPlan
    ) -> PlanningPlan:
        """Harden planning output against low-signal or overly lead-specific subtasks."""
        normalized_subtasks: list[Subtask] = []
        seen_descriptions: set[str] = set()

        for subtask in research_plan.subtasks:
            description = " ".join(subtask.description.split()).strip()
            success_criteria = " ".join(subtask.success_criteria.split()).strip()
            candidate = Subtask(
                order=subtask.order,
                description=description,
                success_criteria=success_criteria,
            )

            if self._subtask_is_low_signal(candidate):
                logger.warning(
                    "Replacing low-signal planning subtask for task %s: %s",
                    task.name,
                    description[:180],
                )
                candidate = Subtask(
                    order=subtask.order,
                    description=(
                        "Validate promising leads against authoritative sources and discard low-signal or non-official pages."
                    ),
                    success_criteria=(
                        "Confirm which opportunities are supported by official program, .gov, .edu, or primary organization pages and remove weak leads."
                    ),
                )
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

            normalized_key = candidate.description.lower()
            if normalized_key in seen_descriptions:
                continue
            seen_descriptions.add(normalized_key)
            normalized_subtasks.append(candidate)

        if not normalized_subtasks:
            return self._fallback_planning_plan(
                task, "plan normalization removed all subtasks"
            )

        has_validation_step = any(
            any(
                token in subtask.description.lower()
                for token in ("verify", "validation", "authoritative", "official")
            )
            for subtask in normalized_subtasks
        )
        if not has_validation_step:
            normalized_subtasks.insert(
                min(1, len(normalized_subtasks)),
                Subtask(
                    order=2,
                    description=(
                        "Verify the strongest opportunities against authoritative sources before drawing conclusions."
                    ),
                    success_criteria=(
                        "Cross-check deadlines, eligibility, and application details on official program, .gov, .edu, or primary organization pages."
                    ),
                ),
            )

        for idx, subtask in enumerate(normalized_subtasks, start=1):
            subtask.order = idx

        research_plan.subtasks = normalized_subtasks
        return research_plan

    def _build_task_evidence_appendix(self, task: Task, task_state: TaskState) -> str:
        """Aggregate unique URLs and key query evidence across subtasks."""
        urls: list[str] = []
        seen_urls: set[str] = set()
        query_lines: list[str] = []
        authoritative_links: list[str] = []
        for idx, agent in sorted(task_state.subtask_agents.items()):
            try:
                agent_state = getattr(agent, "_state", {}).get(task.name)
                if agent_state is None:
                    continue
                for entry in list(getattr(agent_state, "search_history", []) or []):
                    query = str(entry.get("query", "")).strip()
                    results_count = int(entry.get("results_count", 0))
                    if query:
                        query_lines.append(
                            f"- Subtask {idx + 1}: {query} (results={results_count})"
                        )
                    for hit in list(entry.get("results", []) or []):
                        url = str(hit.get("url", "")).strip()
                        if not url or url in seen_urls:
                            continue
                        seen_urls.add(url)
                        urls.append(url)
                        if int(hit.get("authority_score", 0) or 0) >= 3:
                            domain = str(hit.get("domain", "")).strip() or "unknown"
                            authoritative_links.append(
                                f"- Subtask {idx + 1}: {domain} | {url}"
                            )
            except Exception:
                continue

        if not urls and not query_lines:
            return ""

        lines = ["## Evidence Appendix"]
        if query_lines:
            lines.append("")
            lines.append("### Queries Attempted")
            lines.extend(query_lines[:24])
        if urls:
            lines.append("")
            lines.append("### Source Links")
            for url in urls[:30]:
                lines.append(f"- {url}")
        if authoritative_links:
            lines.append("")
            lines.append("### Highest-Trust Sources")
            lines.extend(authoritative_links[:20])
        return "\n".join(lines)

    def _build_task_evidence_brief(self, task: Task, task_state: TaskState) -> str:
        """Create a compact evidence-quality brief for final synthesis."""
        lines = ["## Evidence Quality Brief"]
        for idx, agent in sorted(task_state.subtask_agents.items()):
            try:
                agent_state = getattr(agent, "_state", {}).get(task.name)
                if agent_state is None:
                    continue
                authoritative = 0
                official = 0
                domains: set[str] = set()
                top_hits: list[str] = []
                for entry in list(getattr(agent_state, "search_history", []) or []):
                    for hit in list(entry.get("results", []) or []):
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
                    f"- Subtask {idx + 1}: authoritative_results={authoritative}, "
                    f"official_results={official}, unique_domains={len(domains)}"
                )
                for hit in top_hits:
                    lines.append(f"  - {hit}")
            except Exception:
                continue
        return "\n".join(lines)

    @staticmethod
    def _tokenize_claim_text(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
            if token not in CLAIM_STOPWORDS
        }

    def _extract_claim_candidates(self, text: str) -> list[str]:
        claims: list[str] = []
        seen: set[str] = set()
        for raw_line in (text or "").splitlines():
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
            if len(line) < 40:
                continue
            for chunk in re.split(r"(?<=[.!?])\s+", line):
                candidate = " ".join(chunk.split()).strip()
                candidate = self._sanitize_claim_candidate(candidate)
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
            r"^\[(?:SUMMARY|SEARCH|THOUGHT|CHECKPOINT|COMPLETE)\]\s*", "", cleaned
        )
        cleaned = re.sub(r"^(?:Query|query)\s*:\s*", "", cleaned)
        cleaned = re.sub(r"\s*\[(?:primary|partial)\s+support:.*?\]\s*$", "", cleaned)
        cleaned = re.sub(r"https?://\S+", "", cleaned)
        cleaned = re.sub(r"[*_`#]+", " ", cleaned)
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.")

        lowered = cleaned.lower()
        if len(cleaned) < 40 or len(cleaned) > 240:
            return None
        if any(
            phrase in lowered
            for phrase in (
                "executive summary",
                "final report",
                "this report identifies",
                "confirmed as of",
                "report generated at",
                "critical to verify",
                "contact information is not consistently",
                "precise eligibility requirements",
                "ongoing changes",
                "funding details are not provided",
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
        if any(
            token in lowered
            for token in (
                "tentative",
                "uncertain",
                "likely ",
                "typically ",
                "not explicitly",
                "must verify",
                "check each program",
                "dynamic nature",
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

    @staticmethod
    def _build_primary_record_claims(
        authoritative_records: list[dict[str, Any]],
    ) -> list[str]:
        claims: list[str] = []
        seen: set[tuple[str, str]] = set()
        for record in authoritative_records:
            if not bool(record.get("official", False)):
                continue
            title = str(record.get("title", "")).strip()
            domain = str(record.get("domain", "")).strip().lower()
            if not title or not domain:
                continue
            key = (title.lower(), domain)
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                f"- {title} is an official source relevant to this task. [primary support: {domain}]"
            )
            if len(claims) >= 5:
                break
        return claims

    def _build_verified_evidence_items(
        self, task: Task, task_state: TaskState
    ) -> list[VerifiedEvidenceItem]:
        items: list[VerifiedEvidenceItem] = []
        seen_statements: set[str] = set()
        authoritative_records = self._authoritative_evidence_records(task, task_state)

        raw_reports: list[str] = []
        for idx in sorted(task_state.subtask_reports.keys()):
            raw_reports.append(str(task_state.subtask_reports[idx]))
        raw_reports.extend(
            str(item) for item in task_state.all_findings if str(item).strip()
        )
        combined = "\n\n".join(raw_reports[-6:])
        claims = self._extract_claim_candidates(combined)

        for claim in claims:
            claim_tokens = self._tokenize_claim_text(claim)
            if not claim_tokens:
                continue
            primary_supporting = [
                record
                for record in authoritative_records
                if bool(record.get("official", False))
                and len(claim_tokens & record["tokens"]) >= 2
            ]
            primary_supporting.sort(
                key=lambda record: int(record.get("authority_score", 0) or 0),
                reverse=True,
            )
            if not primary_supporting:
                continue
            sanitized_claim = self._sanitize_claim_candidate(claim)
            if not sanitized_claim:
                continue
            key = sanitized_claim.lower()
            if key in seen_statements:
                continue
            seen_statements.add(key)
            top = primary_supporting[0]
            items.append(
                VerifiedEvidenceItem(
                    statement=sanitized_claim,
                    title=str(top.get("title", "")).strip(),
                    domain=str(top.get("domain", "")).strip().lower(),
                    url=str(top.get("url", "")).strip(),
                )
            )
            if len(items) >= 6:
                return items

        seen_sources: set[tuple[str, str]] = set()
        for record in authoritative_records:
            if not bool(record.get("official", False)):
                continue
            title = str(record.get("title", "")).strip()
            domain = str(record.get("domain", "")).strip().lower()
            if not title or not domain:
                continue
            key = (title.lower(), domain)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            statement = f"{title} is an official source relevant to this task."
            if statement.lower() in seen_statements:
                continue
            items.append(
                VerifiedEvidenceItem(
                    statement=statement,
                    title=title,
                    domain=domain,
                    url=str(record.get("url", "")).strip(),
                )
            )
            if len(items) >= 6:
                break
        return items

    def _authoritative_evidence_records(
        self, task: Task, task_state: TaskState
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for idx, agent in sorted(task_state.subtask_agents.items()):
            try:
                agent_state = getattr(agent, "_state", {}).get(task.name)
                if agent_state is None:
                    continue
                for entry in list(getattr(agent_state, "search_history", []) or []):
                    query = str(entry.get("query", "")).strip()
                    for hit in list(entry.get("results", []) or []):
                        authority_score = int(hit.get("authority_score", 0) or 0)
                        if authority_score < 3:
                            continue
                        title = str(hit.get("title", "")).strip()
                        domain = str(hit.get("domain", "")).strip().lower()
                        url = str(hit.get("url", "")).strip()
                        official = bool(hit.get("official_source", False))
                        token_source = " ".join(
                            part for part in (title, domain, query) if part
                        )
                        records.append(
                            {
                                "subtask": idx + 1,
                                "title": title,
                                "domain": domain,
                                "url": url,
                                "official": official,
                                "authority_score": authority_score,
                                "tokens": self._tokenize_claim_text(token_source),
                            }
                        )
            except Exception:
                continue
        return records

    def _build_claim_verification_brief(self, task: Task, task_state: TaskState) -> str:
        """Build a deterministic claim verification summary from authoritative hits."""
        authoritative_records = self._authoritative_evidence_records(task, task_state)
        if not authoritative_records:
            return "## Claim Verification Brief\n- No authoritative evidence records available yet."

        raw_reports: list[str] = []
        for idx in sorted(task_state.subtask_reports.keys()):
            raw_reports.append(str(task_state.subtask_reports[idx]))
        raw_reports.extend(
            str(item) for item in task_state.all_findings if str(item).strip()
        )
        combined = "\n\n".join(raw_reports[-6:])
        claims = self._extract_claim_candidates(combined)
        if not claims:
            fallback_verified = self._build_primary_record_claims(authoritative_records)
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
            if not claim_tokens:
                continue
            supporting: list[dict[str, Any]] = []
            for record in authoritative_records:
                overlap = len(claim_tokens & record["tokens"])
                if overlap >= 2:
                    supporting.append(record)
            supporting.sort(
                key=lambda record: (
                    int(record["official"]),
                    int(record["authority_score"]),
                ),
                reverse=True,
            )
            primary_supporting = [
                record for record in supporting if bool(record.get("official", False))
            ]
            if primary_supporting:
                refs = ", ".join(
                    f"{item['domain'] or 'unknown'}" for item in primary_supporting[:2]
                )
                verified.append(f"- {claim} [primary support: {refs}]")
            elif supporting:
                refs = ", ".join(
                    f"{item['domain'] or 'unknown'}" for item in supporting[:2]
                )
                tentative.append(f"- {claim} [partial support: {refs}]")
            else:
                unsupported.append(f"- {claim}")

        lines = ["## Claim Verification Brief"]
        if verified:
            lines.append("### Verified Claims")
            lines.extend(verified[:8])
        elif authoritative_records:
            fallback_verified = self._build_primary_record_claims(authoritative_records)
            if fallback_verified:
                lines.append("### Verified Claims")
                lines.extend(fallback_verified)
        if tentative:
            lines.append("")
            lines.append("### Tentative Claims")
            lines.extend(tentative[:6])
        if unsupported:
            lines.append("")
            lines.append("### Unsupported or Weakly Supported Claims")
            lines.extend(unsupported[:5])
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
            if line.startswith("- "):
                cleaned = re.sub(r"\s*\[primary support:.*?\]\s*$", "", line).strip()
                sanitized = self._sanitize_claim_candidate(cleaned.lstrip("- ").strip())
                if sanitized:
                    lines.append(f"- {sanitized}")
        return lines

    def _enforce_verified_findings_policy(
        self,
        report_text: str,
        verification_brief: str,
        verified_items: list[VerifiedEvidenceItem] | None = None,
    ) -> str:
        """
        Replace the Verified Findings section with only primary-source-supported
        claims from the verification brief. This prevents the synthesis model from
        promoting tentative items into the verified section.
        """
        verified_claims = self._extract_verified_claim_lines(
            verification_brief,
            verified_items=verified_items,
        )
        verified_section = (
            "\n".join(verified_claims)
            if verified_claims
            else "- No primary-source-supported claims met the verification bar."
        )
        pattern = re.compile(
            r"(?ims)(^##\s+Verified Findings\s*\n)(.*?)(?=^##\s+Tentative Findings\b|\Z)"
        )
        if pattern.search(report_text or ""):
            return pattern.sub(rf"\1{verified_section}\n\n", report_text, count=1)
        return (
            report_text or ""
        ).strip() + f"\n\n## Verified Findings\n{verified_section}\n"

    def _sanitize_final_report_output(
        self,
        report_text: str,
        verification_brief: str,
        verified_items: list[VerifiedEvidenceItem] | None = None,
    ) -> str:
        text = (report_text or "").strip()
        if not text:
            return text

        section_markers = [
            "## Verified Findings",
            "## Tentative Findings",
            "## Open Uncertainties",
            "## Next Verification Steps",
        ]
        positions = [text.find(marker) for marker in section_markers if marker in text]
        if positions:
            text = text[min(positions) :].lstrip()
        else:
            text = re.sub(
                r"(?is)^\s*(okay[,!].*?|here(?:'|’)s\b.*?|certainly[,!].*?)\n+",
                "",
                text,
                count=1,
            ).strip()

        text = re.sub(r"(?im)^\s*##\s+Final Report:.*\n?", "", text)
        text = re.sub(
            r"(?is)^\s*\*\*Executive Summary:\*\*.*?(?=^##\s+|\Z)",
            "",
            text,
        ).strip()
        text = re.sub(r"(?im)^\s*---\s*$", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return self._enforce_verified_findings_policy(
            text,
            verification_brief,
            verified_items=verified_items,
        ).strip()

    def get_current_subtask(self, task_name: str) -> str | None:
        """Get current subtask description for a task"""
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
        """Get detailed progress information for a task with subtask agent details"""
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

            # Add agent metrics if available
            if i in task_state.subtask_agents:
                agent = task_state.subtask_agents[i]
                agent_metrics = agent.get_metrics()
                subtask_info.update(
                    {
                        "agent_metrics": agent_metrics,
                    }
                )

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
        """Combines multiple research reports into one cohesive final report with token-aware summarization"""
        if self.report_chains is None:
            raise RuntimeError(
                "LangChain report chains are required for report synthesis"
            )

        preserve_summary_facts = os.getenv(
            "ORCHESTRATOR_SUMMARIZER_PRESERVE_FACTS",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}

        # Deduplicate each report while preserving detail
        deduplicated_findings = []
        for i, report in enumerate(research_reports):
            # Use summarization if report is too long and context management is enabled
            if (
                self.context_management_enabled
                and self.summarizer
                and self.token_budget
            ):
                try:
                    report_tokens = self.token_counter.count_tokens(report)
                    max_report_tokens = self.token_budget.previous_findings // len(
                        research_reports
                    )  # Divide budget across reports

                    if report_tokens > max_report_tokens:
                        # Summarize report before deduplication
                        (
                            summary,
                            level,
                            _facts,
                        ) = await self._summarize_on_demand_with_timeout(
                            report,
                            max_tokens=max_report_tokens,
                            preserve_facts=preserve_summary_facts,
                            label=f"report-{i + 1}",
                        )
                        logger.debug(
                            f"Summarized report {i+1} for synthesis: {report_tokens} -> {self.token_counter.count_tokens(summary)} tokens"
                        )
                        report = summary
                except Exception as e:
                    logger.warning(
                        f"Error summarizing report {i+1}, using full content: {e}"
                    )

            deduplicated_findings.append(
                await self._dedupe_report_with_timeout(
                    report=report,
                    index=i + 1,
                    total=len(research_reports),
                )
            )

        # Combine deduplicated findings
        combined_findings = "\n\n---\n\n".join(deduplicated_findings)

        # Summarize combined findings if too long before final synthesis
        if self.context_management_enabled and self.summarizer and self.token_budget:
            try:
                combined_tokens = self.token_counter.count_tokens(combined_findings)
                max_combined_tokens = self.token_budget.previous_findings

                if combined_tokens > max_combined_tokens:
                    # Summarize combined findings
                    (
                        summary,
                        level,
                        _facts,
                    ) = await self._summarize_on_demand_with_timeout(
                        combined_findings,
                        max_tokens=max_combined_tokens,
                        preserve_facts=preserve_summary_facts,
                        label="combined-findings",
                    )
                    logger.debug(
                        f"Summarized combined findings for final synthesis: {combined_tokens} -> {self.token_counter.count_tokens(summary)} tokens"
                    )
                    combined_findings = summary
            except Exception as e:
                logger.warning(
                    f"Error summarizing combined findings, using full content: {e}"
                )

        # Create final synthesis with full context
        self.metrics["total_llm_calls"] += 1
        evidence_brief = self._build_task_evidence_brief(
            task, self.task_states[task.name]
        )
        verification_brief = self._build_claim_verification_brief(
            task, self.task_states[task.name]
        )
        verified_items = self._build_verified_evidence_items(
            task,
            self.task_states[task.name],
        )
        final_report = await self.report_chains.asynthesize(
            task_name=task.name,
            task_description=task.description,
            combined_findings=combined_findings,
            evidence_brief=evidence_brief,
            verification_brief=verification_brief,
        )
        final_report = self._sanitize_final_report_output(
            final_report,
            verification_brief,
            verified_items=verified_items,
        )
        if self._is_meta_critique_output(final_report, task.description):
            logger.warning(
                "Detected meta-critique drift in final synthesis for task %s; attempting repair pass",
                task.name,
            )
            try:
                self.metrics["total_llm_calls"] += 1
                repaired = await self.report_chains.arepair_synthesis(
                    task_name=task.name,
                    task_description=task.description,
                    invalid_draft=final_report,
                    combined_findings=combined_findings,
                    evidence_brief=evidence_brief,
                    verification_brief=verification_brief,
                )
                repaired = self._sanitize_final_report_output(
                    repaired,
                    verification_brief,
                    verified_items=verified_items,
                )
                if not self._is_meta_critique_output(repaired, task.description):
                    return repaired
                logger.warning(
                    "Repair pass still produced meta-critique drift for task %s; using deterministic fallback",
                    task.name,
                )
            except Exception as e:
                logger.warning(
                    "Repair synthesis failed for task %s: %s; using deterministic fallback",
                    task.name,
                    e,
                )
            return self._fallback_final_report_from_findings(
                task,
                combined_findings,
                reason="meta-critique-drift",
            )
        return final_report

    async def _dedupe_report_with_timeout(
        self,
        *,
        report: str,
        index: int,
        total: int,
    ) -> str:
        """Deduplicate one report with timeout + deterministic fallback."""
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
        """Run synchronous summarizer logic in a bounded worker thread."""
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
        """Async context manager entry"""
        if self.use_mcp:
            await self.initialize_mcp()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.use_mcp:
            await self.cleanup_mcp()
