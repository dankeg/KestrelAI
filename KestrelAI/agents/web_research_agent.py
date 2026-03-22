"""
Consolidated Research Agent for KestrelAI
Replaces all duplicate research agent implementations with a single, configurable agent
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import deque
from datetime import datetime
from typing import Any

from KestrelAI.agents.context_manager import ContextManager, TokenBudget, TokenCounter
from KestrelAI.agents.multi_level_summarizer import MultiLevelSummarizer
from KestrelAI.graphs.schemas import ResearchActionPlan
from KestrelAI.memory.hybrid_retriever import HybridRetriever
from KestrelAI.memory.langchain_retrieval_pipeline import LangChainRetrievalPipeline
from KestrelAI.memory.vector_store import MemoryStore
from KestrelAI.shared.models import Task
from KestrelAI.shared.research_utils import (
    build_research_task_profile,
    canonicalize_search_query,
    derive_topic_terms,
    extract_research_terms,
    normalize_research_text,
    text_is_opportunity_search,
    timeouts_disabled,
)
from KestrelAI.shared.runtime_settings import normalize_max_context_tokens

from .base_agent import AgentState
from .base_agent import ResearchAgent as BaseResearchAgent
from .context_builder import ContextBuilder
from .langchain_action_chains import WebResearchActionChains
from .langchain_adapter import LangChainChatAdapter
from .langchain_report_chains import WebResearchLangChainChains
from .prompt_builder import PromptBuilder
from .research_config import ResearchConfig
from .searxng_service import SearXNGService
from .tool_executor import LangChainToolExecutor
from .url_utils import URLFlagManager

try:
    from KestrelAI.graphs.subtask_runner import LangGraphSubtaskRunner
except Exception:  # pragma: no cover - dependency-gated path
    LangGraphSubtaskRunner = None

logger = logging.getLogger(__name__)

CHECKPOINT_STOPWORDS = frozenset(
    {
        "find",
        "finding",
        "currently",
        "current",
        "open",
        "available",
        "support",
        "supports",
        "supporting",
        "program",
        "programs",
        "opportunity",
        "opportunities",
        "funding",
        "grants",
        "grant",
        "fellowship",
        "fellowships",
        "task",
        "research",
        "students",
        "student",
        "senior",
        "undergraduate",
        "undergraduates",
        "united",
        "states",
        "us",
    }
)

CHECKPOINT_STRONG_OPPORTUNITY_MARKERS = frozenset(
    {
        "fellowship",
        "fellowships",
        "grant",
        "grants",
        "scholarship",
        "scholarships",
        "internship",
        "internships",
        "stipend",
        "stipends",
    }
)

CHECKPOINT_PROGRAM_SUPPORT_MARKERS = frozenset(
    {
        "undergraduate",
        "undergraduates",
        "student",
        "students",
        "summer",
        "research",
        "apply",
        "application",
        "applications",
        "deadline",
        "deadlines",
        "eligibility",
        "stipend",
        "funding",
    }
)

CHECKPOINT_APPLICATION_MARKERS = frozenset(
    {
        "apply",
        "application",
        "applications",
        "deadline",
        "deadlines",
        "eligibility",
        "eligible",
        "accepting",
        "accepting applications",
        "rolling",
        "apply now",
        "apply here",
        "submit",
    }
)

CHECKPOINT_DEGREE_MARKERS = frozenset(
    {
        "admission",
        "degree",
        "degrees",
        "bachelor",
        "bachelors",
        "bachelor's",
        "major",
        "majors",
        "minor",
        "minors",
        "curriculum",
        "academics",
        "admissions",
        "student affairs",
        "student life",
        "undergraduate program",
        "undergraduate programs",
    }
)

CHECKPOINT_NEWS_RELEASE_MARKERS = frozenset(
    {
        "news release",
        "press release",
        "grant and award announcement",
        "award announcement",
        "media contact",
        "for immediate release",
    }
)

CHECKPOINT_PRECOLLEGE_MARKERS = frozenset(
    {
        "high school",
        "high-school",
        "secondary school",
        "middle school",
        "k-12",
        "k12",
        "teen",
        "teens",
        "youth",
    }
)

CHECKPOINT_MIXED_AUDIENCE_PROFESSIONAL_MARKERS = frozenset(
    {
        "early-career",
        "early career",
        "professional",
        "professionals",
        "career researchers",
        "researchers",
        "all levels",
    }
)

CHECKPOINT_INDIRECT_FUNDING_PATTERNS = (
    r"\bawarded funding\b",
    r"\bawarded (?:an? )?grant\b",
    r"\bgrant awarded\b",
    r"\breceived funding\b",
    r"\bsecured funding\b",
    r"\bcontinu(?:e|ing|ation) (?:an? |the )?.*?\b(?:program|initiative|cohort|fellowship|scholarship|internship)\b",
    r"\bfunded to continue\b",
    r"\bannounces? funding\b",
    r"\baward(?:ed)? to continue\b",
)


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _env_timeout_seconds(
    name: str,
    default: str,
    *,
    minimum: float = 1.0,
) -> float | None:
    if timeouts_disabled():
        return None
    raw_value = float(os.getenv(name, default))
    if raw_value <= 0:
        return None
    return max(minimum, raw_value)


class WebResearchAgent(BaseResearchAgent):
    """Web research agent that handles search, analysis, and reporting"""

    def __init__(
        self, agent_id: str, llm, memory: MemoryStore, config: ResearchConfig = None
    ):
        super().__init__(agent_id, llm, memory)
        self.config = config or ResearchConfig()
        self.scratchpad = []

        # LangChain adapter for all model interactions.
        self.langchain_adapter: LangChainChatAdapter | None = None
        try:
            self.langchain_adapter = LangChainChatAdapter.from_llm(llm)
        except Exception as e:
            raise RuntimeError("LangChain adapter unavailable") from e

        self.subtask_graph_runner = None

        # Initialize hybrid retriever
        try:
            self.hybrid_retriever = HybridRetriever(memory, enable_bm25=True)
            self.hybrid_retrieval_enabled = True
            logger.info("Hybrid retrieval enabled (vector + BM25)")
        except Exception as e:
            logger.warning(
                f"Failed to initialize hybrid retriever: {e}. Using vector search only."
            )
            self.hybrid_retriever = None
            self.hybrid_retrieval_enabled = False
        # Initialize URL flag manager
        self.url_flag_manager = URLFlagManager()

        # Initialize SearXNG service
        self.searxng_service = SearXNGService(
            searxng_url=None,  # Use default from env
            search_results=self.config.search_results,
            debug=self.config.debug,
        )

        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(self.config)

        # Tooling and retrieval abstractions (LangChain-backed).
        self.tool_executor: LangChainToolExecutor | None = None
        self.retrieval_pipeline: LangChainRetrievalPipeline | None = None
        self.report_chains: WebResearchLangChainChains | None = None
        self.action_chains: WebResearchActionChains | None = None

        # Initialize context management and summarization
        try:
            # Get model name from LLM wrapper if available (for TokenCounter)
            # Note: We pass llm object (not model_name string) to MultiLevelSummarizer
            model_name = getattr(llm, "model", "gemma3:27b")
            self.token_counter = TokenCounter(model_name=model_name)
            self.token_budget = TokenBudget(
                max_context=normalize_max_context_tokens(
                    getattr(self.config, "max_context_tokens", None)
                )
            )
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
            logger.info(f"Context management enabled with model: {model_name}")
        except Exception as e:
            logger.warning(
                f"Failed to initialize context management: {e}. Continuing without it."
            )
            self.token_counter = None
            self.token_budget = None
            self.context_manager = None
            self.summarizer = None
            self.context_management_enabled = False

        try:
            self.tool_executor = LangChainToolExecutor(
                searxng_service=self.searxng_service,
                url_flag_manager=self.url_flag_manager,
                mcp_manager=self.config.mcp_manager,
                mcp_enabled=lambda: bool(self.config.use_mcp and self.mcp_connected),
            )
        except Exception as e:
            raise RuntimeError("LangChain tool executor unavailable") from e

        try:
            self.retrieval_pipeline = LangChainRetrievalPipeline(
                memory_store=memory,
                hybrid_retriever=(
                    self.hybrid_retriever if self.hybrid_retrieval_enabled else None
                ),
                token_counter=self.token_counter,
                summarizer=self.summarizer,
                context_management_enabled=self.context_management_enabled,
                debug=self.config.debug,
            )
        except Exception as e:
            raise RuntimeError("LangChain retrieval pipeline unavailable") from e

        try:
            self.report_chains = WebResearchLangChainChains(
                model=self.langchain_adapter.client
            )
        except Exception as e:
            raise RuntimeError("LangChain report chains unavailable") from e
        try:
            self.action_chains = WebResearchActionChains(
                model=self.langchain_adapter.client
            )
        except Exception as e:
            raise RuntimeError("LangChain action chains unavailable") from e

        # Initialize context builder (lazy initialization to avoid forward reference)
        self._context_builder_initialized = False
        self.context_builder = None

        # Initialize MCP if configured
        self.mcp_connected = False
        if self.config.use_mcp and self.config.mcp_manager:
            asyncio.create_task(self._initialize_mcp())

    async def _initialize_mcp(self):
        """Initialize MCP manager if configured"""
        try:
            if not self.config.mcp_manager.is_initialized:
                self.mcp_connected = await self.config.mcp_manager.initialize()
            else:
                self.mcp_connected = self.config.mcp_manager.is_initialized
        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}")
            self.mcp_connected = False

    async def run_step(self, task: Task) -> str:
        """Run one research step using the LangGraph runtime."""
        return await self._run_step_langgraph(task)

    async def _run_step_langgraph(self, task: Task) -> str:
        """LangGraph-driven research loop."""
        if not self._context_builder_initialized:
            self._initialize_context_builder()

        state = self._state.setdefault(task.name, AgentState(task_id=task.name))
        if self.subtask_graph_runner is None:
            if LangGraphSubtaskRunner is None:
                raise RuntimeError("LangGraph runner unavailable")
            self.subtask_graph_runner = LangGraphSubtaskRunner(self)

        return await self.subtask_graph_runner.run(task, state)

    async def _finalize_run_after_loops(self, task: Task, state: AgentState) -> str:
        """Finalize run when loop budget is exhausted."""
        if state.action_count % self.config.checkpoint_freq != 0:
            await self._create_checkpoint(task, state)

        final_report = await self._generate_final_report(task, state)
        await self._store_final_report_with_summaries(task, state, final_report)
        return final_report

    def _build_step_feedback(self, task: Task, state: AgentState) -> str:
        """Build a bounded progress update for a single worker step."""
        feedback = (getattr(state, "last_step_feedback", "") or "").strip()
        if feedback:
            return feedback

        recent_history = [str(item).strip() for item in list(state.history)[-3:]]
        recent_history = [item for item in recent_history if item]
        if recent_history:
            return "\n".join(recent_history)

        focus = (state.current_focus or "").strip()
        if focus:
            return f"[PROGRESS] Continuing subtask research with focus: {focus}"

        return f"[PROGRESS] Continuing research for: {task.description}"

    def _rag_write_timeout_seconds(self) -> float | None:
        return _env_timeout_seconds("AGENT_RAG_WRITE_TIMEOUT_SECONDS", "30")

    @staticmethod
    def _sanitize_intermediate_note(text: str, *, default_prefix: str) -> str:
        cleaned = re.sub(r"\r\n?", "\n", str(text or "")).strip()
        if not cleaned:
            return default_prefix

        cleaned = re.sub(
            r"(?im)^\s{0,3}(?:#{1,6}\s*)?(?:final report|actionable shortlist|executive summary|summary report)\s*:?.*$",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"(?is)^\s*(?:okay|here(?:'| i)s|below is|this (?:report|summary)|i found)\b[^\\n]*\n+",
            "",
            cleaned,
        ).strip()

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        bullet_lines: list[str] = []
        for line in lines:
            normalized = re.sub(r"^\s*(?:[-*•]|\d+\.)\s*", "", line).strip()
            lowered = normalized.lower()
            if not normalized:
                continue
            if lowered.startswith(
                (
                    "verified findings",
                    "tentative findings",
                    "open uncertainties",
                    "next verification steps",
                )
            ):
                continue
            if "actionable shortlist" in lowered or "final report" in lowered:
                continue
            bullet_lines.append(normalized)

        if not bullet_lines:
            fallback = cleaned[:400].strip()
            return f"{default_prefix}\n- {fallback}" if fallback else default_prefix

        compact_lines = bullet_lines[:6]
        if not all(line.startswith("- ") for line in compact_lines):
            compact_lines = [f"- {line}" for line in compact_lines]
        return "\n".join(compact_lines)

    @staticmethod
    def _checkpoint_scope_terms(task: Task, max_terms: int = 10) -> set[str]:
        tokens = derive_topic_terms(
            str(task.description or task.name or ""),
            max_terms=max_terms,
            stopwords=CHECKPOINT_STOPWORDS,
        )
        if tokens:
            return set(tokens[:max_terms])
        fallback = extract_research_terms(
            str(task.description or task.name or ""),
            max_terms=max_terms,
            stopwords=CHECKPOINT_STOPWORDS,
        )
        return set(fallback[:max_terms])

    @staticmethod
    def _evidence_focused_excerpt(*texts: object, limit: int = 220) -> str:
        date_pattern = re.compile(
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
            r"nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}(?:,\s+\d{4})?\b",
            flags=re.IGNORECASE,
        )
        for raw_text in texts:
            cleaned = re.sub(r"\s+", " ", str(raw_text or "")).strip()
            if not cleaned:
                continue
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
                if sentence.strip()
            ]
            if sentences:
                best_index = -1
                best_score = -1
                for index, sentence in enumerate(sentences):
                    lowered = sentence.lower()
                    score = 0
                    if "deadline to apply" in lowered:
                        score += 12
                    if "deadline" in lowered or date_pattern.search(sentence):
                        score += 8
                    if "can apply" in lowered and re.search(
                        r"\b(freshman|sophmore|sophomore|junior|senior|undergraduate|student|students)\b",
                        lowered,
                    ):
                        score += 7
                    if any(
                        marker in lowered
                        for marker in (
                            "apply",
                            "application",
                            "applications",
                            "eligible",
                            "eligibility",
                            "accepting",
                        )
                    ):
                        score += 4
                    if re.search(
                        r"\b(undergraduate|student|students|senior)\b", lowered
                    ):
                        score += 3
                    if re.search(
                        r"\b(research|stipend|funding|grant|fellowship|internship|program)\b",
                        lowered,
                    ):
                        score += 1
                    if (
                        re.match(r"^\d+\)", lowered)
                        and "deadline" not in lowered
                        and not date_pattern.search(sentence)
                    ):
                        score -= 3
                    if "statement of purpose" in lowered or "which projects" in lowered:
                        score -= 2
                    if any(
                        phrase in lowered
                        for phrase in (
                            "advice on applying",
                            "website is terrible",
                            "as soon as possible",
                            "i will not be extending",
                        )
                    ):
                        score -= 5
                    if score > best_score:
                        best_score = score
                        best_index = index
                if best_score > 0 and best_index >= 0:
                    window: list[str] = []
                    for index in range(
                        max(0, best_index - 1), min(len(sentences), best_index + 2)
                    ):
                        candidate = " ".join(window + [sentences[index]]).strip()
                        if len(candidate) > limit and window:
                            break
                        if len(candidate) > limit:
                            window = [candidate[: max(40, limit - 3)].rstrip() + "..."]
                            break
                        window.append(sentences[index])
                    preferred = " ".join(window).strip()
                    if preferred:
                        return preferred
            if len(cleaned) <= limit:
                return cleaned
            return cleaned[: max(40, limit - 3)].rstrip() + "..."
        return ""

    @staticmethod
    def _checkpoint_excerpt(hit: dict[str, Any], limit: int = 220) -> str:
        return WebResearchAgent._evidence_focused_excerpt(
            hit.get("content_excerpt", ""),
            hit.get("summary", ""),
            hit.get("snippet", ""),
            limit=limit,
        )

    @staticmethod
    def _task_is_opportunity_search(task: Task) -> bool:
        lowered = str(task.description or task.name or "").lower()
        return WebResearchAgent._text_is_opportunity_search(lowered)

    @staticmethod
    def _text_is_opportunity_search(text: str) -> bool:
        return text_is_opportunity_search(text)

    def _active_task_context_text(self, state: AgentState) -> str:
        parts = [
            getattr(self.config, "subtask_description", ""),
            getattr(self.config, "success_criteria", ""),
            state.current_focus,
            getattr(self.config, "orchestrator_guidance", ""),
        ]
        return " ".join(
            str(part or "").strip() for part in parts if str(part or "").strip()
        )

    def _active_context_is_opportunity_search(self, state: AgentState) -> bool:
        return self._text_is_opportunity_search(self._active_task_context_text(state))

    def _active_topic_anchor_terms(self, state: AgentState) -> set[str]:
        return set(
            derive_topic_terms(
                self._active_task_context_text(state),
                max_terms=6,
                stopwords=CHECKPOINT_STOPWORDS,
            )
        )

    @staticmethod
    def _hit_combined_text(hit: dict[str, Any], query: str = "") -> str:
        return " ".join(
            str(part or "")
            for part in (
                query,
                hit.get("title", ""),
                hit.get("summary", ""),
                hit.get("content_excerpt", ""),
                hit.get("content", ""),
                hit.get("url", ""),
                hit.get("domain", ""),
            )
        ).lower()

    @classmethod
    def _hit_has_application_signal(cls, hit: dict[str, Any]) -> bool:
        combined = cls._hit_combined_text(hit)
        return any(marker in combined for marker in CHECKPOINT_APPLICATION_MARKERS)

    @classmethod
    def _hit_is_indirect_funding_announcement(cls, hit: dict[str, Any]) -> bool:
        combined = cls._hit_combined_text(hit)
        if any(marker in combined for marker in CHECKPOINT_NEWS_RELEASE_MARKERS):
            return True
        if cls._hit_has_application_signal(hit):
            return False
        return any(
            re.search(pattern, combined, flags=re.IGNORECASE)
            for pattern in CHECKPOINT_INDIRECT_FUNDING_PATTERNS
        )

    @staticmethod
    def _checkpoint_has_strong_opportunity_signal(hit: dict[str, Any]) -> bool:
        combined = WebResearchAgent._hit_combined_text(hit)
        tokens = set(re.findall(r"[a-z0-9]{2,}", combined))
        profile = build_research_task_profile(combined)
        if WebResearchAgent._hit_is_indirect_funding_announcement(hit):
            return False
        if tokens.intersection(CHECKPOINT_STRONG_OPPORTUNITY_MARKERS):
            return True
        if any(marker in combined for marker in CHECKPOINT_DEGREE_MARKERS):
            return False
        if profile.target_terms and (profile.audience_terms or profile.evidence_terms):
            return True
        if "funding" in tokens or "award" in tokens or "awards" in tokens:
            return bool(tokens.intersection(CHECKPOINT_PROGRAM_SUPPORT_MARKERS))
        if "program" in tokens or "programs" in tokens:
            return bool(profile.audience_terms or profile.evidence_terms) or bool(
                tokens.intersection(CHECKPOINT_PROGRAM_SUPPORT_MARKERS)
            )
        if "opportunity" in tokens or "opportunities" in tokens:
            return bool(profile.audience_terms or profile.evidence_terms) or bool(
                tokens.intersection(CHECKPOINT_PROGRAM_SUPPORT_MARKERS)
            )
        return False

    @staticmethod
    def _checkpoint_is_degree_program_hit(hit: dict[str, Any]) -> bool:
        combined = " ".join(
            str(hit.get(key, "") or "")
            for key in ("title", "summary", "content_excerpt", "snippet", "url")
        ).lower()
        if not any(marker in combined for marker in CHECKPOINT_DEGREE_MARKERS):
            return False
        return not WebResearchAgent._checkpoint_has_strong_opportunity_signal(hit)

    @staticmethod
    def _checkpoint_is_mixed_audience_hit(hit: dict[str, Any]) -> bool:
        combined = WebResearchAgent._hit_combined_text(hit)
        has_undergrad = any(
            marker in combined
            for marker in (
                "undergraduate",
                "undergraduates",
                "undergrad",
                "college student",
                "college students",
                "senior undergraduate",
                "rising senior",
            )
        )
        has_precollege = any(
            marker in combined for marker in CHECKPOINT_PRECOLLEGE_MARKERS
        )
        if not (has_undergrad and has_precollege):
            return False
        return any(
            marker in combined
            for marker in CHECKPOINT_MIXED_AUDIENCE_PROFESSIONAL_MARKERS
        )

    def _checkpoint_hit_score(self, task: Task, query: str, hit: dict[str, Any]) -> int:
        if hit.get("task_aligned") is False:
            return -10
        scope_terms = self._checkpoint_scope_terms(task)
        title = str(hit.get("title", "") or "")
        summary = str(hit.get("summary", "") or "")
        excerpt = str(hit.get("content_excerpt", "") or "")
        combined = " ".join((query, title, summary, excerpt)).lower()
        tokens = set(re.findall(r"[a-z0-9]{2,}", combined))
        scope_overlap = len(tokens.intersection(scope_terms))
        opportunity_overlap = len(
            tokens.intersection(
                {
                    "fellowship",
                    "fellowships",
                    "grant",
                    "grants",
                    "scholarship",
                    "scholarships",
                    "internship",
                    "internships",
                    "program",
                    "programs",
                    "undergraduate",
                    "undergraduates",
                    "student",
                    "students",
                    "application",
                    "applications",
                    "deadline",
                    "deadlines",
                }
            )
        )
        authority = int(hit.get("authority_score", 0) or 0)
        official = int(bool(hit.get("official_source", False)))
        fetched = int(bool(hit.get("fetched", False)))
        title_lower = title.lower()
        if (
            "arxiv" in str(hit.get("domain", "")).lower()
            or "/abs/" in str(hit.get("url", "") or "").lower()
        ):
            return -10
        if scope_overlap == 0:
            return -5
        if self._task_is_opportunity_search(
            task
        ) and self._checkpoint_is_degree_program_hit(hit):
            return -8
        if self._task_is_opportunity_search(
            task
        ) and self._hit_is_indirect_funding_announcement(hit):
            return -8
        if self._task_is_opportunity_search(
            task
        ) and not self._checkpoint_has_strong_opportunity_signal(hit):
            return -6
        mixed_audience_penalty = 0
        if self._task_is_opportunity_search(
            task
        ) and self._checkpoint_is_mixed_audience_hit(hit):
            mixed_audience_penalty = 5
        if opportunity_overlap == 0 and any(
            marker in str(task.description or "").lower()
            for marker in (
                "fellowship",
                "grant",
                "scholarship",
                "internship",
                "program",
            )
        ):
            return -3
        if any(marker in title_lower for marker in ("survey", "review", "paper")):
            return -2
        return (
            (scope_overlap * 3)
            + (opportunity_overlap * 3)
            + authority
            + (official * 2)
            + fetched
            - mixed_audience_penalty
        )

    def _search_hit_alignment_score(
        self,
        query: str,
        hit: dict[str, Any],
        state: AgentState,
    ) -> int:
        if not self._active_context_is_opportunity_search(state):
            return int(hit.get("authority_score", 0) or 0) + int(
                bool(hit.get("official_source", False))
            )

        if self._checkpoint_is_degree_program_hit(hit):
            return -8
        if self._hit_is_indirect_funding_announcement(hit):
            return -8
        if not self._checkpoint_has_strong_opportunity_signal(hit):
            return -6

        hit_tokens = set(re.findall(r"[a-z0-9]{2,}", self._hit_combined_text(hit)))
        query_terms = {
            token
            for token in re.findall(r"[a-z0-9]{2,}", query.lower())
            if token not in CHECKPOINT_STOPWORDS
            and token not in {"site", "edu", "org", "gov", "com"}
        }
        topical_terms = {
            token
            for token in query_terms
            if token
            not in {
                "fellowship",
                "fellowships",
                "grant",
                "grants",
                "scholarship",
                "scholarships",
                "program",
                "programs",
                "internship",
                "internships",
                "funding",
                "opportunity",
                "opportunities",
                "undergraduate",
                "undergraduates",
                "student",
                "students",
                "us",
                "senior",
            }
        }
        if topical_terms and not hit_tokens.intersection(topical_terms):
            return -5
        active_topic_terms = self._active_topic_anchor_terms(state)
        if active_topic_terms and not hit_tokens.intersection(active_topic_terms):
            return -5
        overlap = len(hit_tokens.intersection(query_terms))
        if overlap == 0:
            return -3
        return (
            overlap * 2
            + len(hit_tokens.intersection(CHECKPOINT_APPLICATION_MARKERS))
            + int(hit.get("authority_score", 0) or 0)
            + int(bool(hit.get("official_source", False)))
            + int(bool(hit.get("fetched", False)))
        )

    def _build_deterministic_checkpoint(self, task: Task, state: AgentState) -> str:
        recent_entries = list(getattr(state, "search_history", []) or [])[-8:]
        ranked_hits: list[tuple[int, dict[str, Any], str]] = []
        for entry in recent_entries:
            query = str(entry.get("query", "")).strip()
            for hit in list(entry.get("results", []) or []):
                score = self._checkpoint_hit_score(task, query, hit)
                if score <= 0:
                    continue
                ranked_hits.append((score, hit, query))

        ranked_hits.sort(
            key=lambda item: (
                item[0],
                int(bool(item[1].get("official_source", False))),
                int(item[1].get("authority_score", 0) or 0),
                int(bool(item[1].get("fetched", False))),
            ),
            reverse=True,
        )

        if not ranked_hits:
            recent_queries = [
                str(entry.get("query", "")).strip()
                for entry in recent_entries[-3:]
                if str(entry.get("query", "")).strip()
            ]
            lines = ["- No primary-source-backed checkpoint evidence available yet."]
            if recent_queries:
                lines.append(
                    "- Recent searches attempted: " + "; ".join(recent_queries[:3])
                )
            lines.append(
                "- Continue by targeting official program pages or primary organization sites with clearer opportunity-specific queries."
            )
            return "\n".join(lines)

        bullets: list[str] = []
        seen_sources: set[tuple[str, str]] = set()
        for score, hit, query in ranked_hits:
            if self._task_is_opportunity_search(task) and not (
                bool(hit.get("fetched", False))
                and self._checkpoint_has_strong_opportunity_signal(hit)
            ):
                continue
            title = str(hit.get("title", "")).strip() or "Untitled source"
            domain = str(hit.get("domain", "")).strip() or "unknown"
            url = str(hit.get("url", "")).strip()
            key = (title.lower(), domain.lower())
            if key in seen_sources:
                continue
            seen_sources.add(key)
            excerpt = self._checkpoint_excerpt(hit)
            evidence_text = self._hit_combined_text(hit).lower()
            label = (
                "Official lead"
                if bool(hit.get("official_source", False))
                else "Authoritative lead"
                if int(hit.get("authority_score", 0) or 0) >= 3
                else "Tentative discovery lead"
            )
            bullet = f"- {label}: {title} ({domain})"
            if excerpt:
                bullet += f" - Evidence: {excerpt}"
            elif url:
                bullet += f" - Source: {url}"
            bullets.append(bullet)
            if not re.search(
                r"\b(rolling|deadline|apply|application|applications|accepting applications)\b",
                evidence_text,
            ):
                bullets.append(
                    f"- Open gap for {title}: current application timing is not directly confirmed in the retained evidence."
                )
            if not re.search(
                r"\b(undergraduate|student|students|senior|freshman|sophmore|sophomore|junior)\b",
                evidence_text,
            ):
                bullets.append(
                    f"- Open gap for {title}: senior-undergraduate fit is not directly confirmed in the retained evidence."
                )
            if len(bullets) >= 6:
                break

        return "\n".join(bullets[:6]).strip() or (
            "- No concrete checkpoint evidence available yet."
        )

    async def _add_to_rag_async(
        self,
        task: Task,
        text: str,
        doc_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist RAG entries off the event loop to prevent step stalls."""
        timeout_seconds = self._rag_write_timeout_seconds()
        try:
            add_call = asyncio.to_thread(
                self._add_to_rag,
                task,
                text,
                doc_type,
                metadata,
            )
            if timeout_seconds is None:
                return await add_call
            return await asyncio.wait_for(add_call, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(
                "RAG write timed out for task %s (type=%s) after %.1fs; skipping",
                task.name,
                doc_type,
                timeout_seconds,
            )
            return ""
        except Exception as e:
            logger.warning(
                "RAG write failed for task %s (type=%s): %s",
                task.name,
                doc_type,
                e,
            )
            return ""

    async def _store_final_report_with_summaries(
        self, task: Task, state: AgentState, final_report: str
    ) -> str:
        """Persist final report and hierarchical summaries in RAG."""
        report_id = await self._add_to_rag_async(
            task,
            final_report,
            "final_report",
            metadata={
                "layer": "episodic",
                "action_count": state.action_count,
                "importance_score": self._calculate_importance(final_report),
                "is_final": True,
            },
        )

        if self.context_management_enabled and self.summarizer:
            try:
                hierarchy = await asyncio.to_thread(
                    self.summarizer.create_summary_hierarchy,
                    final_report,
                    preserve_facts=True,
                )
                summaries = hierarchy.get("summaries", {})
                for level_name, summary_text in summaries.items():
                    if level_name == "detailed":
                        continue
                    layer = "semantic" if level_name in ["medium"] else "summary"
                    await self._add_to_rag_async(
                        task,
                        summary_text,
                        f"final_report_{level_name}",
                        metadata={
                            "report_id": report_id,
                            "layer": layer,
                            "summary_level": level_name,
                            "is_final": True,
                        },
                    )
            except Exception as e:
                logger.warning(f"Failed to create/store final report summaries: {e}")

        return report_id

    def _plan_next_action(self, context: str) -> ResearchActionPlan:
        """Build the next action plan via LangChain structured output."""
        if self.action_chains is None:
            raise RuntimeError("LangChain action chains unavailable for planning")
        return self.action_chains.next_action(
            system_prompt=self.prompt_builder.get_system_prompt(),
            context=context,
        )

    def _initialize_context_builder(self):
        """Initialize context builder after all methods are defined (lazy initialization)"""
        if not self._context_builder_initialized:
            self.context_builder = ContextBuilder(
                config=self.config,
                url_flag_manager=self.url_flag_manager,
                context_manager=self.context_manager,
                token_budget=self.token_budget,
                retrieve_from_rag_func=self._retrieve_from_rag,
            )
            self._context_builder_initialized = True

    async def _handle_think_action(self, plan: dict, state: AgentState):
        """Handle think action"""
        thought = plan.get("thought", "").strip()
        if not thought:
            thought = "Analyzing current information..."

        if self.config.debug:
            print(f"  Thinking: {thought[:100]}...")

        state.history.append(f"[THOUGHT] {thought}")
        state.last_step_activity = "thinking"
        state.last_step_feedback = f"[THOUGHT] {thought}"
        state.current_focus = plan.get("direction", state.current_focus)

        # Track metrics
        state.think_count += 1
        self.metrics["total_thoughts"] += 1

    async def _handle_search_action(self, plan: dict, state: AgentState):
        """Handle search action"""
        query = plan.get("query", "").strip()
        if not query:
            state.history.append("[SKIP] Empty search query")
            state.last_step_activity = "search"
            state.last_step_feedback = "[SKIP] Empty search query"
            return

        canonical_query = self._canonicalize_query(query)
        pathway_id = state.pathway_for_query(canonical_query)

        # Enhanced duplicate detection
        if (
            canonical_query in state.queries
            or state.repeated_queries.get(canonical_query, 0) >= 2
        ):
            state.history.append(f"[SKIP] Already searched: {query}")
            state.last_step_activity = "search"
            state.last_step_feedback = f"[SKIP] Already searched: {query}"
            return

        if self.tool_executor is None:
            raise RuntimeError("LangChain tool executor unavailable for search")

        state.search_attempt_count += 1
        search_result = await self.tool_executor.ainvoke("search_web", {"query": query})
        raw_hits = list(search_result.get("hits", []) or [])
        search_time = float(search_result.get("search_time", 0.0))
        raw_results_count = len(raw_hits)

        def _filter_hits(
            candidate_query: str, candidate_hits: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
            filtered_hits: list[dict[str, Any]] = []
            opportunity_search = self._active_context_is_opportunity_search(state)
            for raw_hit in candidate_hits:
                hit = dict(raw_hit)
                relevance_score = self._search_hit_alignment_score(
                    candidate_query,
                    hit,
                    state,
                )
                hit["task_relevance_score"] = relevance_score
                hit["task_aligned"] = relevance_score > 0
                if opportunity_search and relevance_score <= 0:
                    continue
                filtered_hits.append(hit)
            return filtered_hits

        hits = _filter_hits(query, raw_hits)

        state.queries.add(canonical_query)

        if not hits:
            relaxed_query = self._relax_query(query)
            if relaxed_query:
                relaxed_canonical = self._canonicalize_query(relaxed_query)
                if (
                    relaxed_canonical
                    and relaxed_canonical != canonical_query
                    and relaxed_canonical not in state.queries
                ):
                    relaxed_result = await self.tool_executor.ainvoke(
                        "search_web", {"query": relaxed_query}
                    )
                    state.search_attempt_count += 1
                    relaxed_raw_hits = list(relaxed_result.get("hits", []) or [])
                    relaxed_time = float(relaxed_result.get("search_time", 0.0))
                    state.queries.add(relaxed_canonical)
                    relaxed_hits = _filter_hits(relaxed_query, relaxed_raw_hits)
                    if relaxed_hits:
                        query = relaxed_query
                        hits = relaxed_hits
                        search_time = relaxed_time
                        raw_results_count = len(relaxed_raw_hits)

            if not hits:
                state.record_pathway_attempt(pathway_id, hit=False)
                state.zero_result_search_count += 1
                state.search_history.append(
                    {
                        "timestamp": _now_iso(),
                        "query": query,
                        "results_count": 0,
                        "raw_results_count": raw_results_count,
                        "search_time": search_time,
                        "pathway_id": pathway_id,
                        "results": [],
                    }
                )
                state.history.append(f"[NO RESULTS] {query}")
                state.last_step_activity = "search"
                state.last_step_feedback = f"[NO RESULTS] {query}"
                return

        if self.config.debug:
            print(f"  Searching: {query}")

        state.search_count += 1
        state.record_pathway_attempt(pathway_id, hit=True)
        self.metrics["total_searches"] += 1

        # Track search in detailed history
        search_entry = {
            "timestamp": _now_iso(),
            "query": query,
            "results_count": len(hits),
            "raw_results_count": raw_results_count,
            "search_time": search_time,
            "pathway_id": pathway_id,
            "results": [],
        }

        if not hasattr(state, "snips"):
            state.snips = deque(maxlen=10)

        search_results: list[str] = []
        formatted_results: list[str] = []
        for hit in hits:
            title = str(hit.get("title", ""))
            url = str(hit.get("url", "")).strip()
            snippet = str(hit.get("snippet", ""))
            summary = str(hit.get("summary", ""))
            content = str(hit.get("content", ""))
            domain = str(hit.get("domain", "")).strip()
            source_tier = str(hit.get("source_tier", "")).strip()
            authority_score = int(hit.get("authority_score", 0) or 0)
            official_source = bool(hit.get("official_source", False))
            if snippet:
                state.snips.append(snippet)

            if bool(hit.get("fetched")):
                self.metrics["total_web_fetches"] += 1

            search_results.append(title)
            formatted_results.append(f"{title} ({url})" if url else title)

            # Add to search entry (use cleaned URL)
            search_entry["results"].append(
                {
                    "title": title,
                    "url": url,
                    "fetched": bool(hit.get("fetched")),
                    "summary": summary[:280],
                    "content_excerpt": self._evidence_focused_excerpt(
                        content,
                        summary,
                        snippet,
                        limit=700,
                    ),
                    "snippet": snippet[:700],
                    "domain": domain,
                    "source_tier": source_tier,
                    "authority_score": authority_score,
                    "official_source": official_source,
                    "task_aligned": bool(hit.get("task_aligned", True)),
                    "task_relevance_score": int(
                        hit.get("task_relevance_score", 0) or 0
                    ),
                }
            )

        state.search_history.append(search_entry)
        history_results = ", ".join(formatted_results[:3]) if formatted_results else ""
        state.history.append(f"[SEARCH] {query}\n  Found: {history_results}")
        state.last_step_activity = "search"
        state.last_step_feedback = f"[SEARCH] {query}\n  Found: {history_results}"

        # Track metrics
        self.metrics["total_search_results"] += len(search_results)

    @staticmethod
    def _canonicalize_query(query: str) -> str:
        return canonicalize_search_query(query)

    @staticmethod
    def _relax_query(query: str) -> str:
        lowered = (query or "").lower()
        site_filters = [
            f"site:.{match.group(1)}"
            for match in re.finditer(r"site:\.(gov|edu|org|com)\b", lowered)
        ]
        semantic_tokens = re.findall(r"[a-zA-Z0-9]+", normalize_research_text(lowered))
        if not semantic_tokens and not site_filters:
            return ""

        stop = {
            "the",
            "a",
            "an",
            "for",
            "with",
            "and",
            "or",
            "to",
            "in",
            "of",
            "on",
            "by",
            "from",
            "at",
            "that",
            "this",
            "these",
            "those",
            "2026",
        }
        filtered = [t for t in semantic_tokens if t not in stop]
        if not filtered:
            filtered = semantic_tokens

        preserve_priority = list(
            dict.fromkeys(
                derive_topic_terms(query, max_terms=4)
                + [
                    "undergraduate",
                    "undergraduates",
                    "student",
                    "students",
                    "fellowship",
                    "fellowships",
                    "grant",
                    "grants",
                    "scholarship",
                    "scholarships",
                    "internship",
                    "internships",
                    "funding",
                    "deadline",
                    "eligibility",
                ]
            )
        )

        ordered: list[str] = []
        seen: set[str] = set()

        for token in site_filters:
            if token not in seen:
                seen.add(token)
                ordered.append(token)

        for token in preserve_priority:
            if token in filtered and token not in seen:
                seen.add(token)
                ordered.append(token)

        for token in filtered:
            if token not in seen and token not in {"site", "gov", "edu", "org", "com"}:
                seen.add(token)
                ordered.append(token)

        if not ordered:
            return ""

        max_terms = 8 if site_filters else 7
        return " ".join(ordered[:max_terms]).strip()

    async def _handle_mcp_tool_action(self, plan: dict, state: AgentState):
        """Handle MCP tool action"""
        if not self.config.use_mcp or not self.mcp_connected:
            state.history.append("[SKIP] MCP not available")
            state.last_step_activity = "mcp_tool"
            state.last_step_feedback = "[SKIP] MCP not available"
            return

        tool_name = plan.get("tool_name", "").strip()
        tool_parameters = plan.get("tool_parameters", {})

        if not tool_name:
            state.history.append("[SKIP] No tool name specified")
            state.last_step_activity = "mcp_tool"
            state.last_step_feedback = "[SKIP] No tool name specified"
            return

        if self.config.debug:
            print(f"  Using MCP tool: {tool_name}")

        if self.tool_executor is None:
            raise RuntimeError("LangChain tool executor unavailable for MCP calls")

        result = await self.tool_executor.call_mcp_tool(
            tool_name=tool_name,
            tool_parameters=tool_parameters,
        )

        if result.get("success"):
            state.history.append(f"[MCP_TOOL] {tool_name}: Success")
            state.last_step_activity = "mcp_tool"
            state.last_step_feedback = f"[MCP_TOOL] {tool_name}: Success"
            if not hasattr(state, "snips"):
                state.snips = deque(maxlen=10)
            if result.get("data") is not None:
                state.snips.append(
                    f"MCP Tool Result ({tool_name}):\n"
                    f"{json.dumps(result.get('data'), indent=2)}"
                )
        else:
            error = result.get("error") or "Unknown error"
            state.history.append(f"[MCP_TOOL] {tool_name}: Failed - {error}")
            state.last_step_activity = "mcp_tool"
            state.last_step_feedback = f"[MCP_TOOL] {tool_name}: Failed - {error}"

    async def _handle_summarize_action(self, task: Task, state: AgentState):
        """Handle summarize action"""
        snips = getattr(state, "snips", deque())
        material = "\n---\n".join(snips) if snips else "(no new material)"

        # Replace URLs with flags in material before sending to LLM
        material_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(material)
        url_table = self.url_flag_manager.get_url_reference_table()

        if self.report_chains is None:
            raise RuntimeError("LangChain report chains are required for summarization")
        self.metrics["total_llm_calls"] += 1
        llm_step_timeout_seconds = _env_timeout_seconds(
            "AGENT_LLM_STEP_TIMEOUT_SECONDS",
            "45",
        )
        try:
            summarize_call = self.report_chains.asummarize_notes(
                task_description=task.description,
                material_with_flags=material_with_flags,
                url_reference_table=url_table,
            )
            if llm_step_timeout_seconds is None:
                notes_with_flags = await summarize_call
            else:
                notes_with_flags = await asyncio.wait_for(
                    summarize_call,
                    timeout=llm_step_timeout_seconds,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "Summarize action timed out for task %s after %.1fs; using fallback summary",
                task.name,
                llm_step_timeout_seconds,
            )
            notes_with_flags = material_with_flags[:1200]

        # Replace flags back with URLs using the complete flag mapping
        flag_mapping = self.url_flag_manager.flag_to_url.copy()
        notes = self.url_flag_manager.replace_flags_with_urls(
            notes_with_flags, flag_mapping
        )
        notes = self._sanitize_intermediate_note(
            notes,
            default_prefix="- No concrete evidence extracted yet.",
        )

        state.history.append(f"[SUMMARY] {notes[:200]}...")
        state.last_step_activity = "summary"
        state.last_step_feedback = f"[SUMMARY] {notes[:400]}".strip()

        # Store summary in RAG with metadata
        await self._add_to_rag_async(
            task,
            notes,
            "summary",
            metadata={
                "layer": "semantic",  # Summaries go to semantic layer
                "action_count": state.action_count,
                "importance_score": self._calculate_importance(notes),
            },
        )
        snips.clear()

        # Track metrics
        state.summary_count += 1
        self.metrics["total_summaries"] += 1

        # Reset loop counters on summarize
        state.consecutive_thinks = 0
        state.consecutive_searches = 0

    async def _handle_complete_action(self, task: Task, state: AgentState) -> str:
        """Handle complete action (for subtask agents)"""
        if not self.config.is_subtask_agent:
            state.history.append(
                "[SKIP] Complete action only available for subtask agents"
            )
            state.last_step_activity = "complete"
            state.last_step_feedback = (
                "[SKIP] Complete action only available for subtask agents"
            )
            return await self._generate_final_report(task, state)

        completion_reason = "Subtask objectives met"
        state.history.append(f"[COMPLETE] {completion_reason}")
        state.last_step_activity = "complete"
        state.last_step_feedback = f"[COMPLETE] {completion_reason}"

        # Generate final report for this subtask
        final_report = await self._generate_final_report(task, state)
        await self._store_final_report_with_summaries(task, state, final_report)

        return final_report

    async def _create_checkpoint(self, task: Task, state: AgentState):
        """Create a checkpoint summarizing current progress"""
        if self.config.debug:
            print(f"[CHECKPOINT] Creating checkpoint at action {state.action_count}")
        checkpoint_used_fallback = False
        checkpoint = self._build_deterministic_checkpoint(task, state)
        checkpoint = self._sanitize_intermediate_note(
            checkpoint,
            default_prefix="- No concrete checkpoint evidence available yet.",
        )

        # Store checkpoint in state
        state.last_checkpoint = checkpoint
        state.checkpoints.append(checkpoint)
        state.last_step_activity = "checkpoint"
        state.last_step_feedback = (
            f"[CHECKPOINT] Focus: {state.current_focus or 'General research'}\n"
            f"{checkpoint[:500]}".strip()
        )

        # Store checkpoint in RAG (detailed/episodic layer)
        checkpoint_id = await self._add_to_rag_async(
            task,
            checkpoint,
            "checkpoint",
            metadata={
                "checkpoint_index": state.checkpoint_count,
                "layer": "episodic",  # Detailed layer
                "action_count": state.action_count,
                "importance_score": self._calculate_importance(
                    checkpoint
                ),  # Based on fact count
            },
        )

        # Create and store summary hierarchy if context management is enabled
        if self.context_management_enabled and self.summarizer:
            if checkpoint_used_fallback:
                logger.info(
                    "Skipping checkpoint hierarchy generation for task %s because checkpoint content used fallback",
                    task.name,
                )
            else:
                checkpoint_hierarchy_timeout_seconds = _env_timeout_seconds(
                    "AGENT_CHECKPOINT_HIERARCHY_TIMEOUT_SECONDS",
                    "20",
                )
                try:
                    hierarchy_call = self.summarizer.create_summary_hierarchy_async(
                        checkpoint,
                        preserve_facts=True,
                    )
                    if checkpoint_hierarchy_timeout_seconds is None:
                        hierarchy = await hierarchy_call
                    else:
                        hierarchy = await asyncio.wait_for(
                            hierarchy_call,
                            timeout=checkpoint_hierarchy_timeout_seconds,
                        )
                    summaries = hierarchy.get("summaries", {})
                    facts = hierarchy.get("facts")

                    # Store summaries at different levels in RAG
                    for level_name, summary_text in summaries.items():
                        if level_name == "detailed":
                            continue  # Already stored as checkpoint

                        # Determine layer based on compression level
                        if level_name in ["medium"]:
                            layer = "semantic"
                        elif level_name in ["summary", "executive"]:
                            layer = "summary"
                        else:
                            layer = "semantic"

                        await self._add_to_rag_async(
                            task,
                            summary_text,
                            f"checkpoint_{level_name}",
                            metadata={
                                "checkpoint_index": state.checkpoint_count,
                                "checkpoint_id": checkpoint_id,  # Link to detailed version
                                "layer": layer,
                                "summary_level": level_name,
                                "original_length": len(checkpoint),
                                "compressed_length": len(summary_text),
                                "action_count": state.action_count,
                                "importance_score": self._calculate_importance(
                                    checkpoint
                                ),
                            },
                        )

                    # Store facts separately for quick access
                    if facts and facts.to_text():
                        await self._add_to_rag_async(
                            task,
                            facts.to_text(),
                            "checkpoint_facts",
                            metadata={
                                "checkpoint_index": state.checkpoint_count,
                                "checkpoint_id": checkpoint_id,
                                "layer": "facts",
                                "type": "extracted_facts",
                            },
                        )

                    # Store in memory for quick access
                    if hasattr(state, "checkpoint_summaries"):
                        state.checkpoint_summaries[state.checkpoint_count] = {
                            "checkpoint_id": checkpoint_id,
                            "hierarchy": hierarchy,
                        }
                    else:
                        state.checkpoint_summaries = {
                            state.checkpoint_count: {
                                "checkpoint_id": checkpoint_id,
                                "hierarchy": hierarchy,
                            }
                        }

                    if self.config.debug:
                        logger.debug(
                            f"Stored checkpoint {state.checkpoint_count} with {len(summaries)} summary levels in RAG"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to create/store checkpoint summaries: {e}",
                        exc_info=True,
                    )

        # Track metrics
        state.checkpoint_count += 1
        self.metrics["total_checkpoints"] += 1

        # Progressive compression: Compress old checkpoints periodically
        # Run compression every 3 checkpoints to manage memory growth
        # FIXED: Add error handling for background async task
        if state.checkpoint_count % 3 == 0:

            async def compress_with_error_handling():
                try:
                    await self._compress_old_checkpoints(task, state)
                except Exception as e:
                    logger.error(
                        f"Error in background checkpoint compression for task {task.name}: {e}",
                        exc_info=True,
                    )

            asyncio.create_task(compress_with_error_handling())

        # Clear old history after checkpoint
        state.history.clear()
        state.history.append(
            f"[CHECKPOINT CREATED] Focus: {state.current_focus or 'General research'}"
        )

    async def _generate_final_report(self, task: Task, state: AgentState) -> str:
        """Generate final report from all checkpoints and findings"""
        # Retrieve relevant content from RAG using semantic search
        # Use task description as query to get most relevant information
        rag_read_timeout_seconds = _env_timeout_seconds(
            "AGENT_RAG_READ_TIMEOUT_SECONDS",
            "20",
        )
        try:
            rag_call = asyncio.to_thread(
                self._retrieve_from_rag,
                task,
                task.description,
                self.token_budget.rag_content if self.token_budget else None,
            )
            if rag_read_timeout_seconds is None:
                rag_content = await rag_call
            else:
                rag_content = await asyncio.wait_for(
                    rag_call,
                    timeout=rag_read_timeout_seconds,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "RAG retrieval timed out for task %s after %.1fs; continuing without RAG context",
                task.name,
                rag_read_timeout_seconds,
            )
            rag_content = ""

        # Combine all checkpoints
        all_checkpoints = (
            "\n\n---\n\n".join(state.checkpoints) if state.checkpoints else ""
        )

        # Build context from previous reports if available
        previous_reports_context = ""
        if self.config.previous_reports:
            previous_reports_context = (
                "\n\n--- Previous Research Reports ---\n\n"
                + "\n\n---\n\n".join(self.config.previous_reports)
            )

        # If context management is enabled, use summarization for long content
        if self.context_management_enabled and self.summarizer and self.token_budget:
            try:
                # Summarize checkpoints if too long
                if all_checkpoints:
                    checkpoint_tokens = self.token_counter.count_tokens(all_checkpoints)
                    max_checkpoint_tokens = self.token_budget.checkpoints
                    if checkpoint_tokens > max_checkpoint_tokens:
                        (
                            summary,
                            level,
                            facts,
                        ) = self.summarizer.create_summary_on_demand(
                            all_checkpoints,
                            max_tokens=max_checkpoint_tokens,
                            preserve_facts=True,
                        )
                        all_checkpoints = summary
                        if self.config.debug:
                            logger.debug(
                                f"Summarized checkpoints: {checkpoint_tokens} -> {self.token_counter.count_tokens(summary)} tokens"
                            )

                # Summarize previous reports if too long
                if previous_reports_context:
                    reports_tokens = self.token_counter.count_tokens(
                        previous_reports_context
                    )
                    max_reports_tokens = self.token_budget.previous_findings
                    if reports_tokens > max_reports_tokens:
                        (
                            summary,
                            level,
                            facts,
                        ) = self.summarizer.create_summary_on_demand(
                            previous_reports_context,
                            max_tokens=max_reports_tokens,
                            preserve_facts=True,
                        )
                        previous_reports_context = summary
                        if self.config.debug:
                            logger.debug(
                                f"Summarized previous reports: {reports_tokens} -> {self.token_counter.count_tokens(summary)} tokens"
                            )
            except Exception as e:
                logger.warning(
                    f"Error in report summarization, using full content: {e}"
                )

        # Replace URLs with flags in all content before sending to LLM
        # Process all texts to build up the flag manager's complete mapping
        checkpoints_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(
            all_checkpoints
        )
        rag_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(rag_content)
        previous_reports_with_flags = ""
        if previous_reports_context:
            prev_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(
                previous_reports_context
            )
            previous_reports_with_flags = prev_with_flags

        # Get the complete flag mapping after processing all texts
        # This ensures we have all flags that were created during processing
        combined_flag_mapping = self.url_flag_manager.flag_to_url.copy()
        url_table = self.url_flag_manager.get_url_reference_table()

        if self.report_chains is None:
            raise RuntimeError(
                "LangChain report chains are required for final report generation"
            )
        self.metrics["total_llm_calls"] += 1
        raw_final_timeout_seconds = float(
            os.getenv("AGENT_SUBTASK_FINAL_REPORT_TIMEOUT_SECONDS", "180")
        )
        if timeouts_disabled():
            raw_final_timeout_seconds = 0.0
        try:
            if raw_final_timeout_seconds <= 0:
                final_report_with_flags = (
                    await self.report_chains.agenerate_final_report(
                        task_description=task.description,
                        previous_reports_with_flags=previous_reports_with_flags,
                        checkpoints_with_flags=checkpoints_with_flags,
                        rag_with_flags=rag_with_flags,
                        url_reference_table=url_table,
                    )
                )
            else:
                final_report_with_flags = await asyncio.wait_for(
                    self.report_chains.agenerate_final_report(
                        task_description=task.description,
                        previous_reports_with_flags=previous_reports_with_flags,
                        checkpoints_with_flags=checkpoints_with_flags,
                        rag_with_flags=rag_with_flags,
                        url_reference_table=url_table,
                    ),
                    timeout=max(1.0, raw_final_timeout_seconds),
                )
        except asyncio.TimeoutError:
            logger.warning(
                "Subtask final report generation timed out for task %s after %.1fs; using fallback content",
                task.name,
                max(0.0, raw_final_timeout_seconds),
            )
            final_report_with_flags = (
                checkpoints_with_flags[-2400:]
                if checkpoints_with_flags
                else rag_with_flags[-2400:]
            ) or "Final report fallback: no checkpoint content available."

        # Replace flags back with actual URLs
        final_report = self.url_flag_manager.replace_flags_with_urls(
            final_report_with_flags, combined_flag_mapping
        )

        return final_report

    def _retrieve_from_rag(
        self, task: Task, query: str = None, max_tokens: int = None
    ) -> str:
        """Retrieve relevant content from RAG via LangChain retrieval pipeline."""
        if max_tokens is None and self.token_budget:
            max_tokens = self.token_budget.rag_content

        search_query = query or task.description

        try:
            if self.retrieval_pipeline is None:
                raise RuntimeError("LangChain retrieval pipeline unavailable")

            self.retrieval_pipeline.hybrid_retriever = (
                self.hybrid_retriever if self.hybrid_retrieval_enabled else None
            )
            self.retrieval_pipeline.context_management_enabled = (
                self.context_management_enabled
            )
            self.retrieval_pipeline.token_counter = self.token_counter
            self.retrieval_pipeline.summarizer = self.summarizer
            return self.retrieval_pipeline.retrieve(
                task_name=task.name,
                query=search_query,
                max_tokens=max_tokens,
                fallback_entries=self.scratchpad,
            )

        except Exception as e:
            logger.warning(
                f"Error retrieving from RAG, falling back to scratchpad: {e}",
                exc_info=True,
            )
            # Fallback to scratchpad
            if self.scratchpad:
                recent_entries = (
                    self.scratchpad[-5:]
                    if len(self.scratchpad) > 5
                    else self.scratchpad
                )
                return "\n\n".join(recent_entries)
            return "(No previous findings)"

    def _calculate_importance(self, content: str) -> float:
        """
        Calculate importance score for content based on fact density.
        Higher score = more important (more facts = more important).
        """
        if not content:
            return 0.0

        # Count facts (deadlines, URLs, requirements, etc.)
        fact_indicators = [
            r"\bdeadline\b",
            r"\bdue\b",
            r"\bapplication\b",
            r"https?://",
            r"www\.",
            r"\.edu",
            r"\.gov",
            r"\$\d+",
            r"\d+%",
            r"GPA",
            r"eligibility",
            r"@\w+",
            r"contact",
            r"email",
            r"phone",
        ]

        fact_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in fact_indicators
        )

        # Normalize by content length (facts per 1000 chars)
        content_length = len(content)
        if content_length == 0:
            return 0.0

        importance = (fact_count / content_length) * 1000
        return min(importance, 10.0)  # Cap at 10.0

    async def _compress_old_checkpoints(self, task: Task, state: AgentState):
        """
        Progressive compression: Move old checkpoints to summary layers.
        This is called periodically to manage memory growth.
        """
        if not self.context_management_enabled or not self.summarizer:
            return

        # Compression thresholds
        EPISODIC_LIMIT = 5  # Keep last 5 checkpoints in episodic layer
        SEMANTIC_LIMIT = 10  # Keep checkpoints 6-10 in semantic layer

        # Only compress if we have more than EPISODIC_LIMIT checkpoints
        if state.checkpoint_count <= EPISODIC_LIMIT:
            return

        try:
            # Get old checkpoints that should be compressed
            # Checkpoints are indexed from 0, so older ones have lower indices
            old_checkpoint_indices = list(
                range(
                    max(0, state.checkpoint_count - SEMANTIC_LIMIT),
                    state.checkpoint_count - EPISODIC_LIMIT,
                )
            )

            if not old_checkpoint_indices:
                return

            if self.retrieval_pipeline is None:
                return
            docs = self.retrieval_pipeline.retrieve_documents(
                task_name=task.name,
                query=f"checkpoint task:{task.name}",
                k=50,
            )
            if not docs:
                return

            metadatas = [dict(doc.metadata or {}) for doc in docs]

            # Find checkpoints that need compression
            for doc in docs:
                meta = dict(doc.metadata or {})
                if meta.get("task") != task.name:
                    continue

                checkpoint_idx = meta.get("checkpoint_index", -1)
                current_layer = meta.get("layer", "episodic")

                # Skip if already compressed or not in target range
                if checkpoint_idx not in old_checkpoint_indices:
                    continue

                if current_layer == "episodic":
                    # Move from episodic to semantic layer
                    # Check if semantic version already exists
                    has_semantic = any(
                        m.get("checkpoint_index") == checkpoint_idx
                        and m.get("layer") == "semantic"
                        for m in metadatas
                    )

                    if (
                        not has_semantic
                        and checkpoint_idx in old_checkpoint_indices[:5]
                    ):
                        # Create semantic summary if it doesn't exist
                        # The summary should already exist from checkpoint creation, but verify
                        logger.debug(
                            f"Checkpoint {checkpoint_idx} should be in semantic layer"
                        )

                elif current_layer == "semantic":
                    # Move from semantic to summary layer if very old
                    if checkpoint_idx < state.checkpoint_count - SEMANTIC_LIMIT:
                        # Check if summary version exists
                        has_summary = any(
                            m.get("checkpoint_index") == checkpoint_idx
                            and m.get("layer") == "summary"
                            for m in metadatas
                        )

                        if not has_summary:
                            # Create summary version
                            # Get the semantic version to compress further
                            semantic_doc = doc.page_content
                            (
                                summary,
                                level,
                                facts,
                            ) = self.summarizer.create_summary_on_demand(
                                semantic_doc,
                                max_tokens=500,  # Very compressed
                                preserve_facts=True,
                            )

                            # Store summary version
                            checkpoint_id = meta.get(
                                "checkpoint_id", meta.get("doc_id", "")
                            )
                            await self._add_to_rag_async(
                                task,
                                summary,
                                "checkpoint_summary_compressed",
                                metadata={
                                    "checkpoint_index": checkpoint_idx,
                                    "checkpoint_id": checkpoint_id,
                                    "layer": "summary",
                                    "summary_level": "executive",
                                    "compressed_from": "semantic",
                                    "action_count": meta.get("action_count", 0),
                                    "importance_score": meta.get(
                                        "importance_score", 0.0
                                    ),
                                },
                            )

                            logger.debug(
                                f"Compressed checkpoint {checkpoint_idx} from semantic to summary layer"
                            )

        except Exception as e:
            logger.warning(f"Error in progressive compression: {e}", exc_info=True)
