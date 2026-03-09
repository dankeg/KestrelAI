"""
LangGraph-powered subtask runner for WebResearchAgent.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import TYPE_CHECKING, Any, TypedDict

from KestrelAI.graphs.persistence import get_langgraph_runtime
from KestrelAI.graphs.schemas import ResearchActionPlan

if TYPE_CHECKING:
    from KestrelAI.agents.base_agent import AgentState
    from KestrelAI.agents.web_research_agent import WebResearchAgent
    from KestrelAI.shared.models import Task
else:  # Runtime aliases so LangGraph can resolve TypedDict annotations
    AgentState = Any
    WebResearchAgent = Any
    Task = Any

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - dependency-gated path
    END = START = StateGraph = None

logger = logging.getLogger(__name__)

QUERY_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "or",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "within",
        "about",
        "over",
        "under",
        "your",
        "task",
        "subtask",
        "phase",
        "step",
        "identified",
        "verify",
        "gathering",
        "detail",
        "details",
        "description",
        "descriptions",
        "selection",
        "opportunity",
        "opportunities",
        "program",
        "programs",
        "gather",
        "information",
        "website",
        "websites",
        "official",
        "source",
        "sources",
        "focus",
        "primary",
        "pages",
        "listed",
        "following",
        "submission",
        "requirements",
        "deadline",
        "deadlines",
        "eligibility",
        "criteria",
        "collect",
        "including",
        "current",
        "currently",
        "research",
        "explicitly",
        "related",
        "identify",
        "list",
        "all",
        "active",
        "key",
        "document",
        "create",
        "table",
        "structured",
        "undergraduate",
        "undergraduates",
        "focusing",
        "focuses",
        "site",
        "sites",
        "validation",
        "initial",
        "broad",
        "broaden",
        "lead",
        "leads",
        "discovery",
        "discover",
        "across",
        "multiple",
        "authoritative",
        "incidental",
        "page",
        "pages",
        "angle",
        "angles",
        "available",
        "publicly",
        "candidate",
        "candidates",
        "promising",
        "strongest",
        "supported",
        "verification",
        "deeper",
        "warrant",
        "noted",
        "note",
        "instead",
        "federal",
        "nonprofit",
        "major",
        "relevant",
        "each",
        "test",
        "offering",
        "offerings",
    }
)

QUERY_FILLER_TOKENS = frozenset(
    {
        "to",
        "in",
        "for",
        "with",
        "from",
        "explicitly",
        "currently",
        "current",
        "related",
        "identified",
        "gather",
        "information",
        "official",
        "source",
        "sources",
        "site",
        "sites",
        "validation",
        "initial",
        "broad",
        "or",
        "offering",
        "offerings",
        "broaden",
        "lead",
        "leads",
        "discovery",
        "discover",
        "across",
        "multiple",
        "authoritative",
        "incidental",
        "page",
        "pages",
        "angle",
        "angles",
        "available",
        "publicly",
        "candidate",
        "candidates",
        "promising",
        "strongest",
        "supported",
        "verification",
        "verify",
        "deeper",
        "warrant",
        "note",
        "noted",
        "instead",
        "relevant",
        "description",
        "descriptions",
        "each",
        "test",
    }
)


class SubtaskGraphState(TypedDict):
    task: Task
    agent_state: AgentState
    loop_index: int
    max_loops: int
    plan: dict[str, Any]
    action: str
    done: bool
    exhausted: bool
    result: str


class LangGraphSubtaskRunner:
    """Compiles and executes subtask research flow using LangGraph."""

    def __init__(self, agent: WebResearchAgent):
        if StateGraph is None:
            raise ImportError("langgraph is not installed")
        self.agent = agent
        self.graph = self._build_graph()

    @staticmethod
    def _extract_keywords(text: str, max_terms: int) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9/+._-]*", (text or "").lower())
        out: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in QUERY_STOPWORDS:
                continue
            if len(token) < 2:
                continue
            if token.isdigit() and len(token) < 4:
                continue
            if token not in seen:
                seen.add(token)
                out.append(token)
            if len(out) >= max_terms:
                break
        return out

    def _build_fallback_search_query(self, task: Task, agent_state: AgentState) -> str:
        """Create a compact, high-signal fallback query when planning times out."""
        return self._build_structured_search_query(task, agent_state)

    @staticmethod
    def _compact_keyword_query(text: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9/+._-]*", (text or "").lower())
        compacted: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in QUERY_FILLER_TOKENS:
                continue
            if token not in seen:
                seen.add(token)
                compacted.append(token)
            if len(compacted) >= 8:
                break
        if not compacted:
            compacted = tokens[:8]
        return " ".join(compacted).strip()[:120] or "research topic"

    def _extract_recent_title_terms(
        self, agent_state: AgentState, max_terms: int = 6
    ) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        search_history = list(getattr(agent_state, "search_history", []) or [])
        for entry in reversed(search_history[-4:]):
            for hit in list(entry.get("results", []) or []):
                if not (
                    bool(hit.get("official_source", False))
                    or int(hit.get("authority_score", 0) or 0) >= 3
                ):
                    continue
                title = str(hit.get("title", "")).strip()
                for token in self._extract_keywords(title, max_terms=max_terms):
                    if token not in seen:
                        seen.add(token)
                        terms.append(token)
                    if len(terms) >= max_terms:
                        return terms
        return terms

    def _extract_recent_query_terms(
        self, agent_state: AgentState, max_terms: int = 6
    ) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        search_history = list(getattr(agent_state, "search_history", []) or [])
        for entry in reversed(search_history[-4:]):
            query = str(entry.get("query", "")).strip()
            for token in self._extract_keywords(query, max_terms=max_terms):
                if token not in seen:
                    seen.add(token)
                    terms.append(token)
                if len(terms) >= max_terms:
                    return terms
        return terms

    @staticmethod
    def _evidence_facets() -> dict[str, tuple[str, ...]]:
        return {
            "deadline": ("deadline", "deadlines", "application", "submission"),
            "eligibility": ("eligibility", "requirements", "gpa", "undergraduate"),
            "funding": ("funding", "stipend", "housing", "duration"),
            "people": ("faculty", "advisor", "mentor", "contact"),
            "focus": ("research", "focus", "project", "projects"),
            "official": ("official", "site"),
        }

    def _choose_missing_facets(self, task: Task, agent_state: AgentState) -> list[str]:
        desc = " ".join(
            [
                str(getattr(self.agent.config, "subtask_description", "") or ""),
                str(getattr(self.agent.config, "success_criteria", "") or ""),
                str(task.description or ""),
            ]
        ).lower()
        existing = " ".join(
            str(item) for item in getattr(agent_state, "queries", set()) or set()
        ).lower()
        selected: list[str] = []
        for facet, triggers in self._evidence_facets().items():
            if not any(trigger in desc for trigger in triggers):
                continue
            if any(trigger in existing for trigger in triggers):
                continue
            if facet == "deadline":
                selected.extend(["deadline", "application"])
            elif facet == "eligibility":
                selected.extend(["eligibility", "requirements"])
            elif facet == "funding":
                selected.extend(["funding", "duration"])
            elif facet == "people":
                selected.extend(["faculty", "contact"])
            elif facet == "focus":
                selected.extend(["research", "projects"])
            elif facet == "official":
                selected.extend(["official", "site"])
            if len(selected) >= 3:
                break
        if not selected:
            selected = ["deadline", "application"]
        return selected[:3]

    def _build_structured_search_query(
        self,
        task: Task,
        agent_state: AgentState,
    ) -> str:
        subtask_description = (
            getattr(self.agent.config, "subtask_description", "") or ""
        ).strip()
        focus = (agent_state.current_focus or "").strip()
        anchor_source = " ".join(
            [
                str(task.name or ""),
                str(task.description or ""),
                subtask_description,
                focus,
            ]
        ).strip()
        anchor_terms = self._extract_keywords(anchor_source, max_terms=6)
        prior_query_terms = self._extract_recent_query_terms(agent_state, max_terms=6)
        entity_terms = self._extract_recent_title_terms(agent_state, max_terms=5)
        if anchor_terms:
            anchored_entity_terms = [
                token
                for token in entity_terms
                if token in set(anchor_terms + prior_query_terms)
            ]
            if anchored_entity_terms:
                entity_terms = anchored_entity_terms
            else:
                entity_terms = []
        if not entity_terms:
            entity_terms = prior_query_terms[:5]
        if not entity_terms:
            primary = (
                focus or subtask_description or (task.description or task.name or "")
            )
            if ":" in primary:
                head = primary.split(":", 1)[0].strip()
                if len(head.split()) >= 2:
                    primary = head
            entity_terms = self._extract_keywords(primary, max_terms=5)
        task_terms = self._extract_keywords(
            (task.description or task.name or ""),
            max_terms=3,
        )
        facet_terms = self._choose_missing_facets(task, agent_state)

        merged: list[str] = []
        seen: set[str] = set()
        for token in anchor_terms[:4] + entity_terms + facet_terms + task_terms:
            if token not in seen:
                seen.add(token)
                merged.append(token)
        if not merged:
            return "research topic"
        return self._compact_keyword_query(" ".join(merged[:12]))

    def _sanitize_search_query(
        self,
        task: Task,
        agent_state: AgentState,
        query: str,
    ) -> str:
        cleaned = re.sub(r"(?i)^\s*(?:search|query)\s*:\s*", "", query or "").strip()
        cleaned = re.sub(r"\s+", " ", cleaned.replace("\n", " ")).strip(" ,;:-")
        if not cleaned:
            return self._build_structured_search_query(task, agent_state)

        lowered = cleaned.lower()
        instruction_prefixes = (
            "for each",
            "collect ",
            "create ",
            "assess ",
            "document ",
            "summarize ",
            "complete ",
            "current focus",
            "success criteria",
            "subtask ",
            "each program",
        )
        instruction_markers = (
            "success criteria",
            "create a table",
            "structured document",
            "including:",
            "document the",
            "collect key",
            "for each identified",
            "gather detailed information",
            "from the nsf website",
            "focus on the primary",
        )
        looks_instructional = (
            lowered.startswith(instruction_prefixes)
            or any(marker in lowered for marker in instruction_markers)
            or cleaned.count(",") >= 2
            or cleaned.count(";") >= 1
            or len(cleaned.split()) > 14
        )
        query_terms = self._extract_keywords(cleaned, max_terms=8)
        filler_tokens = QUERY_FILLER_TOKENS | {
            "undergraduate",
            "undergraduates",
            "focusing",
            "focuses",
            "on",
        }
        if not looks_instructional:
            raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9/+._-]*", lowered)
            trailing_filler = bool(raw_tokens and raw_tokens[-1] in filler_tokens)
            stopword_heavy = len(query_terms) <= max(3, len(raw_tokens) // 2)
            contains_filler = any(token in filler_tokens for token in raw_tokens)
            if not trailing_filler and not stopword_heavy and not contains_filler:
                return cleaned[:140]

        merged: list[str] = []
        seen: set[str] = set()
        for token in query_terms + self._choose_missing_facets(task, agent_state):
            if token not in seen and token not in filler_tokens:
                seen.add(token)
                merged.append(token)
        candidate = self._compact_keyword_query(" ".join(merged[:10]).strip())
        if candidate and len(candidate.split()) >= 3:
            return candidate[:120]
        return self._build_structured_search_query(task, agent_state)

    def _normalize_action_plan(
        self,
        task: Task,
        agent_state: AgentState,
        plan: ResearchActionPlan,
    ) -> ResearchActionPlan:
        if plan.action == "search":
            plan.query = self._sanitize_search_query(task, agent_state, plan.query)

        recent_actions = list(getattr(agent_state, "action_pattern", []) or [])
        recent_summary_count = sum(
            1 for action in recent_actions[-3:] if action in {"summarize", "complete"}
        )
        max_recent_summary_actions = max(
            1,
            int(os.getenv("SUBTASK_MAX_RECENT_SUMMARY_ACTIONS", "1")),
        )
        if (
            plan.action == "summarize"
            and recent_summary_count >= max_recent_summary_actions
        ):
            forced_query = self._build_forced_search_query(task, agent_state)
            logger.info(
                "Replacing repeated summarize action with search for task %s; query=%s",
                task.name,
                forced_query,
            )
            plan = ResearchActionPlan(
                direction=plan.direction,
                action="search",
                query=forced_query,
                thought="",
                tool_name="",
                tool_parameters={},
            )
        if plan.action == "think":
            recent_think_count = sum(
                1 for action in recent_actions[-4:] if action == "think"
            )
            max_recent_thinks = max(
                2,
                int(os.getenv("SUBTASK_MAX_RECENT_THINK_ACTIONS", "2")),
            )
            if recent_think_count >= max_recent_thinks:
                forced_query = self._build_forced_search_query(task, agent_state)
                logger.info(
                    "Replacing repeated think action with search for task %s; query=%s",
                    task.name,
                    forced_query,
                )
                plan = ResearchActionPlan(
                    direction=plan.direction,
                    action="search",
                    query=forced_query,
                    thought="",
                    tool_name="",
                    tool_parameters={},
                )
        return plan

    @staticmethod
    def _truthy_env(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    def _timeouts_disabled(self) -> bool:
        return self._truthy_env("GLOBAL_DISABLE_TIMEOUTS", False)

    def _canonicalize_query(self, query: str) -> str:
        canonicalizer = getattr(self.agent, "_canonicalize_query", None)
        if callable(canonicalizer):
            try:
                value = canonicalizer(query)
                if isinstance(value, str):
                    return value
            except Exception:
                pass
        lowered = (query or "").lower()
        lowered = re.sub(r"[^a-zA-Z0-9\s]", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    def _build_forced_search_query(self, task: Task, agent_state: AgentState) -> str:
        """Build a non-duplicate forced-search query to break thought loops."""
        base_query = self._build_fallback_search_query(task, agent_state).strip()
        existing_queries = set(getattr(agent_state, "queries", set()) or set())
        subtask_description = (
            getattr(self.agent.config, "subtask_description", "") or ""
        ).strip()

        candidates: list[str] = []
        if base_query:
            candidates.extend(
                [
                    base_query,
                    f"{base_query} official source",
                    f"{base_query} eligibility requirements",
                    f"{base_query} deadlines application",
                ]
            )
        if subtask_description:
            subtask_terms = self._extract_keywords(subtask_description, max_terms=10)
            if subtask_terms:
                candidates.append(" ".join(subtask_terms[:10]))
        task_terms = self._extract_keywords(
            (task.description or task.name or ""), max_terms=8
        )
        if task_terms:
            candidates.append(" ".join(task_terms[:8]))

        for candidate in candidates:
            normalized = " ".join(candidate.split()).strip()
            if not normalized:
                continue
            normalized = self._compact_keyword_query(normalized)
            canonical = self._canonicalize_query(normalized)
            if canonical and canonical not in existing_queries:
                return normalized[:120]

        # As a last resort, add a lightweight angle suffix so dedupe does not skip.
        suffix = max(1, len(existing_queries) + 1)
        fallback = (base_query or "research topic").strip()
        return f"{fallback} angle {suffix}"[:120]

    def _build_graph(self):
        workflow = StateGraph(SubtaskGraphState)
        workflow.add_node("decide_action", self._decide_action)
        workflow.add_node("do_think", self._do_think)
        workflow.add_node("do_search", self._do_search)
        workflow.add_node("do_mcp_tool", self._do_mcp_tool)
        workflow.add_node("do_summarize", self._do_summarize)
        workflow.add_node("do_complete", self._do_complete)
        workflow.add_node("post_action", self._post_action)
        workflow.add_node("finalize", self._finalize)

        workflow.add_edge(START, "decide_action")
        workflow.add_conditional_edges(
            "decide_action",
            self._route_action,
            {
                "think": "do_think",
                "search": "do_search",
                "mcp_tool": "do_mcp_tool",
                "summarize": "do_summarize",
                "complete": "do_complete",
            },
        )

        workflow.add_edge("do_think", "post_action")
        workflow.add_edge("do_search", "post_action")
        workflow.add_edge("do_mcp_tool", "post_action")
        workflow.add_edge("do_summarize", "post_action")
        workflow.add_edge("do_complete", "finalize")

        workflow.add_conditional_edges(
            "post_action",
            self._route_after_action,
            {"continue": "decide_action", "finalize": "finalize"},
        )
        workflow.add_edge("finalize", END)
        checkpointer, store = get_langgraph_runtime()
        compile_kwargs = {}
        if checkpointer is not None:
            compile_kwargs["checkpointer"] = checkpointer
        if store is not None:
            compile_kwargs["store"] = store
        return workflow.compile(**compile_kwargs)

    async def run(self, task: Task, agent_state: AgentState) -> str:
        initial_state: SubtaskGraphState = {
            "task": task,
            "agent_state": agent_state,
            "loop_index": 0,
            "max_loops": max(
                1,
                int(getattr(self.agent.config, "max_actions_per_step", 1)),
            ),
            "plan": {},
            "action": "think",
            "done": False,
            "exhausted": False,
            "result": "",
        }
        run_config = {
            "configurable": {
                "thread_id": f"subtask:{self.agent.agent_id}:{task.name}",
                "checkpoint_ns": "subtask_runner",
            }
        }
        output = await self.graph.ainvoke(initial_state, config=run_config)
        return output.get("result", "")

    async def _decide_action(self, state: SubtaskGraphState) -> dict[str, Any]:
        task = state["task"]
        agent_state = state["agent_state"]

        if agent_state.is_in_loop():
            plan = ResearchActionPlan(action="summarize")
        else:
            raw_context_timeout_seconds = float(
                os.getenv("SUBTASK_CONTEXT_BUILD_TIMEOUT_SECONDS", "15")
            )
            if self._timeouts_disabled() or raw_context_timeout_seconds <= 0:
                context_timeout_seconds: float | None = None
            else:
                context_timeout_seconds = max(5.0, raw_context_timeout_seconds)
            try:
                context_builder_call = asyncio.to_thread(
                    self.agent.context_builder.build_context,
                    task,
                    agent_state,
                )
                if context_timeout_seconds is None:
                    context = await context_builder_call
                else:
                    context = await asyncio.wait_for(
                        context_builder_call,
                        timeout=context_timeout_seconds,
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "Context build timed out after %.1fs for task %s; using fallback context",
                    context_timeout_seconds,
                    task.name,
                )
                context = (
                    f"Task: {(task.description or task.name or '').strip()}\n"
                    f"Current focus: {(agent_state.current_focus or '').strip()}"
                )
            except Exception as e:
                logger.warning(
                    "Context build failed for task %s: %s; using fallback context",
                    task.name,
                    e,
                )
                context = (
                    f"Task: {(task.description or task.name or '').strip()}\n"
                    f"Current focus: {(agent_state.current_focus or '').strip()}"
                )
            raw_configured_action_timeout = float(
                os.getenv("SUBTASK_ACTION_PLAN_TIMEOUT_SECONDS", "60")
            )
            if self._timeouts_disabled() or raw_configured_action_timeout <= 0:
                action_timeout_seconds: float | None = None
            else:
                configured_action_timeout = max(
                    1.0,
                    raw_configured_action_timeout,
                )
                worker_step_timeout_seconds = max(
                    10.0,
                    float(os.getenv("WORKER_STEP_TIMEOUT_SECONDS", "60")),
                )
                execution_reserve_seconds = max(
                    5.0,
                    float(os.getenv("SUBTASK_ACTION_EXECUTION_RESERVE_SECONDS", "30")),
                )
                action_timeout_seconds = min(
                    configured_action_timeout,
                    max(5.0, worker_step_timeout_seconds - execution_reserve_seconds),
                )
                min_action_timeout_seconds = max(
                    5.0,
                    float(os.getenv("SUBTASK_ACTION_PLAN_MIN_TIMEOUT_SECONDS", "45")),
                )
                if action_timeout_seconds < min_action_timeout_seconds:
                    adjusted_timeout = max(
                        5.0,
                        min(
                            configured_action_timeout,
                            worker_step_timeout_seconds - 5.0,
                        ),
                    )
                    if adjusted_timeout > action_timeout_seconds:
                        logger.info(
                            "Raising subtask action planning timeout for task %s from %.1fs to %.1fs to avoid premature fallback",
                            task.name,
                            action_timeout_seconds,
                            adjusted_timeout,
                        )
                        action_timeout_seconds = adjusted_timeout
            try:
                planner_call = asyncio.to_thread(self.agent._plan_next_action, context)
                if action_timeout_seconds is None:
                    plan = await planner_call
                else:
                    plan = await asyncio.wait_for(
                        planner_call,
                        timeout=action_timeout_seconds,
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "Subtask action planning timed out after %.1fs for task %s; using fallback search action",
                    action_timeout_seconds,
                    task.name,
                )
                fallback_query = self._build_fallback_search_query(task, agent_state)
                plan = ResearchActionPlan(action="search", query=fallback_query)
            except Exception as e:
                logger.warning(
                    "Subtask action planning failed for task %s: %s; using fallback summarize action",
                    task.name,
                    e,
                )
                plan = ResearchActionPlan(action="summarize")

        plan = self._normalize_action_plan(task, agent_state, plan)

        # Guardrail: avoid thought-only loops and premature completion without evidence.
        current_searches = int(getattr(agent_state, "search_count", 0))
        current_unique_queries = len(getattr(agent_state, "queries", set()) or set())
        projected_consecutive_thinks = int(
            getattr(agent_state, "consecutive_thinks", 0)
        ) + (1 if plan.action == "think" else 0)
        max_consecutive_thinks = max(
            1, int(os.getenv("SUBTASK_FORCE_SEARCH_MAX_CONSECUTIVE_THINKS", "2"))
        )
        min_searches_for_think = max(
            0, int(os.getenv("SUBTASK_FORCE_SEARCH_MIN_SEARCHES", "1"))
        )
        min_queries_for_think = max(
            0, int(os.getenv("SUBTASK_FORCE_SEARCH_MIN_UNIQUE_QUERIES", "1"))
        )
        min_searches_before_close = max(
            0, int(os.getenv("SUBTASK_MIN_SEARCHES_BEFORE_CLOSE", "1"))
        )
        min_authoritative_results_before_close = max(
            1, int(os.getenv("SUBTASK_MIN_AUTHORITATIVE_RESULTS_BEFORE_CLOSE", "2"))
        )
        enforce_close_guard = self._truthy_env(
            "SUBTASK_ENFORCE_CLOSE_EVIDENCE_GUARD",
            True,
        )
        authoritative_results = 0
        for entry in list(getattr(agent_state, "search_history", []) or []):
            for hit in list(entry.get("results", []) or []):
                if int(hit.get("authority_score", 0) or 0) >= 3:
                    authoritative_results += 1

        force_search_reason = ""
        if (
            plan.action == "think"
            and projected_consecutive_thinks >= max_consecutive_thinks
            and (
                current_searches < min_searches_for_think
                or current_unique_queries < min_queries_for_think
            )
        ):
            force_search_reason = (
                "consecutive think actions without enough evidence "
                f"(searches={current_searches}, unique_queries={current_unique_queries})"
            )
        elif (
            enforce_close_guard
            and plan.action in {"summarize", "complete"}
            and (
                current_searches < min_searches_before_close
                or authoritative_results < min_authoritative_results_before_close
            )
        ):
            force_search_reason = (
                "summary/complete requested before minimum search evidence "
                f"(searches={current_searches}/{min_searches_before_close}, "
                f"authoritative_results={authoritative_results}/{min_authoritative_results_before_close})"
            )

        if force_search_reason:
            forced_query = self._build_forced_search_query(task, agent_state)
            logger.info(
                "Forcing search action for task %s due to %s; query=%s",
                task.name,
                force_search_reason,
                forced_query,
            )
            plan = ResearchActionPlan(action="search", query=forced_query)

        action = plan.action
        agent_state.action_count += 1
        agent_state.record_action(action, plan.query)
        return {
            "plan": plan.model_dump(),
            "action": action,
        }

    def _route_action(self, state: SubtaskGraphState) -> str:
        action = state.get("action", "think")
        if action not in {"think", "search", "mcp_tool", "summarize", "complete"}:
            return "think"
        return action

    async def _do_think(self, state: SubtaskGraphState) -> dict[str, Any]:
        await self.agent._handle_think_action(
            state.get("plan", {}), state["agent_state"]
        )
        return {}

    async def _do_search(self, state: SubtaskGraphState) -> dict[str, Any]:
        await self.agent._handle_search_action(
            state.get("plan", {}), state["agent_state"]
        )
        return {}

    async def _do_mcp_tool(self, state: SubtaskGraphState) -> dict[str, Any]:
        await self.agent._handle_mcp_tool_action(
            state.get("plan", {}), state["agent_state"]
        )
        return {}

    async def _do_summarize(self, state: SubtaskGraphState) -> dict[str, Any]:
        await self.agent._handle_summarize_action(state["task"], state["agent_state"])
        return {}

    async def _do_complete(self, state: SubtaskGraphState) -> dict[str, Any]:
        result = await self.agent._handle_complete_action(
            state["task"], state["agent_state"]
        )
        return {"done": True, "result": result}

    async def _post_action(self, state: SubtaskGraphState) -> dict[str, Any]:
        task = state["task"]
        agent_state = state["agent_state"]
        loop_index = state.get("loop_index", 0)
        max_loops = state.get("max_loops", self.agent.config.think_loops)

        if state.get("done"):
            return {}

        if agent_state.action_count % self.agent.config.checkpoint_freq == 0:
            await self.agent._create_checkpoint(task, agent_state)

        loop_index += 1
        exhausted = loop_index >= max_loops
        return {"loop_index": loop_index, "exhausted": exhausted}

    def _route_after_action(self, state: SubtaskGraphState) -> str:
        if state.get("done") or state.get("exhausted"):
            return "finalize"
        return "continue"

    async def _finalize(self, state: SubtaskGraphState) -> dict[str, Any]:
        if state.get("result"):
            return {}

        task = state["task"]
        agent_state = state["agent_state"]
        result = self.agent._build_step_feedback(task, agent_state)
        return {"result": result, "done": bool(state.get("done", False))}
