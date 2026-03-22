"""
LangGraph-powered subtask runner for WebResearchAgent.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import TYPE_CHECKING, Any, TypedDict

from pydantic import BaseModel, Field

from KestrelAI.graphs.persistence import get_langgraph_runtime
from KestrelAI.graphs.schemas import ResearchActionPlan
from KestrelAI.shared.research_utils import (
    build_research_task_profile,
    canonicalize_search_query,
    derive_topic_terms,
    infer_research_task_family,
    normalize_research_text,
    task_targets_discrete_opportunities,
    timeouts_disabled,
)

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
        "a",
        "an",
        "as",
        "at",
        "be",
        "by",
        "in",
        "is",
        "of",
        "on",
        "the",
        "to",
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
        "find",
        "search",
        "locate",
        "distinct",
        "are",
        "those",
        "these",
        "open",
        "pivot",
        "support",
        "supports",
        "supporting",
        "provide",
        "provides",
        "providing",
        "offer",
        "offers",
        "offered",
        "funding",
        "specifically",
        "specific",
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
        "diversified",
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
        "a",
        "an",
        "as",
        "at",
        "be",
        "by",
        "in",
        "is",
        "of",
        "on",
        "to",
        "to",
        "for",
        "with",
        "from",
        "find",
        "search",
        "locate",
        "distinct",
        "are",
        "those",
        "these",
        "open",
        "pivot",
        "support",
        "supports",
        "supporting",
        "provide",
        "provides",
        "providing",
        "offer",
        "offers",
        "offered",
        "funding",
        "specific",
        "specifically",
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
        "query",
        "queries",
        "diversified",
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

GUIDANCE_NOISE_TOKENS = frozenset(
    {
        "continue",
        "pivot",
        "to",
        "include",
        "specifically",
        "try",
        "new",
        "query",
        "queries",
        "gathering",
        "gather",
        "targeted",
        "tied",
        "directly",
        "success",
        "criteria",
        "current",
        "more",
        "additional",
        "distinct",
        "variants",
        "increase",
        "diversity",
        "verify",
        "key",
        "claims",
        "before",
        "advancing",
        "advance",
        "blocked",
        "reason",
        "stronger",
        "required",
        "evidence",
        "carry",
        "forward",
        "concrete",
        "findings",
        "preserve",
        "unresolved",
        "uncertainty",
        "transition",
        "aligned",
        "using",
        "avoid",
        "repeating",
        "recent",
        "different",
        "constraint",
        "constraints",
        "low",
        "yield",
        "query",
        "queries",
        "diversified",
    }
)

GUIDANCE_FORCE_PIVOT_PATTERNS = (
    r"\bpivot\b",
    r"\bdifferent angle\b",
    r"\bdifferent source type\b",
    r"\bdifferent source class\b",
    r"\bbroaden\b",
    r"\bdiversif",
    r"\bavoid repeating recent queries\b",
    r"\blow-yield quer",
    r"\bstagnat",
)

GUIDANCE_PIVOT_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        r"\bofficial (?:program )?pages?\b|\bofficial sources?\b|\bprimary sources?\b",
        ("official", "site"),
    ),
    (
        r"\bprimary organizations?\b|\bhost organizations?\b|\borganizations?\b",
        ("organizations",),
    ),
    (
        r"\bdirectory(?:/listing)? pages?\b|\bdirectory listings?\b|\blisting pages?\b|\bdirectories\b|\blistings\b",
        ("directory", "listing"),
    ),
    (r"\buniversit(?:y|ies)\b", ("university",)),
    (r"\bcolleges?\b", ("college",)),
    (r"\binstitut(?:e|es)\b", ("institute",)),
    (r"\blabs?\b|\blaborator(?:y|ies)\b", ("lab",)),
    (r"\bfaculty\b|\bmentors?\b|\badvisors?\b", ("faculty",)),
    (r"\bcompan(?:y|ies)\b|\bindustry\b|\bcorporate\b", ("company",)),
    (r"\bnonprofit\b|\bngo\b|\bfoundation\b", ("nonprofit",)),
)

MECHANICAL_GUIDANCE_PATTERNS = (
    r"(?i)\brun at least \d+ more targeted searches tied directly to the current success criteria\.?",
    r"(?i)\bcover at least \d+ more discovery pathway or source-class routes before advancing\.?",
    r"(?i)\bincrease search diversity with at least \d+ additional distinct query variants\.?",
    r"(?i)\bfind at least \d+ more authoritative sources\s*\([^)]*\)\.?",
    r"(?i)\bfind at least \d+ more authoritative sources\b[^.]*\.?",
    r"(?i)\bverify key claims directly on an official source before advancing\.?",
    r"(?i)\bcreate a checkpoint summary to lock in evidence before deciding to transition\.?",
    r"(?i)\bcurrent line of inquiry is stagnating; pivot to an uncovered pathway or source class instead of paraphrasing prior queries\.?",
    r"(?i)\bcurrent line of inquiry is stagnating; pivot to a different angle, source type, or constraint\.?",
)

SOURCE_CLASS_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        r"\baggregator(?:s)?\b|\bdirector(?:y|ies)\b|\blisting(?:s)?\b|\bcompile(?:s|d)? lists?\b|\bfilter(?:ing)? options?\b",
        ("opportunities", "list"),
    ),
    (
        r"\bgovernment\b|\bfederal\b|\bagenc(?:y|ies)\b|\bgrant(?:s)?\.gov\b",
        ("government", "site:.gov"),
    ),
    (
        r"\buniversit(?:y|ies)\b|\bcolleges?\b|\bcampuses?\b",
        ("university", "site:.edu"),
    ),
    (
        r"\bnonprofit\b|\bfoundation\b|\bassociation\b|\bsociet(?:y|ies)\b",
        ("nonprofit", "site:.org"),
    ),
    (
        r"\binstitut(?:e|es)\b|\blabs?\b|\blaborator(?:y|ies)\b|\bcent(?:er|ers|re|res)\b",
        ("institute", "site:.edu"),
    ),
)

TASK_SCOPE_PRIORITY_TOKENS = frozenset(
    {
        "research",
        "fellowship",
        "fellowships",
        "grant",
        "grants",
        "scholarship",
        "scholarships",
        "funding",
        "undergraduate",
        "undergraduates",
        "students",
        "student",
        "senior",
        "internship",
        "internships",
        "summer",
    }
)

GENERIC_TASK_SCOPE_TOKENS = frozenset({"test", "task", "description"})

GUIDANCE_SCOPE_EXPANSION_TOKENS = frozenset(
    {
        "research",
        "fellowship",
        "fellowships",
        "grant",
        "grants",
        "scholarship",
        "scholarships",
        "award",
        "awards",
        "program",
        "programs",
        "funding",
        "opportunity",
        "opportunities",
        "undergraduate",
        "undergraduates",
        "student",
        "students",
        "senior",
        "rising",
        "open",
        "current",
        "official",
        "organization",
        "organizations",
    }
)

GENERIC_SOURCE_LABEL_TOKENS = frozenset(
    {
        "government",
        "university",
        "nonprofit",
        "institute",
        "college",
        "organization",
        "organizations",
    }
)

DISCOVERY_PATHWAY_SOURCE_TOKENS = frozenset(
    {
        "official",
        "primary",
        "organization",
        "organizations",
        "government",
        "university",
        "college",
        "nonprofit",
        "institute",
        "lab",
        "directory",
        "listing",
        "list",
        "repository",
        "documentation",
        "project",
        "conference",
        "publisher",
        "preprint",
        "site:.edu",
        "site:.gov",
        "site:.org",
    }
)

OPPORTUNITY_FAMILY_TOKENS = frozenset(
    {
        "fellowship",
        "fellowships",
        "grant",
        "grants",
        "program",
        "programs",
        "scholarship",
        "scholarships",
        "internship",
        "internships",
        "funding",
        "opportunity",
        "opportunities",
    }
)

DISCOVERY_AUDIENCE_TOKENS = frozenset(
    {
        "undergraduate",
        "undergraduates",
        "students",
        "student",
        "senior",
        "rising",
        "us",
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


class DiscoverySearchPathway(BaseModel):
    label: str = ""
    source_terms: list[str] = Field(default_factory=list)
    focus_terms: list[str] = Field(default_factory=list)
    evidence_terms: list[str] = Field(default_factory=list)
    rationale: str = ""


class DiscoverySearchPathwaySet(BaseModel):
    pathways: list[DiscoverySearchPathway] = Field(default_factory=list)


class LangGraphSubtaskRunner:
    """Compiles and executes subtask research flow using LangGraph."""

    def __init__(self, agent: WebResearchAgent):
        if StateGraph is None:
            raise ImportError("langgraph is not installed")
        self.agent = agent
        self.graph = self._build_graph()

    @staticmethod
    def _extract_keywords(text: str, max_terms: int) -> list[str]:
        normalized_text = normalize_research_text(text or "")
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9/+._:-]*", normalized_text.lower())
        out: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            token = token.strip("._-:/")
            if not token:
                continue
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

    def _guidance_text(self) -> str:
        return self._config_text("orchestrator_guidance")

    def _planning_guidance_text(self) -> str:
        guidance_text, _ = self._split_guidance_sections(self._guidance_text())
        cleaned = guidance_text
        for pattern in MECHANICAL_GUIDANCE_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _control_hints(self) -> dict[str, Any]:
        config = getattr(self.agent, "config", None)
        raw_value = getattr(config, "orchestrator_control_hints", {})
        if isinstance(raw_value, dict):
            return dict(raw_value)
        return {}

    def _config_text(self, attribute: str) -> str:
        config = getattr(self.agent, "config", None)
        raw_value = getattr(config, attribute, "")
        if not isinstance(raw_value, str):
            return ""
        return raw_value.strip()

    def _discovery_pathway_context_key(self, task: Task) -> str:
        return " | ".join(
            [
                self._get_current_subtask_mode(task),
                self._task_query_text(task),
                self._config_text("subtask_description"),
                self._config_text("success_criteria"),
                self._planning_guidance_text(),
            ]
        ).strip()

    def _normalize_discovery_pathway_terms(
        self,
        values: list[str] | tuple[str, ...],
        *,
        max_terms: int,
        source_terms: bool = False,
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            tokens = self._extract_keywords(str(value or ""), max_terms=max_terms + 4)
            if source_terms:
                raw_value = str(value or "").strip().lower()
                if (
                    raw_value in DISCOVERY_PATHWAY_SOURCE_TOKENS
                    and raw_value not in tokens
                ):
                    tokens = [raw_value] + tokens
            for token in tokens:
                if source_terms and token not in DISCOVERY_PATHWAY_SOURCE_TOKENS:
                    continue
                if token not in seen:
                    seen.add(token)
                    normalized.append(token)
                if len(normalized) >= max_terms:
                    return normalized
        return normalized[:max_terms]

    @staticmethod
    def _allowed_discovery_source_terms(task_family: str) -> set[str]:
        common = {
            "official",
            "primary",
            "organization",
            "organizations",
            "directory",
            "listing",
            "site:.edu",
            "site:.gov",
            "site:.org",
        }
        if task_family == "opportunity":
            return common | {
                "government",
                "university",
                "college",
                "nonprofit",
                "institute",
                "lab",
            }
        if task_family == "papers":
            return common | {
                "conference",
                "publisher",
                "preprint",
                "lab",
                "project",
            }
        if task_family == "ecosystem":
            return common | {
                "repository",
                "documentation",
                "project",
                "university",
                "lab",
                "nonprofit",
            }
        return common

    def _sanitize_pathway_source_terms(
        self,
        source_terms: list[str],
        *,
        task_family: str,
        max_terms: int = 3,
    ) -> list[str]:
        allowed = self._allowed_discovery_source_terms(task_family)
        selected: list[str] = []
        seen: set[str] = set()
        site_filter_added = False
        for token in source_terms:
            if token not in allowed:
                continue
            if token.startswith("site:."):
                if site_filter_added:
                    continue
                site_filter_added = True
            if token not in seen:
                seen.add(token)
                selected.append(token)
            if len(selected) >= max_terms:
                break
        return selected[:max_terms]

    def _serialize_discovery_pathways(
        self,
        task: Task,
        pathways: list[DiscoverySearchPathway],
        *,
        task_family: str,
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        seen_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for index, pathway in enumerate(pathways, start=1):
            raw_source_terms = self._normalize_discovery_pathway_terms(
                list(pathway.source_terms),
                max_terms=3,
                source_terms=True,
            )
            raw_focus_terms = self._normalize_discovery_pathway_terms(
                list(pathway.focus_terms),
                max_terms=4,
            )
            source_terms = self._normalize_stored_pathway_source_terms(
                raw_source_terms,
                task_family=task_family,
            )
            focus_terms = self._normalize_stored_pathway_focus_terms(
                task,
                source_terms=source_terms,
                focus_terms=raw_focus_terms,
                task_family=task_family,
            )
            evidence_terms = self._normalize_discovery_pathway_terms(
                list(pathway.evidence_terms),
                max_terms=3,
            )
            if not source_terms or not focus_terms:
                continue
            signature = (tuple(source_terms), tuple(focus_terms))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            label = (pathway.label or "").strip()[:80]
            if len(label.split()) < 2:
                label = " ".join((source_terms + focus_terms)[:6]).strip().title()[:80]
            serialized.append(
                {
                    "id": f"pathway_{index}",
                    "label": label,
                    "source_terms": source_terms,
                    "focus_terms": focus_terms,
                    "evidence_terms": evidence_terms,
                    "rationale": (pathway.rationale or "").strip()[:160],
                    "attempt_count": 0,
                    "hit_count": 0,
                }
            )
        return serialized

    @staticmethod
    def _discovery_source_rules(
        task_family: str,
    ) -> tuple[dict[str, int], tuple[str, ...], dict[str, tuple[str, ...]]]:
        if task_family == "opportunity":
            return (
                {
                    "university": 0,
                    "lab": 0,
                    "organization": 1,
                    "nonprofit": 1,
                    "official": 2,
                    "primary": 2,
                    "directory": 3,
                    "listing": 3,
                    "government": 4,
                },
                ("site:.edu", "site:.org", "site:.gov"),
                {
                    "university": ("site:.edu",),
                    "college": ("site:.edu",),
                    "lab": ("site:.edu",),
                    "government": ("site:.gov",),
                    "nonprofit": ("site:.org",),
                    "directory": ("site:.org",),
                    "listing": ("site:.org",),
                    "organization": ("site:.org", "site:.edu"),
                },
            )
        if task_family == "papers":
            return (
                {
                    "conference": 0,
                    "publisher": 0,
                    "preprint": 1,
                    "project": 1,
                    "lab": 2,
                    "official": 3,
                    "organization": 4,
                },
                ("site:.edu", "site:.org"),
                {
                    "lab": ("site:.edu",),
                    "project": ("site:.org", "site:.edu"),
                    "conference": ("site:.org",),
                    "publisher": ("site:.org",),
                },
            )
        if task_family == "ecosystem":
            return (
                {
                    "repository": 0,
                    "documentation": 1,
                    "official": 1,
                    "organization": 2,
                    "project": 2,
                    "directory": 3,
                    "listing": 3,
                    "lab": 4,
                    "university": 4,
                    "nonprofit": 4,
                },
                ("site:.org", "site:.edu"),
                {
                    "organization": ("site:.org",),
                    "project": ("site:.org", "site:.edu"),
                    "repository": ("site:.org",),
                    "documentation": ("site:.org",),
                },
            )
        return (
            {
                "official": 0,
                "organization": 1,
                "directory": 2,
                "listing": 2,
            },
            ("site:.org", "site:.edu", "site:.gov"),
            {},
        )

    def _normalize_stored_pathway_source_terms(
        self,
        source_terms: list[str],
        *,
        task_family: str,
    ) -> list[str]:
        sanitized = self._sanitize_pathway_source_terms(
            source_terms,
            task_family=task_family,
            max_terms=4,
        )
        if not sanitized:
            return []

        (
            priorities,
            family_site_preference,
            aligned_sites,
        ) = self._discovery_source_rules(task_family)
        site_filters = [token for token in sanitized if token.startswith("site:.")]
        labels = [token for token in sanitized if not token.startswith("site:.")]
        if not labels and site_filters:
            fallback_label = next(
                (
                    label
                    for label, aligned in aligned_sites.items()
                    if site_filters[0] in aligned
                ),
                "official",
            )
            labels = [fallback_label]

        label_order = {token: index for index, token in enumerate(labels)}
        candidate_labels = list(labels)
        if site_filters:
            site_aligned_labels = [
                label
                for label in labels
                if any(site in aligned_sites.get(label, ()) for site in site_filters)
            ]
            if site_aligned_labels:
                candidate_labels = site_aligned_labels
        ordered_labels = sorted(
            candidate_labels,
            key=lambda token: (priorities.get(token, 10), label_order.get(token, 99)),
        )
        if not ordered_labels:
            return []

        primary_label = ordered_labels[0]
        normalized: list[str] = []
        if primary_label in {"directory", "listing"}:
            normalized.extend(["directory", "listing"])
        else:
            normalized.append(primary_label)

        aligned_site = next(
            (
                site
                for site in aligned_sites.get(primary_label, ())
                if site in site_filters
            ),
            None,
        )
        if aligned_site is None:
            aligned_site = next(
                (site for site in family_site_preference if site in site_filters),
                None,
            )
        if aligned_site is None and primary_label in aligned_sites:
            aligned_site = next(iter(aligned_sites.get(primary_label, ())), None)
        if aligned_site and aligned_site not in normalized:
            normalized.append(aligned_site)

        if (
            primary_label in {"official", "primary"}
            and "organization" in ordered_labels
            and "organization" not in normalized
        ):
            normalized.append("organization")

        deduped: list[str] = []
        seen: set[str] = set()
        for token in normalized:
            if token and token not in seen:
                seen.add(token)
                deduped.append(token)
        return deduped[:3]

    def _normalize_stored_pathway_focus_terms(
        self,
        task: Task,
        *,
        source_terms: list[str],
        focus_terms: list[str],
        task_family: str,
    ) -> list[str]:
        temporary_pathway = {
            "source_terms": source_terms,
            "focus_terms": focus_terms,
        }
        normalized_focus = self._pathway_query_focus_terms(
            task,
            temporary_pathway,
            task_family=task_family,
            max_terms=3,
        )
        if normalized_focus:
            return normalized_focus

        fallback_terms = self._task_artifact_terms(
            task, max_terms=2
        ) + self._discovery_topic_terms(
            task,
            max_terms=2,
        )
        return self._normalize_discovery_pathway_terms(fallback_terms, max_terms=3)

    def _merge_discovery_pathways(
        self,
        *,
        generated: list[dict[str, Any]],
        fallback: list[dict[str, Any]],
        task_family: str,
        max_pathways: int = 4,
    ) -> list[dict[str, Any]]:
        combined: list[dict[str, Any]] = []
        seen_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        source_roots: set[str] = set()

        def _add(pathway: dict[str, Any]) -> None:
            if len(combined) >= max_pathways:
                return
            source_terms = list(pathway.get("source_terms", []) or [])
            focus_terms = list(pathway.get("focus_terms", []) or [])
            if not source_terms or not focus_terms:
                return
            signature = (tuple(source_terms), tuple(focus_terms))
            if signature in seen_signatures:
                return
            seen_signatures.add(signature)
            combined.append(pathway)
            source_root = next(
                (token for token in source_terms if not token.startswith("site:.")),
                source_terms[0],
            )
            source_roots.add(source_root)

        for pathway in generated:
            _add(pathway)

        min_distinct_roots = (
            2 if task_family in {"opportunity", "papers", "ecosystem"} else 1
        )
        if len(source_roots) < min_distinct_roots or len(combined) < min(
            3, max_pathways
        ):
            for pathway in fallback:
                _add(pathway)

        return combined[:max_pathways]

    def _fallback_discovery_pathways(
        self,
        task: Task,
        agent_state: AgentState,
    ) -> list[dict[str, Any]]:
        task_family = self._task_family(task)
        task_profile = self._discovery_task_profile(task)
        focus_terms = self._discovery_focus_terms(task, agent_state, max_terms=5)
        if not focus_terms:
            focus_terms = list(
                task_profile.target_terms or task_profile.topic_terms[:3]
            )
        evidence_terms = list(task_profile.evidence_terms[:2])
        strategies = self._preferred_discovery_source_strategies(task, agent_state)
        if not strategies:
            strategies = [
                ["official"],
                ["primary", "organization"],
                ["directory", "listing"],
            ]
        pathways: list[dict[str, Any]] = []
        seen_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for index, strategy in enumerate(strategies[:5], start=1):
            pathway_focus = (
                [focus_terms[(index - 1) % len(focus_terms)]] if focus_terms else []
            )
            if ("opportunities" in strategy and "list" in strategy) or (
                "directory" in strategy and "listing" in strategy
            ):
                pathway_focus = pathway_focus + [
                    token
                    for token in ("directory", "listing")
                    if token not in pathway_focus
                ]
            source_terms = self._normalize_discovery_pathway_terms(
                list(strategy),
                max_terms=3,
                source_terms=True,
            )
            focus_tokens = self._normalize_discovery_pathway_terms(
                pathway_focus or list(task_profile.topic_terms[:2]),
                max_terms=4,
            )
            source_terms = self._normalize_stored_pathway_source_terms(
                source_terms,
                task_family=task_family,
            )
            focus_tokens = self._normalize_stored_pathway_focus_terms(
                task,
                source_terms=source_terms,
                focus_terms=focus_tokens,
                task_family=task_family,
            )
            if not source_terms or not focus_tokens:
                continue
            signature = (tuple(source_terms), tuple(focus_tokens))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            pathways.append(
                {
                    "id": f"pathway_{index}",
                    "label": " ".join((source_terms + focus_tokens)[:6])[:80],
                    "source_terms": source_terms,
                    "focus_terms": focus_tokens,
                    "evidence_terms": evidence_terms[:],
                    "rationale": "",
                    "attempt_count": 0,
                    "hit_count": 0,
                }
            )
        return pathways[:5]

    async def _generate_discovery_pathways(
        self,
        task: Task,
        agent_state: AgentState,
    ) -> list[dict[str, Any]]:
        fallback = self._fallback_discovery_pathways(task, agent_state)
        adapter = getattr(self.agent, "langchain_adapter", None)
        if adapter is None or not hasattr(adapter, "chat_structured_async"):
            return fallback

        recent_queries = [
            str(entry.get("query", "")).strip()
            for entry in list(getattr(agent_state, "search_history", []) or [])[-6:]
            if str(entry.get("query", "")).strip()
        ]
        if not recent_queries:
            recent_queries = [
                str(query).strip()
                for query in (getattr(agent_state, "queries", set()) or set())
                if str(query).strip()
            ][:6]
        task_profile = self._discovery_task_profile(task)
        task_family = self._task_family(task)
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate 4 concise, orthogonal web research pathways for the current discovery subtask. "
                    "Differentiate the pathways by source class or discovery angle, not by trivial word reordering. "
                    "Use only short generic source tokens such as official, organization, university, government, "
                    "nonprofit, lab, directory, listing, repository, documentation, project, conference, publisher, preprint, "
                    "site:.edu, site:.gov, site:.org. "
                    "Keep focus_terms short and task-relevant. Avoid inventing named entities unless they already appear in the task. "
                    "Match the pathway source classes and artifact types to the task family."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task family: {task_family}\n"
                    f"Task: {self._task_query_text(task)}\n"
                    f"Subtask: {self._config_text('subtask_description')}\n"
                    f"Success criteria: {self._config_text('success_criteria')}\n"
                    f"Guidance: {self._planning_guidance_text()}\n"
                    f"Recent queries to avoid repeating: {recent_queries}\n"
                    f"Task profile topic terms: {list(task_profile.topic_terms)}\n"
                    f"Task profile target terms: {list(task_profile.target_terms)}\n"
                    f"Task profile source terms: {list(task_profile.source_terms)}\n"
                    f"Task profile evidence terms: {list(task_profile.evidence_terms)}"
                ),
            },
        ]
        try:
            timeout_seconds = max(
                5.0,
                float(os.getenv("SUBTASK_DISCOVERY_PATHWAY_TIMEOUT_SECONDS", "20")),
            )
            generated = await adapter.chat_structured_async(
                messages,
                DiscoverySearchPathwaySet,
                timeout_seconds=timeout_seconds,
                retries=0,
                fallback_factory=lambda _error: DiscoverySearchPathwaySet(pathways=[]),
            )
            serialized = self._serialize_discovery_pathways(
                task,
                list(generated.pathways),
                task_family=task_family,
            )
            return (
                self._merge_discovery_pathways(
                    generated=serialized,
                    fallback=fallback,
                    task_family=task_family,
                )
                or fallback
            )
        except Exception:
            return fallback

    async def _ensure_discovery_pathways(
        self,
        task: Task,
        agent_state: AgentState,
    ) -> None:
        if self._get_current_subtask_mode(task) != "discovery":
            return
        context_key = self._discovery_pathway_context_key(task)
        current_pathways = list(getattr(agent_state, "search_pathways", []) or [])
        refresh_needed = (
            not current_pathways
            or getattr(agent_state, "pathway_context_key", "") != context_key
        )
        if not refresh_needed and current_pathways:
            zero_result_search_count = int(
                getattr(agent_state, "zero_result_search_count", 0) or 0
            )
            exhausted = all(
                int(pathway.get("attempt_count", 0) or 0) >= 1
                for pathway in current_pathways
            )
            refresh_needed = exhausted and zero_result_search_count >= len(
                current_pathways
            )
        if not refresh_needed:
            return
        agent_state.search_pathways = await self._generate_discovery_pathways(
            task, agent_state
        )
        agent_state.pathway_query_map.clear()
        agent_state.pathway_context_key = context_key

    def _discovery_pathway_priority(
        self,
        pathway: dict[str, Any],
        *,
        task_family: str,
    ) -> int:
        source_terms = list(pathway.get("source_terms", []) or [])
        priorities, _, _ = self._discovery_source_rules(task_family)
        return min((priorities.get(token, 10) for token in source_terms), default=10)

    @staticmethod
    def _discovery_pathway_coverage_bucket(pathway: dict[str, Any]) -> int:
        attempt_count = int(pathway.get("attempt_count", 0) or 0)
        hit_count = int(pathway.get("hit_count", 0) or 0)
        if hit_count <= 0 and attempt_count <= 0:
            return 0
        if hit_count <= 0:
            return 1
        return 2

    def _rank_discovery_pathways(
        self,
        task: Task,
        agent_state: AgentState,
    ) -> list[dict[str, Any]]:
        pathways = list(getattr(agent_state, "search_pathways", []) or [])
        task_family = self._task_family(task)
        control_hints = self._control_hints()
        preferred_pathway_ids = [
            str(pathway_id)
            for pathway_id in list(control_hints.get("preferred_pathway_ids", []) or [])
            if str(pathway_id).strip()
        ]
        preferred_order = {
            pathway_id: index for index, pathway_id in enumerate(preferred_pathway_ids)
        }
        return sorted(
            pathways,
            key=lambda pathway: (
                self._discovery_pathway_coverage_bucket(pathway),
                int(pathway.get("attempt_count", 0) or 0),
                preferred_order.get(str(pathway.get("id", "")), len(preferred_order)),
                self._discovery_pathway_priority(
                    pathway,
                    task_family=task_family,
                ),
                str(pathway.get("id", "")),
            ),
        )

    def _pathway_query_source_terms(
        self,
        pathway: dict[str, Any],
        *,
        task_family: str,
        max_terms: int = 2,
    ) -> list[str]:
        source_terms = self._sanitize_pathway_source_terms(
            list(pathway.get("source_terms", []) or []),
            task_family=task_family,
            max_terms=4,
        )
        if not source_terms:
            return []
        (
            family_source_priority,
            family_site_preference,
            aligned_sites,
        ) = self._discovery_source_rules(task_family)
        site_filters = [token for token in source_terms if token.startswith("site:.")]
        labels = [token for token in source_terms if not token.startswith("site:.")]

        ordered_labels = sorted(
            labels,
            key=lambda token: (
                family_source_priority.get(token, 10),
                source_terms.index(token),
            ),
        )

        selected: list[str] = []
        primary_label = ordered_labels[0] if ordered_labels else ""
        if primary_label:
            selected.append(primary_label)

        aligned_site = None
        if primary_label:
            aligned_site = next(
                (
                    site
                    for site in aligned_sites.get(primary_label, ())
                    if site in site_filters
                ),
                None,
            )
        if aligned_site is None:
            aligned_site = next(
                (site for site in family_site_preference if site in site_filters),
                None,
            )
        if (
            aligned_site is None
            and task_family == "opportunity"
            and primary_label in aligned_sites
        ):
            aligned_site = next(iter(aligned_sites.get(primary_label, ())), None)
        if aligned_site and aligned_site not in selected:
            selected.append(aligned_site)

        for label in ordered_labels[1:]:
            if len(selected) >= max_terms:
                break
            if (
                label in {"official", "primary"}
                and primary_label
                and primary_label
                not in {
                    "official",
                    "primary",
                }
            ):
                continue
            if label not in selected:
                selected.append(label)

        if len(selected) < max_terms and not aligned_site:
            fallback_site = next(
                (site for site in family_site_preference if site in site_filters),
                None,
            )
            if fallback_site and fallback_site not in selected:
                selected.append(fallback_site)

        return selected[:max_terms]

    def _pathway_query_focus_terms(
        self,
        task: Task,
        pathway: dict[str, Any],
        *,
        task_family: str,
        max_terms: int = 2,
    ) -> list[str]:
        focus_terms = self._normalize_discovery_pathway_terms(
            list(pathway.get("focus_terms", []) or []),
            max_terms=4,
        )
        artifact_terms = self._task_artifact_terms(task, max_terms=3)
        topic_terms = self._discovery_topic_terms(task, max_terms=4)
        source_hint_terms = set(
            self._sanitize_pathway_source_terms(
                list(pathway.get("source_terms", []) or []),
                task_family=task_family,
                max_terms=4,
            )
        )
        selected: list[str] = []

        def _push(token: str) -> None:
            if token and token not in selected:
                selected.append(token)

        if task_family == "opportunity":
            allowed_focus = {
                "research",
                "internship",
                "fellowship",
                "grant",
                "scholarship",
                "program",
                "funding",
                "directory",
                "listing",
            }
            opportunity_focus_selected = False
            for token in focus_terms:
                if token in allowed_focus:
                    if token in OPPORTUNITY_FAMILY_TOKENS:
                        if opportunity_focus_selected:
                            continue
                        opportunity_focus_selected = True
                    _push(token)
            if (
                source_hint_terms.intersection({"lab", "university", "college"})
                and "research" not in selected
            ):
                _push("research")
            if not opportunity_focus_selected:
                if source_hint_terms.intersection({"lab", "university", "college"}):
                    _push("program")
                    opportunity_focus_selected = True
                else:
                    for token in artifact_terms:
                        if token in OPPORTUNITY_FAMILY_TOKENS:
                            _push(token)
                            opportunity_focus_selected = True
                            break
            if not selected and "research" in topic_terms:
                _push("research")
        elif task_family == "papers":
            allowed_focus = {
                "benchmark",
                "survey",
                "proceedings",
                "paper",
                "evaluation",
                "project",
            }
            for token in focus_terms:
                if token in allowed_focus:
                    _push(token)
            if "evaluation" in topic_terms:
                _push("evaluation")
            for token in artifact_terms:
                _push(token)
        elif task_family == "ecosystem":
            allowed_focus = {
                "framework",
                "repository",
                "documentation",
                "project",
                "organization",
            }
            for token in focus_terms:
                if token in allowed_focus:
                    _push(token)
            for token in artifact_terms:
                _push(token)
        else:
            for token in focus_terms + artifact_terms:
                _push(token)

        return selected[:max_terms]

    def _build_query_from_pathway(
        self,
        task: Task,
        agent_state: AgentState,
        pathway: dict[str, Any],
    ) -> list[str]:
        topic_terms = self._discovery_topic_terms(task, max_terms=3)
        artifact_terms = self._task_artifact_terms(task, max_terms=2)
        constraint_terms = self._task_constraint_terms(task, max_terms=2)
        audience_terms = [
            token
            for token in self._task_scope_terms(task, max_terms=8)
            if token in DISCOVERY_AUDIENCE_TOKENS
            and token not in {"senior", "rising"}
            and token not in set(constraint_terms)
            and not (
                token in {"student", "students"}
                and "undergraduate" in set(constraint_terms)
            )
        ][:1]
        task_family = self._task_family(task)
        source_terms = self._pathway_query_source_terms(
            pathway,
            task_family=task_family,
            max_terms=2,
        )
        focus_terms = self._pathway_query_focus_terms(
            task,
            pathway,
            task_family=task_family,
            max_terms=2,
        )
        evidence_terms = self._normalize_discovery_pathway_terms(
            list(pathway.get("evidence_terms", []) or []),
            max_terms=1,
        )
        secondary_artifact_terms = [
            token for token in artifact_terms if token not in focus_terms
        ][:1]
        preserve_terms = (
            set(source_terms)
            | set(focus_terms)
            | set(topic_terms)
            | set(secondary_artifact_terms)
            | set(constraint_terms)
            | set(audience_terms)
        )
        ordered_variants: list[list[str]] = []
        if task_family == "papers":
            ordered_variants.extend(
                [
                    source_terms + topic_terms + focus_terms,
                    source_terms
                    + topic_terms
                    + secondary_artifact_terms
                    + focus_terms[:1],
                ]
            )
        elif task_family == "ecosystem":
            ordered_variants.extend(
                [
                    source_terms + topic_terms + focus_terms,
                    source_terms
                    + focus_terms
                    + topic_terms[:2]
                    + secondary_artifact_terms,
                ]
            )
        elif task_family == "opportunity":
            opportunity_evidence_terms = (
                evidence_terms[:1]
                if any(token in {"research", "program"} for token in focus_terms)
                and not any(
                    token in {"fellowship", "grant", "scholarship", "internship"}
                    for token in focus_terms
                )
                else []
            )
            ordered_variants.extend(
                [
                    source_terms
                    + topic_terms
                    + focus_terms
                    + opportunity_evidence_terms
                    + audience_terms
                    + constraint_terms,
                    source_terms
                    + topic_terms
                    + secondary_artifact_terms
                    + audience_terms
                    + constraint_terms,
                ]
            )
        else:
            ordered_variants.extend(
                [
                    source_terms + topic_terms + focus_terms + constraint_terms,
                    source_terms
                    + focus_terms
                    + topic_terms[:2]
                    + secondary_artifact_terms,
                ]
            )
        if evidence_terms:
            ordered_variants.append(
                source_terms
                + topic_terms
                + focus_terms[:1]
                + evidence_terms
                + constraint_terms[:1]
            )
        candidate_queries: list[str] = []
        for ordered_terms in ordered_variants:
            candidate = self._compact_keyword_query(
                " ".join(ordered_terms),
                preserve_terms,
                max_terms=10,
            )
            if candidate and candidate not in candidate_queries:
                candidate_queries.append(candidate)
        return candidate_queries

    def _register_pathway_candidate(
        self,
        agent_state: AgentState,
        query: str,
        pathway_id: str,
    ) -> None:
        canonical = self._canonicalize_query(query)
        if canonical:
            agent_state.register_pathway_query(canonical, pathway_id)

    def _discovery_pathway_stats(self, agent_state: AgentState) -> dict[str, int]:
        pathways = list(getattr(agent_state, "search_pathways", []) or [])
        attempted = 0
        productive = 0
        uncovered = 0
        for pathway in pathways:
            attempt_count = int(pathway.get("attempt_count", 0) or 0)
            hit_count = int(pathway.get("hit_count", 0) or 0)
            if attempt_count > 0:
                attempted += 1
            else:
                uncovered += 1
            if hit_count > 0:
                productive += 1
        return {
            "total": len(pathways),
            "attempted": attempted,
            "productive": productive,
            "uncovered": uncovered,
        }

    def _discovery_search_should_follow_pathway(
        self,
        task: Task,
        agent_state: AgentState,
        query: str,
    ) -> str:
        if self._get_current_subtask_mode(task) != "discovery":
            return ""

        pathway_candidates = self._build_discovery_search_candidates(task, agent_state)
        if not pathway_candidates:
            return ""

        ranked_pathways = self._rank_discovery_pathways(task, agent_state)
        if not ranked_pathways:
            return ""

        top_pathway = ranked_pathways[0]
        top_pathway_id = str(top_pathway.get("id", "") or "")
        top_candidate = pathway_candidates[0]
        top_candidate_canonical = self._canonicalize_query(top_candidate)
        current_canonical = self._canonicalize_query(query)
        if not top_candidate_canonical or current_canonical == top_candidate_canonical:
            return ""

        current_pathway_id = agent_state.pathway_for_query(current_canonical)
        if current_pathway_id == top_pathway_id:
            return ""

        top_bucket = self._discovery_pathway_coverage_bucket(top_pathway)
        current_terms = set(self._extract_keywords(query, max_terms=12))
        desired_terms = set(
            self._pathway_query_source_terms(
                top_pathway,
                task_family=self._task_family(task),
                max_terms=2,
            )
            + self._pathway_query_focus_terms(
                task,
                top_pathway,
                task_family=self._task_family(task),
                max_terms=2,
            )
        )
        if not desired_terms:
            return ""

        overlap = len(current_terms.intersection(desired_terms))
        zero_result_search_count = int(
            getattr(agent_state, "zero_result_search_count", 0) or 0
        )

        if top_bucket == 0 and overlap < min(2, len(desired_terms)):
            return top_candidate
        if top_bucket == 1 and zero_result_search_count > 0 and overlap == 0:
            return top_candidate
        return ""

    @staticmethod
    def _clean_task_name_for_search(task_name: str) -> str:
        cleaned = (task_name or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(
            r"(?i)\b(?:re[- ]?run|validation|debug|manual|smoke|dry[- ]?run|benchmark|eval)\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(
            r"(?i)\b(?:gemma[\w.:/-]*|qwen[\w.:/-]*|llama[\w.:/-]*|mistral[\w.:/-]*|gpt[\w.:/-]*)\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"(?i)\brun\b\s*$", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_:")
        return cleaned

    def _task_query_text(self, task: Task) -> str:
        description = str(task.description or "").strip()
        if description:
            return description
        return self._clean_task_name_for_search(str(task.name or ""))

    def _discovery_task_profile(self, task: Task):
        return build_research_task_profile(
            self._task_query_text(task),
            self._config_text("subtask_description"),
            self._config_text("success_criteria"),
            self._planning_guidance_text(),
        )

    def _task_family(self, task: Task) -> str:
        return infer_research_task_family(
            self._task_query_text(task),
            self._config_text("subtask_description"),
            self._config_text("success_criteria"),
            self._planning_guidance_text(),
        )

    def _task_artifact_terms(
        self,
        task: Task,
        *,
        max_terms: int = 3,
    ) -> list[str]:
        profile = self._discovery_task_profile(task)
        ordered: list[str] = []
        for token in profile.target_terms:
            if token not in ordered:
                ordered.append(token)
            if len(ordered) >= max_terms:
                break
        return ordered[:max_terms]

    def _task_discovery_core_terms(
        self,
        task: Task,
        *,
        max_terms: int = 8,
    ) -> list[str]:
        profile = self._discovery_task_profile(task)
        selected: list[str] = []

        def _push(token: str) -> None:
            if token and token not in selected:
                selected.append(token)

        for token in self._task_topic_terms(task, max_terms=min(4, max_terms)):
            _push(token)
        for token in profile.target_terms:
            _push(token)
            if len(selected) >= max_terms:
                return selected[:max_terms]
        for token in self._task_constraint_terms(task, max_terms=3):
            _push(token)
            if len(selected) >= max_terms:
                return selected[:max_terms]
        for token in self._task_scope_terms(task, max_terms=10):
            if token in DISCOVERY_AUDIENCE_TOKENS:
                continue
            _push(token)
            if len(selected) >= max_terms:
                break
        return selected[:max_terms]

    def _task_constraint_terms(self, task: Task, max_terms: int = 3) -> list[str]:
        text = self._task_query_text(task).lower()
        ordered: list[str] = []

        def _push(token: str) -> None:
            if token and token not in ordered:
                ordered.append(token)

        # "senior" is often a retrieval trap that surfaces senior-fellow or
        # graduate-level programs; retain the population constraint via
        # undergraduate/student wording instead.
        if "senior undergraduate" in text:
            _push("undergraduate")
        elif "senior undergraduates" in text:
            _push("undergraduate")
        elif "undergraduate" in text or "undergraduates" in text:
            _push("undergraduate")
        elif "student" in text or "students" in text:
            _push("students")

        if "united states" in text or re.search(r"\bu\.?s\.?\b", text):
            _push("us")

        return ordered[:max_terms]

    def _task_scope_terms(self, task: Task, max_terms: int = 8) -> list[str]:
        scope_terms = [
            token
            for token in self._extract_keywords(
                self._task_query_text(task), max_terms=12
            )
            if token not in GENERIC_TASK_SCOPE_TOKENS
        ]
        prioritized = [
            token for token in scope_terms if token in TASK_SCOPE_PRIORITY_TOKENS
        ]
        ordered = prioritized + [
            token for token in scope_terms if token not in set(prioritized)
        ]
        deduped: list[str] = []
        seen: set[str] = set()
        for token in ordered:
            if token not in seen:
                seen.add(token)
                deduped.append(token)
            if len(deduped) >= max_terms:
                break
        return deduped[:max_terms]

    def _task_topic_terms(self, task: Task, max_terms: int = 4) -> list[str]:
        profile = self._discovery_task_profile(task)
        topic_terms = list(profile.topic_terms)
        if topic_terms:
            return topic_terms[:max_terms]
        return derive_topic_terms(
            self._task_query_text(task),
            max_terms=max_terms,
            stopwords=QUERY_STOPWORDS,
            extra_exclude={"current", "currently", "open"},
        )

    def _opportunity_query_core_terms(
        self,
        task: Task,
        max_terms: int = 8,
    ) -> list[str]:
        task_text = self._task_query_text(task)
        text = task_text.lower()
        profile = self._discovery_task_profile(task)
        selected: list[str] = []

        def _push(token: str) -> None:
            if token and token not in selected:
                selected.append(token)

        for token in self._task_topic_terms(task, max_terms=min(4, max_terms)):
            _push(token)

        for token in profile.target_terms:
            if len(selected) >= max_terms:
                break
            _push(token)

        for token in profile.audience_terms:
            if token == "undergraduate":
                _push("undergraduate")
            elif token == "student":
                _push("students")

        if len(selected) < max_terms and "program" in text:
            _push("program")
        if len(selected) < max_terms and "research" in text:
            _push("research")

        for token in self._task_scope_terms(task, max_terms=8):
            _push(token)
            if len(selected) >= max_terms:
                break
        return selected[:max_terms]

    def _task_targets_discrete_opportunities(self, task: Task) -> bool:
        return task_targets_discrete_opportunities(task)

    def _task_opportunity_family_terms(
        self,
        task: Task,
        *,
        max_terms: int = 3,
    ) -> list[str]:
        if not self._task_targets_discrete_opportunities(task):
            return []
        ordered_terms: list[str] = []
        for token in self._opportunity_query_core_terms(task, max_terms=8):
            if token not in OPPORTUNITY_FAMILY_TOKENS:
                continue
            if token not in ordered_terms:
                ordered_terms.append(token)
            if len(ordered_terms) >= max_terms:
                break
        return ordered_terms[:max_terms]

    def _discovery_expansion_terms(
        self,
        task: Task,
        agent_state: AgentState,
        *,
        max_terms: int = 4,
    ) -> list[str]:
        task_text = self._task_query_text(task)
        task_profile = self._discovery_task_profile(task)
        task_family = self._task_family(task)
        task_text_lower = task_text.lower()
        subtask_text = " ".join(
            [
                self._config_text("subtask_description"),
                self._config_text("success_criteria"),
            ]
        ).lower()
        existing_text = " ".join(
            str(item) for item in getattr(agent_state, "queries", set()) or set()
        ).lower()
        candidates: list[str] = []

        def _push(token: str) -> None:
            if token and token not in candidates:
                candidates.append(token)

        if re.search(
            r"\baggregator(?:s)?\b|\bdirector(?:y|ies)\b|\blisting(?:s)?\b|\bfilter(?:ing)? capabilities?\b",
            subtask_text,
        ):
            _push("directory")
            _push("listing")
            return candidates[:max_terms]

        for token in task_profile.target_terms:
            if token not in existing_text:
                _push(token)
        if (
            task_family == "opportunity"
            and "funding" in task_text_lower
            and "funding" not in existing_text
        ):
            _push("funding")
        if task_family == "opportunity" and any(
            token in existing_text for token in ("fellowship", "grant", "program")
        ):
            for token in ("research", "internship", "summer"):
                if token not in existing_text:
                    _push(token)
        elif task_family == "papers":
            for token in ("benchmark", "survey", "proceedings", "paper", "evaluation"):
                if token not in existing_text:
                    _push(token)
        elif task_family == "ecosystem":
            for token in ("framework", "repository", "documentation", "project"):
                if token not in existing_text:
                    _push(token)
        if not candidates:
            for token in self._task_topic_terms(task, max_terms=max_terms + 2):
                if token not in existing_text:
                    _push(token)
                if len(candidates) >= max_terms:
                    break
            for token in self._task_discovery_core_terms(task, max_terms=max_terms + 2):
                if token not in DISCOVERY_AUDIENCE_TOKENS:
                    _push(token)
                if len(candidates) >= max_terms:
                    break
        return candidates[:max_terms]

    def _discovery_topic_terms(self, task: Task, max_terms: int = 4) -> list[str]:
        topic_terms = self._task_topic_terms(task, max_terms=max_terms)
        if topic_terms:
            return topic_terms[:max_terms]
        selected: list[str] = []
        for token in self._task_scope_terms(task, max_terms=8):
            if token in OPPORTUNITY_FAMILY_TOKENS:
                continue
            if token in DISCOVERY_AUDIENCE_TOKENS:
                continue
            if token in {"current", "currently", "open"}:
                continue
            if token not in selected:
                selected.append(token)
            if len(selected) >= max_terms:
                break
        return selected[:max_terms]

    def _discovery_focus_terms(
        self,
        task: Task,
        agent_state: AgentState,
        *,
        max_terms: int = 4,
    ) -> list[str]:
        text = self._task_query_text(task).lower()
        task_family = self._task_family(task)
        existing_text = " ".join(
            str(item) for item in (getattr(agent_state, "queries", set()) or set())
        ).lower()
        selected: list[str] = []

        def _push(token: str) -> None:
            if token and token not in selected:
                selected.append(token)

        low_priority_patterns: tuple[tuple[str, str], ...] = ()

        primary_patterns: tuple[tuple[str, str], ...]
        fallback_tokens: tuple[str, ...]
        skip_expansion_tokens: set[str]
        if task_family == "opportunity":
            primary_patterns = (
                (r"\bfellowships?\b", "fellowship"),
                (r"\binternships?\b", "internship"),
                (r"\bscholarships?\b", "scholarship"),
                (r"\bgrants?\b", "grant"),
            )
            low_priority_patterns = ((r"\bprograms?\b", "program"),)
            fallback_tokens = (
                "fellowship",
                "internship",
                "grant",
                "scholarship",
                "program",
            )
            skip_expansion_tokens = {"research", "summer"}
        elif task_family == "papers":
            primary_patterns = (
                (r"\bbenchmarks?\b", "benchmark"),
                (r"\bsurveys?\b", "survey"),
                (r"\bproceedings?\b", "proceedings"),
                (r"\bpapers?\b", "paper"),
            )
            low_priority_patterns = ((r"\breviews?\b", "survey"),)
            fallback_tokens = ("benchmark", "survey", "proceedings", "paper")
            skip_expansion_tokens = set()
        elif task_family == "ecosystem":
            primary_patterns = (
                (r"\bframeworks?\b", "framework"),
                (r"\brepositor(?:y|ies)\b|\brepos?\b", "repository"),
                (r"\bdocs?\b|\bdocumentation\b", "documentation"),
                (r"\bprojects?\b", "project"),
            )
            low_priority_patterns = ((r"\borganizations?\b", "organization"),)
            fallback_tokens = ("framework", "repository", "documentation", "project")
            skip_expansion_tokens = set()
        else:
            primary_patterns = ()
            fallback_tokens = ("official", "organization", "directory", "listing")
            skip_expansion_tokens = set()

        for pattern, token in primary_patterns:
            if re.search(pattern, text) and token not in existing_text:
                _push(token)

        if (
            task_family == "opportunity"
            and "funding" in text
            and "funding" not in existing_text
        ):
            _push("funding")

        for token in self._discovery_expansion_terms(
            task, agent_state, max_terms=max_terms + 2
        ):
            if token in skip_expansion_tokens:
                continue
            if token in existing_text:
                continue
            _push(token)
            if len(selected) >= max_terms:
                break

        if len(selected) < max_terms:
            for pattern, token in low_priority_patterns:
                if re.search(pattern, text) and token not in existing_text:
                    _push(token)
                if len(selected) >= max_terms:
                    break

        if not selected:
            for token in fallback_tokens:
                if token not in existing_text:
                    _push(token)
                if len(selected) >= max_terms:
                    break

        return selected[:max_terms]

    def _preferred_discovery_source_strategies(
        self,
        task: Task,
        agent_state: AgentState,
    ) -> list[list[str]]:
        strategies: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        search_count = int(getattr(agent_state, "search_count", 0) or 0)
        zero_result_search_count = int(
            getattr(agent_state, "zero_result_search_count", 0) or 0
        )
        source_variants = self._subtask_source_variants(task, agent_state)
        subtask_text = " ".join(
            [
                self._config_text("subtask_description"),
                self._config_text("success_criteria"),
            ]
        ).lower()
        task_family = self._task_family(task)
        discrete_opportunity_task = self._task_targets_discrete_opportunities(task)

        def _add(tokens: list[str]) -> None:
            normalized = [token for token in tokens if token]
            key = tuple(normalized)
            if normalized and key not in seen:
                seen.add(key)
                strategies.append(normalized)

        if re.search(
            r"\baggregator(?:s)?\b|\bdirector(?:y|ies)\b|\blisting(?:s)?\b|\bfilter(?:ing)? capabilities?\b",
            subtask_text,
        ):
            _add(["directory", "listing"])

        if task_family == "opportunity":
            if search_count == 0 and zero_result_search_count == 0:
                _add(["primary", "organization"])
                _add(["university", "site:.edu"])
                _add(["official"])
                if discrete_opportunity_task:
                    _add(["directory", "listing"])
            else:
                _add(["primary", "organization"])
                _add(["university", "site:.edu"])
                if discrete_opportunity_task:
                    _add(["directory", "listing"])
                _add(["official"])
        elif task_family == "papers":
            _add(["conference"])
            _add(["publisher"])
            _add(["preprint"])
            _add(["lab", "site:.edu"])
            _add(["project"])
            _add(["official"])
        elif task_family == "ecosystem":
            _add(["repository"])
            _add(["documentation"])
            _add(["organization"])
            _add(["official"])
            _add(["directory", "listing"])
        else:
            _add(["official"])
            _add(["organization"])
            _add(["directory", "listing"])

        if (
            search_count > 0
            or zero_result_search_count > 0
            or any(
                token in self._planning_guidance_text().lower()
                for token in ("directories", "directory", "listing", "listings")
            )
        ):
            _add(["directory", "listing"])

        preferred_site_variants = (
            ("site:.edu",),
            ("site:.org",),
            ("site:.gov",),
        )
        for prefix in preferred_site_variants:
            for variant in source_variants:
                if variant and variant[0] == prefix[0]:
                    _add(prefix)

        if not strategies:
            _add(["official"])

        return strategies

    def _build_discovery_search_candidates(
        self,
        task: Task,
        agent_state: AgentState,
    ) -> list[str]:
        pathway_candidate_groups: list[list[str]] = []
        seen_pathway_candidates: set[str] = set()
        existing_queries = {
            self._canonicalize_query(str(query))
            for query in (getattr(agent_state, "queries", set()) or set())
            if str(query).strip()
        }
        for pathway in self._rank_discovery_pathways(task, agent_state):
            pathway_id = str(pathway.get("id", "") or "")
            pathway_group: list[str] = []
            for candidate in self._build_query_from_pathway(task, agent_state, pathway):
                canonical = self._canonicalize_query(candidate)
                if (
                    not canonical
                    or canonical in existing_queries
                    or canonical in seen_pathway_candidates
                ):
                    continue
                seen_pathway_candidates.add(canonical)
                self._register_pathway_candidate(agent_state, candidate, pathway_id)
                pathway_group.append(candidate)
            if pathway_group:
                pathway_candidate_groups.append(pathway_group)

        if pathway_candidate_groups:
            interleaved_candidates: list[str] = []
            max_group_length = max(len(group) for group in pathway_candidate_groups)
            for index in range(max_group_length):
                for group in pathway_candidate_groups:
                    if index < len(group):
                        interleaved_candidates.append(group[index])
            return interleaved_candidates

        fallback_candidates: list[str] = []
        seen_fallback: set[str] = set()
        for pathway in self._fallback_discovery_pathways(task, agent_state):
            for candidate in self._build_query_from_pathway(task, agent_state, pathway):
                canonical = self._canonicalize_query(candidate)
                if (
                    not canonical
                    or canonical in existing_queries
                    or canonical in seen_fallback
                ):
                    continue
                seen_fallback.add(canonical)
                fallback_candidates.append(candidate)
                if len(fallback_candidates) >= 4:
                    return fallback_candidates
        return fallback_candidates

    def _subtask_source_variants(
        self,
        task: Task,
        agent_state: AgentState,
    ) -> list[list[str]]:
        guidance_text = self._planning_guidance_text()
        text = " ".join(
            [
                self._config_text("subtask_description"),
                self._config_text("success_criteria"),
                guidance_text,
            ]
        ).lower()
        if not text:
            return []

        variants: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        task_scope = self._task_discovery_core_terms(task, max_terms=7)
        core_terms = task_scope[:6]

        if re.search(
            r"\bofficial (?:program )?pages?\b|\bofficial sources?\b|\bprimary organizations?\b|\bprimary organization sources?\b",
            text,
            flags=re.IGNORECASE,
        ):
            for broad_variant in (
                ["official"] + core_terms,
                ["primary", "organization"] + core_terms,
            ):
                deduped: list[str] = []
                local_seen: set[str] = set()
                for token in broad_variant:
                    if token and token not in local_seen:
                        local_seen.add(token)
                        deduped.append(token)
                key = tuple(deduped)
                if deduped and key not in seen:
                    seen.add(key)
                    variants.append(deduped)

        for pattern, variant_terms in SOURCE_CLASS_HINTS:
            if not re.search(pattern, text, flags=re.IGNORECASE):
                continue
            has_site_filter = any(token.startswith("site:.") for token in variant_terms)
            candidate = [
                token
                for token in variant_terms
                if not (has_site_filter and token in GENERIC_SOURCE_LABEL_TOKENS)
            ] + core_terms
            deduped: list[str] = []
            local_seen: set[str] = set()
            for token in candidate:
                if token and token not in local_seen:
                    local_seen.add(token)
                    deduped.append(token)
            key = tuple(deduped)
            if deduped and key not in seen:
                seen.add(key)
                variants.append(deduped)

        if re.search(
            r"\.gov\b|\.edu\b|\.org\b|\bofficial (?:program )?pages?\b|\bprimary organizations?\b|\bprimary organization sources?\b",
            text,
            flags=re.IGNORECASE,
        ):
            for domain_filter in ("site:.gov", "site:.edu", "site:.org"):
                domain_terms = list(core_terms)
                if domain_filter == "site:.org":
                    domain_terms = [
                        token for token in domain_terms if token != "program"
                    ] or domain_terms
                candidate = [domain_filter] + domain_terms
                deduped: list[str] = []
                local_seen: set[str] = set()
                for token in candidate:
                    if token and token not in local_seen:
                        local_seen.add(token)
                        deduped.append(token)
                key = tuple(deduped)
                if deduped and key not in seen:
                    seen.add(key)
                    variants.append(deduped)
        return variants

    def _query_matches_task_scope(
        self,
        task: Task,
        query: str,
    ) -> bool:
        query_terms = set(self._extract_keywords(query, max_terms=12))
        if not query_terms:
            return False

        task_scope_terms = set(self._task_scope_terms(task, max_terms=8))
        task_topic_terms = set(self._task_topic_terms(task, max_terms=6))
        opportunity_terms = set(self._task_opportunity_family_terms(task, max_terms=4))
        constraint_terms = set(self._task_constraint_terms(task, max_terms=3))
        if len(task_scope_terms) < 2 and not constraint_terms:
            return len(query_terms) >= 2
        scope_overlap = len(query_terms.intersection(task_scope_terms))
        topic_overlap = len(query_terms.intersection(task_topic_terms))
        opportunity_overlap = len(query_terms.intersection(opportunity_terms))
        constraint_overlap = len(query_terms.intersection(constraint_terms))

        if scope_overlap >= 3:
            return True
        if scope_overlap >= 2 and constraint_overlap >= 1:
            return True
        if topic_overlap >= 1 and (constraint_overlap >= 1 or opportunity_overlap >= 1):
            return True
        if opportunity_overlap >= 1 and constraint_overlap >= 1:
            return True
        if constraint_overlap >= 1 and bool(
            query_terms.intersection(
                task_scope_terms | task_topic_terms | opportunity_terms
            )
        ):
            return True
        return False

    def _anchor_terms(
        self,
        task: Task,
        agent_state: AgentState,
        max_terms: int = 8,
    ) -> list[str]:
        subtask_description = self._config_text("subtask_description")
        focus = (agent_state.current_focus or "").strip()
        task_query_text = self._task_query_text(task)
        anchor_source = " ".join(
            [
                task_query_text,
                subtask_description,
                focus,
            ]
        ).strip()
        return self._extract_keywords(anchor_source, max_terms=max_terms)

    @staticmethod
    def _split_guidance_sections(guidance: str) -> tuple[str, list[str]]:
        normalized = " ".join((guidance or "").split()).strip()
        if not normalized:
            return "", []
        recent_queries: list[str] = []
        match = re.search(
            r"(?i)avoid repeating recent queries:\s*(.+?)(?=(?:transition blocked reason:|$))",
            normalized,
        )
        if match:
            raw_recent = match.group(1).strip()
            recent_queries = [
                chunk.strip(" ;,")
                for chunk in raw_recent.split(";")
                if chunk.strip(" ;,")
            ]
            normalized = (
                normalized[: match.start()] + " " + normalized[match.end() :]
            ).strip()
        normalized = re.sub(
            r"(?i)transition blocked reason:\s*.+$",
            "",
            normalized,
        ).strip()
        return normalized, recent_queries

    def _extract_repeated_recent_terms(
        self,
        agent_state: AgentState,
        guidance_recent_queries: list[str] | None = None,
        *,
        min_occurrences: int = 2,
        max_terms: int = 8,
    ) -> list[str]:
        queries = list(guidance_recent_queries or [])
        if not queries:
            search_history = list(getattr(agent_state, "search_history", []) or [])
            queries = [
                str(entry.get("query", "")).strip() for entry in search_history[-4:]
            ]
        counts: dict[str, int] = {}
        order: list[str] = []
        for query in queries:
            seen_in_query: set[str] = set()
            for token in self._extract_keywords(query, max_terms=max_terms):
                if token in seen_in_query:
                    continue
                seen_in_query.add(token)
                counts[token] = counts.get(token, 0) + 1
                if token not in order:
                    order.append(token)
        repeated = [token for token in order if counts.get(token, 0) >= min_occurrences]
        return repeated[:max_terms]

    def _derive_guidance_constraints(
        self,
        task: Task,
        agent_state: AgentState,
    ) -> dict[str, object]:
        guidance = self._planning_guidance_text()
        if not guidance:
            return {
                "force_pivot": False,
                "pivot_terms": [],
                "guidance_terms": [],
                "avoid_terms": set(),
            }

        stripped_guidance, recent_queries = self._split_guidance_sections(guidance)
        force_pivot = any(
            re.search(pattern, stripped_guidance, flags=re.IGNORECASE)
            for pattern in GUIDANCE_FORCE_PIVOT_PATTERNS
        )
        anchor_terms = set(self._anchor_terms(task, agent_state, max_terms=8))
        repeated_recent_terms = set(
            self._extract_repeated_recent_terms(
                agent_state,
                recent_queries,
                min_occurrences=2,
                max_terms=10,
            )
        )
        avoid_terms = repeated_recent_terms - anchor_terms

        pivot_terms: list[str] = []
        seen_pivot_terms: set[str] = set()
        for pattern, mapped_terms in GUIDANCE_PIVOT_HINTS:
            if re.search(pattern, stripped_guidance, flags=re.IGNORECASE):
                for token in mapped_terms:
                    if token not in seen_pivot_terms:
                        seen_pivot_terms.add(token)
                        pivot_terms.append(token)

        guidance_terms: list[str] = []
        seen_guidance_terms: set[str] = set(seen_pivot_terms)
        allowed_force_pivot_terms = (
            TASK_SCOPE_PRIORITY_TOKENS | GUIDANCE_SCOPE_EXPANSION_TOKENS | anchor_terms
        )
        for token in self._extract_keywords(stripped_guidance, max_terms=14):
            token_parts = [part for part in re.split(r"[-_/]", token) if part]
            if token in GUIDANCE_NOISE_TOKENS or any(
                part in GUIDANCE_NOISE_TOKENS for part in token_parts
            ):
                continue
            if token in avoid_terms and token not in anchor_terms:
                continue
            if force_pivot and token not in allowed_force_pivot_terms:
                continue
            if token not in seen_guidance_terms:
                seen_guidance_terms.add(token)
                guidance_terms.append(token)

        return {
            "force_pivot": force_pivot,
            "pivot_terms": pivot_terms[:6],
            "guidance_terms": guidance_terms[:6],
            "avoid_terms": avoid_terms,
        }

    def _get_current_subtask_mode(self, task: Task) -> str:
        fallback_mode = self._infer_subtask_mode_from_config()
        orchestrator = getattr(self.agent, "orchestrator", None)
        task_states = getattr(orchestrator, "task_states", {}) if orchestrator else {}
        task_state = task_states.get(task.name)
        if not task_state or not getattr(task_state, "research_plan", None):
            return fallback_mode
        subtasks = getattr(task_state.research_plan, "subtasks", []) or []
        raw_subtask_index = getattr(task_state, "subtask_index", 0)
        if not isinstance(raw_subtask_index, (int, float, str)):
            return fallback_mode
        try:
            subtask_index = int(raw_subtask_index or 0)
        except (TypeError, ValueError):
            return fallback_mode
        if 0 <= subtask_index < len(subtasks):
            raw_subtask_type = getattr(
                subtasks[subtask_index], "subtask_type", "general"
            )
            if hasattr(raw_subtask_type, "value"):
                raw_subtask_type = raw_subtask_type.value
            return str(raw_subtask_type or "general").strip().lower()
        return fallback_mode

    def _infer_subtask_mode_from_config(self) -> str:
        text = " ".join(
            [
                self._config_text("subtask_description"),
                self._config_text("success_criteria"),
            ]
        ).lower()
        if not text:
            return "general"
        if re.search(
            r"^\s*synthesis:|\bcompile\b|\bconsolidated list\b|\bcurated list\b|\bfinal\b.*\breport\b",
            text,
        ):
            return "synthesis"
        if re.search(
            r"^\s*comparison:|\bcompare\b|\bcomparison\b|\bstructured comparison table\b|\bcompare and contrast\b",
            text,
        ):
            return "comparison"
        if re.search(
            r"^\s*(?:authoritative\s+)?verification:|\btrustworthiness\b|\beditorial oversight\b|\baccuracy and currency\b",
            text,
        ):
            return "verification"
        if re.search(
            r"^\s*discovery:",
            text,
        ):
            return "discovery"
        if (
            self._task_targets_discrete_opportunities(
                type(
                    "_TaskLike",
                    (),
                    {
                        "name": "",
                        "description": text,
                    },
                )()
            )
            and any(
                token in text for token in ("find", "identify", "discover", "locate")
            )
            and not any(
                token in text
                for token in ("deadline", "eligibility", "requirements", "status")
            )
        ):
            return "discovery"
        return "general"

    @staticmethod
    def _count_authoritative_results(agent_state: AgentState) -> int:
        authoritative_results = 0
        for entry in list(getattr(agent_state, "search_history", []) or []):
            for hit in list(entry.get("results", []) or []):
                if int(hit.get("authority_score", 0) or 0) >= 3:
                    authoritative_results += 1
        return authoritative_results

    def _synthesis_ready_for_write(self, agent_state: AgentState) -> bool:
        min_searches = max(2, int(os.getenv("SUBTASK_SYNTHESIS_MIN_SEARCHES", "3")))
        min_checkpoints = max(
            1, int(os.getenv("SUBTASK_SYNTHESIS_MIN_CHECKPOINTS", "1"))
        )
        min_authoritative = max(
            2, int(os.getenv("SUBTASK_SYNTHESIS_MIN_AUTHORITATIVE_RESULTS", "6"))
        )
        search_count = int(getattr(agent_state, "search_count", 0) or 0)
        checkpoint_count = int(getattr(agent_state, "checkpoint_count", 0) or 0)
        authoritative_results = self._count_authoritative_results(agent_state)
        return (
            search_count >= min_searches
            and checkpoint_count >= min_checkpoints
            and authoritative_results >= min_authoritative
        )

    def _build_synthesis_gap_query(self, task: Task, agent_state: AgentState) -> str:
        focus_terms = self._extract_keywords(task.description, 4)
        detail_terms = self._verification_detail_terms(
            task, agent_state
        ) or self._choose_missing_facets(task, agent_state)
        query_terms = (
            (["official"] if self._task_targets_discrete_opportunities(task) else [])
            + focus_terms
            + detail_terms[:2]
        )
        candidate = self._compact_keyword_query(" ".join(query_terms))
        return candidate or self._build_structured_search_query(task, agent_state)

    def _verification_detail_terms(
        self, task: Task, agent_state: AgentState
    ) -> list[str]:
        desc = " ".join(
            [
                self._config_text("subtask_description"),
                self._config_text("success_criteria"),
                str(task.description or ""),
            ]
        )
        profile = build_research_task_profile(desc)
        terms: list[str] = []
        if {"deadline", "application"}.intersection(profile.evidence_terms):
            terms.extend(["deadline", "application"])
        if "eligibility" in profile.evidence_terms:
            terms.extend(["eligibility", "requirements"])
        if "status" in profile.evidence_terms:
            terms.extend(["current", "status"])
        if "funding" in profile.evidence_terms and "funding" not in terms:
            terms.extend(["funding", "stipend"])
        if "contact" in profile.evidence_terms and "contact" not in terms:
            terms.extend(["contact", "faculty"])

        if not terms:
            terms = self._choose_missing_facets(task, agent_state)

        opportunity_family_terms = self._task_opportunity_family_terms(
            task, max_terms=1
        )
        if opportunity_family_terms and not any(
            token in OPPORTUNITY_FAMILY_TOKENS for token in terms
        ):
            terms = opportunity_family_terms + terms

        deduped: list[str] = []
        seen: set[str] = set()
        for token in terms:
            if token not in seen:
                seen.add(token)
                deduped.append(token)
        return deduped[:4]

    def _extract_recent_verification_focus_terms(
        self, agent_state: AgentState, max_terms: int = 6
    ) -> list[str]:
        search_history = list(getattr(agent_state, "search_history", []) or [])
        for entry in reversed(search_history[-6:]):
            for hit in list(entry.get("results", []) or []):
                if not (
                    bool(hit.get("official_source", False))
                    or int(hit.get("authority_score", 0) or 0) >= 3
                ):
                    continue
                title = str(hit.get("title", "")).strip()
                if not title:
                    continue
                title_parts = [
                    part.strip()
                    for part in re.split(r"\s+[|\-:]\s+|\|", title)
                    if part.strip()
                ]
                preferred_part = title
                for part in title_parts:
                    if len(part.split()) >= 3:
                        preferred_part = part
                        break
                terms = self._extract_keywords(preferred_part, max_terms=max_terms)
                if terms:
                    return terms[:max_terms]
        previous_findings = self._config_text("previous_findings")
        if not previous_findings:
            return []

        candidate_lines: list[str] = []
        for raw_line in previous_findings.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if "Official lead:" in line:
                extracted = line.split("Official lead:", 1)[1].strip()
                extracted = extracted.split("(", 1)[0].strip(" -:\u2026")
                if extracted:
                    candidate_lines.append(extracted)
                    continue
            if line.startswith("- "):
                stripped = line[2:].strip()
                stripped = re.sub(r"\s+https?://\S+$", "", stripped).strip()
                stripped = stripped.split(" - Evidence:", 1)[0].strip()
                if len(stripped.split()) >= 3:
                    candidate_lines.append(stripped)

        for candidate in candidate_lines:
            terms = self._extract_keywords(candidate, max_terms=max_terms)
            if terms:
                return terms[:max_terms]
        return []

    @staticmethod
    def _compact_keyword_query(
        text: str,
        preserve_tokens: set[str] | None = None,
        *,
        max_terms: int = 10,
    ) -> str:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9/+._:-]*", (text or "").lower())
        preserved = set(preserve_tokens or set())
        ordered_tokens: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            token = token.strip("._-:/")
            if not token:
                continue
            if token in QUERY_FILLER_TOKENS and token not in preserved:
                continue
            if token not in seen:
                seen.add(token)
                ordered_tokens.append(token)
        if not ordered_tokens:
            ordered_tokens = [
                token.strip("._-:/")
                for token in tokens[:max_terms]
                if token.strip("._-:/")
            ]
        compacted = ordered_tokens[:max_terms]
        if preserved:
            preserved_in_order = [
                token for token in ordered_tokens if token in preserved
            ]
            for token in preserved_in_order:
                if token in compacted:
                    continue
                replace_index = next(
                    (
                        index
                        for index in range(len(compacted) - 1, -1, -1)
                        if compacted[index] not in preserved
                    ),
                    None,
                )
                if replace_index is None:
                    if len(compacted) < max_terms:
                        compacted.append(token)
                    continue
                compacted[replace_index] = token
        return " ".join(compacted).strip()[:160] or "research topic"

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
            "directory": (
                "aggregator",
                "aggregators",
                "directory",
                "directories",
                "listing",
                "listings",
                "filtering",
                "filters",
            ),
            "deadline": ("deadline", "deadlines", "application", "submission"),
            "eligibility": ("eligibility", "requirements", "gpa", "undergraduate"),
            "funding": ("funding", "stipend", "housing", "duration"),
            "people": ("faculty", "advisor", "mentor", "contact"),
            "focus": ("research", "focus", "project", "projects"),
            "official": ("official", "site"),
        }

    def _choose_missing_facets(self, task: Task, agent_state: AgentState) -> list[str]:
        subtask_mode = self._get_current_subtask_mode(task)
        opportunity_family_terms = self._task_opportunity_family_terms(
            task, max_terms=1
        )
        if subtask_mode == "discovery":
            discovery_terms = self._discovery_expansion_terms(task, agent_state)
            if discovery_terms:
                return discovery_terms[:3]

        desc = " ".join(
            [
                self._config_text("subtask_description"),
                self._config_text("success_criteria"),
                str(task.description or ""),
            ]
        )
        profile = build_research_task_profile(desc, self._task_query_text(task))
        existing = " ".join(
            str(item) for item in getattr(agent_state, "queries", set()) or set()
        ).lower()
        selected: list[str] = []
        requested_facets = set(profile.evidence_terms)
        if "application" in requested_facets:
            requested_facets.add("deadline")
        if "contact" in requested_facets:
            requested_facets.add("people")
        if "focus" in requested_facets:
            requested_facets.add("focus")
        if any(
            token in profile.source_terms
            for token in (
                "official",
                "government",
                "university",
                "nonprofit",
                "organization",
            )
        ):
            requested_facets.add("official")
        if any(token in profile.source_terms for token in ("directory",)):
            requested_facets.add("directory")
        for facet, triggers in self._evidence_facets().items():
            if requested_facets and facet not in requested_facets:
                continue
            if any(trigger in existing for trigger in triggers):
                continue
            if facet == "directory":
                selected.extend(["opportunities", "list"])
            elif facet == "deadline":
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
            if "directory" in profile.source_terms:
                selected = ["opportunities", "list"]
            elif subtask_mode == "discovery":
                selected = self._discovery_expansion_terms(task, agent_state) or [
                    "grant",
                    "program",
                ]
            else:
                selected = ["current", "status"]
        if subtask_mode != "discovery" and opportunity_family_terms:
            merged = opportunity_family_terms + [
                token
                for token in selected
                if token not in set(opportunity_family_terms)
            ]
            return merged[:3]
        return selected[:3]

    def _build_structured_search_query(
        self,
        task: Task,
        agent_state: AgentState,
    ) -> str:
        subtask_mode = self._get_current_subtask_mode(task)
        if subtask_mode == "discovery":
            for candidate in self._build_discovery_search_candidates(task, agent_state):
                if self._query_matches_task_scope(task, candidate):
                    return candidate

        subtask_description = self._config_text("subtask_description")
        anchor_terms = self._anchor_terms(task, agent_state, max_terms=6)
        anchor_term_set = set(anchor_terms)
        prior_query_terms = self._extract_recent_query_terms(agent_state, max_terms=6)
        entity_terms = self._extract_recent_title_terms(agent_state, max_terms=5)
        raw_verification_focus_terms = (
            self._extract_recent_verification_focus_terms(agent_state, max_terms=6)
            if subtask_mode == "verification"
            else []
        )
        verification_focus_terms = list(raw_verification_focus_terms)
        opportunity_family_terms = self._task_opportunity_family_terms(
            task, max_terms=1
        )
        guidance = self._derive_guidance_constraints(task, agent_state)
        pivot_terms = list(guidance["pivot_terms"])
        guidance_terms = list(guidance["guidance_terms"])
        constraint_terms = self._task_constraint_terms(task, max_terms=3)
        opportunity_family_terms = self._task_opportunity_family_terms(
            task, max_terms=1
        )
        source_variants = self._subtask_source_variants(task, agent_state)
        source_terms: list[str] = []
        source_seen: set[str] = set()
        site_filter_added = False
        if source_variants and (
            subtask_mode == "discovery"
            or guidance["force_pivot"]
            or (not entity_terms and not prior_query_terms)
        ):
            selected_variant = source_variants[0]
            token_limit = (
                4
                if selected_variant and selected_variant[0].startswith("site:.")
                else 3
            )
            for token in selected_variant[:token_limit]:
                if token in source_seen:
                    continue
                if token.startswith("site:."):
                    if site_filter_added:
                        continue
                    site_filter_added = True
                source_seen.add(token)
                source_terms.append(token)
                if len(source_terms) >= 3:
                    break
        avoid_terms = set(guidance["avoid_terms"]) - anchor_term_set

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
            focus = (agent_state.current_focus or "").strip()
            primary = focus or subtask_description or self._task_query_text(task)
            if ":" in primary:
                head = primary.split(":", 1)[0].strip()
                if len(head.split()) >= 2:
                    primary = head
            entity_terms = self._extract_keywords(primary, max_terms=5)
        if verification_focus_terms:
            verification_set = set(verification_focus_terms)
            entity_terms = verification_focus_terms + [
                token for token in entity_terms if token not in verification_set
            ]
        if (
            subtask_mode in {"verification", "comparison", "general"}
            and opportunity_family_terms
            and not any(token in OPPORTUNITY_FAMILY_TOKENS for token in entity_terms)
        ):
            entity_terms = opportunity_family_terms + [
                token
                for token in entity_terms
                if token not in set(opportunity_family_terms)
            ]
        if avoid_terms:
            entity_terms = [
                token
                for token in entity_terms
                if token not in avoid_terms or token in anchor_term_set
            ]
        task_terms = self._task_scope_terms(task, max_terms=6)
        task_topic_terms = self._task_topic_terms(task, max_terms=4)
        if not task_topic_terms:
            task_topic_terms = [
                token for token in task_terms if token not in OPPORTUNITY_FAMILY_TOKENS
            ][:4]
        facet_terms = (
            self._verification_detail_terms(task, agent_state)
            if subtask_mode == "verification"
            else self._choose_missing_facets(task, agent_state)
        )
        if avoid_terms:
            facet_terms = [token for token in facet_terms if token not in avoid_terms]
        if guidance["force_pivot"] and pivot_terms and subtask_mode != "verification":
            facet_terms = facet_terms[:1]
        if (
            subtask_mode in {"verification", "comparison", "general"}
            and opportunity_family_terms
            and not any(token in OPPORTUNITY_FAMILY_TOKENS for token in facet_terms)
        ):
            facet_terms = opportunity_family_terms[:1] + [
                token
                for token in facet_terms
                if token not in set(opportunity_family_terms)
            ]

        if guidance["force_pivot"] and (pivot_terms or guidance_terms):
            guidance_candidate = self._compact_keyword_query(
                " ".join(
                    source_terms
                    + pivot_terms
                    + guidance_terms
                    + task_topic_terms
                    + opportunity_family_terms
                    + constraint_terms
                ),
                set(source_terms)
                | set(pivot_terms)
                | set(guidance_terms)
                | set(task_topic_terms)
                | set(opportunity_family_terms)
                | set(constraint_terms),
                max_terms=12,
            )
            guidance_candidate_terms = set(
                self._extract_keywords(guidance_candidate, max_terms=12)
            )
            if (
                guidance_candidate
                and guidance_candidate_terms.intersection(
                    set(task_topic_terms)
                    | set(opportunity_family_terms)
                    | set(constraint_terms)
                )
                and guidance_candidate_terms.intersection(
                    set(pivot_terms) | set(guidance_terms) | set(source_terms)
                )
            ):
                return guidance_candidate

        search_count = int(getattr(agent_state, "search_count", 0) or 0)
        if (
            subtask_mode == "discovery"
            and search_count == 0
            and not guidance["force_pivot"]
        ):
            bootstrap_terms: list[str] = []
            bootstrap_seen: set[str] = set()
            bootstrap_source = next(
                (
                    variant
                    for variant in source_variants
                    if variant
                    and not any(token.startswith("site:.") for token in variant)
                ),
                [],
            )
            for token in (
                bootstrap_source[:5] + facet_terms + task_terms + constraint_terms
            ):
                if token not in bootstrap_seen:
                    bootstrap_seen.add(token)
                    bootstrap_terms.append(token)
            bootstrap_candidate = self._compact_keyword_query(
                " ".join(bootstrap_terms[:12]),
                set(bootstrap_source) | set(constraint_terms),
            )
            if bootstrap_candidate and self._query_matches_task_scope(
                task, bootstrap_candidate
            ):
                return bootstrap_candidate

        if subtask_mode == "verification" and not raw_verification_focus_terms:
            verification_bootstrap_terms: list[str] = []
            verification_seen: set[str] = set()
            for token in (
                source_terms
                + task_topic_terms
                + opportunity_family_terms
                + constraint_terms
                + facet_terms
            ):
                if token and token not in verification_seen:
                    verification_seen.add(token)
                    verification_bootstrap_terms.append(token)
            verification_candidate = self._compact_keyword_query(
                " ".join(verification_bootstrap_terms),
                set(source_terms)
                | set(constraint_terms)
                | set(opportunity_family_terms)
                | set(task_topic_terms),
                max_terms=12,
            )
            if verification_candidate and self._query_matches_task_scope(
                task, verification_candidate
            ):
                return verification_candidate

        merged: list[str] = []
        seen: set[str] = set()

        if subtask_mode == "verification" and raw_verification_focus_terms:
            ordered_terms = (
                verification_focus_terms[:5]
                + opportunity_family_terms
                + facet_terms
                + constraint_terms
                + pivot_terms
                + guidance_terms
                + entity_terms
                + anchor_terms[:3]
                + source_terms
                + task_terms
            )
        else:
            anchor_slice = (
                anchor_terms[:2] if guidance["force_pivot"] else anchor_terms[:4]
            )
            ordered_terms = (
                verification_focus_terms[:5]
                + source_terms
                + pivot_terms
                + guidance_terms
                + constraint_terms
                + opportunity_family_terms
                + task_topic_terms
                + anchor_slice
                + entity_terms
                + facet_terms
                + task_terms
            )
        for token in ordered_terms:
            if token in avoid_terms and token not in anchor_term_set:
                continue
            if token not in seen:
                seen.add(token)
                merged.append(token)
        if not merged:
            return "research topic"
        preserve_tokens = set(
            pivot_terms
            + source_terms
            + constraint_terms
            + opportunity_family_terms
            + guidance_terms
            + task_topic_terms
            + [token for token in task_terms if token in {"undergraduate", "students"}]
        )
        candidate = self._compact_keyword_query(" ".join(merged[:12]), preserve_tokens)
        if guidance["force_pivot"]:
            candidate_terms = set(self._extract_keywords(candidate, max_terms=12))
            required_terms = set(pivot_terms[:2]) | set(guidance_terms[:2])
            if required_terms and not candidate_terms.intersection(required_terms):
                candidate = self._compact_keyword_query(
                    " ".join(
                        pivot_terms
                        + guidance_terms
                        + task_topic_terms
                        + opportunity_family_terms
                        + constraint_terms
                        + facet_terms
                    ),
                    preserve_tokens | required_terms,
                    max_terms=12,
                )
        if self._query_matches_task_scope(task, candidate):
            return candidate
        candidate_terms = set(self._extract_keywords(candidate, max_terms=12))
        if (
            candidate_terms
            and candidate_terms.intersection(
                set(task_topic_terms) | set(opportunity_family_terms)
            )
            and candidate_terms.intersection(set(pivot_terms) | set(guidance_terms))
        ):
            return candidate
        return self._compact_keyword_query(
            " ".join(self._task_scope_terms(task, max_terms=6) + constraint_terms),
            preserve_tokens,
        )

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
            "from the official website",
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
        anchor_terms = self._anchor_terms(task, agent_state, max_terms=6)
        anchor_term_set = set(anchor_terms)
        opportunity_family_terms = self._task_opportunity_family_terms(
            task, max_terms=1
        )
        guidance = self._derive_guidance_constraints(task, agent_state)
        pivot_terms = list(guidance["pivot_terms"])
        guidance_terms = list(guidance["guidance_terms"])
        avoid_terms = set(guidance["avoid_terms"]) - anchor_term_set
        query_term_set = set(query_terms)
        filler_tokens = QUERY_FILLER_TOKENS | {
            "focusing",
            "focuses",
            "on",
        }
        missing_pivot = bool(
            guidance["force_pivot"]
            and pivot_terms
            and not (query_term_set & set(pivot_terms))
        )
        missing_opportunity_anchor = bool(
            opportunity_family_terms
            and not (query_term_set & OPPORTUNITY_FAMILY_TOKENS)
            and bool(
                query_term_set
                & {
                    "deadline",
                    "deadlines",
                    "application",
                    "eligibility",
                    "requirements",
                    "status",
                    "official",
                }
            )
        )
        repeated_overlap = len((query_term_set - anchor_term_set) & avoid_terms) >= 2
        if self._get_current_subtask_mode(task) == "discovery" and (
            looks_instructional or missing_pivot or repeated_overlap
        ):
            pathway_candidates = self._build_discovery_search_candidates(
                task, agent_state
            )
            if pathway_candidates:
                return pathway_candidates[0][:120]
        if not looks_instructional:
            raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9/+._:-]*", lowered)
            trailing_filler = bool(raw_tokens and raw_tokens[-1] in filler_tokens)
            stopword_heavy = len(query_terms) <= max(3, len(raw_tokens) // 2)
            contains_filler = any(token in filler_tokens for token in raw_tokens)
            if (
                not trailing_filler
                and not stopword_heavy
                and not contains_filler
                and not missing_pivot
                and not missing_opportunity_anchor
                and not repeated_overlap
            ):
                return cleaned[:140]

        merged: list[str] = []
        seen: set[str] = set()
        facet_terms = self._choose_missing_facets(task, agent_state)
        if avoid_terms:
            facet_terms = [token for token in facet_terms if token not in avoid_terms]
        if (
            guidance["force_pivot"]
            and pivot_terms
            and self._get_current_subtask_mode(task) != "verification"
        ):
            facet_terms = facet_terms[:1]
        ordered_terms = (
            opportunity_family_terms
            + query_terms
            + pivot_terms
            + guidance_terms
            + facet_terms
        )
        for token in ordered_terms:
            if token in avoid_terms and token not in anchor_term_set:
                continue
            if token not in seen and token not in filler_tokens:
                seen.add(token)
                merged.append(token)
        candidate = self._compact_keyword_query(
            " ".join(merged[:10]).strip(),
            set(pivot_terms),
        )
        if (
            candidate
            and len(candidate.split()) >= 3
            and self._query_matches_task_scope(task, candidate)
        ):
            return candidate[:120]
        return self._build_structured_search_query(task, agent_state)

    def _normalize_action_plan(
        self,
        task: Task,
        agent_state: AgentState,
        plan: ResearchActionPlan,
    ) -> ResearchActionPlan:
        subtask_mode = self._get_current_subtask_mode(task)
        if plan.action == "search":
            plan.query = self._sanitize_search_query(task, agent_state, plan.query)
            pathway_override = self._discovery_search_should_follow_pathway(
                task,
                agent_state,
                plan.query,
            )
            if pathway_override:
                logger.info(
                    "Replacing discovery search query with pathway-driven query for task %s; query=%s",
                    task.name,
                    pathway_override,
                )
                plan.query = pathway_override

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

        if subtask_mode == "synthesis":
            if self._synthesis_ready_for_write(agent_state):
                if plan.action in {"think", "search", "mcp_tool"}:
                    logger.info(
                        "Rewriting synthesis-stage action to summarize for task %s because evidence is sufficient for report assembly",
                        task.name,
                    )
                    plan = ResearchActionPlan(
                        direction=plan.direction
                        or "Assemble the final report from the strongest verified and tentative evidence.",
                        action="summarize",
                    )
            else:
                if plan.action == "think":
                    forced_query = self._build_synthesis_gap_query(task, agent_state)
                    logger.info(
                        "Rewriting synthesis-stage think action to targeted verification search for task %s; query=%s",
                        task.name,
                        forced_query,
                    )
                    plan = ResearchActionPlan(
                        direction=plan.direction,
                        action="search",
                        query=forced_query,
                    )
                elif plan.action == "search":
                    plan.query = self._build_synthesis_gap_query(task, agent_state)
        return plan

    @staticmethod
    def _truthy_env(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    def _timeouts_disabled(self) -> bool:
        return timeouts_disabled()

    def _canonicalize_query(self, query: str) -> str:
        canonicalizer = getattr(self.agent, "_canonicalize_query", None)
        if callable(canonicalizer):
            try:
                value = canonicalizer(query)
                if isinstance(value, str):
                    return value
            except Exception:
                pass
        return canonicalize_search_query(query)

    def _build_forced_search_query(self, task: Task, agent_state: AgentState) -> str:
        """Build a non-duplicate forced-search query to break thought loops."""
        base_query = self._build_fallback_search_query(task, agent_state).strip()
        existing_queries = {
            self._canonicalize_query(str(query))
            for query in (getattr(agent_state, "queries", set()) or set())
            if str(query).strip()
        }
        subtask_description = self._config_text("subtask_description")
        subtask_mode = self._get_current_subtask_mode(task)
        guidance = self._derive_guidance_constraints(task, agent_state)
        pivot_terms = list(guidance["pivot_terms"])
        guidance_terms = list(guidance["guidance_terms"])
        constraint_terms = self._task_constraint_terms(task, max_terms=3)
        task_topic_terms = self._task_topic_terms(task, max_terms=4)
        opportunity_family_terms = self._task_opportunity_family_terms(
            task, max_terms=2
        )
        source_variants = self._subtask_source_variants(task, agent_state)

        if subtask_mode == "discovery":
            for candidate in self._build_discovery_search_candidates(task, agent_state):
                normalized = " ".join(candidate.split()).strip()
                if not normalized:
                    continue
                canonical = self._canonicalize_query(normalized)
                if canonical and canonical not in existing_queries:
                    return normalized[:120]

        task_preserve_terms = set(self._task_topic_terms(task, max_terms=5)).union(
            {
                token
                for token in self._opportunity_query_core_terms(task, max_terms=8)
                if token in {"undergraduate", "students"}
            }
        )
        preserve_tokens = (
            set(pivot_terms)
            | set(constraint_terms)
            | task_preserve_terms
            | set(opportunity_family_terms)
        )
        raw_verification_focus_terms = (
            self._extract_recent_verification_focus_terms(agent_state, max_terms=6)
            if subtask_mode == "verification"
            else []
        )
        verification_focus_terms = list(raw_verification_focus_terms)
        discovery_expansion_terms = (
            self._discovery_expansion_terms(task, agent_state, max_terms=4)
            if subtask_mode == "discovery"
            else []
        )
        verification_detail_terms = (
            self._verification_detail_terms(task, agent_state)
            if subtask_mode == "verification"
            else []
        )
        if (
            subtask_mode in {"verification", "comparison", "general"}
            and opportunity_family_terms
            and not any(
                token in OPPORTUNITY_FAMILY_TOKENS for token in verification_focus_terms
            )
        ):
            verification_focus_terms = opportunity_family_terms + [
                token
                for token in verification_focus_terms
                if token not in set(opportunity_family_terms)
            ]
        opportunity_family_tokens = {
            "fellowship",
            "fellowships",
            "grant",
            "grants",
            "program",
            "programs",
            "scholarship",
            "scholarships",
            "internship",
            "internships",
            "funding",
        }
        discovery_anchor_terms = [
            token
            for token in (task_topic_terms + self._task_scope_terms(task, max_terms=8))
            if token not in opportunity_family_tokens
            and token not in DISCOVERY_AUDIENCE_TOKENS
        ]
        if not discovery_anchor_terms:
            discovery_anchor_terms = [
                token
                for token in self._opportunity_query_core_terms(task, max_terms=8)
                if token not in opportunity_family_tokens
            ]
        discovery_anchor_terms = discovery_anchor_terms[:5]
        verification_topic_terms = (task_topic_terms + discovery_anchor_terms)[:4]
        verification_source_context = [
            token
            for token in (source_variants[0] if source_variants else [])
            if token.startswith("site:.")
            or token in {"official", "primary", "organization", "organizations"}
        ][:2]

        candidates: list[str] = []
        if raw_verification_focus_terms:
            verification_preserve = preserve_tokens | {"official", "site"}
            candidates.extend(
                [
                    self._compact_keyword_query(
                        " ".join(
                            verification_focus_terms
                            + opportunity_family_terms
                            + verification_detail_terms
                        ),
                        verification_preserve,
                        max_terms=12,
                    ),
                    self._compact_keyword_query(
                        " ".join(
                            verification_focus_terms
                            + opportunity_family_terms
                            + ["official", "site"]
                            + verification_detail_terms
                        ),
                        verification_preserve,
                        max_terms=12,
                    ),
                    self._compact_keyword_query(
                        " ".join(
                            verification_focus_terms
                            + opportunity_family_terms
                            + ["deadline", "eligibility"]
                        ),
                        verification_preserve,
                        max_terms=12,
                    ),
                ]
            )
        elif subtask_mode in {"verification", "comparison", "general"}:
            verification_preserve = preserve_tokens | {"official", "site"}
            candidates.extend(
                [
                    self._compact_keyword_query(
                        " ".join(
                            verification_source_context
                            + verification_topic_terms
                            + opportunity_family_terms
                            + constraint_terms
                            + verification_detail_terms
                        ),
                        verification_preserve | set(verification_source_context),
                        max_terms=12,
                    ),
                    self._compact_keyword_query(
                        " ".join(
                            verification_topic_terms
                            + opportunity_family_terms
                            + constraint_terms
                            + verification_detail_terms
                        ),
                        verification_preserve,
                        max_terms=12,
                    ),
                    self._compact_keyword_query(
                        " ".join(
                            (verification_source_context or ["official"])
                            + verification_topic_terms
                            + opportunity_family_terms
                            + constraint_terms
                            + verification_detail_terms
                        ),
                        verification_preserve | set(verification_source_context),
                        max_terms=12,
                    ),
                ]
            )
        if discovery_expansion_terms:
            discovery_preserve = preserve_tokens | set(discovery_expansion_terms)
            candidates.extend(
                [
                    self._compact_keyword_query(
                        " ".join(
                            discovery_anchor_terms
                            + constraint_terms
                            + discovery_expansion_terms
                        ),
                        discovery_preserve,
                    ),
                    self._compact_keyword_query(
                        " ".join(
                            ["official"]
                            + discovery_anchor_terms
                            + constraint_terms
                            + discovery_expansion_terms
                        ),
                        discovery_preserve | {"official"},
                    ),
                ]
            )
            for variant in source_variants:
                source_context = [
                    token
                    for token in variant
                    if token.startswith("site:.")
                    or token in GENERIC_SOURCE_LABEL_TOKENS
                    or token in {"official", "primary", "organization", "organizations"}
                ]
                if not source_context:
                    continue
                candidates.append(
                    self._compact_keyword_query(
                        " ".join(
                            source_context
                            + discovery_anchor_terms
                            + constraint_terms
                            + discovery_expansion_terms
                        ),
                        discovery_preserve | set(source_context),
                    )
                )
        source_base_candidates: list[str] = []
        source_official_candidates: list[str] = []
        source_directory_candidates: list[str] = []
        for variant in source_variants:
            source_query_core = (
                (task_topic_terms[:3] or discovery_anchor_terms[:3])
                + opportunity_family_terms[:1]
                + constraint_terms
            )
            source_base_candidates.append(
                self._compact_keyword_query(
                    " ".join(variant + source_query_core),
                    preserve_tokens | set(variant) | set(source_query_core),
                )
            )
            source_official_candidates.append(
                self._compact_keyword_query(
                    " ".join(variant + source_query_core + ["official"]),
                    preserve_tokens
                    | set(variant)
                    | set(source_query_core)
                    | {"official"},
                )
            )
            source_directory_candidates.append(
                self._compact_keyword_query(
                    " ".join(variant + source_query_core + ["directory", "listing"]),
                    preserve_tokens | set(variant) | set(source_query_core),
                )
            )
        candidates.extend(source_base_candidates)
        candidates.extend(source_official_candidates)
        candidates.extend(source_directory_candidates)
        if pivot_terms:
            guidance_query = self._compact_keyword_query(
                " ".join(
                    self._anchor_terms(task, agent_state, max_terms=4)
                    + constraint_terms
                    + pivot_terms
                    + guidance_terms
                ),
                preserve_tokens,
            )
            if guidance_query:
                candidates.extend(
                    [
                        guidance_query,
                        self._compact_keyword_query(
                            f"{guidance_query} directory listing",
                            preserve_tokens,
                        ),
                        self._compact_keyword_query(
                            f"{guidance_query} official site",
                            preserve_tokens | {"official", "site"},
                        ),
                    ]
                )
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
        task_terms = self._extract_keywords(self._task_query_text(task), max_terms=8)
        if task_terms:
            candidates.append(" ".join(task_terms[:8]))

        for candidate in candidates:
            normalized = " ".join(candidate.split()).strip()
            if not normalized:
                continue
            if not self._query_matches_task_scope(task, normalized):
                continue
            canonical = self._canonicalize_query(normalized)
            if canonical and canonical not in existing_queries:
                return normalized[:120]

        # Terminal fallback: pivot to a real alternate evidence angle instead of
        # fabricating uniqueness with a numeric suffix.
        pivot_candidates: list[str] = []
        if base_query:
            pivot_candidates.extend(
                [
                    f"{base_query} official site",
                    f"{base_query} primary source",
                    f"{base_query} program overview",
                    f"{base_query} contact faculty lab",
                    f"{base_query} directory listing",
                ]
            )
        if subtask_description:
            subtask_terms = self._extract_keywords(subtask_description, max_terms=6)
            if subtask_terms:
                joined = " ".join(subtask_terms[:6])
                pivot_candidates.extend(
                    [
                        f"{joined} official site",
                        f"{joined} requirements deadline",
                        f"{joined} program list",
                    ]
                )

        seen_canonicals = set(existing_queries)
        for candidate in pivot_candidates:
            normalized = self._compact_keyword_query(candidate, preserve_tokens)
            if not self._query_matches_task_scope(task, normalized):
                continue
            canonical = self._canonicalize_query(normalized)
            if canonical and canonical not in seen_canonicals:
                return normalized[:120]

        fallback = self._compact_keyword_query(
            base_query or " ".join(self._task_scope_terms(task, max_terms=6)),
            preserve_tokens,
        )
        if self._query_matches_task_scope(task, fallback):
            return fallback
        return self._compact_keyword_query(
            " ".join(self._task_scope_terms(task, max_terms=6) + constraint_terms),
            preserve_tokens,
        )

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
            try:
                await self._ensure_discovery_pathways(task, agent_state)
            except Exception as e:
                logger.debug(
                    "Discovery pathway generation failed for task %s: %s",
                    task.name,
                    e,
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
                agent_state.consecutive_planner_failures = 0
            except asyncio.TimeoutError:
                agent_state.planner_fallback_count += 1
                agent_state.consecutive_planner_failures += 1
                logger.warning(
                    "Subtask action planning timed out after %.1fs for task %s; using fallback search action",
                    action_timeout_seconds,
                    task.name,
                )
                fallback_query = self._build_fallback_search_query(task, agent_state)
                plan = ResearchActionPlan(action="search", query=fallback_query)
            except Exception as e:
                agent_state.planner_fallback_count += 1
                agent_state.consecutive_planner_failures += 1
                logger.warning(
                    "Subtask action planning failed for task %s: %s; using fallback search action",
                    task.name,
                    e,
                )
                fallback_query = self._build_fallback_search_query(task, agent_state)
                plan = ResearchActionPlan(action="search", query=fallback_query)

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
        authoritative_results = self._count_authoritative_results(agent_state)
        subtask_mode = self._get_current_subtask_mode(task)

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
            if subtask_mode == "synthesis":
                forced_query = self._build_synthesis_gap_query(task, agent_state)
            logger.info(
                "Forcing search action for task %s due to %s; query=%s",
                task.name,
                force_search_reason,
                forced_query,
            )
            plan = ResearchActionPlan(action="search", query=forced_query)

        if subtask_mode == "synthesis" and self._synthesis_ready_for_write(agent_state):
            if plan.action == "summarize":
                logger.info(
                    "Promoting synthesis-stage summarize action to complete for task %s because evidence is sufficient",
                    task.name,
                )
                plan = ResearchActionPlan(action="complete")
            elif plan.action in {"think", "search"}:
                logger.info(
                    "Promoting synthesis-stage action to summarize for task %s because evidence is sufficient",
                    task.name,
                )
                plan = ResearchActionPlan(action="summarize")

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
