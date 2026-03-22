"""
LangChain tool execution layer for research actions.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from KestrelAI.mcp.langchain_toolkit import LangChainMCPToolkit
from KestrelAI.shared.research_utils import (
    build_research_task_profile,
    derive_topic_terms,
    normalize_research_text,
    source_looks_low_signal,
    task_requests_low_signal_artifacts,
    url_looks_low_signal_source,
)

from .url_utils import clean_url

try:
    from langchain_core.tools import StructuredTool
except ImportError:  # pragma: no cover - dependency-gated path
    StructuredTool = None

logger = logging.getLogger(__name__)

UNDERGRAD_MARKERS = (
    "undergraduate",
    "undergraduates",
    "undergrad",
    "college student",
    "college students",
    "senior undergraduate",
    "rising senior",
)

GRAD_ONLY_MARKERS = (
    "graduate",
    "graduates",
    "graduate student",
    "graduate students",
    "graduate fellowship",
    "graduate research fellowship",
    "doctoral",
    "doctorate",
    "phd",
    "ph.d",
    "postdoc",
    "postdoctoral",
    "post-doctoral",
    "senior fellows",
    "senior fellow",
)

PRECOLLEGE_MARKERS = (
    "high school",
    "high-school",
    "secondary school",
    "middle school",
    "k-12",
    "k12",
    "teen",
    "teens",
    "youth",
)

MIXED_AUDIENCE_PROFESSIONAL_MARKERS = (
    "early-career",
    "early career",
    "professional",
    "professionals",
    "career researchers",
    "researchers",
    "all levels",
)

STUDENT_FIT_MARKERS = UNDERGRAD_MARKERS + (
    "student",
    "students",
    "college",
    "college students",
    "intern",
    "internship",
    "internships",
)

GENERIC_RESEARCH_PORTAL_MARKERS = (
    "office of undergraduate research",
    "office of undergraduate research and fellowships",
    "office of research and fellowships",
    "undergraduate research office",
    "undergraduate research and fellowships",
    "office of fellowships",
)

APPLICATION_MARKERS = (
    "apply",
    "application",
    "applications",
    "deadline",
    "deadlines",
    "eligibility",
    "eligible",
    "accepting applications",
    "rolling",
    "apply now",
    "submit",
)

STUDENT_AFFAIRS_MARKERS = (
    "student affairs",
    "student life",
    "student engagement",
    "campus life",
    "enrollment",
    "financial aid",
    "admission",
    "admissions",
)

NEWS_RELEASE_MARKERS = (
    "news release",
    "press release",
    "grant and award announcement",
    "award announcement",
    "media contact",
    "for immediate release",
)

INSTITUTIONAL_FUNDING_MARKERS = (
    "funded projects",
    "funding opportunities",
    "submission of proposals",
    "submit proposals",
    "proposal submission",
    "proposal & award",
    "proposal and award",
    "award search",
    "research.gov",
    "principal investigator",
    "directorate",
    "solicitation",
)

INDIRECT_FUNDING_PATTERNS = (
    r"\bawarded funding\b",
    r"\bawarded (?:an? )?grant\b",
    r"\bgrant awarded\b",
    r"\breceived funding\b",
    r"\bsecured funding\b",
    r"\bfunded to continue\b",
    r"\bannounces? funding\b",
    r"\baward(?:ed)? to continue\b",
    r"\bcontinu(?:e|ing|ation)\b.*?\b(?:program|initiative|cohort|fellowship|scholarship|internship)\b",
)

STRONG_OPPORTUNITY_MARKERS = (
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
    "funding opportunity",
    "funding opportunities",
    "research experience for undergraduates",
    "summer research",
    "research opportunity",
    "research opportunities",
)

PROGRAM_CONTEXT_MARKERS = (
    "undergraduate",
    "undergraduates",
    "student",
    "students",
    "summer",
    "research",
    "application",
    "applications",
    "apply",
    "deadline",
    "deadlines",
    "eligibility",
    "stipend",
    "funding",
)

DEGREE_PROGRAM_MARKERS = (
    "undergraduate degree",
    "bachelor",
    "bachelors",
    "bachelor's",
    "b.s.",
    "b.a.",
    "major",
    "majors",
    "minor",
    "minors",
    "academics",
    "curriculum",
    "admissions",
    "apply to",
    "undergraduate program",
    "undergraduate programs",
)


def _classify_source(url: str) -> dict[str, Any]:
    host = (urlparse(url).netloc or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]

    tier = "general"
    score = 1
    official = False

    high_trust_suffixes = (".gov", ".edu")
    research_hosts = (
        "arxiv.org",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
    )
    medium_trust_suffixes = (".org",)

    if host.endswith(high_trust_suffixes):
        tier = "authoritative"
        score = 4
        official = True
    elif host in research_hosts:
        tier = "authoritative"
        score = 4
        official = False
    elif any(host.endswith(suffix) for suffix in medium_trust_suffixes):
        tier = "trusted_org"
        score = 3
    elif host.endswith(".com"):
        tier = "commercial"
        score = 2

    if url_looks_low_signal_source(url):
        tier = "low_signal"
        score = 0
        official = False

    return {
        "domain": host,
        "source_tier": tier,
        "authority_score": score,
        "official_source": official,
    }


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _query_terms_for_ranking(query: str) -> set[str]:
    ignored = {
        "site",
        "gov",
        "edu",
        "org",
        "com",
        "www",
        "official",
        "source",
        "sources",
    }
    return {
        tok
        for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]*", (query or "").lower())
        if len(tok) >= 2 and tok not in ignored and not tok.isdigit()
    }


def _query_topic_terms(query: str) -> set[str]:
    return set(derive_topic_terms(query, max_terms=6))


def _site_filters(query: str) -> set[str]:
    return {
        match.group(1).lower()
        for match in re.finditer(r"site:\.(gov|edu|org|com)\b", (query or "").lower())
    }


def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in markers)


def _query_requires_undergraduate(query: str) -> bool:
    return _contains_any_marker(query, UNDERGRAD_MARKERS)


def _has_application_signal(text: str) -> bool:
    return _contains_any_marker(text, APPLICATION_MARKERS)


def _is_degree_misaligned_result(query: str, title: str, body: str, href: str) -> bool:
    if not _query_requires_undergraduate(query):
        return False
    combined = " ".join(part for part in (title, body, href) if part).lower()
    has_undergrad = _contains_any_marker(combined, UNDERGRAD_MARKERS)
    has_grad_only = _contains_any_marker(combined, GRAD_ONLY_MARKERS)
    return has_grad_only and not has_undergrad


def _is_precollege_misaligned_result(
    query: str,
    title: str,
    body: str,
    href: str,
) -> bool:
    if not _query_requires_undergraduate(query):
        return False
    combined = " ".join(part for part in (title, body, href) if part).lower()
    has_undergrad = _contains_any_marker(combined, UNDERGRAD_MARKERS)
    has_precollege = _contains_any_marker(combined, PRECOLLEGE_MARKERS)
    return has_precollege and not has_undergrad


def _is_mixed_audience_result(
    query: str,
    title: str,
    body: str,
    href: str,
) -> bool:
    if not _query_requires_undergraduate(query):
        return False
    combined = " ".join(part for part in (title, body, href) if part).lower()
    has_undergrad = _contains_any_marker(combined, UNDERGRAD_MARKERS)
    has_precollege = _contains_any_marker(combined, PRECOLLEGE_MARKERS)
    if not (has_undergrad and has_precollege):
        return False
    return _contains_any_marker(combined, MIXED_AUDIENCE_PROFESSIONAL_MARKERS)


def _has_undergraduate_fit_signal(text: str) -> bool:
    return _contains_any_marker(text, STUDENT_FIT_MARKERS)


def _has_topic_fit_signal(text: str, topic_terms: set[str]) -> bool:
    if not topic_terms:
        return False
    terms = set(re.findall(r"[a-z0-9]{2,}", normalize_research_text(text).lower()))
    return bool(terms.intersection(topic_terms))


def _has_prominent_topic_fit_signal(text: str, topic_terms: set[str]) -> bool:
    if not topic_terms:
        return False
    prominent_terms = set(
        re.findall(r"[a-z0-9]{2,}", normalize_research_text(text).lower())
    )
    return bool(prominent_terms.intersection(topic_terms))


def _is_generic_research_portal_result(
    query: str,
    title: str,
    body: str,
    href: str,
    topic_terms: set[str] | None = None,
) -> bool:
    if not _is_opportunity_query(query):
        return False
    combined = " ".join(part for part in (title, body, href) if part).lower()
    if not _contains_any_marker(combined, GENERIC_RESEARCH_PORTAL_MARKERS):
        return False
    if topic_terms:
        prominent = " ".join(part for part in (title, href) if part)
        return not _has_prominent_topic_fit_signal(prominent, topic_terms)
    return False


def _is_opportunity_query(query: str) -> bool:
    query_terms = _query_terms_for_ranking(query)
    return bool(
        query_terms.intersection(
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
                "funding",
                "undergraduate",
                "undergraduates",
                "student",
                "students",
                "deadline",
                "deadlines",
                "application",
                "applications",
            }
        )
    )


def _is_student_affairs_result(title: str, body: str, href: str) -> bool:
    combined = " ".join(part for part in (title, body, href) if part).lower()
    return _contains_any_marker(combined, STUDENT_AFFAIRS_MARKERS)


def _is_indirect_funding_announcement(
    title: str,
    body: str,
    href: str,
) -> bool:
    combined = " ".join(part for part in (title, body, href) if part).lower()
    if _contains_any_marker(combined, NEWS_RELEASE_MARKERS):
        return True
    return any(re.search(pattern, combined) for pattern in INDIRECT_FUNDING_PATTERNS)


def _is_institutional_funding_result(
    query: str,
    title: str,
    body: str,
    href: str,
) -> bool:
    if not _query_requires_undergraduate(query):
        return False
    combined = " ".join(part for part in (title, body, href) if part).lower()
    if not _contains_any_marker(combined, INSTITUTIONAL_FUNDING_MARKERS):
        return False
    prominent = " ".join(part for part in (title, href) if part).lower()
    return not _contains_any_marker(prominent, UNDERGRAD_MARKERS + STUDENT_FIT_MARKERS)


def _has_strong_opportunity_signal(text: str) -> bool:
    lowered = (text or "").lower()
    profile = build_research_task_profile(lowered)
    if any(marker in lowered for marker in STRONG_OPPORTUNITY_MARKERS):
        return True
    if any(marker in lowered for marker in DEGREE_PROGRAM_MARKERS):
        return False
    if profile.target_terms and (
        profile.audience_terms or profile.evidence_terms or "research" in lowered
    ):
        return True
    if "program" in lowered or "programs" in lowered:
        return bool(profile.audience_terms or profile.evidence_terms) or any(
            marker in lowered for marker in PROGRAM_CONTEXT_MARKERS
        )
    if "opportunity" in lowered or "opportunities" in lowered:
        return (
            bool(profile.audience_terms or profile.evidence_terms)
            or "research" in lowered
        )
    return False


def _is_degree_program_result(title: str, body: str, href: str) -> bool:
    combined = " ".join(part for part in (title, body, href) if part).lower()
    if not any(marker in combined for marker in DEGREE_PROGRAM_MARKERS):
        return False
    return not any(marker in combined for marker in STRONG_OPPORTUNITY_MARKERS)


def _is_publication_like_result(url: str, title: str, body: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]
    lowered = " ".join(part for part in (title, body, parsed.path) if part).lower()
    if host in {"arxiv.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov"}:
        return True
    publication_markers = (
        "arxiv",
        "preprint",
        "survey",
        "paper",
        "journal",
        "abstract",
        "/abs/",
    )
    opportunity_markers = (
        "fellowship",
        "grant",
        "scholarship",
        "internship",
        "program",
        "students",
        "undergraduate",
        "applications",
        "deadline",
        "apply",
    )
    return any(marker in lowered for marker in publication_markers) and not any(
        marker in lowered for marker in opportunity_markers
    )


def _is_low_signal_result(query: str, title: str, body: str, href: str) -> bool:
    if task_requests_low_signal_artifacts(query):
        return False
    return source_looks_low_signal(title, body, href, url=href)


def _matches_site_filters(url: str, required_suffixes: set[str]) -> bool:
    if not required_suffixes:
        return True
    host = (urlparse(url).netloc or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return any(
        host.endswith(f".{suffix}") or host == suffix for suffix in required_suffixes
    )


def _rank_search_hit(
    query: str, hit: dict[str, Any], source_meta: dict[str, Any]
) -> tuple[int, int, int, int, int]:
    title = str(hit.get("title", "")).lower()
    body = str(hit.get("body", "")).lower()
    href = str(hit.get("href", "")).lower()
    title_terms = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]*", title))
    body_terms = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]*", f"{body} {href}"))
    query_terms = _query_terms_for_ranking(query)
    topic_terms = _query_topic_terms(query)
    prominent = " ".join(part for part in (title, href) if part)
    combined = " ".join(part for part in (title, body, href) if part)

    title_overlap = len(query_terms.intersection(title_terms))
    body_overlap = len(query_terms.intersection(body_terms))
    opportunity_keywords = {
        "fellowship",
        "fellowships",
        "grant",
        "grants",
        "scholarship",
        "scholarships",
        "program",
        "programs",
        "undergraduate",
        "undergraduates",
        "student",
        "students",
        "internship",
        "internships",
        "summer",
        "research",
    }
    opportunity_bonus = sum(
        1
        for token in opportunity_keywords
        if token in title_terms or token in body_terms
    )
    domain_bonus = 1 if source_meta.get("official_source") else 0
    profile_score = 0
    if _is_student_affairs_result(title, body, href):
        profile_score -= 4
    if _is_indirect_funding_announcement(title, body, href):
        profile_score -= 6
    if _is_institutional_funding_result(query, title, body, href):
        profile_score -= 6
    if _is_mixed_audience_result(query, title, body, href):
        profile_score -= 4
    if topic_terms:
        if _has_prominent_topic_fit_signal(prominent, topic_terms):
            profile_score += 3
        elif _has_topic_fit_signal(body, topic_terms):
            profile_score += 1
        else:
            profile_score -= 3
    if _query_requires_undergraduate(query):
        if _contains_any_marker(prominent, UNDERGRAD_MARKERS):
            profile_score += 3
        elif _contains_any_marker(body, UNDERGRAD_MARKERS):
            profile_score += 1
        elif _is_degree_misaligned_result(query, title, body, href):
            profile_score -= 5
        elif _is_precollege_misaligned_result(query, title, body, href):
            profile_score -= 5
        else:
            profile_score -= 2
    if _is_opportunity_query(query) and source_meta.get("source_tier") == "low_signal":
        profile_score -= 2

    return (
        profile_score,
        int(source_meta.get("authority_score", 0) or 0) + domain_bonus,
        title_overlap,
        body_overlap,
        opportunity_bonus,
    )


class SearchToolInput(BaseModel):
    query: str = Field(min_length=1)


class MCPToolInput(BaseModel):
    tool_name: str = Field(min_length=1)
    tool_parameters: dict[str, Any] = Field(default_factory=dict)


class LangChainToolExecutor:
    """Unified tool executor backed by LangChain StructuredTool."""

    def __init__(
        self,
        *,
        searxng_service: Any,
        url_flag_manager: Any,
        mcp_manager: Any | None = None,
        mcp_enabled: Callable[[], bool] | None = None,
    ):
        if StructuredTool is None:
            raise ImportError("langchain_core is not installed")

        self.searxng_service = searxng_service
        self.url_flag_manager = url_flag_manager
        self.mcp_manager = mcp_manager
        self.mcp_enabled = mcp_enabled
        self.mcp_toolkit: LangChainMCPToolkit | None = None

        self._tools = {
            "search_web": StructuredTool.from_function(
                func=self._search_web,
                name="search_web",
                description="Search the web and return normalized snippets.",
                args_schema=SearchToolInput,
            ),
            "mcp_call": StructuredTool.from_function(
                coroutine=self._call_mcp_tool,
                name="mcp_call",
                description="Call an MCP tool with structured parameters.",
                args_schema=MCPToolInput,
            ),
        }

        if self.mcp_manager is not None:
            try:
                self.mcp_toolkit = LangChainMCPToolkit(
                    mcp_manager=self.mcp_manager,
                    mcp_enabled=self.mcp_enabled,
                )
                self._tools.update(self.mcp_toolkit.build_tools())
            except Exception as e:
                logger.debug("Failed to initialize MCP toolkit tools: %s", e)
                self.mcp_toolkit = None

    async def ainvoke(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool '{tool_name}'")
        result = await tool.ainvoke(payload)
        return result if isinstance(result, dict) else {"result": result}

    def get_available_tools(self) -> list[str]:
        """List registered tool names."""
        return sorted(self._tools.keys())

    def resolve_mcp_tool_name(self, tool_name: str) -> str | None:
        """Resolve a logical MCP tool name to a registered executor tool key."""
        if self.mcp_toolkit is None:
            return None
        return self.mcp_toolkit.resolve_tool_name(tool_name)

    async def call_mcp_tool(
        self, tool_name: str, tool_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke MCP tool using toolkit-specific tool if available."""
        resolved = self.resolve_mcp_tool_name(tool_name)
        if resolved:
            return await self.ainvoke(resolved, tool_parameters)
        return await self.ainvoke(
            "mcp_call",
            {"tool_name": tool_name, "tool_parameters": tool_parameters},
        )

    def _search_web(self, query: str) -> dict[str, Any]:
        start = time.perf_counter()
        hits = self.searxng_service.search(query)
        processed_hits: list[dict[str, Any]] = []
        disable_timeouts = os.getenv(
            "GLOBAL_DISABLE_TIMEOUTS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        raw_total_timeout_seconds = _env_float(
            "TOOL_SEARCH_TOTAL_TIMEOUT_SECONDS", 25.0
        )
        total_timeout_seconds: float | None
        if disable_timeouts or raw_total_timeout_seconds <= 0:
            total_timeout_seconds = None
        else:
            total_timeout_seconds = max(1.0, raw_total_timeout_seconds)
        max_fetched_hits = max(
            1,
            _env_int("TOOL_SEARCH_MAX_FETCHED_HITS", 4),
        )
        budget_exhausted = False
        required_site_filters = _site_filters(query)
        ranked_hits: list[dict[str, Any]] = []
        opportunity_query = _is_opportunity_query(query)
        query_topic_terms = _query_topic_terms(query)
        for hit in hits:
            href = str(hit.get("href", ""))
            clean_href = clean_url(href)
            if clean_href is None:
                logger.warning(
                    "Skipping invalid URL from search result: %s", href[:100]
                )
                continue
            if required_site_filters and not _matches_site_filters(
                clean_href, required_site_filters
            ):
                continue
            if opportunity_query and _is_publication_like_result(
                clean_href,
                str(hit.get("title", "")),
                str(hit.get("body", "")),
            ):
                continue
            source_meta = _classify_source(clean_href)
            if opportunity_query and _is_low_signal_result(
                query,
                str(hit.get("title", "")),
                str(hit.get("body", "")),
                clean_href,
            ):
                continue
            if opportunity_query and _is_degree_misaligned_result(
                query,
                str(hit.get("title", "")),
                str(hit.get("body", "")),
                clean_href,
            ):
                continue
            if opportunity_query and _is_precollege_misaligned_result(
                query,
                str(hit.get("title", "")),
                str(hit.get("body", "")),
                clean_href,
            ):
                continue
            if opportunity_query and _is_institutional_funding_result(
                query,
                str(hit.get("title", "")),
                str(hit.get("body", "")),
                clean_href,
            ):
                continue
            if opportunity_query and _is_generic_research_portal_result(
                query,
                str(hit.get("title", "")),
                str(hit.get("body", "")),
                clean_href,
                query_topic_terms,
            ):
                continue
            if opportunity_query and _is_indirect_funding_announcement(
                str(hit.get("title", "")),
                str(hit.get("body", "")),
                clean_href,
            ):
                continue
            if opportunity_query and _is_degree_program_result(
                str(hit.get("title", "")),
                str(hit.get("body", "")),
                clean_href,
            ):
                continue
            ranked_hits.append(
                {
                    **hit,
                    "href": clean_href,
                    "_source_meta": source_meta,
                    "_rank": _rank_search_hit(query, hit, source_meta),
                }
            )

        ranked_hits.sort(key=lambda item: item["_rank"], reverse=True)

        for hit in ranked_hits[:max_fetched_hits]:
            if (
                total_timeout_seconds is not None
                and (time.perf_counter() - start) >= total_timeout_seconds
            ):
                budget_exhausted = True
                logger.warning(
                    "Tool search budget exhausted for query '%s' after %.2fs",
                    query,
                    time.perf_counter() - start,
                )
                break

            clean_href = str(hit.get("href", "")).strip()
            source_meta = dict(hit.get("_source_meta", {}))
            body = self.searxng_service.extract_text(clean_href)
            url_flag = self.url_flag_manager.get_or_create_flag(clean_href)
            if url_flag is None:
                continue

            title = str(hit.get("title", ""))
            summary = str(hit.get("body", ""))
            combined = " ".join(
                part for part in (title, summary, body, clean_href) if part
            )
            if opportunity_query and query_topic_terms:
                prominent = " ".join(part for part in (title, clean_href) if part)
                if not _has_prominent_topic_fit_signal(
                    prominent, query_topic_terms
                ) and not _has_topic_fit_signal(body, query_topic_terms):
                    continue
            if opportunity_query and _query_requires_undergraduate(query):
                if _is_degree_misaligned_result(
                    query, title, f"{summary} {body}", clean_href
                ):
                    continue
                if _is_precollege_misaligned_result(
                    query,
                    title,
                    f"{summary} {body}",
                    clean_href,
                ):
                    continue
                if _is_institutional_funding_result(
                    query,
                    title,
                    f"{summary} {body}",
                    clean_href,
                ):
                    continue
                if not _has_undergraduate_fit_signal(combined):
                    continue
            if opportunity_query and _is_student_affairs_result(
                title, body, clean_href
            ):
                prominent = " ".join(part for part in (title, clean_href) if part)
                if (
                    query_topic_terms
                    and not _has_prominent_topic_fit_signal(
                        prominent, query_topic_terms
                    )
                    and not _has_topic_fit_signal(body, query_topic_terms)
                ):
                    continue
                if not _has_application_signal(body):
                    continue
            if opportunity_query and _is_indirect_funding_announcement(
                title,
                body,
                clean_href,
            ):
                continue
            if opportunity_query and not _has_strong_opportunity_signal(combined):
                continue
            snippet = (
                f"Title: {title}\n"
                f"URL: {url_flag} (see URL reference table)\n"
                f"Summary: {summary[:200]}\n"
                f"Content: {body[:500]}"
            )
            processed_hits.append(
                {
                    "title": title,
                    "url": clean_href,
                    "summary": summary,
                    "content": body,
                    "fetched": bool(body),
                    "snippet": snippet,
                    **source_meta,
                }
            )

        processed_hits.sort(
            key=lambda item: (
                int(item.get("authority_score", 0)),
                int(bool(item.get("fetched"))),
                len(str(item.get("content", ""))),
            ),
            reverse=True,
        )

        return {
            "query": query,
            "search_time": time.perf_counter() - start,
            "hits": processed_hits,
            "budget_exhausted": budget_exhausted,
            "max_fetched_hits": max_fetched_hits,
        }

    async def _call_mcp_tool(
        self, tool_name: str, tool_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        if self.mcp_toolkit is not None:
            return await self.mcp_toolkit.execute_tool(tool_name, tool_parameters)

        if self.mcp_manager is None:
            return {
                "success": False,
                "tool_name": tool_name,
                "data": None,
                "error": "MCP manager unavailable",
            }

        if self.mcp_enabled is not None and not self.mcp_enabled():
            return {
                "success": False,
                "tool_name": tool_name,
                "data": None,
                "error": "MCP not available",
            }

        try:
            result = await self.mcp_manager.call_tool(tool_name, tool_parameters)
            return {
                "success": bool(result.success),
                "tool_name": tool_name,
                "data": result.data,
                "error": result.error,
            }
        except Exception as e:
            return {
                "success": False,
                "tool_name": tool_name,
                "data": None,
                "error": str(e),
            }
