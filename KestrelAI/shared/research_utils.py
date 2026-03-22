from __future__ import annotations

import os
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

DISCRETE_OPPORTUNITY_TERMS = (
    "fellowship",
    "fellowships",
    "grant",
    "grants",
    "scholarship",
    "scholarships",
    "internship",
    "internships",
    "funding opportunity",
    "funding opportunities",
    "opportunity",
    "opportunities",
    "program",
    "programs",
)

OPPORTUNITY_SEARCH_MARKERS = DISCRETE_OPPORTUNITY_TERMS + ("funding",)

RESEARCH_AUDIENCE_TERMS = frozenset(
    {
        "undergraduate",
        "undergraduates",
        "undergrad",
        "student",
        "students",
        "senior",
        "rising",
        "freshman",
        "freshmen",
        "sophomore",
        "sophomores",
        "junior",
        "juniors",
        "graduate",
        "graduates",
        "doctoral",
        "doctorate",
        "phd",
        "postdoc",
        "postdoctoral",
        "youth",
        "teen",
        "teens",
        "us",
        "u.s",
        "united",
        "states",
    }
)

RESEARCH_SOURCE_CLASS_TERMS = frozenset(
    {
        "official",
        "primary",
        "site",
        "sites",
        "source",
        "sources",
        "organization",
        "organizations",
        "government",
        "federal",
        "agency",
        "agencies",
        "university",
        "universities",
        "college",
        "colleges",
        "institute",
        "institutes",
        "lab",
        "labs",
        "laboratory",
        "laboratories",
        "center",
        "centers",
        "centre",
        "centres",
        "directory",
        "directories",
        "listing",
        "listings",
        "portal",
        "portals",
        "website",
        "websites",
        "page",
        "pages",
        "nonprofit",
        "foundation",
        "association",
        "society",
        "societies",
    }
)

RESEARCH_EVIDENCE_TERMS = frozenset(
    {
        "deadline",
        "deadlines",
        "eligibility",
        "eligible",
        "application",
        "applications",
        "apply",
        "applying",
        "accepting",
        "funding",
        "stipend",
        "stipends",
        "amount",
        "amounts",
        "award",
        "awards",
        "requirements",
        "criteria",
        "contact",
        "contacts",
        "status",
        "open",
        "current",
        "currently",
        "available",
        "availability",
    }
)

RESEARCH_PROCESS_TERMS = frozenset(
    {
        "find",
        "search",
        "searches",
        "map",
        "mapping",
        "locate",
        "identify",
        "discover",
        "gather",
        "collect",
        "verify",
        "compare",
        "summarize",
        "summary",
        "report",
        "research",
        "task",
        "subtask",
        "phase",
        "step",
        "focus",
        "goal",
        "goals",
        "success",
        "criteria",
    }
)

GENERIC_RESEARCH_TOPIC_EXCLUDE_TOKENS = frozenset(
    set(DISCRETE_OPPORTUNITY_TERMS)
    | set(RESEARCH_AUDIENCE_TERMS)
    | set(RESEARCH_SOURCE_CLASS_TERMS)
    | set(RESEARCH_EVIDENCE_TERMS)
    | set(RESEARCH_PROCESS_TERMS)
    | {
        "are",
        "and",
        "or",
        "the",
        "a",
        "an",
        "at",
        "be",
        "by",
        "for",
        "in",
        "is",
        "of",
        "on",
        "to",
        "with",
        "from",
        "into",
        "within",
        "about",
        "across",
        "support",
        "supports",
        "supporting",
        "offer",
        "offers",
        "offered",
        "that",
        "their",
        "them",
        "there",
        "these",
        "those",
        "which",
        "while",
        "include",
        "including",
        "provided",
        "public",
        "publicly",
        "distinct",
        "strong",
        "recent",
        "latest",
        "newest",
        "academic",
        "major",
        "conference",
        "conferences",
        "method",
        "methods",
        "list",
        "lists",
        "opportunity",
        "opportunities",
    }
)

TARGET_TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "fellowship": ("fellowship", "fellowships"),
    "grant": ("grant", "grants"),
    "scholarship": ("scholarship", "scholarships"),
    "internship": ("internship", "internships"),
    "program": ("program", "programs"),
    "funding": ("funding", "funding opportunity", "funding opportunities"),
    "opportunity": ("opportunity", "opportunities"),
    "paper": ("paper", "papers", "article", "articles", "preprint", "preprints"),
    "proceedings": (
        "proceeding",
        "proceedings",
        "conference proceeding",
        "conference proceedings",
        "workshop proceeding",
        "workshop proceedings",
    ),
    "survey": ("survey", "surveys", "review", "reviews"),
    "benchmark": ("benchmark", "benchmarks", "leaderboard", "leaderboards"),
    "framework": (
        "framework",
        "frameworks",
        "toolkit",
        "toolkits",
        "library",
        "libraries",
    ),
    "repository": ("repository", "repositories", "repo", "repos"),
}

SOURCE_CLASS_ALIASES: dict[str, tuple[str, ...]] = {
    "official": ("official", "primary"),
    "government": ("government", "federal", "agency", "agencies", ".gov", "site:.gov"),
    "university": (
        "university",
        "universities",
        "college",
        "colleges",
        ".edu",
        "site:.edu",
    ),
    "nonprofit": (
        "nonprofit",
        "foundation",
        "association",
        "society",
        "societies",
        ".org",
        "site:.org",
    ),
    "organization": ("organization", "organizations", "institution", "institutions"),
    "directory": (
        "directory",
        "directories",
        "listing",
        "listings",
        "portal",
        "portals",
        "database",
        "databases",
    ),
    "lab": (
        "lab",
        "labs",
        "laboratory",
        "laboratories",
        "center",
        "centers",
        "centre",
        "centres",
        "institute",
        "institutes",
    ),
    "repository": ("repository", "repositories", "repo", "repos", "github", "gitlab"),
    "documentation": ("docs", "documentation", "readme", "manual", "guide", "guides"),
    "project": ("project", "projects"),
    "conference": (
        "conference",
        "conferences",
        "workshop",
        "workshops",
        "symposium",
        "symposia",
    ),
    "publisher": ("publisher", "publishers", "journal", "journals", "proceedings"),
    "preprint": ("arxiv", "preprint", "preprints"),
}

EVIDENCE_FACET_ALIASES: dict[str, tuple[str, ...]] = {
    "deadline": ("deadline", "deadlines", "submission", "submissions"),
    "application": ("apply", "application", "applications", "submit", "submission"),
    "eligibility": (
        "eligibility",
        "eligible",
        "requirement",
        "requirements",
        "criteria",
        "gpa",
    ),
    "status": (
        "open",
        "opened",
        "current",
        "currently",
        "available",
        "availability",
        "closed",
        "active",
    ),
    "funding": (
        "funding",
        "stipend",
        "stipends",
        "award",
        "awards",
        "amount",
        "amounts",
        "housing",
        "duration",
    ),
    "contact": (
        "contact",
        "contacts",
        "faculty",
        "advisor",
        "advisors",
        "mentor",
        "mentors",
    ),
    "focus": (
        "research",
        "project",
        "projects",
        "focus",
        "focuses",
        "topic",
        "topics",
        "theme",
        "themes",
    ),
}

AUDIENCE_ALIASES: dict[str, tuple[str, ...]] = {
    "undergraduate": ("undergraduate", "undergraduates", "undergrad", "undergrads"),
    "student": ("student", "students", "college student", "college students"),
    "graduate": (
        "graduate",
        "graduates",
        "doctoral",
        "doctorate",
        "phd",
        "postdoc",
        "postdoctoral",
    ),
    "us": ("us", "u.s", "united states"),
}


@dataclass(frozen=True)
class ResearchTaskProfile:
    topic_terms: tuple[str, ...]
    target_terms: tuple[str, ...]
    source_terms: tuple[str, ...]
    evidence_terms: tuple[str, ...]
    audience_terms: tuple[str, ...]


def infer_research_task_family(*texts: str) -> str:
    joined_text = " ".join(str(text or "") for text in texts if str(text or "").strip())
    normalized = normalize_research_text(joined_text).lower()
    profile = build_research_task_profile(joined_text)
    target_terms = set(profile.target_terms)

    if target_terms.intersection(
        {
            "fellowship",
            "grant",
            "scholarship",
            "internship",
            "program",
            "funding",
            "opportunity",
        }
    ) or text_is_opportunity_search(normalized):
        return "opportunity"

    if target_terms.intersection(
        {"paper", "proceedings", "survey", "benchmark"}
    ) or any(
        token in normalized
        for token in (
            "paper",
            "papers",
            "proceedings",
            "benchmark",
            "benchmarks",
            "survey",
            "surveys",
            "arxiv",
            "journal",
        )
    ):
        return "papers"

    if target_terms.intersection({"framework", "repository"}) or any(
        token in normalized
        for token in (
            "framework",
            "frameworks",
            "repository",
            "repositories",
            "repo",
            "repos",
            "open-source",
            "opensource",
            "maintainer",
            "maintainers",
            "ecosystem",
            "docs",
            "documentation",
        )
    ):
        return "ecosystem"

    return "general"


LOW_SIGNAL_TEXT_PATTERNS = (
    r"\bblog (?:post|article)\b",
    r"\bforum (?:post|thread|discussion)\b",
    r"\bcommunity (?:post|thread|discussion)\b",
    r"\bquestion(?:s)? and answers?\b",
    r"\bq\s*&\s*a\b",
    r"\bwiki(?:pedia)?\b",
    r"\bencyclopedia\b",
    r"\branking(?:s)?\b",
    r"\blisticle\b",
    r"\broundup\b",
    r"\bultimate guide\b",
    r"\bcomprehensive guide\b",
    r"\bdirectory of\b",
    r"\blist of\b",
    r"\b\d+\+?\s+(?:verified\s+)?(?:fellowships?|grants?|scholarships?|programs?|opportunities?)\b",
)

LOW_SIGNAL_URL_PATH_MARKERS = (
    "/blog/",
    "/blogs/",
    "/forum/",
    "/forums/",
    "/community/",
    "/communities/",
    "/thread/",
    "/threads/",
    "/question/",
    "/questions/",
    "/answer/",
    "/answers/",
    "/wiki/",
    "/rankings/",
    "/best-",
    "/top-",
    "/jobs/",
    "/careers/",
)

LOW_SIGNAL_HOST_LABEL_MARKERS = frozenset(
    {
        "blog",
        "blogs",
        "forum",
        "forums",
        "community",
        "communities",
        "wiki",
        "jobs",
        "careers",
        "rankings",
    }
)

LOW_SIGNAL_REFERENCE_CONTEXT_MARKERS = (
    "listed on",
    "ranking",
    "rankings",
    "blog",
    "forum",
    "community",
    "directory",
    "listicle",
    "roundup",
)

LOW_SIGNAL_ARTIFACT_REQUEST_TERMS = frozenset(
    {
        "blog",
        "blogs",
        "forum",
        "forums",
        "community",
        "communities",
        "thread",
        "threads",
        "wiki",
        "ranking",
        "rankings",
        "listicle",
        "newsletter",
    }
)


def _contains_alias(text: str, alias: str) -> bool:
    normalized = str(text or "").lower()
    lowered_alias = alias.lower()
    if any(char in lowered_alias for char in ".:"):
        return lowered_alias in normalized
    pattern = r"(?<![a-z0-9])" + re.escape(lowered_alias) + r"(?![a-z0-9])"
    return re.search(pattern, normalized) is not None


def _derive_canonical_terms(
    text: str,
    aliases: dict[str, tuple[str, ...]],
    *,
    max_terms: int,
) -> list[str]:
    normalized = normalize_research_text(text).lower()
    selected: list[str] = []
    for canonical, variants in aliases.items():
        if any(_contains_alias(normalized, variant) for variant in variants):
            selected.append(canonical)
        if len(selected) >= max_terms:
            break
    return selected[:max_terms]


def _expand_alias_terms(
    selected: Collection[str],
    aliases: dict[str, tuple[str, ...]],
) -> set[str]:
    expanded: set[str] = set()
    for canonical in selected:
        normalized = str(canonical or "").lower()
        if not normalized:
            continue
        expanded.add(normalized)
        for variant in aliases.get(normalized, ()):
            expanded.add(str(variant).lower())
    return expanded


def derive_target_terms(text: str, *, max_terms: int = 4) -> list[str]:
    return _derive_canonical_terms(text, TARGET_TERM_ALIASES, max_terms=max_terms)


def derive_source_class_terms(text: str, *, max_terms: int = 4) -> list[str]:
    return _derive_canonical_terms(text, SOURCE_CLASS_ALIASES, max_terms=max_terms)


def derive_evidence_terms(text: str, *, max_terms: int = 4) -> list[str]:
    return _derive_canonical_terms(text, EVIDENCE_FACET_ALIASES, max_terms=max_terms)


def derive_audience_terms(text: str, *, max_terms: int = 3) -> list[str]:
    return _derive_canonical_terms(text, AUDIENCE_ALIASES, max_terms=max_terms)


def build_research_task_profile(
    *texts: str,
    max_topic_terms: int = 6,
    max_category_terms: int = 4,
) -> ResearchTaskProfile:
    joined_text = " ".join(str(text or "") for text in texts if str(text or "").strip())
    target_terms = tuple(derive_target_terms(joined_text, max_terms=max_category_terms))
    source_terms = tuple(
        derive_source_class_terms(joined_text, max_terms=max_category_terms)
    )
    evidence_terms = tuple(
        derive_evidence_terms(joined_text, max_terms=max_category_terms)
    )
    audience_terms = tuple(
        derive_audience_terms(joined_text, max_terms=max_category_terms)
    )
    topic_exclude = (
        _expand_alias_terms(target_terms, TARGET_TERM_ALIASES)
        | _expand_alias_terms(source_terms, SOURCE_CLASS_ALIASES)
        | _expand_alias_terms(evidence_terms, EVIDENCE_FACET_ALIASES)
        | _expand_alias_terms(audience_terms, AUDIENCE_ALIASES)
    )
    topic_terms = tuple(
        derive_topic_terms(
            joined_text,
            max_terms=max_topic_terms,
            stopwords={"is", "are", "be", "that", "these", "those", "which", "while"},
            extra_exclude=topic_exclude,
        )
    )
    return ResearchTaskProfile(
        topic_terms=topic_terms,
        target_terms=target_terms,
        source_terms=source_terms,
        evidence_terms=evidence_terms,
        audience_terms=audience_terms,
    )


def extract_domains_from_text(text: str) -> list[str]:
    domains: list[str] = []
    for raw in re.findall(r"https?://[^\s)]+", text or ""):
        host = (urlparse(raw).netloc or "").lower().strip()
        if host.startswith("www."):
            host = host[4:]
        if host:
            domains.append(host)
    for raw in re.findall(
        r"\b(?:www\.)?[a-z0-9.-]+\.(?:com|org|net|edu|gov|us)(?:/[^\s)]*)?",
        text or "",
        flags=re.IGNORECASE,
    ):
        host = raw.split("/", 1)[0].lower().strip()
        if host.startswith("www."):
            host = host[4:]
        if host:
            domains.append(host)
    return domains


def url_looks_low_signal_source(url: str) -> bool:
    parsed = urlparse(url if "://" in str(url or "") else f"https://{url}")
    host = (parsed.netloc or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]
    host_labels = {token for token in re.split(r"[.-]", host) if token}
    path = (parsed.path or "").lower()
    if host_labels.intersection(LOW_SIGNAL_HOST_LABEL_MARKERS):
        return True
    return any(marker in path for marker in LOW_SIGNAL_URL_PATH_MARKERS)


def text_looks_low_signal_source(text: str) -> bool:
    normalized = normalize_research_text(text).lower()
    return any(re.search(pattern, normalized) for pattern in LOW_SIGNAL_TEXT_PATTERNS)


def source_looks_low_signal(*parts: str, url: str = "") -> bool:
    combined = " ".join(str(part or "") for part in parts if str(part or "").strip())
    return text_looks_low_signal_source(combined) or url_looks_low_signal_source(url)


def task_requests_low_signal_artifacts(text: str) -> bool:
    normalized = normalize_research_text(text).lower()
    tokens = set(re.findall(r"[a-z][a-z0-9]{1,}", normalized))
    return bool(tokens.intersection(LOW_SIGNAL_ARTIFACT_REQUEST_TERMS))


def subtask_looks_low_signal(
    subtask_text: str,
    *,
    task_text: str = "",
) -> bool:
    combined = normalize_research_text(subtask_text).lower()
    if task_text and task_requests_low_signal_artifacts(task_text):
        return False
    if text_looks_low_signal_source(combined):
        return True
    domains = extract_domains_from_text(combined)
    if not domains:
        return False
    has_low_signal_context = any(
        marker in combined for marker in LOW_SIGNAL_REFERENCE_CONTEXT_MARKERS
    )
    if not has_low_signal_context:
        return False
    return any(
        url_looks_low_signal_source(domain) or not domain.endswith((".gov", ".edu"))
        for domain in domains
    )


def timeouts_disabled() -> bool:
    return os.getenv("GLOBAL_DISABLE_TIMEOUTS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def canonicalize_search_query(query: str) -> str:
    query = re.sub(r"[^a-zA-Z0-9\s]", " ", str(query or "").lower())
    return re.sub(r"\s+", " ", query).strip()


def normalize_research_text(text: str) -> str:
    normalized = str(text or "")
    return re.sub(
        r"(?i)\b([a-z0-9][a-z0-9._+-]{1,})\s*/\s*([a-z0-9][a-z0-9._+-]{1,})\b",
        r" \1 \2 ",
        normalized,
    )


def extract_research_terms(
    text: str,
    *,
    max_terms: int = 12,
    stopwords: Collection[str] | None = None,
    exclude: Collection[str] | None = None,
) -> list[str]:
    stopword_set = {str(token).lower() for token in (stopwords or ())}
    exclude_set = {str(token).lower() for token in (exclude or ())}
    tokens = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9/+._:-]*",
        normalize_research_text(text).lower(),
    )
    extracted: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = token.strip("._-:/")
        if not token:
            continue
        if token in stopword_set or token in exclude_set:
            continue
        if len(token) < 2:
            continue
        if token.isdigit() and len(token) < 4:
            continue
        if token not in seen:
            seen.add(token)
            extracted.append(token)
        if len(extracted) >= max_terms:
            break
    return extracted


def derive_topic_terms(
    text: str,
    *,
    max_terms: int = 6,
    stopwords: Collection[str] | None = None,
    extra_exclude: Collection[str] | None = None,
) -> list[str]:
    exclude = set(GENERIC_RESEARCH_TOPIC_EXCLUDE_TOKENS)
    exclude.update(str(token).lower() for token in (extra_exclude or ()))
    return extract_research_terms(
        text,
        max_terms=max_terms,
        stopwords=stopwords,
        exclude=exclude,
    )


def text_is_opportunity_search(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in OPPORTUNITY_SEARCH_MARKERS)


def task_targets_discrete_opportunities(task: Any) -> bool:
    task_text = " ".join(
        [
            str(getattr(task, "name", "") or ""),
            str(getattr(task, "description", "") or ""),
        ]
    ).lower()
    return any(term in task_text for term in DISCRETE_OPPORTUNITY_TERMS)
