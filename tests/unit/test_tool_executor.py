from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from KestrelAI.agents.tool_executor import LangChainToolExecutor, _rank_search_hit


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_normalizes_hits_and_urls():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "Low Signal",
            "href": "https://reddit.com/r/test",
            "body": "forum discussion",
        },
        {
            "title": "Valid",
            "href": "https://nsf.gov/reu/program",
            "body": "body text",
        },
        {
            "title": "Invalid",
            "href": "not-a-url",
            "body": "ignored",
        },
    ]
    searxng.extract_text.return_value = (
        "machine learning fellowship content for students"
    )

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke("search_web", {"query": "nsf reu"})

    assert result["query"] == "nsf reu"
    assert len(result["hits"]) == 2
    assert result["hits"][0]["title"] == "Valid"
    assert result["hits"][0]["url"] == "https://nsf.gov/reu/program"
    assert result["hits"][0]["source_tier"] == "authoritative"
    assert result["hits"][0]["official_source"] is True
    assert result["hits"][0]["authority_score"] >= result["hits"][1]["authority_score"]
    assert "[URL_1]" in result["hits"][0]["snippet"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_reranks_authoritative_hits_before_fetch():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "AI/ML Scholarships 2026",
            "href": "https://scholarshipsandgrants.us/major/ai-ml/",
            "body": "commercial listing",
        },
        {
            "title": "AI for Air Traffic Management for Undergraduate Students | NASA",
            "href": "https://stemgateway.nasa.gov/public/s/course-offering/a0BSJ000000x6Cf2AI/ai-ml-for-air-traffic-management-for-undergraduate-students",
            "body": "ai ml opportunity for undergraduate students",
        },
    ]
    searxng.extract_text.return_value = (
        "machine learning fellowship content for students"
    )

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke(
        "search_web",
        {"query": "ai ml fellowship undergraduate site:.edu OR site:.gov"},
    )

    assert len(result["hits"]) == 1
    assert result["hits"][0]["url"].startswith("https://stemgateway.nasa.gov/")
    assert result["hits"][0]["official_source"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_enforces_site_filters():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "arXiv paper",
            "href": "http://arxiv.org/abs/1234.5678",
            "body": "paper content",
        },
        {
            "title": "NASA opportunity",
            "href": "https://stemgateway.nasa.gov/public/s/course-offering/a0BSJ000000x6Cf2AI/ai-ml-for-air-traffic-management-for-undergraduate-students",
            "body": "undergraduate students ai ml opportunity",
        },
    ]
    searxng.extract_text.return_value = (
        "machine learning fellowship content for students"
    )

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke(
        "search_web",
        {"query": "site:.gov ai/ml fellowship undergraduate"},
    )

    assert len(result["hits"]) == 1
    assert result["hits"][0]["url"].startswith("https://stemgateway.nasa.gov/")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_skips_publication_hits_for_opportunity_queries():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "Revitalization of an undergraduate physics program",
            "href": "http://arxiv.org/abs/physics/0004028v1",
            "body": "undergraduate program paper",
        },
        {
            "title": "AI for Good Foundation Fellowships - AI for Good Foundation",
            "href": "https://ai4good.org/what-we-do/fellowships/",
            "body": "machine learning fellowship program for students",
        },
    ]
    searxng.extract_text.return_value = (
        "machine learning fellowship content for students"
    )

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke(
        "search_web",
        {"query": "site:.org machine learning fellowship undergraduate"},
    )

    assert len(result["hits"]) == 1
    assert result["hits"][0]["url"] == "https://ai4good.org/what-we-do/fellowships/"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_skips_graduate_only_hits_for_undergraduate_queries():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "NSF Graduate Research Fellowships Program (GRFP)",
            "href": "https://www.nsfgrfp.org/",
            "body": "graduate fellowship for doctoral students in science and engineering",
        },
        {
            "title": "Undergraduate AI Research Fellowship",
            "href": "https://example.edu/ai-undergraduate-fellowship",
            "body": "artificial intelligence fellowship for undergraduate students",
        },
    ]
    searxng.extract_text.return_value = "extracted content"

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke(
        "search_web",
        {"query": "official ai machine learning undergraduate fellowship program"},
    )

    assert len(result["hits"]) == 1
    assert result["hits"][0]["url"] == "https://example.edu/ai-undergraduate-fellowship"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_skips_low_signal_hits_for_opportunity_queries():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "National Space Grant College and Fellowship Program",
            "href": "https://en.wikipedia.org/wiki/National_Space_Grant_College_and_Fellowship_Program",
            "body": "reference page",
        },
        {
            "title": "AI for Good Undergraduate Fellowship",
            "href": "https://ai4good.org/what-we-do/fellowships/",
            "body": "fellowship program for undergraduate students in artificial intelligence",
        },
        {
            "title": "AI/ML Scholarships 2026 — 20+ Verified Fellowships, Grants",
            "href": "https://scholarshipsandgrants.us/major/ai-ml/",
            "body": "directory of ai and machine learning grants and scholarships",
        },
        {
            "title": "16 AI Fellowships for Scientists, Programmers and Tech",
            "href": "https://www.profellow.com/fellowships/ai-fellowships/",
            "body": "directory of ai fellowships",
        },
        {
            "title": "Scholarships for AI and Machine Learning | 2026 UPDATED List",
            "href": "https://www.linkedin.com/pulse/scholarships-ai-machine-learning-2026-updated-list-abroadin-1cnle",
            "body": "linkedin listicle of ai scholarships",
        },
        {
            "title": "2026 Best Undergraduate Artificial Intelligence Programs",
            "href": "https://www.usnews.com/best-colleges/rankings/computer-science/artificial-intelligence",
            "body": "college rankings and best undergraduate artificial intelligence programs",
        },
    ]
    searxng.extract_text.return_value = "extracted content"

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke(
        "search_web",
        {"query": "official ai undergraduate fellowship program"},
    )

    assert len(result["hits"]) == 1
    assert result["hits"][0]["url"] == "https://ai4good.org/what-we-do/fellowships/"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_requires_explicit_student_fit_for_undergraduate_queries():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "Applied Machine Learning Fellowship",
            "href": "https://example.gov/applied-machine-learning-fellowship",
            "body": "artificial intelligence fellowship for advanced researchers",
        },
        {
            "title": "AI Research Program",
            "href": "https://example.edu/ai-research-program",
            "body": "machine learning program",
        },
    ]

    def extract_text(url: str) -> str:
        if "example.edu" in url:
            return "Artificial intelligence research program for undergraduate students with faculty mentorship."
        return "Artificial intelligence fellowship program for advanced research staff."

    searxng.extract_text.side_effect = extract_text

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke(
        "search_web",
        {"query": "official ai machine learning undergraduate fellowship program"},
    )

    assert len(result["hits"]) == 1
    assert result["hits"][0]["url"] == "https://example.edu/ai-research-program"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_requires_explicit_ai_ml_fit_for_ai_queries():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "Summer Undergraduate Research Fellowship",
            "href": "https://example.edu/surf",
            "body": "undergraduate research fellowship for students",
        },
        {
            "title": "AI Research Internship for Undergraduates",
            "href": "https://example.edu/ai-research-internship",
            "body": "artificial intelligence internship for undergraduate students",
        },
    ]

    def extract_text(url: str) -> str:
        if "ai-research-internship" in url:
            return "Machine learning and artificial intelligence internship for undergraduate students."
        return "Summer undergraduate research fellowship across multiple disciplines."

    searxng.extract_text.side_effect = extract_text

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke(
        "search_web",
        {"query": "official ai machine learning undergraduate internship fellowship"},
    )

    assert len(result["hits"]) == 1
    assert result["hits"][0]["url"] == "https://example.edu/ai-research-internship"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_rejects_generic_research_portals_for_ai_opportunity_queries():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "Home | Office of Undergraduate Research and Fellowships",
            "href": "https://uraf.harvard.edu/",
            "body": "undergraduate research and fellowships portal with multiple opportunities including artificial intelligence topics",
        },
        {
            "title": "AI Research Fellowship for Undergraduates",
            "href": "https://example.edu/ai-research-fellowship",
            "body": "artificial intelligence fellowship for undergraduate students",
        },
    ]

    def extract_text(url: str) -> str:
        if "uraf.harvard.edu" in url:
            return "Office of Undergraduate Research and Fellowships resources for students across many fields."
        return "Machine learning and artificial intelligence fellowship for undergraduate students."

    searxng.extract_text.side_effect = extract_text

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke(
        "search_web",
        {"query": "site:.edu machine learning ai research undergraduate fellowship"},
    )

    assert len(result["hits"]) == 1
    assert result["hits"][0]["url"] == "https://example.edu/ai-research-fellowship"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_rejects_generic_ai_academic_pages_without_opportunity_signal():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "AI Education | University of Houston",
            "href": "https://www.uh.edu/ai/education/",
            "body": "artificial intelligence education faculty colleges resources",
        },
        {
            "title": "AI and Machine Learning REU | Undergraduate Research",
            "href": "https://researchops.web.illinois.edu/opportunity/ai-and-machine-learning-reu",
            "body": "undergraduate research experience with application timeline and faculty mentorship",
        },
    ]

    def extract_text(url: str) -> str:
        if "uh.edu" in url:
            return (
                "Artificial intelligence education, faculty, colleges, and resources."
            )
        return "AI and machine learning research experience for undergraduates with an application deadline and program details."

    searxng.extract_text.side_effect = extract_text

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke(
        "search_web",
        {"query": "site:.edu machine learning ai research undergraduate us program"},
    )

    assert len(result["hits"]) == 1
    assert (
        result["hits"][0]["url"]
        == "https://researchops.web.illinois.edu/opportunity/ai-and-machine-learning-reu"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_rejects_degree_and_admissions_pages_for_opportunity_queries():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "Artificial Intelligence and Machine Learning Undergraduate Degree",
            "href": "https://drexel.edu/cci/academics/undergraduate-programs/bs-artificial-intelligence-machine-learning/",
            "body": "undergraduate degree admissions curriculum application deadlines",
        },
        {
            "title": "AI Research Summer Fellowship Program",
            "href": "https://www.stevens.edu/stevens-institute-for-artificial-intelligence/fellowship",
            "body": "fellowship for undergraduate researchers in artificial intelligence",
        },
    ]

    def extract_text(url: str) -> str:
        if "drexel.edu" in url:
            return "Undergraduate degree admissions, curriculum, and application deadlines."
        return "Fellowship for undergraduate students with artificial intelligence research projects."

    searxng.extract_text.side_effect = extract_text

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke(
        "search_web",
        {
            "query": "site:.edu ai machine learning undergraduate deadline application eligibility requirements"
        },
    )

    assert len(result["hits"]) == 1
    assert (
        result["hits"][0]["url"]
        == "https://www.stevens.edu/stevens-institute-for-artificial-intelligence/fellowship"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_rejects_student_affairs_fellowship_pages_with_only_snippet_level_ai_match():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "Daben Liu Research Fellowship - Admission & Student Engagement | USC Viterbi",
            "href": "https://viterbigradadmission.usc.edu/fellowships/daben-liu-research-fellowship/",
            "body": "AI and machine learning fellowships for students with application information",
        },
        {
            "title": "AI Research Summer Fellowship Program",
            "href": "https://www.stevens.edu/stevens-institute-for-artificial-intelligence/fellowship",
            "body": "fellowship for undergraduate researchers in artificial intelligence",
        },
    ]

    def extract_text(url: str) -> str:
        if "usc.edu" in url:
            return (
                "Admission and student engagement fellowship for incoming students. "
                "Selection and campus programming details are provided by student affairs."
            )
        return (
            "Artificial intelligence research fellowship for undergraduate students with "
            "application deadlines, eligibility, and faculty mentorship."
        )

    searxng.extract_text.side_effect = extract_text

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke(
        "search_web",
        {
            "query": "site:.edu ai machine learning undergraduate fellowship deadline eligibility"
        },
    )

    assert len(result["hits"]) == 1
    assert (
        result["hits"][0]["url"]
        == "https://www.stevens.edu/stevens-institute-for-artificial-intelligence/fellowship"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_rejects_indirect_funding_news_announcements_for_opportunity_queries():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "NCSA awarded funding to continue AI-focused summer program",
            "href": "https://www.eurekalert.org/news-releases/123456",
            "body": "press release about AI funding for a summer initiative",
        },
        {
            "title": "AI Research Internship for Undergraduates",
            "href": "https://example.edu/ai-research-internship",
            "body": "artificial intelligence internship for undergraduate students",
        },
    ]

    def extract_text(url: str) -> str:
        if "eurekalert.org" in url:
            return (
                "News release. Media contact. NCSA awarded funding to continue an AI-focused "
                "research program. This announcement does not include application instructions."
            )
        return (
            "Machine learning and artificial intelligence internship for undergraduate students "
            "with an application deadline and eligibility requirements."
        )

    searxng.extract_text.side_effect = extract_text

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke(
        "search_web",
        {"query": "official ai machine learning undergraduate fellowship program"},
    )

    assert len(result["hits"]) == 1
    assert result["hits"][0]["url"] == "https://example.edu/ai-research-internship"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_rejects_precollege_fellowships_for_undergraduate_queries():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "Astral Fellows - AI Research Fellowship for High School Students",
            "href": "https://www.astralfellows.org/",
            "body": "AI research fellowship for high school students",
        },
        {
            "title": "AI Research Summer Fellowship Program",
            "href": "https://www.stevens.edu/stevens-institute-for-artificial-intelligence/fellowship",
            "body": "fellowship for undergraduate researchers in artificial intelligence",
        },
    ]

    def extract_text(url: str) -> str:
        if "astralfellows.org" in url:
            return (
                "Artificial intelligence research fellowship for high school students "
                "with mentorship and summer programming."
            )
        return (
            "Artificial intelligence research fellowship for undergraduate students "
            "with application deadlines and faculty mentorship."
        )

    searxng.extract_text.side_effect = extract_text

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke(
        "search_web",
        {"query": "official ai machine learning undergraduate fellowship program"},
    )

    assert len(result["hits"]) == 1
    assert (
        result["hits"][0]["url"]
        == "https://www.stevens.edu/stevens-institute-for-artificial-intelligence/fellowship"
    )


@pytest.mark.unit
def test_rank_search_hit_penalizes_mixed_audience_results_for_undergraduate_queries():
    query = "official machine learning ai research undergraduate us fellowship"
    mixed_hit = {
        "title": "Astral Fellows - AI Research Fellowship for High School and Undergraduate Students",
        "href": "https://www.astralfellows.org/",
        "body": "AI research fellowship for high school, undergraduate, and early-career researchers.",
    }
    official_hit = {
        "title": "Research Experience for Undergraduates (REU) Combinatorics, Algorithms, and AI for Real Problems",
        "href": "https://www.cs.umd.edu/projects/reucaar/",
        "body": "Undergraduate AI research experience with stipend, application deadline, and summer program details.",
    }

    mixed_rank = _rank_search_hit(
        query,
        mixed_hit,
        {"authority_score": 3, "official_source": False, "source_tier": "trusted_org"},
    )
    official_rank = _rank_search_hit(
        query,
        official_hit,
        {"authority_score": 4, "official_source": True, "source_tier": "authoritative"},
    )

    assert official_rank > mixed_rank


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_rejects_institutional_funding_calls_for_undergraduate_opportunity_queries():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "Advancing Research at the intersection of Biology and Artificial Intelligence",
            "href": "https://www.nsf.gov/funding/opportunities/dcl-advancing-research-intersection-biology-artificial",
            "body": "NSF encourages the submission of proposals that advance biological research using AI/ML",
        },
        {
            "title": "AI and Machine Learning REU | Undergraduate Research Opportunities",
            "href": "https://researchops.web.illinois.edu/opportunity/ai-and-machine-learning-reu",
            "body": "undergraduate research experience with application timeline and faculty mentorship",
        },
    ]

    def extract_text(url: str) -> str:
        if "nsf.gov" in url:
            return (
                "NSF funding opportunity. The directorate encourages the submission of proposals. "
                "Proposal and award policies are available on Research.gov."
            )
        return (
            "AI and machine learning research experience for undergraduates with application "
            "deadlines, eligibility details, and program dates."
        )

    searxng.extract_text.side_effect = extract_text

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke(
        "search_web",
        {"query": "ai machine learning undergraduate grant official research"},
    )

    assert len(result["hits"]) == 1
    assert (
        result["hits"][0]["url"]
        == "https://researchops.web.illinois.edu/opportunity/ai-and-machine-learning-reu"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_tool_uses_manager_and_returns_payload():
    mcp_manager = Mock()
    mcp_manager.call_tool = AsyncMock(
        return_value=SimpleNamespace(success=True, data={"k": "v"}, error=None)
    )

    executor = LangChainToolExecutor(
        searxng_service=Mock(),
        url_flag_manager=Mock(),
        mcp_manager=mcp_manager,
        mcp_enabled=lambda: True,
    )

    result = await executor.ainvoke(
        "mcp_call",
        {"tool_name": "query_database", "tool_parameters": {"sql": "select 1"}},
    )

    assert result["success"] is True
    assert result["data"] == {"k": "v"}
    mcp_manager.call_tool.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_tool_respects_enabled_gate():
    mcp_manager = Mock()
    mcp_manager.call_tool = AsyncMock()

    executor = LangChainToolExecutor(
        searxng_service=Mock(),
        url_flag_manager=Mock(),
        mcp_manager=mcp_manager,
        mcp_enabled=lambda: False,
    )

    result = await executor.ainvoke(
        "mcp_call",
        {"tool_name": "query_database", "tool_parameters": {}},
    )

    assert result["success"] is False
    assert "not available" in result["error"].lower()
    mcp_manager.call_tool.assert_not_awaited()
