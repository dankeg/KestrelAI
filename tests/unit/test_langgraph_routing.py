from unittest.mock import AsyncMock, Mock, patch

import pytest

try:
    from KestrelAI.agents.base_agent import AgentState
    from KestrelAI.agents.research_orchestrator import ResearchOrchestrator
    from KestrelAI.agents.web_research_agent import WebResearchAgent
    from KestrelAI.graphs.schemas import ResearchActionPlan
    from KestrelAI.graphs.subtask_runner import (
        DiscoverySearchPathway,
        LangGraphSubtaskRunner,
    )
    from KestrelAI.shared.models import ResearchPlan, Subtask
except ImportError:
    from agents.base_agent import AgentState
    from agents.research_orchestrator import ResearchOrchestrator
    from agents.web_research_agent import WebResearchAgent
    from graphs.schemas import ResearchActionPlan
    from graphs.subtask_runner import DiscoverySearchPathway, LangGraphSubtaskRunner
    from shared.models import ResearchPlan, Subtask


@pytest.mark.unit
@pytest.mark.asyncio
async def test_web_research_agent_routes_to_langgraph_engine(mock_llm, mock_task):
    mock_memory = Mock()
    agent = WebResearchAgent("test-agent", mock_llm, mock_memory)
    runner = Mock()
    runner.run = AsyncMock(return_value="langgraph-result")
    agent.subtask_graph_runner = runner

    result = await agent.run_step(mock_task)
    assert result == "langgraph-result"
    runner.run.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_orchestrator_routes_to_langgraph_engine(mock_llm, mock_task):
    with patch("KestrelAI.agents.research_orchestrator.MemoryStore"):
        with patch("KestrelAI.agents.research_orchestrator.WebResearchAgent"):
            orchestrator = ResearchOrchestrator(
                [mock_task], mock_llm, profile="kestrel"
            )

    runner = Mock()
    runner.run = AsyncMock(return_value="orchestrator-langgraph-result")
    orchestrator.langgraph_runner = runner

    result = await orchestrator.next_action(mock_task)
    assert result == "orchestrator-langgraph-result"
    runner.run.assert_awaited_once_with(mock_task)


@pytest.mark.unit
def test_subtask_runner_sanitizes_instructional_queries(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "For each identified program, collect key program details including "
            "program name institution duration funding details research focus"
        )
    )
    runner._extract_keywords = LangGraphSubtaskRunner._extract_keywords
    runner._build_fallback_search_query = (
        LangGraphSubtaskRunner._build_fallback_search_query.__get__(runner)
    )

    agent_state = AgentState(task_id=mock_task.name)
    query = (
        "For each identified program, collect key program details including "
        "research current NSF REU in AI and machine learning"
    )
    sanitized = LangGraphSubtaskRunner._sanitize_search_query(
        runner,
        mock_task,
        agent_state,
        query,
    )

    assert "for each identified" not in sanitized.lower()
    assert "collect key" not in sanitized.lower()
    assert len(sanitized.split()) <= 10


@pytest.mark.unit
def test_subtask_runner_compacts_filler_heavy_query(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(subtask_description="NSF REU AI machine learning")
    runner._extract_keywords = LangGraphSubtaskRunner._extract_keywords
    runner._build_fallback_search_query = (
        LangGraphSubtaskRunner._build_fallback_search_query.__get__(runner)
    )

    agent_state = AgentState(task_id=mock_task.name)
    sanitized = LangGraphSubtaskRunner._sanitize_search_query(
        runner,
        mock_task,
        agent_state,
        "nsf reu explicitly to ai machine learning in",
    )

    assert "nsf" in sanitized
    assert "reu" in sanitized
    assert "machine" in sanitized
    assert "learning" in sanitized
    assert "for each" not in sanitized.lower()
    assert "collect" not in sanitized.lower()


@pytest.mark.unit
def test_subtask_runner_compacts_short_filler_query(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(subtask_description="NSF REU AI machine learning")
    runner._extract_keywords = LangGraphSubtaskRunner._extract_keywords
    runner._build_fallback_search_query = (
        LangGraphSubtaskRunner._build_fallback_search_query.__get__(runner)
    )

    agent_state = AgentState(task_id=mock_task.name)
    sanitized = LangGraphSubtaskRunner._sanitize_search_query(
        runner,
        mock_task,
        agent_state,
        "reu site nsf in ai machine",
    )

    assert "reu" in sanitized
    assert "nsf" in sanitized
    assert "ai" in sanitized
    assert "machine" in sanitized
    assert "site" not in sanitized
    assert " in " not in f" {sanitized} "
    assert " to " not in f" {sanitized} "


@pytest.mark.unit
def test_subtask_runner_fallback_query_compacts_filler_terms(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Currently identify NSF REU sites explicitly related to AI machine learning in 2024"
    )
    runner._extract_keywords = LangGraphSubtaskRunner._extract_keywords
    runner._extract_recent_title_terms = (
        LangGraphSubtaskRunner._extract_recent_title_terms.__get__(runner)
    )
    runner._choose_missing_facets = (
        LangGraphSubtaskRunner._choose_missing_facets.__get__(runner)
    )
    runner._compact_keyword_query = LangGraphSubtaskRunner._compact_keyword_query
    runner._evidence_facets = LangGraphSubtaskRunner._evidence_facets

    agent_state = AgentState(task_id=mock_task.name)
    agent_state.current_focus = "currently nsf reu sites to ai machine in"

    query = LangGraphSubtaskRunner._build_fallback_search_query(
        runner,
        mock_task,
        agent_state,
    )

    assert "nsf" in query
    assert "reu" in query
    assert "machine" in query
    assert "currently" not in query
    assert "explicitly" not in query
    assert " to " not in f" {query} "
    assert " in " not in f" {query} "


@pytest.mark.unit
def test_subtask_runner_forced_query_does_not_fabricate_angle_suffix(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Find ML fellowship opportunities with deadlines and eligibility."
    )
    runner._extract_keywords = LangGraphSubtaskRunner._extract_keywords
    runner._build_fallback_search_query = Mock(
        return_value="ml fellowships deadline application grants"
    )
    runner._compact_keyword_query = LangGraphSubtaskRunner._compact_keyword_query
    runner._canonicalize_query = LangGraphSubtaskRunner._canonicalize_query.__get__(
        runner
    )

    agent_state = AgentState(task_id=mock_task.name)
    agent_state.queries.update(
        {
            "ml fellowships deadline application grants",
            "ml fellowships deadline application grants official source",
            "ml fellowships deadline application grants eligibility requirements",
            "ml fellowships deadline application grants deadlines application",
        }
    )

    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        mock_task,
        agent_state,
    )

    assert "angle " not in query.lower()
    assert query


@pytest.mark.unit
def test_subtask_runner_structured_query_uses_missing_facets(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Verify deadlines and eligibility criteria for each program.",
        success_criteria="Confirm deadline and eligibility requirements on official pages.",
    )
    runner._extract_keywords = LangGraphSubtaskRunner._extract_keywords
    runner._extract_recent_title_terms = (
        LangGraphSubtaskRunner._extract_recent_title_terms.__get__(runner)
    )
    runner._choose_missing_facets = (
        LangGraphSubtaskRunner._choose_missing_facets.__get__(runner)
    )
    runner._compact_keyword_query = LangGraphSubtaskRunner._compact_keyword_query
    runner._evidence_facets = LangGraphSubtaskRunner._evidence_facets

    agent_state = AgentState(task_id=mock_task.name)
    agent_state.search_history = [
        {
            "query": "brown ai reu",
            "results": [
                {
                    "title": "Brown AI for Computational Creativity REU",
                    "official_source": True,
                    "authority_score": 4,
                }
            ],
        }
    ]

    query = LangGraphSubtaskRunner._build_structured_search_query(
        runner,
        mock_task,
        agent_state,
    )

    assert "brown" in query
    assert ("deadline" in query) or ("application" in query)


@pytest.mark.unit
def test_subtask_runner_structured_query_preserves_domain_anchors_over_irrelevant_titles(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Identify NSF REU opportunities relevant to AI and machine learning.",
        success_criteria="Find official opportunities and verify deadlines.",
    )
    runner._extract_keywords = LangGraphSubtaskRunner._extract_keywords
    runner._extract_recent_title_terms = (
        LangGraphSubtaskRunner._extract_recent_title_terms.__get__(runner)
    )
    runner._extract_recent_query_terms = (
        LangGraphSubtaskRunner._extract_recent_query_terms.__get__(runner)
    )
    runner._choose_missing_facets = (
        LangGraphSubtaskRunner._choose_missing_facets.__get__(runner)
    )
    runner._compact_keyword_query = LangGraphSubtaskRunner._compact_keyword_query
    runner._evidence_facets = LangGraphSubtaskRunner._evidence_facets

    agent_state = AgentState(task_id=mock_task.name)
    agent_state.search_history = [
        {
            "query": "nsf reu ai machine deadline application eligibility",
            "results": [
                {
                    "title": "REU in Combinatorics - Experience for Undergraduates",
                    "official_source": True,
                    "authority_score": 4,
                }
            ],
        }
    ]

    query = LangGraphSubtaskRunner._build_structured_search_query(
        runner,
        mock_task,
        agent_state,
    )

    assert "nsf" in query
    assert "reu" in query
    assert "combinatorics" not in query


@pytest.mark.unit
def test_subtask_runner_structured_query_uses_orchestrator_guidance_pivots(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Find open machine learning fellowship opportunities for students.",
        success_criteria="Find reputable opportunities with official pages.",
        orchestrator_guidance=(
            "Current line of inquiry is stagnating; pivot to a different angle, source type, or constraint. "
            "Continue the current subtask, but pivot away from low-yield queries toward official sources, "
            "primary organizations, and directory/listing pages. "
            "Avoid repeating recent queries: ml fellowships find open deadline application grants; "
            "ml fellowships find open grants deadline application; "
            "ml fellowships find open application deadline grants."
        ),
    )

    agent_state = AgentState(task_id=mock_task.name)
    agent_state.search_history = [
        {"query": "ml fellowships find open deadline application grants"},
        {"query": "ml fellowships find open grants deadline application"},
        {"query": "ml fellowships find open application deadline grants"},
    ]

    query = LangGraphSubtaskRunner._build_structured_search_query(
        runner,
        mock_task,
        agent_state,
    )

    assert any(
        token in query
        for token in ("site:.edu", "site:.gov", "site:.org", "official", "primary")
    )
    assert any(
        token in query for token in ("organization", "research", "test", "description")
    )
    assert "deadline" not in query
    assert "application" not in query
    assert "grants" not in query


@pytest.mark.unit
def test_subtask_runner_verification_query_targets_recent_named_lead(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Verify the current status, deadlines, and eligibility details for the identified fellowships.",
        success_criteria="Confirm deadlines, eligibility criteria, and current status from official pages.",
        orchestrator_guidance="",
    )

    agent_state = AgentState(task_id=mock_task.name)
    agent_state.search_history = [
        {
            "query": "site:.gov machine learning fellowship undergraduate",
            "results": [
                {
                    "title": "Applied Machine Learning Fellowship | Los Alamos National Laboratory",
                    "official_source": True,
                    "authority_score": 4,
                }
            ],
        }
    ]

    runner._get_current_subtask_mode = Mock(return_value="verification")

    query = LangGraphSubtaskRunner._build_structured_search_query(
        runner,
        mock_task,
        agent_state,
    )

    assert "applied" in query
    assert "fellowship" in query
    assert ("deadline" in query) or ("application" in query)
    assert ("eligibility" in query) or ("requirements" in query)


@pytest.mark.unit
def test_subtask_runner_forced_verification_query_targets_recent_named_lead(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Verify the current status, deadlines, and eligibility details for the identified fellowships.",
        success_criteria="Confirm deadlines, eligibility criteria, and current status from official pages.",
        orchestrator_guidance="",
    )
    runner._build_fallback_search_query = (
        LangGraphSubtaskRunner._build_fallback_search_query.__get__(runner)
    )

    agent_state = AgentState(task_id=mock_task.name)
    agent_state.search_history = [
        {
            "query": "site:.gov machine learning fellowship undergraduate",
            "results": [
                {
                    "title": "Applied Machine Learning Fellowship | Los Alamos National Laboratory",
                    "official_source": True,
                    "authority_score": 4,
                }
            ],
        }
    ]
    runner._get_current_subtask_mode = Mock(return_value="verification")

    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        mock_task,
        agent_state,
    )

    assert "applied" in query
    assert "fellowship" in query
    assert ("deadline" in query) or ("application" in query)


@pytest.mark.unit
def test_subtask_runner_verification_query_uses_previous_findings_when_local_history_empty(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Verify the current status, deadlines, and eligibility details for the identified fellowships.",
        success_criteria="Confirm deadlines, eligibility criteria, and current status from official pages.",
        orchestrator_guidance="",
        previous_findings=(
            "[CHECKPOINT] Focus: General research\n"
            "- Official lead: Applied Machine Learning Fellowship | Los Alamos National Laboratory (lanl.gov)\n"
            "- Official lead: Student Awards and Fellowships - Machine Learning - CMU (ml.cmu.edu)\n"
        ),
    )
    runner._get_current_subtask_mode = Mock(return_value="verification")

    agent_state = AgentState(task_id=mock_task.name)

    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        mock_task,
        agent_state,
    )

    assert ("applied" in query) or ("machine" in query)
    assert "fellowship" in query
    assert ("deadline" in query) or ("application" in query)


@pytest.mark.unit
def test_subtask_runner_forced_verification_query_keeps_opportunity_anchor_without_local_lead(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Verify candidate opportunities and program requirements directly from official program "
            "and sponsoring organization sources."
        ),
        success_criteria=(
            "Confirm which opportunities are real, current, and relevant by checking official "
            "program pages for deadlines, eligibility, and research focus."
        ),
        orchestrator_guidance=(
            "Broaden the search to adjacent in-scope opportunity types and prioritize official pages."
        ),
        previous_findings="",
    )
    runner._build_fallback_search_query = (
        LangGraphSubtaskRunner._build_fallback_search_query.__get__(runner)
    )
    runner._get_current_subtask_mode = Mock(return_value="verification")

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)

    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert any(
        token in query for token in ("fellowship", "grant", "program", "funding")
    )
    assert any(
        token in query
        for token in ("deadline", "application", "eligibility", "requirements")
    )
    assert any(
        token in query for token in ("site:.edu", "site:.org", "site:.gov", "official")
    )


@pytest.mark.unit
def test_subtask_runner_sanitize_verification_query_rebuilds_generic_detail_only_query(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Verify candidate opportunities and program requirements directly from official program "
            "and sponsoring organization sources."
        ),
        success_criteria=(
            "Confirm which opportunities are real, current, and relevant by checking official "
            "program pages for deadlines, eligibility, and research focus."
        ),
        orchestrator_guidance="",
        previous_findings="",
    )
    runner._build_fallback_search_query = (
        LangGraphSubtaskRunner._build_fallback_search_query.__get__(runner)
    )
    runner._get_current_subtask_mode = Mock(return_value="verification")

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)

    query = LangGraphSubtaskRunner._sanitize_search_query(
        runner,
        constrained_task,
        agent_state,
        "site:.edu machine learning ai research deadline application eligibility requirements",
    )

    assert any(
        token in query for token in ("fellowship", "grant", "program", "funding")
    )
    assert any(
        token in query
        for token in ("deadline", "application", "eligibility", "requirements")
    )


@pytest.mark.unit
def test_subtask_runner_structured_query_ignores_task_run_metadata_in_name(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Find official AI/ML fellowship opportunities for undergraduates.",
        success_criteria="Find reputable opportunities with official pages.",
        orchestrator_guidance="",
    )

    noisy_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships Gemma Run",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate students "
                "in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=noisy_task.name)
    query = LangGraphSubtaskRunner._build_structured_search_query(
        runner,
        noisy_task,
        agent_state,
    )

    assert "gemma" not in query
    assert "run" not in query
    assert "fellowships" in query or "fellowship" in query
    assert "undergraduate" in query or "research" in query


@pytest.mark.unit
def test_subtask_runner_structured_query_cleans_task_name_when_description_missing(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="",
        success_criteria="Find reputable opportunities with official pages.",
        orchestrator_guidance="",
    )

    noisy_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships Gemma Rerun Validation",
            "description": "",
        }
    )

    agent_state = AgentState(task_id=noisy_task.name)
    query = LangGraphSubtaskRunner._build_structured_search_query(
        runner,
        noisy_task,
        agent_state,
    )

    assert "gemma" not in query
    assert "rerun" not in query
    assert "validation" not in query
    assert "ml" in query or "fellowships" in query


@pytest.mark.unit
def test_subtask_runner_structured_query_preserves_key_task_constraints(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Find official AI/ML fellowship opportunities for senior undergraduates.",
        success_criteria="Find official opportunities with enough detail to verify research fit.",
        orchestrator_guidance="",
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    query = LangGraphSubtaskRunner._build_structured_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert "undergraduate" in query
    assert "find" not in query
    assert "support" not in query
    assert "ai/ml" not in query
    assert any(
        token in query
        for token in ("ai", "ml", "machine", "learning", "artificial", "intelligence")
    )


@pytest.mark.unit
def test_subtask_runner_structured_query_drops_hyphenated_guidance_boilerplate(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Find official AI/ML fellowship opportunities for undergraduates.",
        success_criteria="Find official opportunities with enough detail to verify research fit.",
        orchestrator_guidance=(
            "Continue gathering evidence for the current subtask using diversified, "
            "success-criteria-aligned queries. Pivot toward official sources and "
            "primary organizations."
        ),
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    query = LangGraphSubtaskRunner._build_structured_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert "success" not in query
    assert "criteria" not in query
    assert "aligned" not in query
    assert "queries" not in query
    assert "undergraduate" in query


@pytest.mark.unit
def test_subtask_runner_sanitize_query_rebuilds_repetitive_query_when_guidance_forces_pivot(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Find open machine learning fellowship opportunities for students.",
        success_criteria="Find reputable opportunities with official pages.",
        orchestrator_guidance=(
            "Pivot to a different angle. Focus on official sources, primary organizations, "
            "and directory/listing pages. Avoid repeating recent queries: "
            "ml fellowships find open deadline application grants; "
            "ml fellowships find open grants deadline application; "
            "ml fellowships find open application deadline grants."
        ),
    )

    agent_state = AgentState(task_id=mock_task.name)
    agent_state.search_history = [
        {"query": "ml fellowships find open deadline application grants"},
        {"query": "ml fellowships find open grants deadline application"},
        {"query": "ml fellowships find open application deadline grants"},
    ]

    sanitized = LangGraphSubtaskRunner._sanitize_search_query(
        runner,
        mock_task,
        agent_state,
        "ml fellowships find open deadline application grants program",
    )

    assert any(
        token in sanitized
        for token in (
            "site:.edu",
            "site:.gov",
            "site:.org",
            "primary",
            "organization",
            "official",
        )
    )
    assert "deadline" not in sanitized
    assert "application" not in sanitized


@pytest.mark.unit
def test_subtask_runner_forced_query_prioritizes_uncovered_discovery_pathway(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Find open machine learning fellowship opportunities for students.",
        success_criteria="Find reputable opportunities with official pages.",
        orchestrator_guidance="Pivot to different source classes and broader discovery angles.",
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    agent_state.search_pathways = [
        {
            "id": "pathway_1",
            "label": "official fellowship",
            "source_terms": ["official"],
            "focus_terms": ["fellowship"],
            "evidence_terms": ["deadline"],
            "attempt_count": 1,
            "hit_count": 1,
        },
        {
            "id": "pathway_2",
            "label": "organization lab path",
            "source_terms": ["organization", "lab", "site:.edu"],
            "focus_terms": ["research", "internship"],
            "evidence_terms": [],
            "attempt_count": 0,
            "hit_count": 0,
        },
    ]

    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert "lab" in query
    assert "site:.edu" in query
    assert "senior" not in query
    assert agent_state.pathway_query_map
    assert (
        agent_state.pathway_query_map.get(
            LangGraphSubtaskRunner._canonicalize_query(runner, query)
        )
        == "pathway_2"
    )


@pytest.mark.unit
def test_subtask_runner_normalize_search_plan_prefers_uncovered_pathway(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Find open machine learning fellowship opportunities for students.",
        success_criteria="Find reputable opportunities with official pages.",
        orchestrator_guidance="Pivot to different source classes and broader discovery angles.",
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    agent_state.search_pathways = [
        {
            "id": "pathway_1",
            "label": "official fellowship",
            "source_terms": ["official"],
            "focus_terms": ["fellowship"],
            "evidence_terms": ["deadline"],
            "attempt_count": 1,
            "hit_count": 1,
        },
        {
            "id": "pathway_2",
            "label": "organization lab path",
            "source_terms": ["organization", "lab", "site:.edu"],
            "focus_terms": ["research", "internship"],
            "evidence_terms": [],
            "attempt_count": 0,
            "hit_count": 0,
        },
    ]

    plan = ResearchActionPlan(
        action="search", query="official ai ml fellowship undergraduate us"
    )

    normalized = LangGraphSubtaskRunner._normalize_action_plan(
        runner,
        constrained_task,
        agent_state,
        plan,
    )

    assert "lab" in normalized.query
    assert "site:.edu" in normalized.query
    assert "senior" not in normalized.query
    assert (
        agent_state.pathway_query_map.get(
            LangGraphSubtaskRunner._canonicalize_query(runner, normalized.query)
        )
        == "pathway_2"
    )


@pytest.mark.unit
def test_subtask_runner_discovery_pathway_context_key_ignores_mechanical_guidance_noise(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Discovery: broaden to lab, organization, and directory sources.",
        success_criteria="Find official or primary sources across multiple source classes.",
        orchestrator_guidance=(
            "Broaden to lab, organization, and directory sources. "
            "Cover at least 1 more discovery pathway or source-class routes before advancing. "
            "Avoid repeating recent queries: ai fellowship lab; ai fellowship directory."
        ),
    )

    first_key = LangGraphSubtaskRunner._discovery_pathway_context_key(
        runner,
        mock_task,
    )

    runner.agent.config.orchestrator_guidance = (
        "Broaden to lab, organization, and directory sources. "
        "Cover at least 2 more discovery pathway or source-class routes before advancing. "
        "Avoid repeating recent queries: ai fellowship university; ai fellowship organization."
    )

    second_key = LangGraphSubtaskRunner._discovery_pathway_context_key(
        runner,
        mock_task,
    )

    assert first_key == second_key
    assert "cover at least" not in first_key.lower()
    assert "avoid repeating recent queries" not in first_key.lower()


@pytest.mark.unit
def test_subtask_runner_planning_guidance_strips_authoritative_source_counter_noise(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Discovery: broaden to lab, organization, and directory sources.",
        success_criteria="Find official or primary sources across multiple source classes.",
        orchestrator_guidance=(
            "Broaden the search to adjacent in-scope opportunity types and prioritize official pages, "
            "primary organizations, and directories instead of repeating narrow lead-specific queries. "
            "Find at least 1 more authoritative sources (.gov, .edu, official program pages, or primary organizations)."
        ),
    )

    guidance = LangGraphSubtaskRunner._planning_guidance_text(runner)

    assert "Broaden the search" in guidance
    assert "find at least 1 more authoritative sources" not in guidance.lower()
    assert ".gov" not in guidance.lower()
    assert ".edu" not in guidance.lower()


@pytest.mark.unit
def test_subtask_runner_rank_discovery_pathways_respects_orchestrator_preference(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="Discovery: map multiple source classes.",
        success_criteria="Cover official and directory sources.",
        orchestrator_guidance="Broaden to multiple source classes.",
        orchestrator_control_hints={
            "discovery_mode": "pathway_first",
            "preferred_pathway_ids": ["pathway_2", "pathway_1"],
        },
    )

    agent_state = AgentState(task_id=mock_task.name)
    agent_state.search_pathways = [
        {
            "id": "pathway_1",
            "label": "official path",
            "source_terms": ["official"],
            "focus_terms": ["research"],
            "evidence_terms": [],
            "attempt_count": 0,
            "hit_count": 0,
        },
        {
            "id": "pathway_2",
            "label": "alternate official path",
            "source_terms": ["official"],
            "focus_terms": ["research"],
            "evidence_terms": [],
            "attempt_count": 0,
            "hit_count": 0,
        },
    ]

    ranked = LangGraphSubtaskRunner._rank_discovery_pathways(
        runner,
        mock_task,
        agent_state,
    )

    assert [pathway["id"] for pathway in ranked[:2]] == ["pathway_2", "pathway_1"]


@pytest.mark.unit
def test_subtask_runner_serialized_pathways_normalize_mixed_source_terms(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Discovery: identify distinct publicly available AI/ML research opportunities "
            "across universities, organizations, labs, and directories."
        ),
        success_criteria="Cover multiple source classes and surface official or primary pages.",
        orchestrator_guidance="Broaden to lab, organization, and directory sources.",
    )
    task = mock_task.model_copy(
        update={
            "name": "ML Opportunities",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    serialized = LangGraphSubtaskRunner._serialize_discovery_pathways(
        runner,
        task,
        [
            DiscoverySearchPathway(
                label="Government & Nonprofit AI/ML Funding Programs",
                source_terms=["government", "nonprofit", "site:.gov"],
                focus_terms=["ai", "ml", "undergrad", "scholarship"],
                evidence_terms=["deadline"],
            ),
            DiscoverySearchPathway(
                label="Academic & Industry Preprints & Conference Funding",
                source_terms=[],
                focus_terms=["ai", "ml", "undergrad"],
                evidence_terms=[],
            ),
        ],
        task_family="opportunity",
    )

    assert len(serialized) == 1
    assert serialized[0]["source_terms"] == ["government", "site:.gov"]
    assert serialized[0]["focus_terms"]


@pytest.mark.unit
def test_subtask_runner_merges_generated_and_fallback_pathways_for_coverage(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Discovery: identify distinct publicly available AI/ML research opportunities "
            "across universities, organizations, labs, and directories."
        ),
        success_criteria="Cover multiple source classes and surface official or primary pages.",
        orchestrator_guidance="Broaden to lab, organization, and directory sources.",
    )
    task = mock_task.model_copy(
        update={
            "name": "ML Opportunities",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )
    agent_state = AgentState(task_id=task.name)

    generated = LangGraphSubtaskRunner._serialize_discovery_pathways(
        runner,
        task,
        [
            DiscoverySearchPathway(
                label="Government & Nonprofit AI/ML Funding Programs",
                source_terms=["government", "nonprofit", "site:.gov"],
                focus_terms=["ai", "ml", "undergrad", "scholarship"],
                evidence_terms=["deadline"],
            ),
            DiscoverySearchPathway(
                label="Empty Source Path",
                source_terms=[],
                focus_terms=["ai", "ml", "undergrad"],
                evidence_terms=[],
            ),
        ],
        task_family="opportunity",
    )
    fallback = LangGraphSubtaskRunner._fallback_discovery_pathways(
        runner,
        task,
        agent_state,
    )

    merged = LangGraphSubtaskRunner._merge_discovery_pathways(
        runner,
        generated=generated,
        fallback=fallback,
        task_family="opportunity",
    )

    assert len(merged) >= 3
    assert all(pathway["source_terms"] for pathway in merged)
    assert all(pathway["focus_terms"] for pathway in merged)
    assert (
        len(
            {
                next(
                    (
                        token
                        for token in pathway["source_terms"]
                        if not token.startswith("site:.")
                    ),
                    pathway["source_terms"][0],
                )
                for pathway in merged
            }
        )
        >= 2
    )


@pytest.mark.unit
def test_subtask_runner_paper_discovery_query_is_not_opportunity_shaped(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Discovery: find recent papers, proceedings, benchmark papers, and surveys "
            "on evaluation of retrieval-augmented generation agents."
        ),
        success_criteria=(
            "Identify strong paper leads from conference proceedings, arXiv, publisher pages, "
            "lab pages, or official project pages."
        ),
        orchestrator_guidance=(
            "Broaden to proceedings, benchmark papers, survey papers, project pages, and lab pages. "
            "Avoid repeating recent queries: rag eval agents benchmark; rag evaluation agent benchmark."
        ),
    )

    task = mock_task.model_copy(
        update={
            "name": "RAG Eval Papers",
            "description": (
                "Find strong recent academic papers and conference proceedings on evaluation methods "
                "for retrieval-augmented generation agents."
            ),
        }
    )

    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        task,
        AgentState(task_id=task.name),
    )

    assert "opportunities list" not in query
    assert any(
        token in query for token in ("paper", "proceedings", "benchmark", "evaluation")
    )


@pytest.mark.unit
def test_subtask_runner_ecosystem_discovery_query_prefers_repo_doc_artifacts(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Discovery: find open-source local-first AI agent frameworks, maintainers, organizations, "
            "and project directories."
        ),
        success_criteria=(
            "Identify strong official repos, docs pages, organization pages, and ecosystem maps."
        ),
        orchestrator_guidance=(
            "Pivot away from repeated framework-name searches. Broaden to organization pages, repo indexes, "
            "official docs, and ecosystem directories."
        ),
    )

    task = mock_task.model_copy(
        update={
            "name": "Local Agent Ecosystem",
            "description": (
                "Map the open-source local-first AI agent ecosystem, including major frameworks, repos, "
                "maintainers, and active organizations."
            ),
        }
    )

    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        task,
        AgentState(task_id=task.name),
    )

    assert "opportunities list" not in query
    assert any(
        token in query
        for token in (
            "framework",
            "repository",
            "documentation",
            "project",
            "organization",
        )
    )


@pytest.mark.unit
def test_subtask_runner_structured_query_keeps_safe_scope_expansions_from_pivot_feedback(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Identify currently open fellowships, grants, programs, or funding opportunities "
            "relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance=(
            "Current line of inquiry is stagnating; pivot to a different angle. "
            "Expand the search to include grants, scholarships, research awards, and programs. "
            "Specifically search for opportunities mentioning senior undergraduate or rising senior. "
            "Try IEEE, ACM, Stanford, and MIT as well."
        ),
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    query = LangGraphSubtaskRunner._build_structured_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert any(
        token in query for token in ("fellowship", "scholarship", "award", "program")
    )
    assert "undergraduate" in query
    assert "ieee" not in query
    assert "acm" not in query
    assert "stanford" not in query
    assert "mit" not in query


@pytest.mark.unit
def test_subtask_runner_source_variants_ignore_recent_query_suffix_text(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Identify currently open fellowships, grants, programs, or funding opportunities "
            "relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance=(
            "Broaden the search to adjacent in-scope fields. "
            "Avoid repeating recent queries: site:.edu machine learning fellowship undergraduate; "
            "site:.gov machine learning fellowship undergraduate; "
            "nsf site:.gov machine learning fellowship undergraduate"
        ),
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    variants = LangGraphSubtaskRunner._subtask_source_variants(
        runner,
        constrained_task,
        agent_state,
    )

    assert all("nsf" not in variant for variant in variants)
    assert any("site:.gov" in variant for variant in variants)


@pytest.mark.unit
def test_subtask_runner_choose_missing_facets_prefers_opportunity_types_for_discovery(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Identify currently open fellowships, grants, programs, or funding opportunities "
            "relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance="",
    )
    runner._get_current_subtask_mode = Mock(return_value="discovery")

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    facets = LangGraphSubtaskRunner._choose_missing_facets(
        runner,
        constrained_task,
        agent_state,
    )

    assert "deadline" not in facets
    assert "application" not in facets
    assert any(token in facets for token in ("grant", "program", "fellowship"))


@pytest.mark.unit
def test_subtask_runner_discovery_expansion_pivots_to_adjacent_opportunity_types_after_core_terms_used(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Identify currently open fellowships, grants, programs, or funding opportunities "
            "relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance="",
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    agent_state.queries = {
        "official ai machine learning fellowship grant program undergraduate"
    }

    terms = LangGraphSubtaskRunner._discovery_expansion_terms(
        runner,
        constrained_task,
        agent_state,
        max_terms=4,
    )

    assert any(token in terms for token in ("research", "internship", "summer", "reu"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("subtask_description", "success_criteria", "expected_mode"),
    [
        (
            "Discovery: Identify currently open fellowships and grants from official sources.",
            "List multiple distinct candidate opportunities from official program pages.",
            "discovery",
        ),
        (
            "Authoritative Verification: Verify the accuracy and currency of the opportunities.",
            "Confirm trustworthiness with evidence from About Us pages and editorial policies.",
            "verification",
        ),
        (
            "Comparison: Compare the identified opportunities based on deadline and funding.",
            "Provide a structured comparison table.",
            "comparison",
        ),
        (
            "Synthesis: Compile a consolidated list of verified opportunities.",
            "Produce a final curated list in report form.",
            "synthesis",
        ),
    ],
)
def test_subtask_runner_infers_subtask_mode_from_config_without_orchestrator(
    mock_task,
    subtask_description,
    success_criteria,
    expected_mode,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=subtask_description,
        success_criteria=success_criteria,
        orchestrator_guidance="",
    )

    mode = LangGraphSubtaskRunner._get_current_subtask_mode(runner, mock_task)

    assert mode == expected_mode


@pytest.mark.unit
def test_subtask_runner_reads_enum_subtask_type_from_orchestrator_state(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description="",
        success_criteria="",
        orchestrator_guidance="",
    )
    orchestrator = Mock()
    task_state = Mock()
    task_state.subtask_index = 0
    task_state.research_plan = ResearchPlan(
        restated_task="Test task",
        subtasks=[
            Subtask(
                order=1,
                description="Discovery: Identify opportunities.",
                success_criteria="List distinct opportunities.",
                subtask_type="discovery",
            )
        ],
    )
    orchestrator.task_states = {mock_task.name: task_state}
    runner.agent.orchestrator = orchestrator

    mode = LangGraphSubtaskRunner._get_current_subtask_mode(runner, mock_task)

    assert mode == "discovery"


@pytest.mark.unit
def test_subtask_runner_forced_query_broadens_discovery_after_initial_core_query(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Discovery: Search for university-specific AI/ML scholarship and fellowship programs. "
            "Prioritize official pages and primary organizations."
        ),
        success_criteria=(
            "A list of at least 5 university programs, each with details on eligibility, "
            "deadlines, and funding amounts."
        ),
        orchestrator_guidance=(
            "Continue gathering evidence using diversified queries. Pivot to primary "
            "organizations and broader opportunity families."
        ),
    )
    runner._get_current_subtask_mode = Mock(return_value="discovery")
    runner._build_fallback_search_query = Mock(
        return_value="official ai machine learning fellowship grant program undergraduate"
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    agent_state.queries = {
        "official ai machine learning fellowship grant program undergraduate"
    }

    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert (
        query != "official ai machine learning fellowship grant program undergraduate"
    )
    assert "undergraduate" in query
    assert any(
        token in query
        for token in ("official", "organization", "site:.edu", "site:.org", "site:.gov")
    )


@pytest.mark.unit
def test_subtask_runner_structured_query_bootstraps_with_broad_official_query(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Identify currently open fellowships, grants, programs, or funding opportunities "
            "relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance="",
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    query = LangGraphSubtaskRunner._build_structured_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert any(site in query for site in ("site:.org", "site:.edu", "site:.gov"))
    assert any(
        token in query
        for token in ("official", "organization", "site:.edu", "site:.org", "site:.gov")
    )
    assert "undergraduate" in query


@pytest.mark.unit
def test_subtask_runner_sanitize_query_rejects_low_scope_fragment(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Identify distinct NSF, university, and nonprofit program opportunities "
            "relevant to the task from official sources."
        ),
        success_criteria="Find official opportunities with enough detail to verify research fit.",
        orchestrator_guidance="",
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    sanitized = LangGraphSubtaskRunner._sanitize_search_query(
        runner,
        constrained_task,
        agent_state,
        "distinct nsf university",
    )

    assert "distinct" not in sanitized
    assert any(
        token in sanitized
        for token in ("fellowship", "fellowships", "grants", "research", "ai", "ml")
    )


@pytest.mark.unit
def test_subtask_runner_forced_query_uses_source_class_variants_with_task_scope(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Identify distinct NSF, university, and nonprofit program opportunities "
            "relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct opportunities supported by official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance="Pivot toward official sources.",
    )
    runner._build_fallback_search_query = Mock(
        return_value="grants fellowships ai ml research senior undergraduate us official"
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)

    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert "distinct" not in query
    assert "undergraduate" in query
    assert any(
        token in query for token in ("official", "site:.gov", "site:.edu", "site:.org")
    )
    assert any(
        token in query
        for token in (
            "ai",
            "ml",
            "machine",
            "learning",
            "fellowship",
            "grant",
            "fellowships",
            "grants",
        )
    )


@pytest.mark.unit
def test_subtask_runner_forced_query_derives_official_domain_filters_from_success_criteria(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Identify currently open fellowships, grants, programs, or funding "
            "opportunities relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance="",
    )
    runner._build_fallback_search_query = Mock(
        return_value="grants fellowships ai ml research senior undergraduate us official"
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert any(
        token in query for token in ("official", "site:.gov", "site:.edu", "site:.org")
    )
    assert "undergraduate" in query
    assert any(
        token in query
        for token in (
            "ai",
            "ml",
            "machine",
            "learning",
            "artificial",
            "intelligence",
            "research",
            "fellowship",
            "fellowships",
            "grant",
            "grants",
        )
    )


@pytest.mark.unit
def test_subtask_runner_structured_query_honors_aggregator_discovery_subtask(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Discovery: Search for national and regional scholarship/fellowship aggregators "
            "that list AI/ML opportunities for undergraduates."
        ),
        success_criteria=(
            "A list of reputable scholarship/fellowship aggregators with links to their "
            "sites and filtering capabilities."
        ),
        orchestrator_guidance="",
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    query = LangGraphSubtaskRunner._build_structured_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert any(
        token in query for token in ("directory", "listing", "organization", "official")
    )
    assert "undergraduate" in query or "student" in query
    assert any(
        token in query
        for token in ("machine", "learning", "ai", "ml", "artificial", "intelligence")
    )


@pytest.mark.unit
def test_subtask_runner_forced_query_honors_aggregator_discovery_subtask(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Discovery: Search for national and regional scholarship/fellowship aggregators "
            "that list AI/ML opportunities for undergraduates."
        ),
        success_criteria=(
            "A list of reputable scholarship/fellowship aggregators with links to their "
            "sites and filtering capabilities."
        ),
        orchestrator_guidance="",
    )
    runner._build_fallback_search_query = Mock(
        return_value="machine learning fellowship undergraduate"
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert any(
        token in query for token in ("directory", "listing", "organization", "official")
    )
    assert "undergraduate" in query or "student" in query


@pytest.mark.unit
def test_subtask_runner_forced_query_preserves_undergraduate_and_ai_scope_terms(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Identify currently open fellowships, grants, programs, or funding "
            "opportunities relevant to the task from primary organizations."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance="Pivot toward primary organizations and official sources.",
    )
    runner._build_fallback_search_query = Mock(
        return_value="primary organization machine learning fellowship grant program undergraduate ai"
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert "undergraduate" in query
    assert any(token in query for token in ("ai", "machine", "learning"))


@pytest.mark.unit
def test_subtask_runner_discovery_queries_use_clean_source_scoped_candidates(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Discovery: Identify currently open fellowships, grants, programs, or funding "
            "opportunities relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources, preserving details needed to "
            "verify deadlines, eligibility, and research fit."
        ),
        orchestrator_guidance=(
            "Broaden the search to adjacent in-scope opportunity types and prioritize official "
            "pages, primary organizations, and directories instead of repeating narrow "
            "lead-specific queries."
        ),
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)

    structured = LangGraphSubtaskRunner._build_structured_search_query(
        runner,
        constrained_task,
        agent_state,
    )
    forced = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    for query in (structured, forced):
        assert "undergraduate" in query
        assert "us" in query
        assert any(
            token in query
            for token in ("site:.edu", "site:.org", "site:.gov", "official")
        )
        assert not ("grants" in query and "grant" in query)
        assert not ("fellowships" in query and "fellowship" in query)
        assert "opportunities list" not in query
        assert len(query.split()) <= 8


@pytest.mark.unit
def test_subtask_runner_discovery_forced_query_rotates_opportunity_family_after_first_search(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Discovery: Identify currently open fellowships, grants, programs, or funding "
            "opportunities relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance="Broaden the search to adjacent in-scope opportunity types.",
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    first_query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )
    agent_state.queries.add(first_query)
    second_query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert first_query != second_query
    assert "fellowship" in first_query
    assert any(
        token in second_query
        for token in ("grant", "funding", "scholarship", "internship", "reu", "program")
    )
    assert "undergraduate" in second_query
    assert "us" in second_query


@pytest.mark.unit
def test_subtask_runner_discovery_forced_query_pivots_source_strategy_after_repeated_zero_results(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Discovery: Identify currently open fellowships, grants, programs, or funding "
            "opportunities relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance=(
            "Broaden the search to adjacent in-scope opportunity types and prioritize official "
            "pages, primary organizations, and directories."
        ),
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    agent_state.zero_result_search_count = 1
    agent_state.queries.update(
        {
            "site:.edu machine learning ai research undergraduate us fellowship",
        }
    )

    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert any(
        token in query
        for token in (
            "site:.org",
            "site:.gov",
            "official",
            "organization",
            "opportunities",
            "list",
        )
    )
    assert "site:.edu" not in query


@pytest.mark.unit
def test_subtask_runner_discovery_prefers_broader_official_and_directory_strategies_before_domain_filters(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Discovery: Identify currently open fellowships, grants, programs, or funding "
            "opportunities relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance=(
            "Broaden the search to adjacent in-scope opportunity types and prioritize official "
            "pages, primary organizations, and directories."
        ),
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)

    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert any(
        token in query
        for token in ("official", "organization", "opportunities", "list")
    )
    assert "site:.edu" not in query


@pytest.mark.unit
def test_subtask_runner_discovery_moves_off_official_strategy_after_first_query(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Discovery: Identify currently open fellowships, grants, programs, or funding "
            "opportunities relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance=(
            "Broaden the search to adjacent in-scope opportunity types and prioritize official "
            "pages, primary organizations, and directories."
        ),
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    agent_state.search_count = 1
    agent_state.queries.add(
        "official machine learning ai research undergraduate us fellowship"
    )

    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert any(token in query for token in ("organization", "opportunities", "list"))
    assert any(
        token in query
        for token in (
            "fellowship",
            "grant",
            "program",
            "reu",
            "internship",
            "scholarship",
            "funding",
        )
    )
    assert not query.startswith("official ")


@pytest.mark.unit
def test_subtask_runner_discovery_demotes_grant_after_first_search_for_undergraduate_research_tasks(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Discovery: Identify currently open fellowships, grants, programs, or funding "
            "opportunities relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance=(
            "Broaden the search to adjacent in-scope opportunity types and prioritize official "
            "pages, primary organizations, and directories."
        ),
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    agent_state.search_count = 1
    agent_state.queries.add(
        "official machine learning ai research undergraduate us fellowship"
    )

    candidates = LangGraphSubtaskRunner._build_discovery_search_candidates(
        runner,
        constrained_task,
        agent_state,
    )[:4]

    assert any(
        token in candidates[0]
        for token in ("organization", "directory", "listing", "official")
    )
    assert "grant" not in candidates[0] or "program" not in candidates[0]


@pytest.mark.unit
def test_subtask_runner_discovery_prefers_more_specific_adjacent_types_before_generic_program(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Discovery: Identify currently open fellowships, grants, programs, or funding "
            "opportunities relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance="Broaden the search to adjacent in-scope opportunity types.",
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    first_query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )
    agent_state.queries.add(first_query)
    second_query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )
    agent_state.queries.add(second_query)
    third_query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert "program" not in third_query or third_query.count("program") <= 1
    assert any(
        token in third_query
        for token in ("organization", "directory", "listing", "official", "research")
    )


@pytest.mark.unit
def test_subtask_runner_forced_query_rotates_domain_filters_before_suffix_churn(
    mock_task,
):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(
        subtask_description=(
            "Identify currently open fellowships, grants, programs, or funding "
            "opportunities relevant to the task from official sources."
        ),
        success_criteria=(
            "List multiple distinct candidate opportunities from official program pages, "
            ".gov, .edu, or primary organization sources."
        ),
        orchestrator_guidance="",
    )
    runner._build_fallback_search_query = Mock(
        return_value="grants fellowships ai ml research senior undergraduate us official"
    )

    constrained_task = mock_task.model_copy(
        update={
            "name": "ML Fellowships",
            "description": (
                "Find currently open grants, programs, fellowships, or funding opportunities "
                "that support AI/ML research and are available to senior undergraduate "
                "students in the United States."
            ),
        }
    )

    agent_state = AgentState(task_id=constrained_task.name)
    agent_state.queries.add("site gov ai ml fellowship undergraduate program")

    query = LangGraphSubtaskRunner._build_forced_search_query(
        runner,
        constrained_task,
        agent_state,
    )

    assert "site:.gov official" not in query
    assert any(token in query for token in ("official", "site:.edu", "site:.org"))


@pytest.mark.unit
def test_subtask_runner_replaces_repeated_summarize_with_search(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(subtask_description="Brown AI REU deadline eligibility")
    runner._build_forced_search_query = Mock(
        return_value="brown ai reu deadline eligibility"
    )
    runner._sanitize_search_query = (
        LangGraphSubtaskRunner._sanitize_search_query.__get__(runner)
    )
    runner._extract_keywords = LangGraphSubtaskRunner._extract_keywords
    runner._build_fallback_search_query = (
        LangGraphSubtaskRunner._build_fallback_search_query.__get__(runner)
    )

    agent_state = AgentState(task_id=mock_task.name)
    agent_state.action_pattern.extend(["search", "summarize", "summarize"])
    plan = ResearchActionPlan(action="summarize")

    normalized = LangGraphSubtaskRunner._normalize_action_plan(
        runner,
        mock_task,
        agent_state,
        plan,
    )

    assert normalized.action == "search"
    assert normalized.query == "brown ai reu deadline eligibility"


@pytest.mark.unit
def test_subtask_runner_replaces_repeated_think_with_search(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(subtask_description="Brown AI REU deadline eligibility")
    runner._build_forced_search_query = Mock(
        return_value="brown ai reu deadline eligibility"
    )
    runner._sanitize_search_query = (
        LangGraphSubtaskRunner._sanitize_search_query.__get__(runner)
    )
    runner._extract_keywords = LangGraphSubtaskRunner._extract_keywords
    runner._build_fallback_search_query = (
        LangGraphSubtaskRunner._build_fallback_search_query.__get__(runner)
    )

    agent_state = AgentState(task_id=mock_task.name)
    agent_state.action_pattern.extend(["think", "search", "think", "think"])
    plan = ResearchActionPlan(action="think", thought="Need to reason more")

    normalized = LangGraphSubtaskRunner._normalize_action_plan(
        runner,
        mock_task,
        agent_state,
        plan,
    )

    assert normalized.action == "search"
    assert normalized.query == "brown ai reu deadline eligibility"


@pytest.mark.unit
def test_subtask_runner_keeps_synthesis_stage_on_write_path(mock_task):
    runner = LangGraphSubtaskRunner.__new__(LangGraphSubtaskRunner)
    runner.agent = Mock()
    runner.agent.config = Mock(subtask_description="Compile final shortlist")
    runner._build_forced_search_query = Mock(return_value="fallback query")
    runner._build_synthesis_gap_query = Mock(
        return_value="official nsf ai reu deadline eligibility"
    )
    runner._sanitize_search_query = (
        LangGraphSubtaskRunner._sanitize_search_query.__get__(runner)
    )
    runner._extract_keywords = LangGraphSubtaskRunner._extract_keywords
    runner._build_fallback_search_query = (
        LangGraphSubtaskRunner._build_fallback_search_query.__get__(runner)
    )
    runner._get_current_subtask_mode = Mock(return_value="synthesis")
    runner._synthesis_ready_for_write = Mock(return_value=True)

    agent_state = AgentState(task_id=mock_task.name)
    agent_state.search_count = 4
    agent_state.checkpoint_count = 1
    plan = ResearchActionPlan(action="summarize")

    normalized = LangGraphSubtaskRunner._normalize_action_plan(
        runner,
        mock_task,
        agent_state,
        plan,
    )

    assert normalized.action == "summarize"
