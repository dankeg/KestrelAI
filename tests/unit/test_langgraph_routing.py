from unittest.mock import AsyncMock, Mock, patch

import pytest

try:
    from KestrelAI.agents.base_agent import AgentState
    from KestrelAI.agents.research_orchestrator import ResearchOrchestrator
    from KestrelAI.agents.web_research_agent import WebResearchAgent
    from KestrelAI.graphs.schemas import ResearchActionPlan
    from KestrelAI.graphs.subtask_runner import LangGraphSubtaskRunner
except ImportError:
    from agents.base_agent import AgentState
    from agents.research_orchestrator import ResearchOrchestrator
    from agents.web_research_agent import WebResearchAgent
    from graphs.schemas import ResearchActionPlan
    from graphs.subtask_runner import LangGraphSubtaskRunner


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
    assert ("eligibility" in query) or ("requirements" in query)


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
