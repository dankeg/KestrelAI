# Unit tests for agent components
from unittest.mock import AsyncMock, Mock, patch

import pytest

try:
    from KestrelAI.agents.base_agent import AgentState
    from KestrelAI.agents.research_orchestrator import (
        OrchestratorDecision,
        PlanningPlan,
        PrePlanningAction,
        ResearchOrchestrator,
        SourceEvidenceRecord,
        VerifiedEvidenceItem,
    )
    from KestrelAI.agents.web_research_agent import WebResearchAgent
    from KestrelAI.shared.models import ResearchPlan, Subtask, SubtaskType
except ImportError:
    from agents.base_agent import AgentState
    from agents.research_orchestrator import (
        OrchestratorDecision,
        PlanningPlan,
        PrePlanningAction,
        ResearchOrchestrator,
        SourceEvidenceRecord,
        VerifiedEvidenceItem,
    )
    from agents.web_research_agent import WebResearchAgent
    from shared.models import ResearchPlan, Subtask, SubtaskType


@pytest.mark.unit
class TestResearchOrchestrator:
    """Test research orchestrator functionality."""

    @pytest.fixture
    def orchestrator(self, mock_llm, mock_task):
        """Create orchestrator instance for testing."""
        # Patch MemoryStore and WebResearchAgent so orchestrator tests don't
        # hit the real vector store or network-dependent agent logic.
        with patch("KestrelAI.agents.research_orchestrator.MemoryStore"), patch(
            "KestrelAI.agents.research_orchestrator.WebResearchAgent"
        ) as mock_agent_cls:
            mock_agent = Mock()
            mock_agent.run_step = AsyncMock(return_value="Test subtask result")
            mock_agent.get_metrics.return_value = {}
            mock_agent._state = {}
            mock_agent_cls.return_value = mock_agent
            yield ResearchOrchestrator([mock_task], mock_llm, profile="kestrel")

    def test_orchestrator_initialization(self, mock_llm, mock_task):
        """Test orchestrator initialization."""
        with patch("KestrelAI.agents.research_orchestrator.MemoryStore"):
            with patch(
                "KestrelAI.agents.research_orchestrator.WebResearchAgent"
            ) as mock_agent_cls:
                mock_agent = Mock()
                mock_agent.run_step = AsyncMock(return_value="Test subtask result")
                mock_agent.get_metrics.return_value = {}
                mock_agent._state = {}
                mock_agent_cls.return_value = mock_agent

                orchestrator = ResearchOrchestrator(
                    [mock_task], mock_llm, profile="kestrel"
                )

                assert orchestrator.llm == mock_llm
                assert len(orchestrator.tasks) == 1
                assert orchestrator.tasks[mock_task.name] == mock_task

    def test_subtask_target_count_caps_impractical_discovery_targets(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_MAX_DISCOVERY_TARGET_COUNT", "6")
        subtask = Subtask(
            order=1,
            description="Identify at least 10 distinct authoritative opportunities before verification.",
            success_criteria="Produce a list of 10 distinct official opportunities.",
        )

        assert ResearchOrchestrator._subtask_target_count(subtask) == 6

    def test_orchestrator_task_states(self, orchestrator, mock_task):
        """Test orchestrator task states."""
        assert mock_task.name in orchestrator.task_states
        task_state = orchestrator.task_states[mock_task.name]
        assert task_state.task == mock_task
        assert task_state.research_plan is None

    def test_create_subtask_agent_attaches_orchestrator_reference(
        self, orchestrator, mock_task
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = ResearchPlan(
            restated_task="Test task",
            subtasks=[
                Subtask(
                    order=1,
                    description="Discovery: Identify currently open opportunities from official sources.",
                    success_criteria="List multiple distinct candidate opportunities.",
                    subtask_type="discovery",
                )
            ],
        )

        agent = task_state.create_subtask_agent(
            0, orchestrator.llm, orchestrator.memory
        )

        assert agent.orchestrator is orchestrator
        assert agent.parent_task_name == mock_task.name

    @pytest.mark.asyncio
    async def test_planning_phase(self, orchestrator, mock_task):
        """Test planning phase execution."""
        with patch.object(orchestrator.control_chains, "planning_plan") as mock_plan:
            with patch.object(
                orchestrator, "_run_preplanning_exploration", new=AsyncMock()
            ) as mock_preplanning:
                mock_preplanning.return_value = "preplanning findings"
                mock_plan.return_value = PlanningPlan(
                    restated_task="Test restated task",
                    subtasks=[
                        {
                            "order": 1,
                            "description": "Test subtask 1",
                            "success_criteria": "Test criteria 1",
                        }
                    ],
                )
                await orchestrator._planning_phase(mock_task)

                task_state = orchestrator.task_states[mock_task.name]
                assert task_state.research_plan is not None
                assert task_state.research_plan.restated_task == "Test restated task"
                assert len(task_state.research_plan.subtasks) >= 3
                assert task_state.research_plan.subtasks[0].subtask_type == "discovery"
                assert task_state.research_plan.subtasks[-1].subtask_type == "synthesis"
                mock_plan.assert_called_once()
                assert (
                    mock_plan.call_args.kwargs.get("preplanning_context")
                    == "preplanning findings"
                )

    @pytest.mark.asyncio
    async def test_preplanning_exploration_generates_context(
        self, orchestrator, mock_task, monkeypatch
    ):
        """Test bounded pre-planning exploration loop produces context notes."""
        monkeypatch.setenv("ORCHESTRATOR_PREPLANNING_MAX_STEPS", "2")
        monkeypatch.setenv("ORCHESTRATOR_PREPLANNING_STEP_TIMEOUT_SECONDS", "5")

        with patch.object(
            orchestrator.control_chains, "preplanning_action"
        ) as mock_preplanning_action:
            mock_preplanning_action.side_effect = [
                PrePlanningAction(action="search", query="focused test query"),
                PrePlanningAction(action="done", reasoning="Enough initial context"),
            ]

            with patch.object(orchestrator.preplanning_search, "search") as mock_search:
                mock_search.return_value = [
                    {
                        "title": "Relevant result",
                        "href": "https://example.com/test",
                        "body": "Useful snippet for planning context.",
                    }
                ]
                context = await orchestrator._run_preplanning_exploration(mock_task)

                assert "focused test query" in context
                assert "Relevant result" in context
                assert "Enough initial context" in context
                assert mock_preplanning_action.call_count == 2
                mock_search.assert_called_once_with("focused test query")

    @pytest.mark.asyncio
    async def test_planning_phase_retry(self, orchestrator, mock_task):
        """Test planning phase retry mechanism."""
        with patch.object(orchestrator.control_chains, "planning_plan") as mock_plan:
            with patch.object(
                orchestrator, "_run_preplanning_exploration", new=AsyncMock()
            ) as mock_preplanning:
                mock_preplanning.return_value = ""
                # First call fails, second succeeds
                mock_plan.side_effect = [
                    Exception("First attempt fails"),
                    PlanningPlan(
                        restated_task="Test restated task",
                        subtasks=[
                            {
                                "order": 1,
                                "description": "Test subtask 1",
                                "success_criteria": "Test criteria 1",
                            }
                        ],
                    ),
                ]

                await orchestrator._planning_phase(mock_task)

                task_state = orchestrator.task_states[mock_task.name]
                assert task_state.research_plan is not None
                assert mock_plan.call_count == 2

    @pytest.mark.asyncio
    async def test_next_action_advances_subtask_on_review(
        self, orchestrator, mock_task, mock_research_plan
    ):
        """Test next_action delegates execution to the LangGraph runner."""
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan

        mocked_runner = Mock()
        mocked_runner.run = AsyncMock(return_value="Test orchestration result")
        orchestrator.langgraph_runner = mocked_runner

        result = await orchestrator.next_action(mock_task)

        assert result == "Test orchestration result"
        mocked_runner.run.assert_awaited_once_with(mock_task)

    @pytest.mark.asyncio
    async def test_apply_review_decision_sets_active_guidance(
        self, orchestrator, mock_task, mock_research_plan
    ):
        """Guidance from orchestrator decisions should reach active subtask agent."""
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = 4
        agent_state.search_count = 2
        agent_state.checkpoint_count = 1
        agent_state.queries.update({"q1", "q2"})

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        decision = OrchestratorDecision(
            reasoning="Need deeper evidence in this angle",
            decision="continue",
            feedback="Focus on official program pages and capture deadlines + eligibility.",
            subtask="stay",
            next_task=mock_task.name,
        )

        result = await orchestrator._apply_review_decision(
            mock_task, "Latest notes", decision
        )

        assert "[ORCHESTRATOR FEEDBACK]" in result
        assert "official program pages" in result
        assert (
            task_state.get_subtask_guidance(0)
            == mock_subtask_agent.config.orchestrator_guidance
        )
        assert "official program pages" in task_state.get_subtask_guidance(0)

    @pytest.mark.asyncio
    async def test_apply_review_decision_blocks_premature_progression(
        self, orchestrator, mock_task, mock_research_plan
    ):
        """Proceed/done should be blocked when exploration depth is insufficient."""
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = 1
        agent_state.search_count = 0
        agent_state.checkpoint_count = 0

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        decision = OrchestratorDecision(
            reasoning="Looks good enough",
            decision="done",
            feedback="Complete this subtask.",
            subtask="proceed",
            next_task=mock_task.name,
        )

        result = await orchestrator._apply_review_decision(
            mock_task, "Latest notes", decision
        )

        assert "[ORCHESTRATOR GUARD]" in result
        assert task_state.subtask_index == 0
        assert 0 not in task_state.completed_subtasks
        assert "Insufficient exploration depth" in result

    def test_evidence_stats_ignore_task_unaligned_hits(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.search_history = [
            {
                "query": "site:.org machine learning ai research undergraduate us funding",
                "results": [
                    {
                        "title": "NCSA awarded funding to continue AI-focused NSF REU program",
                        "url": "https://www.eurekalert.org/news-releases/123",
                        "domain": "eurekalert.org",
                        "authority_score": 4,
                        "official_source": False,
                        "fetched": True,
                        "task_aligned": False,
                    }
                ],
            }
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        task_state.subtask_agents[0] = mock_subtask_agent

        stats = orchestrator._get_current_subtask_evidence_stats(mock_task, task_state)

        assert stats["authoritative_results"] == 0
        assert stats["official_results"] == 0
        assert stats["unique_domains"] == 0

    @pytest.mark.asyncio
    async def test_apply_review_decision_does_not_force_progression_at_iteration_ceiling(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = orchestrator.max_iterations_per_subtask
        agent_state.search_count = 2
        agent_state.summary_count = 1
        agent_state.checkpoint_count = 1
        agent_state.queries.update({"q1", "q2"})
        agent_state.search_history = [
            {
                "query": "weak evidence query",
                "results_count": 1,
                "results": [
                    {
                        "title": "Blog post",
                        "url": "https://example.com/post",
                        "domain": "example.com",
                        "source_tier": "commercial",
                        "authority_score": 2,
                        "official_source": False,
                        "fetched": True,
                    }
                ],
            }
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        decision = OrchestratorDecision(
            reasoning="Switch to the next subtask.",
            decision="switch",
            feedback="Proceed to next subtask.",
            subtask="proceed",
            next_task=mock_task.name,
        )

        result = await orchestrator._apply_review_decision(
            mock_task, "Latest notes", decision
        )

        assert "[ORCHESTRATOR GUARD]" in result
        assert task_state.subtask_index == 0
        assert 0 not in task_state.completed_subtasks

    @pytest.mark.asyncio
    async def test_review_forces_progression_at_hard_subtask_limit(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        hard_limit = orchestrator._hard_subtask_iteration_limit()
        agent_state.action_count = hard_limit
        agent_state.search_attempt_count = hard_limit
        agent_state.search_count = 1
        agent_state.checkpoint_count = 0
        agent_state.queries.update({"q1"})
        agent_state.search_history = [
            {
                "query": "weak repetitive query",
                "results_count": 1,
                "results": [
                    {
                        "title": "Blog post",
                        "url": "https://example.com/post",
                        "domain": "example.com",
                        "source_tier": "commercial",
                        "authority_score": 1,
                        "official_source": False,
                        "fetched": False,
                    }
                ],
            }
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        decision = await orchestrator._review(mock_task, "Latest notes")

        assert decision.decision == "switch"
        assert decision.subtask == "proceed"
        assert "Hard per-subtask iteration limit reached" in decision.reasoning

    @pytest.mark.asyncio
    async def test_apply_review_decision_allows_progression_at_hard_subtask_limit(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        hard_limit = orchestrator._hard_subtask_iteration_limit()
        agent_state.action_count = hard_limit
        agent_state.search_attempt_count = hard_limit
        agent_state.search_count = 1
        agent_state.checkpoint_count = 0
        agent_state.queries.update({"q1"})
        agent_state.search_history = [
            {
                "query": "weak repetitive query",
                "results_count": 1,
                "results": [
                    {
                        "title": "Blog post",
                        "url": "https://example.com/post",
                        "domain": "example.com",
                        "source_tier": "commercial",
                        "authority_score": 1,
                        "official_source": False,
                        "fetched": False,
                    }
                ],
            }
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        decision = OrchestratorDecision(
            reasoning="Hard per-subtask iteration limit reached; forcing progression.",
            decision="switch",
            feedback="Stop the current subtask now and advance.",
            subtask="proceed",
            next_task=mock_task.name,
        )

        result = await orchestrator._apply_review_decision(
            mock_task, "Latest notes", decision
        )

        assert "[ORCHESTRATOR GUARD]" not in result
        assert task_state.subtask_index == 1
        assert 0 in task_state.completed_subtasks

    def test_build_guidance_filters_synthetic_recent_queries(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = 4
        agent_state.search_count = 2
        agent_state.checkpoint_count = 1
        agent_state.queries.update({"q1", "q2"})
        agent_state.search_history = [
            {"query": "ml fellowships find open deadline application grants angle 18"},
            {"query": "ml fellowships find open deadline application grants angle 19"},
            {"query": "ml fellowships find open deadline application grants angle 20"},
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        guidance = orchestrator._build_guidance_from_decision(
            mock_task,
            task_state,
            OrchestratorDecision(
                reasoning="Need more breadth",
                decision="continue",
                feedback="Continue gathering evidence for the current subtask.",
                subtask="stay",
                next_task=mock_task.name,
            ),
        )

        assert "angle 18" not in guidance
        assert "angle 19" not in guidance
        assert "angle 20" not in guidance
        assert (
            guidance.count("ml fellowships find open deadline application grants") <= 1
        )

    def test_build_guidance_suppresses_ungrounded_specific_queries_when_evidence_is_thin(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = 2
        agent_state.search_count = 0
        agent_state.checkpoint_count = 0
        agent_state.queries.update(
            {"nsf site gov machine learning fellowship undergraduate"}
        )
        agent_state.search_history = [
            {
                "query": "nsf site:.gov machine learning fellowship undergraduate",
                "results": [],
            }
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        guidance = orchestrator._build_guidance_from_decision(
            mock_task,
            task_state,
            OrchestratorDecision(
                reasoning="Need more breadth",
                decision="continue",
                feedback=(
                    'Switch to a new query: "NSF research opportunities undergraduate AI" site:.gov. '
                    "Then investigate NSF REU pages."
                ),
                subtask="stay",
                next_task=mock_task.name,
            ),
        )

        assert "NSF research opportunities undergraduate AI" not in guidance
        assert "investigate NSF REU pages" not in guidance
        assert "Run at least" in guidance or "Find at least" in guidance

    def test_build_guidance_sanitizes_ungrounded_named_entities_but_keeps_scope_expansion(
        self, orchestrator, mock_task
    ):
        opportunity_task = mock_task.model_copy(
            update={
                "description": (
                    "Find currently open grants, programs, fellowships, or funding opportunities "
                    "that support AI/ML research and are available to senior undergraduate students "
                    "in the United States."
                ),
                "budgetMinutes": 60,
            }
        )
        task_state = orchestrator.task_states[opportunity_task.name]
        task_state.research_plan = ResearchPlan(
            restated_task="Identify and validate current AI/ML opportunities.",
            subtasks=[
                Subtask(
                    order=1,
                    description=(
                        "Identify currently open fellowships, grants, programs, or funding "
                        "opportunities relevant to the task from official sources."
                    ),
                    success_criteria=(
                        "List multiple distinct candidate opportunities from official program pages, "
                        ".gov, .edu, or primary organization sources."
                    ),
                    subtask_type="discovery",
                    status="in_progress",
                    findings=[],
                )
            ],
            current_subtask_index=0,
        )
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=opportunity_task.name)
        agent_state.action_count = 5
        agent_state.search_count = 1
        agent_state.queries.update(
            {
                "site:.edu machine learning fellowship undergraduate",
                "site:.gov machine learning fellowship undergraduate",
            }
        )
        agent_state.search_history = [
            {
                "query": "site:.gov machine learning fellowship undergraduate",
                "results": [
                    {
                        "title": "Applied Machine Learning Fellowship | Los Alamos National Laboratory",
                        "url": "https://www.lanl.gov/engage/collaboration/internships/summer-schools/applied-machine-learning-fellowship",
                        "domain": "lanl.gov",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
                "results_count": 1,
            }
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {opportunity_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        guidance = orchestrator._build_guidance_from_decision(
            opportunity_task,
            task_state,
            OrchestratorDecision(
                reasoning="Need broader discovery",
                decision="continue",
                feedback=(
                    "Specifically, broaden the search to include funding opportunities related to "
                    "computational science, data science, statistical learning, and algorithmic research "
                    "alongside AI/ML. Also, investigate funding sources known to support research in these "
                    "areas, such as the Hertz Foundation, Schmidt Futures, and the Fannie and John Hertz "
                    "Foundation. Use queries like: computational science fellowship undergraduate; data science "
                    "grant undergraduate; Hertz Foundation undergraduate research. Prioritize .edu, .gov, and "
                    "primary organization websites."
                ),
                subtask="stay",
                next_task=opportunity_task.name,
            ),
        )

        assert "computational science" in guidance
        assert "data science" in guidance
        assert "Hertz Foundation" not in guidance
        assert "Schmidt Futures" not in guidance
        assert "Use queries like" not in guidance

    def test_build_guidance_uses_paper_specific_fallback_language(
        self, orchestrator, mock_task
    ):
        paper_task = mock_task.model_copy(
            update={
                "name": "RAG Eval Papers",
                "description": (
                    "Find strong papers and conference proceedings about evaluation "
                    "methods for retrieval-augmented generation agents."
                ),
            }
        )
        task_state = orchestrator.task_states[mock_task.name]
        orchestrator.task_states[paper_task.name] = task_state
        task_state.task = paper_task
        task_state.research_plan = ResearchPlan(
            restated_task=paper_task.description,
            subtasks=[
                Subtask(
                    order=1,
                    description="Discover papers, proceedings, and lab or publisher sources across different source classes.",
                    success_criteria="Cover multiple source classes and find authoritative benchmark or evaluation papers.",
                    subtask_type=SubtaskType.DISCOVERY,
                )
            ],
            current_subtask_index=0,
        )
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=paper_task.name)
        agent_state.action_count = 1
        agent_state.search_count = 0
        agent_state.checkpoint_count = 0

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {paper_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        guidance = orchestrator._build_guidance_from_decision(
            paper_task,
            task_state,
            OrchestratorDecision(
                reasoning="Need more breadth",
                decision="continue",
                feedback="Continue gathering evidence for the current subtask.",
                subtask="stay",
                next_task=paper_task.name,
            ),
        )

        lowered = guidance.lower()
        assert "opportunity types" not in lowered
        assert "proceedings" in lowered or "publisher" in lowered
        assert "official program pages" not in lowered

    def test_build_guidance_uses_ecosystem_specific_authority_language(
        self, orchestrator, mock_task
    ):
        ecosystem_task = mock_task.model_copy(
            update={
                "name": "Local Agent Ecosystem",
                "description": (
                    "Map the current open-source local-first AI agent ecosystem, "
                    "including frameworks, repositories, and documentation hubs."
                ),
            }
        )
        task_state = orchestrator.task_states[mock_task.name]
        orchestrator.task_states[ecosystem_task.name] = task_state
        task_state.task = ecosystem_task
        task_state.research_plan = ResearchPlan(
            restated_task=ecosystem_task.description,
            subtasks=[
                Subtask(
                    order=1,
                    description="Discover repository, documentation, and ecosystem index sources across different source classes.",
                    success_criteria="Cover multiple source classes and identify authoritative framework or repository sources.",
                    subtask_type=SubtaskType.DISCOVERY,
                )
            ],
            current_subtask_index=0,
        )
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=ecosystem_task.name)
        agent_state.action_count = 1
        agent_state.search_count = 0
        agent_state.checkpoint_count = 0

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {ecosystem_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        guidance = orchestrator._build_guidance_from_decision(
            ecosystem_task,
            task_state,
            OrchestratorDecision(
                reasoning="Need more breadth",
                decision="continue",
                feedback="Continue gathering evidence for the current subtask.",
                subtask="stay",
                next_task=ecosystem_task.name,
            ),
        )

        lowered = guidance.lower()
        assert "opportunity types" not in lowered
        assert (
            "official repositories" in lowered or "maintainer documentation" in lowered
        )
        assert "official program pages" not in lowered

    def test_get_current_subtask(self, orchestrator, mock_task, mock_research_plan):
        """Test getting current subtask."""
        # Set up research plan
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan

        current_subtask = orchestrator.get_current_subtask(mock_task.name)

        assert current_subtask is not None
        # Returns description plus success criteria
        assert current_subtask.startswith("Test subtask 1")

    def test_get_current_subtask_no_plan(self, orchestrator, mock_task):
        """Test getting current subtask when no plan exists."""
        current_subtask = orchestrator.get_current_subtask(mock_task.name)
        assert current_subtask is None

    def test_get_current_subtask_completed(self, orchestrator, mock_task):
        """Test getting current subtask when all subtasks completed."""
        # Create a completed research plan
        completed_plan = ResearchPlan(
            restated_task="Test restated task",
            subtasks=[
                Subtask(
                    order=1,
                    description="Test subtask 1",
                    success_criteria="Test criteria 1",
                    status="completed",
                    findings=["result1"],
                )
            ],
            current_subtask_index=1,  # Beyond available subtasks
        )

        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = completed_plan
        task_state.subtask_index = 1  # Set to beyond available subtasks

        current_subtask = orchestrator.get_current_subtask(mock_task.name)
        assert current_subtask == "All subtasks completed"

    def test_subtask_completion_readiness_requires_authoritative_sources(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = 6
        agent_state.search_count = 3
        agent_state.summary_count = 1
        agent_state.checkpoint_count = 1
        agent_state.queries.update({"q1", "q2", "q3"})
        agent_state.search_history = [
            {
                "query": "test query",
                "results_count": 2,
                "results": [
                    {
                        "title": "Blog post",
                        "url": "https://example.com/post",
                        "domain": "example.com",
                        "source_tier": "commercial",
                        "authority_score": 2,
                        "official_source": False,
                        "fetched": True,
                    },
                    {
                        "title": "Forum post",
                        "url": "https://reddit.com/r/test",
                        "domain": "reddit.com",
                        "source_tier": "low_signal",
                        "authority_score": 0,
                        "official_source": False,
                        "fetched": True,
                    },
                ],
            }
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        ready, reason = orchestrator._evaluate_subtask_completion_readiness(
            mock_task, task_state
        )

        assert ready is False
        assert "authoritative_results" in reason

    def test_subtask_completion_readiness_accepts_diverse_authoritative_sources(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = 6
        agent_state.search_count = 3
        agent_state.summary_count = 1
        agent_state.checkpoint_count = 1
        agent_state.queries.update({"q1", "q2", "q3"})
        agent_state.search_history = [
            {
                "query": "official query",
                "results_count": 2,
                "results": [
                    {
                        "title": "NSF REU",
                        "url": "https://nsf.gov/reu",
                        "domain": "nsf.gov",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "University REU page",
                        "url": "https://mit.edu/reu",
                        "domain": "mit.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                ],
            }
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        ready, reason = orchestrator._evaluate_subtask_completion_readiness(
            mock_task, task_state
        )

        assert ready is True
        assert "sufficient" in reason.lower()

    def test_subtask_completion_readiness_requires_distinct_discovery_results(
        self, orchestrator, mock_task
    ):
        plan = ResearchPlan(
            restated_task="Research REU opportunities",
            subtasks=[
                Subtask(
                    order=1,
                    description="Discover and list all publicly available NSF REU opportunities in AI and machine learning for 2024.",
                    success_criteria="Compile a list of at least 5 distinct NSF REU opportunities related to AI and machine learning.",
                    status="in_progress",
                    findings=[],
                )
            ],
            current_subtask_index=0,
        )
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = 6
        agent_state.search_count = 4
        agent_state.summary_count = 1
        agent_state.checkpoint_count = 1
        agent_state.queries.update({"q1", "q2", "q3", "q4"})
        agent_state.search_history = [
            {
                "query": "reu ai opportunities",
                "results_count": 3,
                "results": [
                    {
                        "title": "CMU NSF REU in AI",
                        "url": "https://cmu.edu/reu-ai",
                        "domain": "cmu.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "NSF REU opportunities",
                        "url": "https://nsf.gov/reu",
                        "domain": "nsf.gov",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "OSU AI-EDGE summer program",
                        "url": "https://osu.edu/ai-edge-reu",
                        "domain": "osu.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                ],
            }
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        ready, reason = orchestrator._evaluate_subtask_completion_readiness(
            mock_task, task_state
        )

        assert ready is False
        assert "distinct_authoritative_titles" in reason

    def test_subtask_completion_readiness_requires_pathway_coverage_for_broad_discovery(
        self, orchestrator, mock_task
    ):
        plan = ResearchPlan(
            restated_task="Research undergraduate AI/ML opportunities",
            subtasks=[
                Subtask(
                    order=1,
                    description="Discover and list distinct publicly available AI/ML research opportunities for undergraduates.",
                    success_criteria="Compile a shortlist of distinct opportunities from multiple source classes.",
                    status="in_progress",
                    findings=[],
                )
            ],
            current_subtask_index=0,
        )
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = 6
        agent_state.search_count = 4
        agent_state.summary_count = 1
        agent_state.checkpoint_count = 1
        agent_state.queries.update({"q1", "q2", "q3", "q4"})
        agent_state.search_pathways = [
            {"id": "pathway_1", "attempt_count": 1, "hit_count": 1},
            {"id": "pathway_2", "attempt_count": 0, "hit_count": 0},
            {"id": "pathway_3", "attempt_count": 0, "hit_count": 0},
        ]
        agent_state.search_history = [
            {
                "query": "official opportunity query",
                "results_count": 3,
                "results": [
                    {
                        "title": "Program A",
                        "url": "https://a.edu/program-a",
                        "domain": "a.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "Program B",
                        "url": "https://b.edu/program-b",
                        "domain": "b.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "Program C",
                        "url": "https://c.org/program-c",
                        "domain": "c.org",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                ],
            }
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        ready, reason = orchestrator._evaluate_subtask_completion_readiness(
            mock_task, task_state
        )

        assert ready is False
        assert "pathways_attempted" in reason

    def test_ceiling_progression_readiness_is_stricter_than_base_readiness(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = orchestrator.max_iterations_per_subtask
        agent_state.search_count = 3
        agent_state.summary_count = 1
        agent_state.checkpoint_count = 1
        agent_state.queries.update({"q1", "q2", "q3"})
        agent_state.search_history = [
            {
                "query": "official query",
                "results_count": 2,
                "results": [
                    {
                        "title": "NSF REU",
                        "url": "https://nsf.gov/reu",
                        "domain": "nsf.gov",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "University REU page",
                        "url": "https://mit.edu/reu",
                        "domain": "mit.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                ],
            }
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        base_ready, _ = orchestrator._evaluate_subtask_completion_readiness(
            mock_task, task_state
        )
        (
            ceiling_ready,
            ceiling_reason,
        ) = orchestrator._evaluate_ceiling_progression_readiness(mock_task, task_state)

        assert base_ready is True
        assert ceiling_ready is False
        assert "Ceiling progression blocked" in ceiling_reason

    def test_build_guidance_prefers_uncovered_pathway_coverage_for_broad_discovery(
        self, orchestrator, mock_task
    ):
        plan = ResearchPlan(
            restated_task="Research undergraduate AI/ML opportunities",
            subtasks=[
                Subtask(
                    order=1,
                    description="Discover and list distinct publicly available AI/ML research opportunities for undergraduates.",
                    success_criteria="Compile a shortlist of distinct opportunities from multiple source classes.",
                    status="in_progress",
                    findings=[],
                )
            ],
            current_subtask_index=0,
        )
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = plan
        task_state.subtask_index = 0
        task_state.subtask_stagnation_rounds[0] = 2

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = 4
        agent_state.search_count = 2
        agent_state.checkpoint_count = 1
        agent_state.queries.update({"q1", "q2"})
        agent_state.search_pathways = [
            {"id": "pathway_1", "attempt_count": 1, "hit_count": 0},
            {"id": "pathway_2", "attempt_count": 0, "hit_count": 0},
            {"id": "pathway_3", "attempt_count": 0, "hit_count": 0},
        ]
        agent_state.search_history = [
            {"query": "official ai ml fellowship undergraduate"},
            {"query": "official ai ml funding undergraduate"},
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        guidance = orchestrator._build_guidance_from_decision(
            mock_task,
            task_state,
            OrchestratorDecision(
                reasoning="Need more breadth",
                decision="continue",
                feedback="Continue gathering evidence for the current subtask.",
                subtask="stay",
                next_task=mock_task.name,
            ),
        )

        assert "discovery pathway" in guidance or "source-class routes" in guidance
        assert "pivot to an uncovered pathway or source class" in guidance

    def test_set_subtask_guidance_syncs_structured_control_hints(
        self, orchestrator, mock_task
    ):
        plan = ResearchPlan(
            restated_task="Research undergraduate AI/ML opportunities",
            subtasks=[
                Subtask(
                    order=1,
                    description="Discover and identify distinct publicly available AI/ML research opportunities for undergraduates.",
                    success_criteria="Compile a shortlist of distinct opportunities from multiple source classes.",
                    status="in_progress",
                    findings=[],
                    subtask_type=SubtaskType.DISCOVERY,
                )
            ],
            current_subtask_index=0,
        )
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.search_pathways = [
            {"id": "pathway_1", "attempt_count": 0, "hit_count": 0},
            {"id": "pathway_2", "attempt_count": 1, "hit_count": 0},
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(
            orchestrator_guidance="",
            orchestrator_control_hints={},
        )
        task_state.subtask_agents[0] = mock_subtask_agent

        orchestrator._set_subtask_guidance(
            mock_task,
            task_state,
            0,
            "Broaden to official lab, organization, and directory sources.",
        )

        assert (
            mock_subtask_agent.config.orchestrator_guidance
            == "Broaden to official lab, organization, and directory sources."
        )
        assert mock_subtask_agent.config.orchestrator_control_hints == {
            "discovery_mode": "pathway_first",
            "preferred_pathway_ids": ["pathway_1", "pathway_2"],
            "pathway_count": 2,
            "pathway_attempted_count": 1,
            "pathway_hit_count": 0,
            "pathway_uncovered_count": 1,
            "stagnating": False,
        }

    def test_plateau_progression_allows_near_target_discovery_exit(
        self, orchestrator, mock_task
    ):
        plan = ResearchPlan(
            restated_task="Research REU opportunities",
            subtasks=[
                Subtask(
                    order=1,
                    description="Discover and list all publicly available NSF REU opportunities in AI and machine learning for 2024.",
                    success_criteria="Compile a list of at least 5 distinct NSF REU opportunities related to AI and machine learning.",
                    status="in_progress",
                    findings=[],
                )
            ],
            current_subtask_index=0,
        )
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = plan
        task_state.subtask_index = 0
        task_state.subtask_stagnation_rounds[0] = 3

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = 25
        agent_state.search_count = 5
        agent_state.summary_count = 8
        agent_state.checkpoint_count = 2
        agent_state.queries.update({"q1", "q2", "q3", "q4", "q5"})
        agent_state.search_history = [
            {
                "query": "reu ai opportunities",
                "results_count": 4,
                "results": [
                    {
                        "title": "CMU NSF REU in AI",
                        "url": "https://cmu.edu/reu-ai",
                        "domain": "cmu.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "NSF REU opportunities",
                        "url": "https://nsf.gov/reu",
                        "domain": "nsf.gov",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "OSU AI-EDGE summer program",
                        "url": "https://osu.edu/ai-edge-reu",
                        "domain": "osu.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "Brown AI REU",
                        "url": "https://brown.edu/ai-reu",
                        "domain": "brown.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                ],
            }
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        (
            plateau_ready,
            plateau_reason,
        ) = orchestrator._evaluate_plateau_progression_readiness(mock_task, task_state)

        assert plateau_ready is True
        assert "Discovery appears saturated near target" in plateau_reason

    def test_plateau_progression_still_blocks_shallow_discovery(
        self, orchestrator, mock_task
    ):
        plan = ResearchPlan(
            restated_task="Research REU opportunities",
            subtasks=[
                Subtask(
                    order=1,
                    description="Discover and list all publicly available NSF REU opportunities in AI and machine learning for 2024.",
                    success_criteria="Compile a list of at least 5 distinct NSF REU opportunities related to AI and machine learning.",
                    status="in_progress",
                    findings=[],
                )
            ],
            current_subtask_index=0,
        )
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = plan
        task_state.subtask_index = 0
        task_state.subtask_stagnation_rounds[0] = 4

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = 20
        agent_state.search_count = 2
        agent_state.summary_count = 6
        agent_state.checkpoint_count = 1
        agent_state.queries.update({"q1", "q2"})
        agent_state.search_history = [
            {
                "query": "reu ai opportunities",
                "results_count": 2,
                "results": [
                    {
                        "title": "CMU NSF REU in AI",
                        "url": "https://cmu.edu/reu-ai",
                        "domain": "cmu.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "NSF REU opportunities",
                        "url": "https://nsf.gov/reu",
                        "domain": "nsf.gov",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                ],
            }
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        (
            plateau_ready,
            plateau_reason,
        ) = orchestrator._evaluate_plateau_progression_readiness(mock_task, task_state)

        assert plateau_ready is False
        assert "practical sufficiency" in plateau_reason.lower()

    @pytest.mark.asyncio
    async def test_review_does_not_force_progress_non_discovery_without_stagnation(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 1
        orchestrator.control_chains = None

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = orchestrator.max_iterations_per_subtask
        agent_state.search_count = 5
        agent_state.summary_count = 1
        agent_state.checkpoint_count = 1
        agent_state.queries.update({"q1", "q2", "q3", "q4"})
        agent_state.search_history = [
            {
                "query": "official opportunities",
                "results_count": 3,
                "results": [
                    {
                        "title": "Brown AI for Computational Creativity REU",
                        "url": "https://aireu.cs.brown.edu/",
                        "domain": "brown.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "WPI Data Science REU",
                        "url": "https://www.wpi.edu/academics/departments/data-science/students/reu-program",
                        "domain": "wpi.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "NSF REU Opportunities",
                        "url": "https://www.nsf.gov/funding/initiatives/reu",
                        "domain": "nsf.gov",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                ],
            }
        ]

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[1] = mock_subtask_agent

        decision = await orchestrator._review(mock_task, "Latest notes")

        assert decision.decision == "continue"
        assert decision.subtask == "stay"

    def test_should_defer_review_returns_false_under_low_progress(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = 4
        agent_state.search_attempt_count = 2
        agent_state.search_count = 0

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        defer, _reason = orchestrator._should_defer_llm_review(mock_task, task_state)

        assert defer is False

    @pytest.mark.asyncio
    async def test_review_fallback_uses_planner_instability_guidance(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0
        orchestrator.control_chains = None

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.action_count = 3
        agent_state.search_attempt_count = 2
        agent_state.search_count = 0
        agent_state.planner_fallback_count = 2
        agent_state.consecutive_planner_failures = 2

        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        decision = await orchestrator._review(mock_task, "Latest notes")

        assert decision.decision == "continue"
        assert "Planner instability detected" in decision.feedback

    def test_planning_normalization_replaces_low_signal_subtasks(
        self, orchestrator, mock_task
    ):
        raw_plan = PlanningPlan(
            restated_task="Research opportunities",
            subtasks=[
                {
                    "order": 1,
                    "description": "Analyze the Shopify article https://www.shopify.com/blog/ai for relevant opportunities.",
                    "success_criteria": "Determine whether the Shopify blog contains useful opportunity information.",
                },
                {
                    "order": 2,
                    "description": "Compile a shortlist from official sources.",
                    "success_criteria": "List strong opportunities with links.",
                },
            ],
        )

        normalized = orchestrator._normalize_research_plan(mock_task, raw_plan)

        assert all(
            "shopify" not in subtask.description.lower()
            for subtask in normalized.subtasks
        )
        assert any(
            "authoritative sources" in subtask.description.lower()
            or "official sources" in subtask.description.lower()
            for subtask in normalized.subtasks
        )

    def test_planning_normalization_replaces_overanchored_lead_subtasks(
        self, orchestrator, mock_task
    ):
        raw_plan = PlanningPlan(
            restated_task="Research opportunities",
            subtasks=[
                {
                    "order": 1,
                    "description": "Investigate the ‘2024 NSF REU Participants - NSF AI Planning Institute for Data ...’ webpage at Carnegie Mellon University for detailed information on the NSF REU program.",
                    "success_criteria": "Extract the program’s description, goals, eligibility requirements, application process, and listed deadlines from the CMU webpage.",
                },
                {
                    "order": 2,
                    "description": "Verify the strongest opportunities against authoritative sources.",
                    "success_criteria": "Cross-check deadlines and eligibility on official pages.",
                },
            ],
        )

        normalized = orchestrator._normalize_research_plan(mock_task, raw_plan)

        assert all(
            "carnegie mellon university" not in subtask.description.lower()
            for subtask in normalized.subtasks
        )
        assert all(
            "identify distinct nsf" not in subtask.description.lower()
            for subtask in normalized.subtasks
        )
        assert any(
            "identify distinct government" in subtask.description.lower()
            or "official sources" in subtask.success_criteria.lower()
            for subtask in normalized.subtasks
        )

    def test_planning_normalization_replaces_out_of_scope_artifacts(
        self, orchestrator, mock_task
    ):
        raw_plan = PlanningPlan(
            restated_task="Research opportunities",
            subtasks=[
                {
                    "order": 1,
                    "description": "Analyze the provided arXiv abstracts to determine whether any associated REU opportunities exist.",
                    "success_criteria": "Identify direct REU opportunities linked to the abstracts and relevant FOAs.",
                },
                {
                    "order": 2,
                    "description": "Compile verified opportunities from official sources.",
                    "success_criteria": "List current opportunities with verified deadlines.",
                },
            ],
        )

        normalized = orchestrator._normalize_research_plan(mock_task, raw_plan)

        assert all(
            "arxiv" not in subtask.description.lower()
            and "foa" not in subtask.description.lower()
            for subtask in normalized.subtasks
        )
        assert any(
            "primary or authoritative program pages" in subtask.description.lower()
            or "official program pages" in subtask.success_criteria.lower()
            for subtask in normalized.subtasks
        )

    def test_planning_normalization_replaces_opportunity_misaligned_subtasks(
        self, orchestrator, mock_task
    ):
        opportunity_task = mock_task.model_copy(
            update={
                "name": "ML Fellowships",
                "description": (
                    "Find currently open grants, programs, fellowships, or funding opportunities "
                    "that support AI/ML research and are available to senior undergraduate "
                    "students in the United States."
                ),
            }
        )
        raw_plan = PlanningPlan(
            restated_task="Research opportunities",
            subtasks=[
                {
                    "order": 1,
                    "description": (
                        "Identify and list major US universities with active AI/ML research programs, "
                        "including research areas and faculty expertise."
                    ),
                    "success_criteria": (
                        "A list of at least 20 universities with AI/ML programs and brief research descriptions."
                    ),
                },
                {
                    "order": 2,
                    "description": "Verify the strongest opportunities against authoritative sources.",
                    "success_criteria": "Cross-check details on official pages.",
                },
            ],
        )

        normalized = orchestrator._normalize_research_plan(opportunity_task, raw_plan)

        assert any(
            "candidate opportunities" in subtask.description.lower()
            or "candidate opportunities" in subtask.success_criteria.lower()
            for subtask in normalized.subtasks
        )
        assert all(
            "faculty expertise" not in subtask.description.lower()
            for subtask in normalized.subtasks
        )
        assert [subtask.subtask_type for subtask in normalized.subtasks] == [
            "discovery",
            "verification",
            "comparison",
            "synthesis",
        ]

    def test_planning_normalization_replaces_organization_mapping_subtasks(
        self, orchestrator, mock_task
    ):
        opportunity_task = mock_task.model_copy(
            update={
                "name": "ML Fellowships",
                "description": (
                    "Find currently open grants, programs, fellowships, or funding opportunities "
                    "that support AI/ML research and are available to senior undergraduate "
                    "students in the United States."
                ),
            }
        )
        raw_plan = PlanningPlan(
            restated_task="Research opportunities",
            subtasks=[
                {
                    "order": 1,
                    "description": (
                        "Discovery: Identify organizations and institutions that commonly offer "
                        "AI/ML fellowships or grants to undergraduate students in the US."
                    ),
                    "success_criteria": (
                        "A list of at least 10 organizations/institutions known to offer "
                        "relevant funding opportunities, with links to their websites or program pages."
                    ),
                },
                {
                    "order": 2,
                    "description": "Verify the strongest opportunities against authoritative sources.",
                    "success_criteria": "Cross-check details on official pages.",
                },
            ],
        )

        normalized = orchestrator._normalize_research_plan(opportunity_task, raw_plan)

        assert normalized.subtasks[0].subtask_type == "discovery"
        assert (
            "organizations and institutions"
            not in normalized.subtasks[0].description.lower()
        )
        assert any(
            phrase in normalized.subtasks[0].description.lower()
            for phrase in (
                "currently open fellowships",
                "candidate opportunities",
                "relevant opportunities from official sources",
            )
        )

    def test_infer_subtask_type_respects_explicit_phase_prefixes(self, orchestrator):
        assert (
            orchestrator._infer_subtask_type(
                "Discovery: Compile a list of authoritative websites.",
                "List at least 5 authoritative websites and organizations.",
            )
            == "discovery"
        )
        assert (
            orchestrator._infer_subtask_type(
                "Verification: Confirm the strongest opportunities on official pages.",
                "Check deadlines and eligibility directly.",
            )
            == "verification"
        )

    def test_planning_normalization_canonicalizes_opportunity_plan(
        self, orchestrator, mock_task
    ):
        opportunity_task = mock_task.model_copy(
            update={
                "name": "ML Fellowships",
                "description": (
                    "Find currently open grants, programs, fellowships, or funding opportunities "
                    "that support AI/ML research and are available to senior undergraduate "
                    "students in the United States."
                ),
            }
        )
        raw_plan = PlanningPlan(
            restated_task="Identify and validate currently open AI/ML opportunities.",
            subtasks=[
                {
                    "order": 1,
                    "description": "Discover a broad set of relevant opportunities from authoritative sources before narrowing the shortlist.",
                    "success_criteria": "Identify multiple distinct candidate opportunities from official program pages, .gov, .edu, or primary organization sources with enough breadth to support later verification.",
                },
                {
                    "order": 2,
                    "description": "Discovery: Identify authoritative websites and organizations that compile lists of fellowships and grants for undergraduate students in STEM fields, specifically AI/ML.",
                    "success_criteria": "A list of at least 5 authoritative websites/organizations that regularly publish information on undergraduate STEM fellowships and grants.",
                },
                {
                    "order": 3,
                    "description": "Discovery: Using the websites identified in Subtask 1, discover specific AI/ML fellowships, grants, and programs targeted towards senior undergraduate students in the United States. Prioritize opportunities with clear eligibility criteria and application deadlines.",
                    "success_criteria": "A comprehensive list of at least 15 distinct AI/ML funding opportunities, including the name, sponsoring organization, eligibility requirements, and deadlines.",
                },
                {
                    "order": 4,
                    "description": "Authoritative Verification: For each opportunity identified in Subtask 2, verify its current status (open/closed) and eligibility requirements through direct reference to the sponsoring organization's official website.",
                    "success_criteria": "Confirmation of current status and eligibility criteria for each opportunity, with discrepancies noted.",
                },
            ],
        )

        normalized = orchestrator._normalize_research_plan(opportunity_task, raw_plan)

        assert [subtask.subtask_type for subtask in normalized.subtasks] == [
            "discovery",
            "verification",
            "comparison",
            "synthesis",
        ]
        assert len(normalized.subtasks) == 4
        assert (
            "authoritative websites" not in normalized.subtasks[1].description.lower()
        )
        assert "current status" in normalized.subtasks[1].success_criteria.lower()

    def test_planning_normalization_rejects_bare_low_signal_domain_anchors(
        self, orchestrator, mock_task
    ):
        opportunity_task = mock_task.model_copy(
            update={
                "name": "ML Fellowships",
                "description": (
                    "Find currently open grants, programs, fellowships, or funding opportunities "
                    "that support AI/ML research and are available to senior undergraduate "
                    "students in the United States."
                ),
            }
        )
        raw_plan = PlanningPlan(
            restated_task="Identify and validate currently open AI/ML opportunities.",
            subtasks=[
                {
                    "order": 1,
                    "description": "Identify currently open fellowships, grants, programs, or funding opportunities relevant to the task from official sources.",
                    "success_criteria": "List multiple distinct candidate opportunities from official program pages, .gov, .edu, or primary organization sources.",
                },
                {
                    "order": 2,
                    "description": (
                        "Authoritative Verification: Verify the accuracy and currency of the opportunities "
                        "listed on scholarshipsandgrants.us/major/ai-ml/."
                    ),
                    "success_criteria": (
                        "Cross-check each listed opportunity against official program pages and note discrepancies."
                    ),
                },
            ],
        )

        normalized = orchestrator._normalize_research_plan(opportunity_task, raw_plan)

        assert all(
            "scholarshipsandgrants.us" not in subtask.description.lower()
            and "scholarshipsandgrants.us" not in subtask.success_criteria.lower()
            for subtask in normalized.subtasks
        )

    def test_truncate_research_plan_preserves_synthesis_phase(
        self, orchestrator, mock_task
    ):
        plan = PlanningPlan(
            restated_task="Research opportunities",
            subtasks=[
                {
                    "order": 1,
                    "description": "Discover broad opportunities",
                    "success_criteria": "List broad opportunities",
                    "subtask_type": "discovery",
                },
                {
                    "order": 2,
                    "description": "Discover authoritative websites",
                    "success_criteria": "List authoritative websites",
                    "subtask_type": "discovery",
                },
                {
                    "order": 3,
                    "description": "Verify opportunities",
                    "success_criteria": "Check deadlines and eligibility",
                    "subtask_type": "verification",
                },
                {
                    "order": 4,
                    "description": "Compare strongest opportunities",
                    "success_criteria": "Rank the options",
                    "subtask_type": "comparison",
                },
                {
                    "order": 5,
                    "description": "Synthesize final report",
                    "success_criteria": "Produce final report",
                    "subtask_type": "synthesis",
                },
            ],
        )

        truncated = orchestrator._truncate_research_plan_preserving_phases(
            mock_task, plan, max_subtasks=4
        )

        assert len(truncated.subtasks) == 4
        assert truncated.subtasks[0].subtask_type == "discovery"
        assert truncated.subtasks[-1].subtask_type == "synthesis"
        assert any(
            subtask.subtask_type == "comparison" for subtask in truncated.subtasks
        )

    def test_planning_normalization_forces_discovery_to_start_and_synthesis_to_end(
        self, orchestrator, mock_task
    ):
        raw_plan = PlanningPlan(
            restated_task="Research opportunities",
            subtasks=[
                {
                    "order": 1,
                    "description": "Synthesis: Create a concise shortlist of the strongest opportunities.",
                    "success_criteria": "Produce a useful final shortlist.",
                },
                {
                    "order": 2,
                    "description": "Verify the strongest opportunities against authoritative sources.",
                    "success_criteria": "Cross-check details on official pages.",
                },
            ],
        )

        normalized = orchestrator._normalize_research_plan(mock_task, raw_plan)

        assert len(normalized.subtasks) >= 3
        assert normalized.subtasks[0].subtask_type == "discovery"
        assert normalized.subtasks[-1].subtask_type == "synthesis"
        assert any(
            subtask.subtask_type == "verification" for subtask in normalized.subtasks
        )

    def test_planning_normalization_adds_comparison_for_too_short_plan(
        self, orchestrator, mock_task
    ):
        raw_plan = PlanningPlan(
            restated_task="Research opportunities",
            subtasks=[
                {
                    "order": 1,
                    "description": "Identify distinct opportunities from official sources.",
                    "success_criteria": "List multiple distinct opportunities.",
                },
                {
                    "order": 2,
                    "description": "Synthesize findings into a final deliverable.",
                    "success_criteria": "Produce a coherent final output.",
                },
            ],
        )

        normalized = orchestrator._normalize_research_plan(mock_task, raw_plan)

        assert len(normalized.subtasks) >= 3
        assert any(
            subtask.subtask_type == "verification" for subtask in normalized.subtasks
        )
        assert normalized.subtasks[0].subtask_type == "discovery"
        assert normalized.subtasks[-1].subtask_type == "synthesis"

    def test_claim_verification_brief_labels_claims_by_support(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0
        task_state.subtask_reports[
            0
        ] = """
        The NSF REU program at Example University has a February 1 deadline for undergraduate applicants.
        This opportunity includes an AI and machine learning research track for summer 2026.
        """

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.search_history = [
            {
                "query": "example university reu ai machine learning",
                "results": [
                    {
                        "title": "Example University REU in AI and Machine Learning",
                        "url": "https://example.edu/reu-ai",
                        "domain": "example.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "NSF REU Site",
                        "url": "https://nsf.gov/reu/example",
                        "domain": "nsf.gov",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                ],
            }
        ]
        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        brief = orchestrator._build_claim_verification_brief(mock_task, task_state)

        assert "Verified Claims" in brief
        assert "primary support" in brief

    def test_evidence_status_appendix_policy_uses_only_primary_supported_claims(
        self, orchestrator
    ):
        report = """
## Executive Summary
Summary text.

## Evidence Status Appendix
### Verified Findings
- Unverified item from draft

### Tentative Findings
- Tentative item
"""
        verification_brief = """
## Claim Verification Brief
### Verified Claims
- Brown University has an REU deadline in March. [primary support: brown.edu]

### Tentative Claims
- Another item. [partial support: example.com]
"""

        rewritten = orchestrator._enforce_evidence_status_appendix_policy(
            report,
            verification_brief,
        )

        assert "## Evidence Status Appendix" in rewritten
        assert "### Verified Findings" in rewritten
        assert "Brown University has an REU deadline in March" in rewritten
        assert "Unverified item from draft" not in rewritten

    def test_sanitize_final_report_output_strips_preamble_and_keeps_sections(
        self, orchestrator
    ):
        report = """
Okay, here's a consolidated, actionable final report.

## Final Report: Example

**Executive Summary:** This report identifies opportunities.

## Executive Summary
Brown University appears to have a relevant REU track.

## Scope and Method
Used official program pages and checkpoint notes.

## Findings
- Brown University has an REU deadline in March.

## Comparative Analysis
- Brown is more clearly documented than other examples.

## Limitations and Open Questions
- Missing funding amount

## Recommended Next Steps
1. Check the official page
"""
        verification_brief = """
## Claim Verification Brief
### Verified Claims
- Brown University has an REU deadline in March. [primary support: brown.edu]
"""

        rewritten = orchestrator._sanitize_final_report_output(
            report,
            verification_brief,
        )

        assert not rewritten.startswith("Okay,")
        assert "## Final Report:" not in rewritten
        assert "**Executive Summary:**" not in rewritten
        assert rewritten.startswith("## Executive Summary")
        assert "## Evidence Status Appendix" in rewritten
        assert "### Verified Findings" in rewritten

    def test_sanitize_final_report_output_normalizes_inline_headings_and_replaces_appendix(
        self, orchestrator
    ):
        report = (
            "## Executive Summary Summary text. "
            "## Scope and Method Used official pages. "
            "## Findings Brown University has an REU deadline in March. "
            "## Comparative Analysis Official pages are stronger. "
            "## Limitations and Open Questions Missing funding amount. "
            "## Recommended Next Steps Check the official page. "
            "## Evidence Status Appendix ### Verified Findings - Draft fragment"
        )
        verification_brief = """
## Claim Verification Brief
### Verified Claims
- Brown University has an REU deadline in March. [primary support: brown.edu]
"""

        rewritten = orchestrator._sanitize_final_report_output(
            report,
            verification_brief,
        )

        assert "\n## Scope and Method\n" in rewritten
        assert "\n## Findings\n" in rewritten
        assert "\n### Verified Findings\n" in rewritten
        assert rewritten.count("## Evidence Status Appendix") == 1
        assert "Draft fragment" not in rewritten

    def test_sanitize_final_report_output_puts_third_level_heading_on_its_own_line(
        self, orchestrator
    ):
        report = """
## Executive Summary
Summary text.

## Scope and Method
Method text.

## Findings
### Current Opportunities * Brown REU is active. * CMU program is selective.

## Comparative Analysis
Comparison text.

## Limitations and Open Questions
Open question.

## Recommended Next Steps
Next step.
"""
        verification_brief = """
## Claim Verification Brief
### Verified Claims
- Brown REU is active. [primary support: brown.edu]
"""

        rewritten = orchestrator._sanitize_final_report_output(
            report,
            verification_brief,
        )

        assert "### Current Opportunities\n* Brown REU is active." in rewritten

    def test_sanitize_final_report_output_dedupes_duplicate_top_level_sections(
        self, orchestrator
    ):
        report = """
## Executive Summary
Summary text.

## Scope and Method
Method text.

## Findings
Finding block one.

## Comparative Analysis
Short comparison.

## Comparative Analysis
Longer comparison with specific Brown University and CMU distinctions.

## Limitations and Open Questions
Open question.

## Recommended Next Steps
Next step.
"""
        verification_brief = """
## Claim Verification Brief
### Verified Claims
- Brown University has an REU deadline in March. [primary support: brown.edu]
"""

        rewritten = orchestrator._sanitize_final_report_output(
            report,
            verification_brief,
        )

        assert rewritten.count("## Comparative Analysis") == 1
        assert (
            "Longer comparison with specific Brown University and CMU distinctions."
            in rewritten
        )

    def test_report_has_unknown_urls_detects_non_evidenced_links(self, orchestrator):
        report = "Use https://example.edu/program and https://invented.example.com/page"
        assert orchestrator._report_has_unknown_urls(
            report,
            allowed_urls=["https://example.edu/program"],
        )

    def test_claim_verification_brief_falls_back_to_primary_source_anchors(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0
        task_state.subtask_reports[
            0
        ] = "Executive Summary: this report identifies several opportunities."

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.search_history = [
            {
                "query": "brown ai reu official",
                "results": [
                    {
                        "title": "Brown AI for Computational Creativity REU",
                        "url": "https://aireu.cs.brown.edu/",
                        "domain": "brown.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
            }
        ]
        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        brief = orchestrator._build_claim_verification_brief(mock_task, task_state)

        assert "### Verified Claims" in brief
        assert (
            "Brown AI for Computational Creativity REU is an official source" in brief
        )

    def test_evidence_status_appendix_policy_prefers_typed_verified_items(
        self, orchestrator
    ):
        report = """
## Executive Summary
Summary text.
"""
        verification_brief = (
            "## Claim Verification Brief\n- No claim candidates extracted yet."
        )
        verified_items = [
            VerifiedEvidenceItem(
                statement="Brown AI for Computational Creativity REU is an official source relevant to this task.",
                title="Brown AI for Computational Creativity REU",
                domain="brown.edu",
                url="https://aireu.cs.brown.edu/",
            )
        ]

        rewritten = orchestrator._enforce_evidence_status_appendix_policy(
            report,
            verification_brief,
            verified_items=verified_items,
        )

        assert "## Evidence Status Appendix" in rewritten
        assert (
            "Brown AI for Computational Creativity REU is an official source relevant to this task."
            in rewritten
        )
        assert "noisy draft fragment" not in rewritten

    def test_build_verified_evidence_items_prefers_primary_supported_claims(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_index = 0
        task_state.subtask_reports[
            0
        ] = "Brown University AI REU deadline is March 15, 2026 according to the official program page."

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.search_history = [
            {
                "query": "brown university ai reu deadline 2026 official",
                "results": [
                    {
                        "title": "Brown AI for Computational Creativity REU",
                        "url": "https://aireu.cs.brown.edu/",
                        "domain": "brown.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
            }
        ]
        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        items = orchestrator._build_verified_evidence_items(mock_task, task_state)

        assert items
        assert any(
            "deadline is march 15, 2026" in item.statement.lower() for item in items
        )
        assert not all(
            "official source relevant to this task" in item.statement.lower()
            for item in items
        )

    def test_build_structured_evidence_packet_includes_lineage_and_excerpts(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_reports[
            0
        ] = """
Subtask 1: Discovery
Latest Notes:
Brown University AI REU deadline is March 15, 2026 according to the official program page.
"""

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.search_history = [
            {
                "query": "brown university ai reu deadline 2026 official",
                "results": [
                    {
                        "title": "Brown AI for Computational Creativity REU",
                        "url": "https://aireu.cs.brown.edu/",
                        "summary": "Summer research program with application information.",
                        "content_excerpt": "The page lists a March 15, 2026 application deadline and undergraduate eligibility details.",
                        "domain": "brown.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
            }
        ]
        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        verified_items = orchestrator._build_verified_evidence_items(
            mock_task, task_state
        )
        verification_brief = orchestrator._build_claim_verification_brief(
            mock_task, task_state
        )
        packet = orchestrator._build_structured_evidence_packet(
            mock_task,
            task_state,
            verification_brief=verification_brief,
            verified_items=verified_items,
        )

        assert "### Verified Findings With Source Lineage" in packet
        assert (
            "Brown AI for Computational Creativity REU | brown.edu | https://aireu.cs.brown.edu/"
            in packet
        )
        assert "March 15, 2026 application deadline" in packet

    def test_build_verified_evidence_items_matches_claims_to_correct_sources(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_reports[
            0
        ] = """
Discovery and verification summary:
- MLH Fellowship is a current official lead with a 12-week remote format and mentorship.
- Fellowship.AI is a direct official program lead focused on AI work and projects.
- Stevens Institute of Technology lists an AI Research Summer Fellowship Program with rolling applications.
- Flagship Pioneering Fellows is a strong but more selective deep-tech fellowship lead.
"""

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.search_history = [
            {
                "query": "ml fellowships official ai research undergraduates",
                "results": [
                    {
                        "title": "MLH Fellowship",
                        "url": "https://fellowship.mlh.io/",
                        "summary": "Official MLH Fellowship overview page.",
                        "content_excerpt": "The MLH Fellowship is a 12-week remote internship alternative where fellows contribute to production software and receive mentorship from engineers and maintainers.",
                        "domain": "fellowship.mlh.io",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "Fellowship.AI",
                        "url": "https://www.fellowship.ai/",
                        "summary": "Official Fellowship.AI landing page.",
                        "content_excerpt": "Fellowship.AI presents an application-based fellowship model centered on AI work and project experience.",
                        "domain": "fellowship.ai",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "Stevens Institute of Technology AI Research Summer Fellowship Program",
                        "url": "https://www.stevens.edu/ai-research-summer-fellowship",
                        "summary": "Official Stevens program page.",
                        "content_excerpt": "The Stevens page describes an AI Research Summer Fellowship Program with rolling applications and research participation for students in an academic setting.",
                        "domain": "stevens.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                    {
                        "title": "Flagship Pioneering Fellows",
                        "url": "https://www.flagshippioneering.com/fellows/",
                        "summary": "Official Flagship fellows page.",
                        "content_excerpt": "Flagship Pioneering describes a fellows program tied to deep-tech venture creation and research-intensive innovation.",
                        "domain": "flagshippioneering.com",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    },
                ],
            }
        ]
        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        items = orchestrator._build_verified_evidence_items(mock_task, task_state)
        url_to_statement = {item.url: item.statement for item in items}

        assert (
            "12-week remote internship alternative"
            in url_to_statement["https://fellowship.mlh.io/"]
        )
        assert (
            "application-based fellowship model"
            in url_to_statement["https://www.fellowship.ai/"]
        )
        assert (
            "rolling applications"
            in url_to_statement["https://www.stevens.edu/ai-research-summer-fellowship"]
        )
        assert (
            "deep-tech"
            in url_to_statement["https://www.flagshippioneering.com/fellows/"]
        )
        assert len(url_to_statement) == len(items)

    def test_build_primary_record_claims_uses_excerpt_backed_statements(
        self, orchestrator
    ):
        claims = orchestrator._build_primary_record_claims(
            [
                SourceEvidenceRecord(
                    subtask=1,
                    title="MLH Fellowship",
                    domain="fellowship.mlh.io",
                    url="https://fellowship.mlh.io/",
                    official=True,
                    authority_score=4,
                    source_tier="authoritative",
                    fetched=True,
                    query="ml fellowship official",
                    content_excerpt="The MLH Fellowship is a 12-week remote internship alternative with mentorship from engineers and maintainers.",
                    tokens=set(),
                    title_tokens={"mlh", "fellowship"},
                    domain_tokens={"fellowship", "mlh"},
                    query_tokens={"ml", "fellowship", "official"},
                    evidence_tokens={
                        "mlh",
                        "fellowship",
                        "12",
                        "week",
                        "remote",
                        "internship",
                        "alternative",
                        "mentorship",
                        "engineers",
                        "maintainers",
                    },
                )
            ]
        )

        assert claims
        assert "12-week remote internship alternative" in claims[0]
        assert "official source relevant to this task" not in claims[0]

    def test_extract_open_uncertainty_lines_preserves_gap_style_bullets(
        self, orchestrator
    ):
        brief = """
## Claim Verification Brief

### Unsupported or Weakly Supported Claims
- MLH Fellowship: Current application timing is not directly confirmed in the retained evidence.
- Fellowship.AI: Senior-undergraduate fit is not directly confirmed in the retained evidence.
"""
        lines = orchestrator._extract_open_uncertainty_lines(brief)

        assert lines == [
            "- MLH Fellowship: Current application timing is not directly confirmed in the retained evidence",
            "- Fellowship.AI: Senior-undergraduate fit is not directly confirmed in the retained evidence",
        ]

    @pytest.mark.asyncio
    async def test_synthesize_final_report_strips_unknown_urls_but_keeps_known_sources(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_reports[
            0
        ] = """
Subtask 1: Discovery
Latest Notes:
Brown University AI REU deadline is March 15, 2026 according to the official program page.
"""

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.search_history = [
            {
                "query": "brown ai reu official deadline",
                "results": [
                    {
                        "title": "Brown AI for Computational Creativity REU",
                        "url": "https://aireu.cs.brown.edu/",
                        "summary": "Official program page for Brown's AI REU.",
                        "content_excerpt": "Applications close March 15, 2026 for undergraduate applicants.",
                        "domain": "brown.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
            }
        ]
        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        orchestrator.report_chains = Mock()
        orchestrator.report_chains.asynthesize = AsyncMock(
            return_value="""
## Executive Summary
Brown University appears to have a relevant REU opportunity for undergraduates. The official page is [Brown AI REU](https://aireu.cs.brown.edu/). Another option is [MLH Fellowship](https://mlhfellowship.com/).

## Scope and Method
Used official pages and prior findings.

## Findings
Brown AI for Computational Creativity REU on brown.edu lists a March 15, 2026 deadline for undergraduates.

## Comparative Analysis
Brown has stronger official documentation than the invented alternative.

## Limitations and Open Questions
Funding amount still needs direct confirmation.

## Recommended Next Steps
Check Brown's official page first.

## Evidence Status Appendix
### Verified Findings
- Placeholder

### Tentative Findings
- Placeholder

### Open Uncertainties
- Placeholder
"""
        )
        orchestrator.report_chains.arepair_synthesis = AsyncMock()

        report = await orchestrator.synthesize_final_report(
            mock_task, ["Subtask report"]
        )

        assert "https://aireu.cs.brown.edu/" in report
        assert "https://mlhfellowship.com/" not in report
        assert "[URL_" not in report

    @pytest.mark.asyncio
    async def test_synthesize_final_report_falls_back_when_generic_report_lacks_grounding(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_reports[
            0
        ] = """
Subtask 1: Discovery
Latest Notes:
Brown University AI REU deadline is March 15, 2026 according to the official program page.
"""

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.search_history = [
            {
                "query": "brown ai reu official deadline",
                "results": [
                    {
                        "title": "Brown AI for Computational Creativity REU",
                        "url": "https://aireu.cs.brown.edu/",
                        "summary": "Official program page for Brown's AI REU.",
                        "content_excerpt": "Applications close March 15, 2026 for undergraduate applicants.",
                        "domain": "brown.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
            }
        ]
        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        generic_report = """
## Executive Summary
This report reviews the strongest available evidence and presents a practical overview for the user.

## Scope and Method
Multiple sources were reviewed and compared to identify likely opportunities.

## Findings
Several promising opportunities exist and the strongest ones should be prioritized first.

## Comparative Analysis
Official sources are generally stronger than aggregator sources for accuracy.

## Limitations and Open Questions
Some details remain unclear and require further checking.

## Recommended Next Steps
Review the strongest opportunities and verify current details.

## Evidence Status Appendix
### Verified Findings
- Placeholder

### Tentative Findings
- Placeholder

### Open Uncertainties
- Placeholder
"""
        orchestrator.report_chains = Mock()
        orchestrator.report_chains.asynthesize = AsyncMock(return_value=generic_report)
        orchestrator.report_chains.arepair_synthesis = AsyncMock(
            return_value=generic_report
        )

        report = await orchestrator.synthesize_final_report(
            mock_task, ["Subtask report"]
        )

        assert "Brown University AI REU deadline is March 15, 2026" in report
        assert "Brown AI for Computational Creativity REU" in report

    @pytest.mark.asyncio
    async def test_synthesize_final_report_rejects_unknown_urls_and_falls_back(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_reports[
            0
        ] = """
Subtask 1: Discovery
Latest Notes:
Brown University AI REU deadline is March 15, 2026 according to the official program page.
"""

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.search_history = [
            {
                "query": "brown ai reu official deadline",
                "results": [
                    {
                        "title": "Brown AI for Computational Creativity REU",
                        "url": "https://aireu.cs.brown.edu/",
                        "summary": "Official Brown AI REU page.",
                        "content_excerpt": "Applications close March 15, 2026 for undergraduate applicants.",
                        "domain": "brown.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
            }
        ]
        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        bad_report = """
## Executive Summary
Brown AI REU is active. Visit https://www.brown.edu/reu for details.

## Scope and Method
Used official pages.

## Findings
Brown AI for Computational Creativity REU appears current.

## Comparative Analysis
Brown is stronger than other programs.

## Limitations and Open Questions
Funding is unclear.

## Recommended Next Steps
Check https://www.brown.edu/reu now.

## Evidence Status Appendix
### Verified Findings
- Placeholder

### Tentative Findings
- Placeholder

### Open Uncertainties
- Placeholder
"""
        orchestrator.report_chains = Mock()
        orchestrator.report_chains.asynthesize = AsyncMock(return_value=bad_report)
        orchestrator.report_chains.arepair_synthesis = AsyncMock(
            return_value=bad_report
        )

        report = await orchestrator.synthesize_final_report(
            mock_task, ["Subtask report"]
        )

        assert "https://www.brown.edu/reu" not in report
        assert "https://aireu.cs.brown.edu/" in report

    @pytest.mark.asyncio
    async def test_synthesize_final_report_rejects_unsupported_certainty_and_prefers_deterministic(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_reports[
            0
        ] = """
Discovery and verification summary:
- Brown University AI REU deadline is March 15, 2026 according to the official program page.
"""

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.search_history = [
            {
                "query": "brown ai reu official deadline",
                "results": [
                    {
                        "title": "Brown AI for Computational Creativity REU",
                        "url": "https://aireu.cs.brown.edu/",
                        "summary": "Official Brown AI REU page.",
                        "content_excerpt": "Applications close March 15, 2026 for undergraduate applicants.",
                        "domain": "brown.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
            }
        ]
        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        overclaiming_report = """
## Executive Summary
This report identifies currently open opportunities that are immediately actionable for senior undergraduate students.

## Scope and Method
Used official pages.

## Findings
Brown AI for Computational Creativity REU is currently open to senior undergraduate students.

## Comparative Analysis
Brown is the clear best current option.

## Limitations and Open Questions
Funding is unclear.

## Recommended Next Steps
Apply immediately.

## Evidence Status Appendix
### Verified Findings
- Placeholder

### Tentative Findings
- Placeholder

### Open Uncertainties
- Placeholder
"""
        orchestrator.report_chains = Mock()
        orchestrator.report_chains.asynthesize = AsyncMock(
            return_value=overclaiming_report
        )
        orchestrator.report_chains.arepair_synthesis = AsyncMock(
            return_value=overclaiming_report
        )

        report = await orchestrator.synthesize_final_report(
            mock_task, ["Subtask report"]
        )

        assert "currently open opportunities" not in report.lower()
        assert "immediately actionable" not in report.lower()
        assert "What the retained evidence supports" in report
        assert "Still unclear from the retained evidence" in report

    def test_sanitize_claim_candidate_rejects_query_and_metadata_fragments(
        self, orchestrator
    ):
        assert (
            orchestrator._sanitize_claim_candidate(
                "[SUMMARY] - The 2026 NSF REU applications are not currently open"
            )
            == "The 2026 NSF REU applications are not currently open"
        )
        assert (
            orchestrator._sanitize_claim_candidate(
                "Query: reu nsf ai detailed of cmu planning deadline (results=3)"
            )
            is None
        )
        assert (
            orchestrator._sanitize_claim_candidate(
                "NSF REU - NSF AI Planning Institute for Data-Driven Discovery | domain=cmu.edu | tier=authoritative | authority=4"
            )
            is None
        )

    def test_meta_critique_detection_triggers_on_review_style_output(
        self, orchestrator
    ):
        """Final synthesis should flag common report-review drift language."""
        bad_output = """
        Okay, this is a remarkably thorough and well-structured report.
        Overall Assessment:
        Strengths:
        Would you like me to elaborate on these suggestions for improvement?
        """
        assert orchestrator._is_meta_critique_output(
            bad_output,
            "Research current grants for undergraduate AI/ML students in the US.",
        )

    def test_meta_critique_detection_allows_explicit_review_tasks(self, orchestrator):
        """If task explicitly requests a critique, review-style language is allowed."""
        review_style_output = (
            "Overall Assessment: Strengths and weaknesses of the report."
        )
        assert not orchestrator._is_meta_critique_output(
            review_style_output,
            "Critique and evaluate this report quality with strengths and weaknesses.",
        )

    def test_strip_control_channel_annotations_removes_orchestrator_feedback(
        self, orchestrator
    ):
        raw = """
[SEARCH] nsf ai reu
Found something useful.

[ORCHESTRATOR FEEDBACK] Continue gathering evidence for the current subtask using diversified, success-criteria-aligned queries.
"""
        cleaned = orchestrator._strip_control_channel_annotations(raw)

        assert "[SEARCH]" not in cleaned
        assert "[ORCHESTRATOR FEEDBACK]" not in cleaned
        assert "Found something useful." in cleaned

    def test_build_subtask_report_snapshot_strips_control_feedback(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan

        snapshot = orchestrator._build_subtask_report_snapshot(
            mock_task,
            task_state,
            0,
            "Concrete finding line.\n\n[ORCHESTRATOR FEEDBACK] Avoid repeating recent queries.",
        )

        assert "Concrete finding line." in snapshot
        assert "[ORCHESTRATOR FEEDBACK]" not in snapshot
        assert "Avoid repeating recent queries" not in snapshot

    def test_sanitize_claim_candidate_rejects_orchestrator_guidance_fragment(
        self, orchestrator
    ):
        assert (
            orchestrator._sanitize_claim_candidate(
                "ml fellowships find open diversified success-criteria-aligned queries"
            )
            is None
        )

    def test_build_verified_evidence_items_ignores_orchestrator_feedback_fragments(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_reports[
            0
        ] = """
Subtask 1: Discovery
Latest Notes:
[ORCHESTRATOR FEEDBACK] Continue gathering evidence for the current subtask using diversified, success-criteria-aligned queries.
Brown University AI REU deadline is March 15, 2026 according to the official program page.
"""

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.search_history = [
            {
                "query": "brown ai reu official deadline",
                "results": [
                    {
                        "title": "Brown AI for Computational Creativity REU",
                        "url": "https://aireu.cs.brown.edu/",
                        "domain": "brown.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
            }
        ]
        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        items = orchestrator._build_verified_evidence_items(mock_task, task_state)

        assert items
        assert any(
            "Brown University AI REU deadline is March 15, 2026" in item.statement
            for item in items
        )
        assert not any(
            "success-criteria-aligned queries" in item.statement for item in items
        )

    @pytest.mark.asyncio
    async def test_synthesize_final_report_rejects_meta_critique_and_uses_deterministic_report(
        self, orchestrator, mock_task, mock_research_plan
    ):
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan
        task_state.subtask_reports[
            0
        ] = """
Subtask 1: Discovery
Latest Notes:
Brown University AI REU deadline is March 15, 2026 according to the official program page.
"""

        agent_state = AgentState(task_id=mock_task.name)
        agent_state.search_history = [
            {
                "query": "brown ai reu official deadline",
                "results": [
                    {
                        "title": "Brown AI for Computational Creativity REU",
                        "url": "https://aireu.cs.brown.edu/",
                        "domain": "brown.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
            }
        ]
        mock_subtask_agent = Mock()
        mock_subtask_agent._state = {mock_task.name: agent_state}
        mock_subtask_agent.config = Mock(orchestrator_guidance="")
        task_state.subtask_agents[0] = mock_subtask_agent

        orchestrator.report_chains = Mock()
        orchestrator.report_chains.asynthesize = AsyncMock(
            return_value="""
Here’s a breakdown of why this report is strong.

**Strengths:**
- Great narrative structure.

**Overall:**
This is an outstanding final research report.
"""
        )
        orchestrator.report_chains.arepair_synthesis = AsyncMock(
            return_value="""
**Minor Suggestions for Refinement:**
- Add more detail.
"""
        )

        report = await orchestrator.synthesize_final_report(
            mock_task,
            ["Subtask report body"],
        )

        assert "## Executive Summary" in report
        assert "## Findings" in report
        assert "## Evidence Status Appendix" in report
        assert "Brown University AI REU deadline is March 15, 2026" in report
        assert "Strengths" not in report
        assert "outstanding final research report" not in report.lower()


@pytest.mark.unit
class TestWebResearchAgent:
    """Test web research agent functionality."""

    @pytest.fixture
    def research_agent(self, mock_llm):
        """Create research agent instance for testing."""
        mock_memory = Mock()
        return WebResearchAgent("test-agent", mock_llm, mock_memory)

    def test_agent_initialization(self, research_agent, mock_llm):
        """Test research agent initialization."""
        assert research_agent.llm == mock_llm
        assert research_agent.config is not None
        assert research_agent.memory is not None

    @pytest.mark.asyncio
    async def test_web_search(self, research_agent):
        """Test web search handling populates agent state."""
        state = AgentState(task_id="test-task")
        plan = {"query": "test query"}
        research_agent.tool_executor = Mock()
        research_agent.tool_executor.ainvoke = AsyncMock(
            return_value={
                "hits": [
                    {
                        "title": "Test Result",
                        "url": "https://example.com",
                        "snippet": "Test summary",
                        "summary": "Official summary text",
                        "content": "Fetched page body with concrete details",
                        "domain": "example.com",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
                "search_time": 0.12,
            }
        )

        await research_agent._handle_search_action(plan, state)

        assert state.search_attempt_count == 1
        assert state.search_count == 1
        assert any("Test Result" in entry for entry in state.history)
        assert state.search_history
        stored = state.search_history[0]["results"][0]
        assert stored["summary"] == "Official summary text"
        assert "Fetched page body" in stored["content_excerpt"]
        assert stored["task_aligned"] is True
        assert stored["task_relevance_score"] > 0

    @pytest.mark.asyncio
    async def test_web_search_records_discovery_pathway_hits(self, research_agent):
        state = AgentState(task_id="test-task")
        state.search_pathways = [
            {
                "id": "pathway_1",
                "label": "official org path",
                "source_terms": ["official", "organization"],
                "focus_terms": ["fellowship"],
                "evidence_terms": ["deadline"],
                "attempt_count": 0,
                "hit_count": 0,
            }
        ]
        state.register_pathway_query("test query", "pathway_1")
        plan = {"query": "test query"}
        research_agent.tool_executor = Mock()
        research_agent.tool_executor.ainvoke = AsyncMock(
            return_value={
                "hits": [
                    {
                        "title": "Official Fellowship Page",
                        "url": "https://example.org/fellowship",
                        "snippet": "Official fellowship application now open.",
                        "summary": "Official fellowship details.",
                        "content": "Deadline April 1. Eligibility includes senior undergraduates.",
                        "domain": "example.org",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
                "search_time": 0.11,
            }
        )

        await research_agent._handle_search_action(plan, state)

        assert state.search_pathways[0]["attempt_count"] == 1
        assert state.search_pathways[0]["hit_count"] == 1
        assert state.search_history[0]["pathway_id"] == "pathway_1"

    @pytest.mark.asyncio
    async def test_web_search_persists_evidence_focused_excerpt(self, research_agent):
        state = AgentState(task_id="test-task")
        plan = {
            "query": "primary organization machine learning ai research undergraduate us reu"
        }
        research_agent.tool_executor = Mock()
        research_agent.tool_executor.ainvoke = AsyncMock(
            return_value={
                "hits": [
                    {
                        "title": "REU Combinatorics, Algorithms, and AI for Real Problems",
                        "url": "https://www.cs.umd.edu/projects/reucaar/",
                        "snippet": "AI research experience for undergraduates.",
                        "summary": "Summer REU in combinatorics, algorithms, and AI for undergraduates.",
                        "content": (
                            "Research Experience for Undergraduates (REU) Combinatorics, Algorithms, and AI for Real Problems. "
                            "Research Experience for Undergraduates (REU) Combinatorics, Algorithms, and AI for Real Problems. "
                            "THE PROGRAM FOR 2026 RUNS JUNE 1-AUG 14. "
                            "0) DEADLINE TO APPLY: March 3, 2026. "
                            "1) Look at the projects. 2) Gather your CV and transcript. "
                            "3) Write a statement of purpose telling us which projects you want to work on. "
                            "4) If you don't have the prereq for a project you want to work on, explain that in your statement. "
                            "5) Anyone who is a Freshman OR Sophmore OR Junior OR Senior in a College in America can apply."
                        ),
                        "domain": "cs.umd.edu",
                        "source_tier": "authoritative",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
                "search_time": 0.12,
            }
        )

        await research_agent._handle_search_action(plan, state)

        stored = state.search_history[0]["results"][0]
        lowered_excerpt = stored["content_excerpt"].lower()
        assert "deadline to apply: march 3, 2026." in lowered_excerpt
        assert "statement of purpose" not in lowered_excerpt

    @pytest.mark.asyncio
    async def test_web_search_failure(self, research_agent):
        """Test web search gracefully handles no results."""
        state = AgentState(task_id="test-task")
        plan = {"query": "test query 2026"}
        research_agent.tool_executor = Mock()
        research_agent.tool_executor.ainvoke = AsyncMock(
            side_effect=[
                {"hits": [], "search_time": 0.08},
                {"hits": [], "search_time": 0.05},
            ]
        )

        await research_agent._handle_search_action(plan, state)

        assert state.search_attempt_count == 2
        assert state.search_count == 0
        assert state.zero_result_search_count == 1
        assert state.history[-1] == "[NO RESULTS] test query 2026"

    @pytest.mark.asyncio
    async def test_web_search_filters_low_signal_funding_news_for_opportunity_tasks(
        self, research_agent
    ):
        state = AgentState(task_id="test-task")
        plan = {
            "query": "site:.org machine learning ai research undergraduate us funding"
        }
        research_agent.config.subtask_description = "Discovery: find currently open AI/ML fellowships, grants, programs, or funding opportunities for senior undergraduates in the US."
        research_agent.config.success_criteria = "Identify authoritative opportunities with direct application or eligibility details."
        research_agent.tool_executor = Mock()
        research_agent.tool_executor.ainvoke = AsyncMock(
            side_effect=[
                {
                    "hits": [
                        {
                            "title": "NCSA awarded funding to continue AI-focused NSF REU program | EurekAlert!",
                            "url": "https://www.eurekalert.org/news-releases/123456",
                            "snippet": "News Release 24-Feb-2025 NCSA awarded funding to continue AI-focused NSF REU program. Applications are expected next cycle.",
                            "summary": "Grant and Award Announcement about renewed NSF REU funding and future applications.",
                            "content": "News release announcing funding to continue the REU program. Media contact included. Applications may open later.",
                            "domain": "eurekalert.org",
                            "source_tier": "authoritative",
                            "authority_score": 4,
                            "official_source": False,
                            "fetched": True,
                        }
                    ],
                    "search_time": 0.12,
                },
                {"hits": [], "search_time": 0.05},
            ]
        )

        await research_agent._handle_search_action(plan, state)

        assert state.search_attempt_count == 2
        assert state.search_count == 0
        assert state.zero_result_search_count == 1
        assert state.search_history
        assert state.search_history[0]["results_count"] == 0
        assert state.search_history[0]["raw_results_count"] == 1
        assert (
            state.history[-1]
            == "[NO RESULTS] site:.org machine learning ai research undergraduate us funding"
        )

    @pytest.mark.asyncio
    async def test_web_search_filters_admissions_style_fellowship_pages_without_topic_match(
        self, research_agent
    ):
        state = AgentState(task_id="test-task")
        plan = {
            "query": "site:.edu machine learning ai research undergraduate us fellowship"
        }
        research_agent.config.subtask_description = "Discovery: find currently open AI/ML fellowships, grants, programs, or funding opportunities for senior undergraduates in the US."
        research_agent.config.success_criteria = "Identify authoritative opportunities with direct application or eligibility details."
        research_agent.tool_executor = Mock()
        research_agent.tool_executor.ainvoke = AsyncMock(
            side_effect=[
                {
                    "hits": [
                        {
                            "title": "Daben Liu Research Fellowship - Admission & Student Engagement",
                            "url": "https://example.edu/research/daben-liu-fellowship",
                            "snippet": "Machine learning AI undergraduate research fellowship result snippet from search indexing.",
                            "summary": "Undergraduate fellowship program run through student engagement.",
                            "content": "A fellowship page hosted under admission and student engagement for campus leadership and community service.",
                            "domain": "example.edu",
                            "source_tier": "authoritative",
                            "authority_score": 4,
                            "official_source": True,
                            "fetched": True,
                        }
                    ],
                    "search_time": 0.12,
                },
                {"hits": [], "search_time": 0.05},
            ]
        )

        await research_agent._handle_search_action(plan, state)

        assert state.search_count == 0
        assert state.zero_result_search_count == 1
        assert state.search_history[0]["results_count"] == 0

    def test_agent_research_cycle(self, research_agent):
        """Test complete research cycle."""
        # Test that we can create a research agent
        assert research_agent is not None
        assert research_agent.llm is not None
        assert research_agent.memory is not None

    def test_build_step_feedback_prefers_last_step_feedback_over_checkpoint(
        self, research_agent
    ):
        state = AgentState(task_id="test-task")
        state.last_checkpoint = "## Old checkpoint report"
        state.last_step_feedback = (
            "[SEARCH] official nsf reu ai\n  Found: Result A, Result B"
        )

        task = Mock()
        task.description = "Test task description"

        feedback = research_agent._build_step_feedback(task, state)

        assert feedback.startswith("[SEARCH]")
        assert "Old checkpoint report" not in feedback

    def test_build_step_feedback_falls_back_to_recent_history_without_last_step_feedback(
        self, research_agent
    ):
        state = AgentState(task_id="test-task")
        state.last_checkpoint = "## Old checkpoint report"
        state.history.append("[THOUGHT] Narrowing the search")
        state.history.append("[SEARCH] nsf reu ai\n  Found: Result A")

        task = Mock()
        task.description = "Test task description"

        feedback = research_agent._build_step_feedback(task, state)

        assert "[SEARCH] nsf reu ai" in feedback
        assert "Old checkpoint report" not in feedback

    def test_sanitize_intermediate_note_removes_report_style_wrapper(
        self, research_agent
    ):
        raw = """## NSF REU Opportunities in AI & Machine Learning – Actionable Shortlist

1. Brown REU deadline is tentative.
2. CMU program page lists application details.
"""

        cleaned = research_agent._sanitize_intermediate_note(
            raw,
            default_prefix="- No concrete evidence extracted yet.",
        )

        assert "Actionable Shortlist" not in cleaned
        assert cleaned.startswith("- ")
        assert "Brown REU deadline is tentative." in cleaned

    def test_build_deterministic_checkpoint_ignores_irrelevant_publication_hits(
        self, research_agent
    ):
        task = Mock()
        task.name = "ML Fellowships"
        task.description = (
            "Find currently open grants, programs, fellowships, or funding "
            "opportunities that support AI/ML research and are available to senior "
            "undergraduate students in the United States."
        )
        state = AgentState(task_id="test-task")
        state.search_history = [
            {
                "query": "site:.org machine learning fellowship undergraduate",
                "results": [
                    {
                        "title": "Revitalization of an undergraduate physics program",
                        "url": "http://arxiv.org/abs/physics/0004028v1",
                        "domain": "arxiv.org",
                        "summary": "undergraduate physics program paper",
                        "content_excerpt": "paper about revitalization of a physics program",
                        "authority_score": 4,
                        "official_source": False,
                        "fetched": True,
                    }
                ],
            }
        ]

        checkpoint = research_agent._build_deterministic_checkpoint(task, state)

        assert "arxiv" not in checkpoint.lower()
        assert "physics program" not in checkpoint.lower()
        assert (
            "No primary-source-backed checkpoint evidence available yet." in checkpoint
        )

    def test_build_deterministic_checkpoint_prefers_evidence_backed_official_leads(
        self, research_agent
    ):
        task = Mock()
        task.name = "ML Fellowships"
        task.description = (
            "Find currently open grants, programs, fellowships, or funding "
            "opportunities that support AI/ML research and are available to senior "
            "undergraduate students in the United States."
        )
        state = AgentState(task_id="test-task")
        state.search_history = [
            {
                "query": "site:.edu machine learning fellowship undergraduate program",
                "results": [
                    {
                        "title": "AI Research Summer Fellowship Program - Stevens Institute of Technology",
                        "url": "https://www.stevens.edu/stevens-institute-for-artificial-intelligence/fellowship",
                        "domain": "stevens.edu",
                        "summary": "AI Research Summer Fellowship Program for undergraduate researchers.",
                        "content_excerpt": "The AI Research Summer Fellowship Program supports undergraduate researchers in artificial intelligence.",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
            }
        ]

        checkpoint = research_agent._build_deterministic_checkpoint(task, state)

        assert "Stevens Institute" in checkpoint
        assert "Official lead" in checkpoint
        assert (
            "Open gap for AI Research Summer Fellowship Program - Stevens Institute of Technology"
            in checkpoint
        )

    def test_build_deterministic_checkpoint_demotes_mixed_audience_leads(
        self, research_agent
    ):
        task = Mock()
        task.description = (
            "Find currently open grants, programs, fellowships, or funding opportunities "
            "that support AI/ML research and are available to senior undergraduate students in the United States."
        )
        state = AgentState(task_id="test-task")
        state.checkpoint_count = 1
        state.search_history = [
            {
                "query": "official machine learning ai research undergraduate us fellowship",
                "results": [
                    {
                        "title": "Astral Fellows - AI Research Fellowship for High School and Undergraduate Students",
                        "url": "https://www.astralfellows.org/",
                        "domain": "astralfellows.org",
                        "summary": "Fellowship for high school, undergraduate, and early-career researchers.",
                        "content_excerpt": "AI research fellowship for high school, undergraduate, and early-career researchers. Apply by March 27.",
                        "authority_score": 3,
                        "official_source": False,
                        "fetched": True,
                        "task_aligned": True,
                    },
                    {
                        "title": "Research Experience for Undergraduates (REU) Combinatorics, Algorithms, and AI for Real Problems",
                        "url": "https://www.cs.umd.edu/projects/reucaar/",
                        "domain": "cs.umd.edu",
                        "summary": "Summer REU in combinatorics, algorithms, and AI for undergraduates.",
                        "content_excerpt": "Research Experience for Undergraduates in Combinatorics, Algorithms, and AI for Real Problems. Deadline to apply: March 3, 2026. Students receive a stipend.",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                        "task_aligned": True,
                    },
                ],
            }
        ]

        checkpoint = research_agent._build_deterministic_checkpoint(task, state)

        first_lead_line = next(
            line for line in checkpoint.splitlines() if "lead:" in line.lower()
        )
        assert "REU" in first_lead_line
        assert "Astral Fellows" in checkpoint

    def test_build_deterministic_checkpoint_prefers_deadline_sentence_in_excerpt(
        self, research_agent
    ):
        task = Mock()
        task.description = "Find currently open AI/ML research programs for senior undergraduate students in the United States."
        state = AgentState(task_id="test-task")
        state.checkpoint_count = 1
        state.search_history = [
            {
                "query": "primary organization machine learning ai research undergraduate us reu",
                "results": [
                    {
                        "title": "Research Experience for Undergraduates (REU) Combinatorics, Algorithms, and AI for Real Problems",
                        "url": "https://www.cs.umd.edu/projects/reucaar/",
                        "domain": "cs.umd.edu",
                        "summary": "Summer REU in combinatorics, algorithms, and AI for undergraduates.",
                        "content_excerpt": (
                            "Research Experience for Undergraduates in Combinatorics, Algorithms, and AI for Real Problems. "
                            "The program runs June 1-August 14 and provides a stipend for participating students. "
                            "Deadline to apply: March 3, 2026. Anyone who is a freshman, sophomore, junior, or senior in a college in America can apply."
                        ),
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                        "task_aligned": True,
                    }
                ],
            }
        ]

        checkpoint = research_agent._build_deterministic_checkpoint(task, state)

        assert "Deadline to apply: March 3, 2026." in checkpoint
        assert "current application timing is not directly confirmed" not in checkpoint
        assert "senior-undergraduate fit is not directly confirmed" not in checkpoint

    def test_build_deterministic_checkpoint_rejects_generic_ai_academic_pages(
        self, research_agent
    ):
        task = Mock()
        task.name = "ML Fellowships"
        task.description = (
            "Find currently open grants, programs, fellowships, or funding "
            "opportunities that support AI/ML research and are available to senior "
            "undergraduate students in the United States."
        )
        state = AgentState(task_id="test-task")
        state.search_history = [
            {
                "query": "site:.edu machine learning ai research undergraduate us program",
                "results": [
                    {
                        "title": "AI Education | University of Houston",
                        "url": "https://www.uh.edu/ai/education/",
                        "domain": "uh.edu",
                        "summary": "Artificial intelligence education faculty colleges resources.",
                        "content_excerpt": "Artificial intelligence at UH with education, faculty, colleges, resources, and news.",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
            }
        ]

        checkpoint = research_agent._build_deterministic_checkpoint(task, state)

        assert "University of Houston" not in checkpoint
        assert (
            "No primary-source-backed checkpoint evidence available yet." in checkpoint
        )

    def test_build_deterministic_checkpoint_rejects_indirect_funding_news_announcements(
        self, research_agent
    ):
        task = Mock()
        task.name = "ML Fellowships"
        task.description = (
            "Find currently open grants, programs, fellowships, or funding "
            "opportunities that support AI/ML research and are available to senior "
            "undergraduate students in the United States."
        )
        state = AgentState(task_id="test-task")
        state.search_history = [
            {
                "query": "site:.org machine learning ai research undergraduate us funding",
                "results": [
                    {
                        "title": "NCSA awarded funding to continue AI-focused NSF REU program | EurekAlert!",
                        "url": "https://www.eurekalert.org/news-releases/123456",
                        "domain": "eurekalert.org",
                        "summary": "Grant and Award Announcement about renewed NSF REU funding and future applications.",
                        "content_excerpt": "News release announcing funding to continue the REU program. Media contact included. Applications may open later.",
                        "authority_score": 4,
                        "official_source": False,
                        "fetched": True,
                    }
                ],
            }
        ]

        checkpoint = research_agent._build_deterministic_checkpoint(task, state)

        assert "EurekAlert" not in checkpoint
        assert (
            "No primary-source-backed checkpoint evidence available yet." in checkpoint
        )

    def test_relax_query_preserves_site_filter_and_opportunity_family(
        self, research_agent
    ):
        relaxed = research_agent._relax_query(
            "site:.edu machine learning ai research undergraduate us fellowship"
        )

        assert "site:.edu" in relaxed
        assert "fellowship" in relaxed
        assert "machine" in relaxed
        assert "learning" in relaxed

    def test_build_deterministic_checkpoint_rejects_degree_pages_with_admissions_language(
        self, research_agent
    ):
        task = Mock()
        task.name = "ML Fellowships"
        task.description = (
            "Find currently open grants, programs, fellowships, or funding "
            "opportunities that support AI/ML research and are available to senior "
            "undergraduate students in the United States."
        )
        state = AgentState(task_id="test-task")
        state.search_history = [
            {
                "query": "site:.edu ai machine learning undergraduate deadline application eligibility requirements",
                "results": [
                    {
                        "title": "Artificial Intelligence and Machine Learning Undergraduate Degree",
                        "url": "https://drexel.edu/cci/academics/undergraduate-programs/bs-artificial-intelligence-machine-learning/",
                        "domain": "drexel.edu",
                        "summary": "Undergraduate degree admissions and curriculum.",
                        "content_excerpt": "Admissions, curriculum, and undergraduate degree application deadlines for the BS in artificial intelligence and machine learning.",
                        "authority_score": 4,
                        "official_source": True,
                        "fetched": True,
                    }
                ],
            }
        ]

        checkpoint = research_agent._build_deterministic_checkpoint(task, state)

        assert "Undergraduate Degree" not in checkpoint
        assert (
            "No primary-source-backed checkpoint evidence available yet." in checkpoint
        )

    def test_agent_performance(self, research_agent, performance_thresholds):
        """Test agent performance."""
        # Test that we can create a research agent quickly
        import time

        start_time = time.time()

        # Just test initialization performance
        assert research_agent is not None

        operation_time = time.time() - start_time
        assert operation_time < performance_thresholds["task_creation_time"]
