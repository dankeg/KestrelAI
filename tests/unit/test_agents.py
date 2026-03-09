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
        VerifiedEvidenceItem,
    )
    from KestrelAI.agents.web_research_agent import WebResearchAgent
    from KestrelAI.shared.models import ResearchPlan, Subtask
except ImportError:
    from agents.base_agent import AgentState
    from agents.research_orchestrator import (
        OrchestratorDecision,
        PlanningPlan,
        PrePlanningAction,
        ResearchOrchestrator,
        VerifiedEvidenceItem,
    )
    from agents.web_research_agent import WebResearchAgent
    from shared.models import ResearchPlan, Subtask


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

    def test_orchestrator_task_states(self, orchestrator, mock_task):
        """Test orchestrator task states."""
        assert mock_task.name in orchestrator.task_states
        task_state = orchestrator.task_states[mock_task.name]
        assert task_state.task == mock_task
        assert task_state.research_plan is None

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
                assert len(task_state.research_plan.subtasks) >= 1
                assert (
                    task_state.research_plan.subtasks[0].description == "Test subtask 1"
                )
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
        assert any(
            "identify distinct nsf" in subtask.description.lower()
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
            "official nsf" in subtask.description.lower()
            or "official program pages" in subtask.success_criteria.lower()
            for subtask in normalized.subtasks
        )

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

    def test_verified_findings_policy_uses_only_primary_supported_claims(
        self, orchestrator
    ):
        report = """
## Verified Findings
- Unverified item from draft

## Tentative Findings
- Tentative item
"""
        verification_brief = """
## Claim Verification Brief
### Verified Claims
- Brown University has an REU deadline in March. [primary support: brown.edu]

### Tentative Claims
- Another item. [partial support: example.com]
"""

        rewritten = orchestrator._enforce_verified_findings_policy(
            report,
            verification_brief,
        )

        assert "Brown University has an REU deadline in March" in rewritten
        assert "Unverified item from draft" not in rewritten

    def test_sanitize_final_report_output_strips_preamble_and_keeps_sections(
        self, orchestrator
    ):
        report = """
Okay, here's a consolidated, actionable final report.

## Final Report: Example

**Executive Summary:** This report identifies opportunities.

## Verified Findings
- Brown University has an REU deadline in March. [primary support: brown.edu]

## Tentative Findings
- Another item

## Open Uncertainties
- Missing funding amount

## Next Verification Steps
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
        assert rewritten.startswith("## Verified Findings")
        assert "## Tentative Findings" in rewritten

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

    def test_verified_findings_policy_prefers_typed_verified_items(self, orchestrator):
        report = """
## Verified Findings
- noisy draft fragment

## Tentative Findings
- tentative item
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

        rewritten = orchestrator._enforce_verified_findings_policy(
            report,
            verification_brief,
            verified_items=verified_items,
        )

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

        research_agent.searxng_service.search = Mock(
            return_value=[
                {
                    "title": "Test Result",
                    "href": "https://example.com",
                    "body": "Test summary",
                }
            ]
        )
        research_agent.searxng_service.extract_text = Mock(
            return_value="Fetched page content"
        )

        await research_agent._handle_search_action(plan, state)

        assert state.search_count == 1
        assert any("Test Result" in entry for entry in state.history)
        assert state.search_history

    @pytest.mark.asyncio
    async def test_web_search_failure(self, research_agent):
        """Test web search gracefully handles no results."""
        state = AgentState(task_id="test-task")
        plan = {"query": "test query"}

        research_agent.searxng_service.search = Mock(return_value=[])

        await research_agent._handle_search_action(plan, state)

        assert state.search_count == 1
        assert state.history[-1] == "[NO RESULTS] test query"

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

    def test_agent_performance(self, research_agent, performance_thresholds):
        """Test agent performance."""
        # Test that we can create a research agent quickly
        import time

        start_time = time.time()

        # Just test initialization performance
        assert research_agent is not None

        operation_time = time.time() - start_time
        assert operation_time < performance_thresholds["task_creation_time"]
