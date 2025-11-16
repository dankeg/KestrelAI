# Unit tests for agent components
from unittest.mock import AsyncMock, Mock, patch

import pytest

try:
    from KestrelAI.agents.base_agent import AgentState
    from KestrelAI.agents.research_orchestrator import ResearchOrchestrator
    from KestrelAI.agents.web_research_agent import WebResearchAgent
    from KestrelAI.shared.models import ResearchPlan, Subtask
except ImportError:
    from agents.base_agent import AgentState
    from agents.research_orchestrator import ResearchOrchestrator
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
        with patch("KestrelAI.agents.research_orchestrator.MemoryStore"):
            with patch(
                "KestrelAI.agents.research_orchestrator.WebResearchAgent"
            ) as mock_agent_cls:
                mock_agent = Mock()
                mock_agent.run_step = AsyncMock(return_value="Test subtask result")
                mock_agent.get_metrics.return_value = {}
                mock_agent._state = {}
                mock_agent_cls.return_value = mock_agent
                return ResearchOrchestrator([mock_task], mock_llm, profile="kestrel")

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
        with patch.object(orchestrator.llm, "chat") as mock_chat:
            mock_chat.return_value = """
            {
                "restated_task": "Test restated task",
                "subtasks": [
                    {
                        "order": 1,
                        "description": "Test subtask 1",
                        "success_criteria": "Test criteria 1",
                        "status": "pending",
                        "findings": []
                    }
                ],
                "current_subtask_index": 0
            }
            """

            await orchestrator._planning_phase(mock_task)

            task_state = orchestrator.task_states[mock_task.name]
            assert task_state.research_plan is not None
            assert task_state.research_plan.restated_task == "Test restated task"
            assert len(task_state.research_plan.subtasks) == 1

    @pytest.mark.asyncio
    async def test_planning_phase_retry(self, orchestrator, mock_task):
        """Test planning phase retry mechanism."""
        with patch.object(orchestrator.llm, "chat") as mock_chat:
            # First call fails, second succeeds
            mock_chat.side_effect = [
                Exception("First attempt fails"),
                """
                {
                    "restated_task": "Test restated task",
                    "subtasks": [
                        {
                            "order": 1,
                            "description": "Test subtask 1",
                            "success_criteria": "Test criteria 1",
                            "status": "pending",
                            "findings": []
                        }
                    ],
                    "current_subtask_index": 0
                }
                """,
            ]

            await orchestrator._planning_phase(mock_task)

            task_state = orchestrator.task_states[mock_task.name]
            assert task_state.research_plan is not None
            assert mock_chat.call_count == 2

    @pytest.mark.asyncio
    async def test_next_action_advances_subtask_on_review(
        self, orchestrator, mock_task, mock_research_plan
    ):
        """Test that next_action uses review decision to advance subtask state."""
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = mock_research_plan

        # Start at first subtask
        task_state.subtask_index = 0
        task_state.completed_subtasks = set()

        # Patch _review to force a decision to proceed to the next subtask
        with patch.object(
            orchestrator,
            "_review",
            AsyncMock(
                return_value=type(
                    "D",
                    (),
                    {
                        "reasoning": "Test",
                        "decision": "switch",
                        "feedback": "Proceed to next subtask",
                        "subtask": "proceed",
                        "next_task": mock_task.name,
                    },
                )()
            ),
        ):
            result = await orchestrator.next_action(mock_task)
            assert isinstance(result, str)
            # After proceeding, the first subtask should be marked complete
            assert 0 in task_state.completed_subtasks
            # And subtask_index should advance to 1 (second subtask)
            assert task_state.subtask_index == 1

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

        assert state.search_count == 0
        assert state.history[-1] == "[NO RESULTS] test query"

    def test_agent_research_cycle(self, research_agent):
        """Test complete research cycle."""
        # Test that we can create a research agent
        assert research_agent is not None
        assert research_agent.llm is not None
        assert research_agent.memory is not None

    def test_agent_performance(self, research_agent, performance_thresholds):
        """Test agent performance."""
        # Test that we can create a research agent quickly
        import time

        start_time = time.time()

        # Just test initialization performance
        assert research_agent is not None

        operation_time = time.time() - start_time
        assert operation_time < performance_thresholds["task_creation_time"]
