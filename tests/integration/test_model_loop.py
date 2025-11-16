# Integration tests for agent workflow
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

try:
    from KestrelAI.model_loop import KestrelAgentWorker
    from KestrelAI.shared.models import (
        ResearchPlan,
        Task,
        TaskStatus,
    )
    from KestrelAI.shared.models import (
        Subtask as SharedSubtask,
    )
except ImportError:
    from model_loop import KestrelAgentWorker
    from shared.models import ResearchPlan, Task, TaskStatus
    from shared.models import Subtask as SharedSubtask


@pytest.mark.integration
class TestKestrelAgentWorkerIntegration:
    """Test model loop integration functionality."""

    @pytest.fixture
    def model_loop(self, mock_redis, mock_llm):
        """Create model loop instance for testing."""
        with patch(
            "KestrelAI.model_loop.get_sync_redis_client", return_value=mock_redis
        ):
            with patch("KestrelAI.model_loop.MemoryStore"):
                with patch("KestrelAI.model_loop.WebResearchAgent"):
                    loop = KestrelAgentWorker()
                    loop.llm = mock_llm
                    return loop

    def test_model_loop_initialization(self, model_loop):
        """Test model loop initialization."""
        assert model_loop.llm is not None
        assert model_loop.redis_client is not None
        assert model_loop.running is False
        assert model_loop.paused is False
        assert model_loop.current_task_id is None

    def test_start_task(self, model_loop, mock_task, mock_redis):
        """Test starting a task."""
        # Mock Redis client wrapper operations
        mock_redis.get_next_command.return_value = None
        mock_redis.send_update.return_value = None

        # Mock orchestrator creation
        with patch(
            "KestrelAI.model_loop.ResearchOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator

            # Mock planning phase
            mock_orchestrator._planning_phase = AsyncMock()
            mock_orchestrator.get_current_subtask.return_value = Mock(
                description="Test subtask"
            )

            # start_task now expects a dict-like config payload
            payload = {
                "name": mock_task.name,
                "description": mock_task.description,
                "budgetMinutes": mock_task.budgetMinutes,
            }
            model_loop.start_task(mock_task.id, payload)
            assert model_loop.current_task_id == mock_task.id
            assert model_loop.running is True
            assert mock_task.id in model_loop.tasks

    def test_process_task_step(self, model_loop, mock_task, mock_redis):
        """Test processing a task step."""
        # Set up task state
        model_loop.current_task_id = mock_task.id
        model_loop.tasks[mock_task.id] = mock_task
        model_loop.running = True

        # Initialize minimal task_metrics entry as start_task would
        model_loop.task_metrics[mock_task.id] = {
            "search_count": 0,
            "think_count": 0,
            "summary_count": 0,
            "checkpoint_count": 0,
            "action_count": 0,
            "searches": [],
            "start_time": time.time(),
            "last_research_plan_state": {
                "subtask_index": -1,
                "completed_subtasks": set(),
            },
        }

        # Mock orchestrator
        class DummyTaskState:
            def __init__(self):
                # Minimal research plan: two subtasks
                self.research_plan = ResearchPlan(
                    restated_task="Restated task",
                    subtasks=[
                        SharedSubtask(
                            order=1,
                            description="Subtask 1",
                            success_criteria="Criteria 1",
                        ),
                        SharedSubtask(
                            order=2,
                            description="Subtask 2",
                            success_criteria="Criteria 2",
                        ),
                    ],
                    current_subtask_index=0,
                )
                self.subtask_index = 0
                self.completed_subtasks = set()
                self.subtask_agents = {}

        mock_orchestrator = Mock()
        mock_orchestrator.task_states = {mock_task.name: DummyTaskState()}
        mock_orchestrator.get_task_progress.return_value = {
            "progress": 0.0,
            "subtasks": [],
            "completed": 0,
            "total": 2,
        }
        mock_orchestrator.next_action = AsyncMock(return_value="Test research result")
        mock_orchestrator.get_current_subtask.return_value = "Test subtask"
        model_loop.orchestrator = mock_orchestrator

        # Mock Redis client wrapper operations
        mock_redis.send_update.return_value = None

        model_loop.process_task_step()

        # Should have called next_action
        mock_orchestrator.next_action.assert_called_once_with(mock_task)

    def test_process_task_step_no_plan(self, model_loop, mock_task):
        """Test processing task step when no research plan exists."""
        # Set up task state
        model_loop.current_task_id = mock_task.id
        model_loop.tasks[mock_task.id] = mock_task
        model_loop.running = True

        # Mock orchestrator with no research plan
        mock_orchestrator = Mock()
        mock_orchestrator.task_states = {mock_task.name: Mock(research_plan=None)}
        model_loop.orchestrator = mock_orchestrator

        # Should not process when no plan exists
        model_loop.process_task_step()

        # Should not have called next_action
        assert (
            not hasattr(mock_orchestrator, "next_action")
            or mock_orchestrator.next_action.call_count == 0
        )

    def test_update_settings(self, model_loop, mock_llm):
        """Test updating settings."""
        settings = {"ollamaMode": "local", "orchestrator": "kestrel"}

        with patch("KestrelAI.model_loop.LlmWrapper") as mock_llm_class:
            mock_llm_class.return_value = mock_llm

            with patch("KestrelAI.model_loop.WebResearchAgent") as mock_agent_class:
                mock_agent = Mock()
                mock_agent_class.return_value = mock_agent

                model_loop.update_settings(settings)

                assert model_loop.app_settings["ollamaMode"] == "local"
                assert model_loop.app_settings["orchestrator"] == "kestrel"
                assert model_loop.llm == mock_llm
                assert model_loop.agent == mock_agent

    def test_send_research_plan_update(self, model_loop, mock_task, mock_redis):
        """Test sending research plan update."""

        # Set up orchestrator with research plan
        class DummyTaskState:
            def __init__(self):
                self.research_plan = ResearchPlan(
                    restated_task="Restated task",
                    subtasks=[
                        SharedSubtask(
                            order=1,
                            description="Subtask 1",
                            success_criteria="Criteria 1",
                        ),
                        SharedSubtask(
                            order=2,
                            description="Subtask 2",
                            success_criteria="Criteria 2",
                        ),
                    ],
                    current_subtask_index=0,
                )
                self.subtask_index = 0
                self.completed_subtasks = set()
                self.subtask_agents = {}

        mock_orchestrator = Mock()
        mock_orchestrator.task_states = {mock_task.name: DummyTaskState()}
        model_loop.orchestrator = mock_orchestrator

        model_loop.send_research_plan_update(mock_task.id, mock_task.name)

        # Should have enqueued an update via the sync Redis client wrapper
        mock_redis.send_update.assert_called()

    def test_send_research_plan_update_no_plan(self, model_loop, mock_task, mock_redis):
        """Test sending research plan update when no plan exists."""
        # Set up orchestrator with no research plan
        mock_orchestrator = Mock()
        mock_orchestrator.task_states = {mock_task.name: Mock(research_plan=None)}
        model_loop.orchestrator = mock_orchestrator

        # Should not update when no plan exists
        model_loop.send_research_plan_update(mock_task.id, mock_task.name)

        # Should not have called Redis set
        mock_redis.set.assert_not_called()

    def test_handle_command_start(self, model_loop, mock_task, mock_redis):
        """Test handling start command."""
        command = {
            "type": "start",
            "taskId": mock_task.id,
            "payload": {
                "name": mock_task.name,
                "description": mock_task.description,
                "budgetMinutes": mock_task.budgetMinutes,
            },
        }

        # Mock Redis client wrapper operations
        mock_redis.get_next_command.return_value = None
        mock_redis.send_update.return_value = None

        # Mock orchestrator creation
        with patch(
            "KestrelAI.model_loop.ResearchOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator
            mock_orchestrator._planning_phase = AsyncMock()
            mock_orchestrator.get_current_subtask.return_value = Mock(
                description="Test subtask"
            )

            model_loop.handle_command(command)

            assert model_loop.current_task_id == mock_task.id
            assert model_loop.running is True

    def test_handle_command_stop(self, model_loop, mock_task):
        """Test handling stop command."""
        # Set up running task
        model_loop.current_task_id = mock_task.id
        model_loop.running = True

        command = {"type": "stop", "task_id": mock_task.id}

        model_loop.handle_command(command)

        assert model_loop.running is False
        assert model_loop.current_task_id is None

    def test_handle_command_pause(self, model_loop, mock_task):
        """Test handling pause command."""
        # Set up running task
        model_loop.current_task_id = mock_task.id
        model_loop.running = True

        command = {"type": "pause", "task_id": mock_task.id}

        model_loop.handle_command(command)

        assert model_loop.paused is True
        assert model_loop.running is True  # Still running, just paused

    def test_handle_command_resume(self, model_loop, mock_task):
        """Test handling resume command."""
        # Set up paused task
        model_loop.current_task_id = mock_task.id
        model_loop.running = True
        model_loop.paused = True

        command = {"type": "resume", "task_id": mock_task.id}

        model_loop.handle_command(command)

        assert model_loop.paused is False
        assert model_loop.running is True

    def test_performance_requirements(self, model_loop, performance_thresholds):
        """Test performance requirements."""
        # Test task creation performance
        start_time = time.time()

        mock_task = Task(
            name="Performance Test",
            description="Test performance",
            budgetMinutes=5,
            status=TaskStatus.ACTIVE,
        )

        # Mock Redis operations
        with patch("KestrelAI.model_loop.get_sync_redis_client") as mock_redis_client:
            mock_redis = Mock()
            mock_redis.get_next_command.return_value = None
            mock_redis.send_update.return_value = None
            mock_redis_client.return_value = mock_redis

            # Mock orchestrator creation
            with patch(
                "KestrelAI.model_loop.ResearchOrchestrator"
            ) as mock_orchestrator_class:
                mock_orchestrator = Mock()
                mock_orchestrator_class.return_value = mock_orchestrator
                mock_orchestrator._planning_phase = AsyncMock()
                mock_orchestrator.get_current_subtask.return_value = Mock(
                    description="Test subtask"
                )

                payload = {
                    "name": mock_task.name,
                    "description": mock_task.description,
                    "budgetMinutes": mock_task.budgetMinutes,
                }
                model_loop.start_task(mock_task.id, payload)

        creation_time = time.time() - start_time
        assert creation_time < performance_thresholds["task_creation_time"]
