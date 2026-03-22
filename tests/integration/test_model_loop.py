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
                    loop.execution_llm = mock_llm
                    loop.control_llm = mock_llm
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
            mock_orchestrator.task_states = {}
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
                            subtask_type="discovery",
                        ),
                        SharedSubtask(
                            order=2,
                            description="Subtask 2",
                            success_criteria="Criteria 2",
                            subtask_type="synthesis",
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
        settings = {
            "llmProvider": "openai_compatible",
            "openaiBaseUrl": "https://api.example.com/v1",
            "openaiApiKey": "sk-test",
            "modelName": "gpt-4.1-mini",
            "executionModelName": "gpt-4.1-mini",
            "controlModelName": "o4-mini",
            "orchestrator": "kestrel",
        }

        with patch("KestrelAI.model_loop.LlmWrapper") as mock_llm_class:
            mock_execution_llm = Mock(name="execution_llm")
            mock_control_llm = Mock(name="control_llm")
            mock_llm_class.side_effect = [mock_execution_llm, mock_control_llm]

            with patch("KestrelAI.model_loop.WebResearchAgent") as mock_agent_class:
                mock_agent = Mock()
                mock_agent_class.return_value = mock_agent

                model_loop.update_settings(settings)

                assert model_loop.app_settings["llmProvider"] == "openai_compatible"
                assert model_loop.app_settings["orchestrator"] == "kestrel"
                assert model_loop.app_settings["executionModelName"] == "gpt-4.1-mini"
                assert model_loop.app_settings["controlModelName"] == "o4-mini"
                assert model_loop.execution_llm == mock_execution_llm
                assert model_loop.control_llm == mock_control_llm
                assert model_loop.llm == mock_execution_llm
                assert model_loop.agent == mock_agent
                assert mock_llm_class.call_count == 2
                assert mock_llm_class.call_args_list[0].kwargs == {
                    "model": "gpt-4.1-mini",
                    "host": "https://api.example.com/v1",
                    "provider": "openai_compatible",
                    "api_key": "sk-test",
                }
                assert mock_llm_class.call_args_list[1].kwargs == {
                    "model": "o4-mini",
                    "host": "https://api.example.com/v1",
                    "provider": "openai_compatible",
                    "api_key": "sk-test",
                }
                mock_agent_class.assert_called_with(
                    "main-agent",
                    mock_execution_llm,
                    model_loop.mem,
                    config=mock_agent_class.call_args.kwargs["config"],
                )

    def test_update_settings_uses_control_model_for_orchestrator(
        self, model_loop, mock_task
    ):
        settings = {
            "llmProvider": "openai_compatible",
            "openaiBaseUrl": "https://api.example.com/v1",
            "openaiApiKey": "sk-test",
            "modelName": "o4-mini",
            "executionModelName": "gpt-4.1-mini",
            "controlModelName": "o4-mini",
            "orchestrator": "kestrel",
        }

        model_loop.current_task_id = mock_task.id
        model_loop.tasks[mock_task.id] = mock_task

        with patch("KestrelAI.model_loop.LlmWrapper") as mock_llm_class:
            mock_execution_llm = Mock(name="execution_llm")
            mock_control_llm = Mock(name="control_llm")
            mock_llm_class.side_effect = [mock_execution_llm, mock_control_llm]

            with patch("KestrelAI.model_loop.WebResearchAgent"):
                with patch(
                    "KestrelAI.model_loop.ResearchOrchestrator"
                ) as mock_orchestrator_class:
                    mock_orchestrator = Mock()
                    mock_orchestrator._planning_phase = AsyncMock()
                    mock_orchestrator.get_current_subtask.return_value = (
                        "Initial research"
                    )
                    mock_orchestrator_class.return_value = mock_orchestrator

                    model_loop.update_settings(settings)

                    mock_orchestrator_class.assert_called_with(
                        [mock_task],
                        mock_control_llm,
                        profile="kestrel",
                        max_context_tokens=model_loop.app_settings["maxContextTokens"],
                    )

    def test_update_settings_ollama_uses_native_provider(self, model_loop, mock_llm):
        settings = {
            "llmProvider": "ollama",
            "ollamaMode": "local",
            "modelName": "qwen3:14b",
            "executionModelName": "gemma3:4b",
            "controlModelName": "qwen3:14b",
            "orchestrator": "kestrel",
        }

        with patch("KestrelAI.model_loop.LlmWrapper") as mock_llm_class:
            mock_execution_llm = Mock(name="execution_llm")
            mock_control_llm = Mock(name="control_llm")
            mock_llm_class.side_effect = [mock_execution_llm, mock_control_llm]

            with patch("KestrelAI.model_loop.WebResearchAgent") as mock_agent_class:
                mock_agent = Mock()
                mock_agent_class.return_value = mock_agent

                with patch.object(
                    model_loop, "_ensure_model_available", return_value=None
                ) as mock_ensure:
                    model_loop.update_settings(settings)

                assert model_loop.execution_llm == mock_execution_llm
                assert model_loop.control_llm == mock_control_llm
                assert mock_llm_class.call_args_list[0].kwargs == {
                    "model": "gemma3:4b",
                    "host": "http://localhost:11434",
                    "provider": "ollama_native",
                    "api_key": None,
                }
                assert mock_llm_class.call_args_list[1].kwargs == {
                    "model": "qwen3:14b",
                    "host": "http://localhost:11434",
                    "provider": "ollama_native",
                    "api_key": None,
                }
                assert mock_ensure.call_count == 2
                ensured_models = {call.args[0] for call in mock_ensure.call_args_list}
                assert ensured_models == {"gemma3:4b", "qwen3:14b"}

    def test_ensure_model_available_starts_background_pull_for_missing_model(
        self, model_loop
    ):
        fake_client = Mock()
        fake_client.list.return_value = {"models": [{"name": "gemma3:12b"}]}

        started = {}

        class DummyThread:
            def __init__(self, *, target, name, daemon):
                started["target"] = target
                started["name"] = name
                started["daemon"] = daemon

            def start(self):
                started["started"] = True

        with patch("ollama.Client", return_value=fake_client):
            with patch("KestrelAI.model_loop.threading.Thread", DummyThread):
                model_loop._ensure_model_available(
                    "qwen3:14b", "http://localhost:11434"
                )

        assert started["started"] is True
        assert started["daemon"] is True
        assert started["name"] == "ollama-pull-qwen3:14b"

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
                            subtask_type="discovery",
                        ),
                        SharedSubtask(
                            order=2,
                            description="Subtask 2",
                            success_criteria="Criteria 2",
                            subtask_type="synthesis",
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
        _, kwargs = mock_redis.send_update.call_args
        plan = kwargs["research_plan"]
        assert plan["subtasks"][0]["subtask_type"] == "discovery"
        assert plan["subtasks"][0]["status"] == "in_progress"
        assert plan["subtasks"][1]["subtask_type"] == "synthesis"

    def test_generate_final_report_embeds_structured_report_without_nested_heading(
        self, model_loop, mock_task, mock_redis, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()

        safe_name = "".join(
            c if c.isalnum() or c in (" ", "-", "_") else "_" for c in mock_task.name
        ).strip()
        (notes_dir / f"{safe_name.upper()}_FINAL_REPORT.txt").write_text(
            "# Final Research Report\n\n## Executive Summary\nA useful summary.\n\n## Findings\n- Verified item.\n",
            encoding="utf-8",
        )

        model_loop.current_task_id = mock_task.id
        model_loop.tasks[mock_task.id] = mock_task
        model_loop.task_metrics[mock_task.id] = {
            "search_count": 3,
            "think_count": 4,
            "summary_count": 1,
            "checkpoint_count": 2,
            "action_count": 0,
            "searches": [],
            "start_time": time.time(),
            "last_research_plan_state": {},
        }

        model_loop.generate_final_report(completed=True)

        _, _, report_content = mock_redis.send_report.call_args.args[:3]
        assert "# Final Research Report" not in report_content
        assert "## Key Findings" not in report_content
        assert "## Executive Summary" in report_content
        assert "## Findings" in report_content

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
