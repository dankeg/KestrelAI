# Unit tests for core components
import time
from unittest.mock import Mock, patch

import pytest

try:
    from KestrelAI.agents.base import LlmWrapper
    from KestrelAI.shared.models import Task, TaskStatus
    from KestrelAI.shared.redis_utils import get_sync_redis_client
except ImportError:
    from agents.base import LlmWrapper
    from shared.models import Task, TaskStatus
    from shared.redis_utils import get_sync_redis_client


@pytest.mark.unit
class TestLlmWrapper:
    """Test LLM wrapper functionality."""

    @patch("ollama.Client")
    def test_llm_initialization(self, mock_client_class, test_config):
        """Test LLM wrapper initialization."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        llm = LlmWrapper(
            model=test_config["model_name"], host=test_config["ollama_host"]
        )
        assert llm.model == test_config["model_name"]
        assert llm.client is not None

    @patch("ollama.Client")
    def test_llm_chat_success(self, mock_client_class, test_config):
        """Test successful LLM chat."""
        # Mock successful response
        mock_client = Mock()
        mock_client.chat.return_value = {"message": {"content": "Test response"}}
        mock_client_class.return_value = mock_client

        llm = LlmWrapper(
            model=test_config["model_name"], host=test_config["ollama_host"]
        )

        result = llm.chat([{"role": "user", "content": "Hello"}])
        assert result == "Test response"
        mock_client.chat.assert_called_once()

    @patch("ollama.Client")
    def test_llm_chat_failure(self, mock_client_class, test_config):
        """Test LLM chat failure handling."""
        # Mock failed response
        mock_client = Mock()
        mock_client.chat.side_effect = Exception("Connection failed")
        mock_client_class.return_value = mock_client

        llm = LlmWrapper(
            model=test_config["model_name"], host=test_config["ollama_host"]
        )

        with pytest.raises(Exception):
            llm.chat([{"role": "user", "content": "Hello"}])

    @patch("ollama.Client")
    def test_llm_performance(
        self, mock_client_class, test_config, performance_thresholds
    ):
        """Test LLM response time performance."""
        mock_client = Mock()
        mock_client.chat.return_value = {"message": {"content": "Fast response"}}
        mock_client_class.return_value = mock_client

        llm = LlmWrapper(
            model=test_config["model_name"], host=test_config["ollama_host"]
        )

        start_time = time.time()
        llm.chat([{"role": "user", "content": "Hello"}])
        response_time = time.time() - start_time

        assert response_time < performance_thresholds["llm_response_time"]


class TestRedisClient:
    """Test Redis client functionality."""

    @patch("redis.Redis")
    def test_redis_connection(self, mock_redis_class):
        """Test Redis connection."""
        mock_redis = Mock()
        mock_redis.brpop.return_value = None  # No commands in queue
        mock_redis_class.return_value = mock_redis

        from KestrelAI.shared.redis_utils import RedisConfig

        client = get_sync_redis_client(RedisConfig(host="localhost", port=6379, db=0))

        # Test that we can get next command (will return None for empty queue)
        result = client.get_next_command(timeout=0)
        assert result is None

    @patch("redis.Redis")
    def test_redis_operations(self, mock_redis_class):
        """Test Redis operations."""
        mock_redis = Mock()
        mock_redis_class.return_value = mock_redis

        from KestrelAI.shared.redis_utils import RedisConfig

        # Test that we can create a Redis client
        client = get_sync_redis_client(RedisConfig(host="localhost", port=6379, db=0))
        assert client is not None
        assert client.config.host == "localhost"
        assert client.config.port == 6379

    def test_redis_performance(self, performance_thresholds):
        """Test Redis operation performance."""
        start_time = time.time()

        # Mock Redis operations to test performance
        with patch("KestrelAI.shared.redis_utils.redis.Redis") as mock_redis_class:
            mock_redis = Mock()
            mock_redis.set.return_value = True
            mock_redis_class.return_value = mock_redis

            from KestrelAI.shared.redis_utils import RedisConfig

            client = get_sync_redis_client(
                RedisConfig(host="localhost", port=6379, db=0)
            )

            client.checkpoint("test_task", {"state": "test"})

        operation_time = time.time() - start_time
        assert operation_time < performance_thresholds["redis_operation_time"]


class TestTaskModel:
    """Test Task model functionality."""

    def test_task_creation(self):
        """Test task creation."""
        task = Task(
            name="Test Task",
            description="Test description",
            budgetMinutes=30,
            status=TaskStatus.ACTIVE,
        )

        assert task.name == "Test Task"
        assert task.description == "Test description"
        assert task.budgetMinutes == 30
        assert task.status == TaskStatus.ACTIVE

    def test_task_serialization(self):
        """Test task serialization."""
        task = Task(
            name="Test Task",
            description="Test description",
            budgetMinutes=30,
            status=TaskStatus.ACTIVE,
        )

        # Test model_dump (Pydantic v2)
        task_dict = task.model_dump()
        assert task_dict["name"] == "Test Task"
        assert task_dict["status"] == "active"

        # Test model_validate (Pydantic v2)
        new_task = Task.model_validate(task_dict)
        assert new_task.name == task.name
        assert new_task.status == task.status

    def test_task_validation(self):
        """Test task validation."""
        # Test valid task
        task = Task(
            name="Valid Task",
            description="Valid description",
            budgetMinutes=30,
            status=TaskStatus.ACTIVE,
        )
        assert task.name is not None

        # Test that negative budget is allowed (Pydantic v2 doesn't validate by default)
        task_negative = Task(
            name="Test Task",
            description="Test description",
            budgetMinutes=-1,  # This is allowed in Pydantic v2
            status=TaskStatus.ACTIVE,
        )
        assert task_negative.budgetMinutes == -1
