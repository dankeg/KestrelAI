# Test configuration and fixtures
import asyncio
import os
import shutil
import tempfile
from unittest.mock import Mock

import pytest

# ---------------------------------------------------------------------------
# Compatibility shims for third-party libraries used in the codebase
# ---------------------------------------------------------------------------
# Some dependencies (e.g. chromadb) still reference numpy aliases that were
# removed in NumPy 2.x (np.float_, np.uint). To keep the test environment
# working without modifying site-packages, we provide lightweight aliases
# here before those libraries are imported.
try:
    import numpy as np  # type: ignore

    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]
    if not hasattr(np, "uint"):
        np.uint = np.uint64  # type: ignore[attr-defined]
except Exception:
    # Tests that don't touch these paths should still run even if numpy is absent
    pass

# Import test utilities

# Set test environment variables
os.environ["PYTHONPATH"] = "/app"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["SEARXNG_URL"] = "http://localhost:8080"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TESTING"] = "1"  # Signal that we're in test mode


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = Mock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = True
    mock_redis.keys.return_value = []
    mock_redis.lpush.return_value = 1
    mock_redis.rpop.return_value = None
    mock_redis.llen.return_value = 0
    return mock_redis


@pytest.fixture
def mock_llm():
    """Mock LLM wrapper for testing."""
    mock_llm = Mock()
    mock_llm.chat.return_value = "Mock LLM response"
    mock_llm.model = "test-model"
    mock_client = Mock()
    mock_client.host = "http://localhost:11434"
    mock_llm.client = mock_client
    return mock_llm


@pytest.fixture
def mock_task():
    """Mock task object for testing."""
    from KestrelAI.shared.models import Task, TaskStatus

    return Task(
        name="Test Task",
        description="Test task description",
        budgetMinutes=5,
        status=TaskStatus.ACTIVE,
    )


@pytest.fixture
def mock_research_plan():
    """Mock research plan for testing."""
    from KestrelAI.shared.models import ResearchPlan, Subtask

    return ResearchPlan(
        restated_task="Test restated task",
        subtasks=[
            Subtask(
                order=1,
                description="Test subtask 1",
                success_criteria="Test criteria 1",
                status="pending",
                findings=[],
            ),
            Subtask(
                order=2,
                description="Test subtask 2",
                success_criteria="Test criteria 2",
                status="pending",
                findings=[],
            ),
        ],
        current_subtask_index=0,
    )


@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        "ollama_host": "http://localhost:11434",
        "redis_host": "localhost",
        "redis_port": 6379,
        "searxng_url": "http://localhost:8080",
        "model_name": "gemma3:27b",
    }


# Performance test thresholds
PERFORMANCE_THRESHOLDS = {
    "llm_response_time": 10.0,  # seconds
    "planning_phase_time": 30.0,  # seconds
    "task_creation_time": 1.0,  # seconds
    "redis_operation_time": 0.1,  # seconds
}


@pytest.fixture
def performance_thresholds():
    """Performance test thresholds."""
    return PERFORMANCE_THRESHOLDS


@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    """Mock SentenceTransformer to avoid slow model loading in tests."""
    from unittest.mock import Mock, patch

    class FakeEmbedding:
        """Minimal replacement for numpy arrays used in SentenceTransformer outputs."""

        def __init__(self, data):
            self._data = data

        @property
        def ndim(self):
            if not self._data:
                return 1
            return 2 if isinstance(self._data[0], list) else 1

        @property
        def shape(self):
            if self.ndim == 1:
                return (len(self._data),)
            inner_len = len(self._data[0]) if self._data and self._data[0] else 0
            return (len(self._data), inner_len)

        def tolist(self):
            return self._data

        def __getitem__(self, idx):
            return self._data[idx]

        def __len__(self):
            return len(self._data)

    mock_model = Mock()

    embedding_dim = 384

    def mock_encode(text):
        if isinstance(text, str):
            return FakeEmbedding([0.1] * embedding_dim)
        elif isinstance(text, list):
            return FakeEmbedding([[0.1] * embedding_dim for _ in text])
        else:
            return FakeEmbedding([0.1] * embedding_dim)

    mock_model.encode.side_effect = mock_encode

    class MockSentenceTransformerClass:
        def __init__(self, model_name=None):
            self.model_name = model_name

        def encode(self, text):
            return mock_encode(text)

    with patch(
        "KestrelAI.memory.vector_store._get_sentence_transformer",
        return_value=MockSentenceTransformerClass,
    ):
        yield mock_model
