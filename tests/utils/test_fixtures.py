"""
Test fixtures and utilities for KestrelAI tests.
"""

import asyncio
import tempfile
from unittest.mock import Mock, patch

import pytest

try:
    from KestrelAI.shared.models import ResearchPlan, Subtask, Task, TaskStatus
except ImportError:
    from shared.models import ResearchPlan, Subtask, Task, TaskStatus


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        id="test_task_1",
        name="Test Research Task",
        description="A test research task for unit testing",
        status=TaskStatus.PENDING,
        createdAt=1704067200000,
        updatedAt=1704067200000,
    )


@pytest.fixture
def sample_research_plan():
    """Create a sample research plan for testing."""
    return ResearchPlan(
        restated_task="Test research task for unit testing",
        subtasks=[
            Subtask(
                order=1,
                description="First subtask",
                success_criteria="Complete first subtask",
                status="pending",
            ),
            Subtask(
                order=2,
                description="Second subtask",
                success_criteria="Complete second subtask",
                status="pending",
            ),
        ],
        current_subtask_index=0,
        created_at=1704067200.0,
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM wrapper for testing."""
    with patch("KestrelAI.agents.base.LlmWrapper") as mock_llm_class:
        mock_llm = Mock()
        mock_llm.generate_response.return_value = "Mocked LLM response"
        mock_llm_class.return_value = mock_llm
        yield mock_llm


@pytest.fixture
def mock_memory_store():
    """Create a mock memory store for testing."""
    with patch("KestrelAI.memory.vector_store.MemoryStore") as mock_memory_class:
        mock_memory = Mock()
        mock_memory.add.return_value = None
        mock_memory.search.return_value = {
            "ids": [["doc1", "doc2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"source": "test"}, {"source": "test"}]],
        }
        mock_memory.delete_all.return_value = None
        mock_memory_class.return_value = mock_memory
        yield mock_memory


@pytest.fixture
def mock_redis():
    """Create a mock Redis client for testing."""
    with patch("redis.Redis") as mock_redis_class:
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis_class.return_value = mock_redis
        yield mock_redis


@pytest.fixture
def mock_ollama_client():
    """Create a mock Ollama client for testing."""
    with patch("ollama.Client") as mock_client_class:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Mocked Ollama response"
        mock_client.chat.return_value = mock_response
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_chromadb():
    """Create a mock ChromaDB client for testing."""
    with patch("chromadb.PersistentClient") as mock_chroma_class:
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.name = "test_collection"
        mock_collection.add.return_value = None
        mock_collection.query.return_value = {
            "ids": [["doc1"]],
            "distances": [[0.1]],
            "metadatas": [[{"source": "test"}]],
        }
        mock_collection.delete.return_value = None
        mock_client.get_collection.return_value = mock_collection
        mock_client.create_collection.return_value = mock_collection
        mock_chroma_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer for testing."""
    with patch("sentence_transformers.SentenceTransformer") as mock_transformer_class:
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        mock_transformer_class.return_value = mock_model
        yield mock_model


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_requests():
    """Create a mock requests module for testing HTTP calls."""
    with (
        patch("requests.get") as mock_get,
        patch("requests.post") as mock_post,
        patch("requests.put") as mock_put,
        patch("requests.delete") as mock_delete,
    ):
        # Default successful responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.text = "Success"

        mock_get.return_value = mock_response
        mock_post.return_value = mock_response
        mock_put.return_value = mock_response
        mock_delete.return_value = mock_response

        yield {
            "get": mock_get,
            "post": mock_post,
            "put": mock_put,
            "delete": mock_delete,
        }


class TestDataFactory:
    """Factory class for creating test data."""

    @staticmethod
    def create_task(
        task_id: str = "test_task",
        name: str = "Test Task",
        description: str = "Test description",
        status: TaskStatus = TaskStatus.PENDING,
    ) -> Task:
        """Create a task with default or custom values."""
        return Task(
            id=task_id,
            name=name,
            description=description,
            status=status,
            createdAt=1704067200000,
            updatedAt=1704067200000,
        )

    @staticmethod
    def create_subtask(
        order: int = 1,
        description: str = "Test subtask",
        success_criteria: str = "Test criteria",
        status: str = "pending",
    ) -> Subtask:
        """Create a subtask with default or custom values."""
        return Subtask(
            order=order,
            description=description,
            success_criteria=success_criteria,
            status=status,
        )

    @staticmethod
    def create_research_plan(
        restated_task: str = "Test restated task", subtasks: list | None = None
    ) -> ResearchPlan:
        """Create a research plan with default or custom values."""
        if subtasks is None:
            subtasks = [
                TestDataFactory.create_subtask(1, "First subtask", "Complete first"),
                TestDataFactory.create_subtask(2, "Second subtask", "Complete second"),
            ]

        return ResearchPlan(
            restated_task=restated_task,
            subtasks=subtasks,
            current_subtask_index=0,
            created_at=1704067200.0,
        )
