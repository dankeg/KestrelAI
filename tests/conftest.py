# Test configuration and fixtures
import os
import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Import test utilities
from .utils.test_fixtures import TestDataFactory

# Set test environment variables
os.environ["PYTHONPATH"] = "/app"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["SEARXNG_URL"] = "http://localhost:8080"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

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
        status=TaskStatus.ACTIVE
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
                findings=[]
            ),
            Subtask(
                order=2,
                description="Test subtask 2", 
                success_criteria="Test criteria 2",
                status="pending",
                findings=[]
            )
        ],
        current_subtask_index=0
    )

@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        "ollama_host": "http://localhost:11434",
        "redis_host": "localhost",
        "redis_port": 6379,
        "searxng_url": "http://localhost:8080",
        "model_name": "gemma3:12b"
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
