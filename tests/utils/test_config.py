"""
Test configuration and constants for KestrelAI tests.
"""

import os
from typing import Any

# Test environment configuration
TEST_CONFIG = {
    "api_base_url": "http://localhost:8000/api/v1",
    "frontend_url": "http://localhost:5173",
    "ollama_url": "http://localhost:11434",
    "searxng_url": "http://localhost:8080",
    "redis_host": "localhost",
    "redis_port": 6379,
    "test_timeout": 30,
    "mock_timeout": 5,
}

# Service availability markers
SERVICE_MARKERS = {
    "requires_services": ["Backend API", "Frontend"],
    "requires_optional_services": ["Ollama", "SearXNG", "Redis"],
}

# Test categories
TEST_CATEGORIES = {
    "unit": "Fast, isolated tests with mocked dependencies",
    "integration": "Tests that verify component interactions",
    "api": "Backend API endpoint tests",
    "frontend": "Frontend UI and theme tests",
    "e2e": "End-to-end workflow tests",
    "performance": "Performance and regression tests",
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "llm_response_time": 10.0,  # seconds
    "planning_phase_time": 30.0,  # seconds
    "task_creation_time": 1.0,  # seconds
    "redis_operation_time": 0.1,  # seconds
    "api_response_time": 2.0,  # seconds
    "frontend_load_time": 3.0,  # seconds
}

# Test data constants
TEST_DATA = {
    "sample_task_name": "Test Research Task",
    "sample_task_description": "A test research task for unit testing",
    "sample_subtask_count": 3,
    "default_timeout": 30,
    "mock_response_delay": 0.1,
}


def get_test_config() -> dict[str, Any]:
    """Get test configuration with environment overrides."""
    config = TEST_CONFIG.copy()

    # Override with environment variables if present
    for key in config:
        env_key = f"TEST_{key.upper()}"
        if env_key in os.environ:
            config[key] = os.environ[env_key]

    return config


def get_service_requirements(category: str) -> list[str]:
    """Get required services for a test category."""
    if category in ["unit"]:
        return []
    elif category in ["integration", "api"]:
        return SERVICE_MARKERS["requires_services"]
    elif category in ["e2e", "performance"]:
        return (
            SERVICE_MARKERS["requires_services"]
            + SERVICE_MARKERS["requires_optional_services"]
        )
    else:
        return SERVICE_MARKERS["requires_services"]


def get_performance_threshold(metric: str) -> float:
    """Get performance threshold for a metric."""
    return PERFORMANCE_THRESHOLDS.get(metric, 5.0)  # Default 5 seconds
