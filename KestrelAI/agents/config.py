"""
Configuration settings for KestrelAI orchestrator and research agents
"""

import os
from dataclasses import dataclass


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator behavior"""

    slice_minutes: int = 15
    max_iterations_per_subtask: int = 10
    max_total_iterations: int = 100
    max_stuck_count: int = 3
    max_retries: int = 3
    planning_timeout_seconds: int = 1200
    review_timeout_seconds: int = 200
    preplanning_max_steps: int = 0
    preplanning_step_timeout_seconds: int = 300


@dataclass
class ResearchAgentConfig:
    """Configuration for research agent behavior"""

    think_loops: int = 6
    search_results: int = 4
    fetch_bytes: int = 30_000
    context_window: int = 20
    checkpoint_freq: int = 5
    max_snippet_length: int = 3000
    max_repeats: int = 3


@dataclass
class SystemConfig:
    """Overall system configuration"""

    debug: bool = True
    searxng_url: str = "http://localhost:8080/search"
    ollama_host: str = "http://localhost:11434"


# Orchestrator profiles
ORCHESTRATOR_PROFILES = {
    "hummingbird": {
        "slice_minutes": 5,
        "max_iterations_per_subtask": 5,
        "max_total_iterations": 50,
        "planning_timeout_seconds": 600,
        "review_timeout_seconds": 200,
        "preplanning_max_steps": 0,
        "preplanning_step_timeout_seconds": 300,
        "description": "Fast, focused research with quick iterations",
    },
    "kestrel": {
        "slice_minutes": 15,
        "max_iterations_per_subtask": 10,
        "max_total_iterations": 100,
        "planning_timeout_seconds": 1200,
        "review_timeout_seconds": 200,
        "preplanning_max_steps": 0,
        "preplanning_step_timeout_seconds": 300,
        "description": "Balanced approach with moderate depth",
    },
    "albatross": {
        "slice_minutes": 30,
        "max_iterations_per_subtask": 15,
        "max_total_iterations": 150,
        "planning_timeout_seconds": 1800,
        "review_timeout_seconds": 200,
        "preplanning_max_steps": 0,
        "preplanning_step_timeout_seconds": 300,
        "description": "Deep, thorough research with extensive exploration",
    },
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def get_orchestrator_config(profile: str = "kestrel") -> OrchestratorConfig:
    """Get orchestrator configuration for a specific profile"""
    if profile not in ORCHESTRATOR_PROFILES:
        profile = "kestrel"

    config = ORCHESTRATOR_PROFILES[profile]
    return OrchestratorConfig(
        slice_minutes=config["slice_minutes"],
        max_iterations_per_subtask=config["max_iterations_per_subtask"],
        max_total_iterations=config["max_total_iterations"],
        planning_timeout_seconds=_env_int(
            "ORCHESTRATOR_PLANNING_TIMEOUT_SECONDS",
            config.get("planning_timeout_seconds", 1200),
        ),
        review_timeout_seconds=_env_int(
            "ORCHESTRATOR_REVIEW_TIMEOUT_SECONDS",
            config.get("review_timeout_seconds", 200),
        ),
        preplanning_max_steps=_env_int(
            "ORCHESTRATOR_PREPLANNING_MAX_STEPS",
            config.get("preplanning_max_steps", 0),
        ),
        preplanning_step_timeout_seconds=_env_int(
            "ORCHESTRATOR_PREPLANNING_STEP_TIMEOUT_SECONDS",
            config.get("preplanning_step_timeout_seconds", 300),
        ),
    )


def get_research_agent_config() -> ResearchAgentConfig:
    """Get default research agent configuration"""
    return ResearchAgentConfig()


def get_system_config() -> SystemConfig:
    """Get system configuration"""
    return SystemConfig()
