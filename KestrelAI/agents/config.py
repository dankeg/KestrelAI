"""
Configuration settings for KestrelAI orchestrator and research agents
"""

from dataclasses import dataclass


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator behavior"""

    slice_minutes: int = 15
    max_iterations_per_subtask: int = 10
    max_total_iterations: int = 100
    max_stuck_count: int = 3
    max_retries: int = 3


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
        "description": "Fast, focused research with quick iterations",
    },
    "kestrel": {
        "slice_minutes": 15,
        "max_iterations_per_subtask": 10,
        "max_total_iterations": 100,
        "description": "Balanced approach with moderate depth",
    },
    "albatross": {
        "slice_minutes": 30,
        "max_iterations_per_subtask": 15,
        "max_total_iterations": 150,
        "description": "Deep, thorough research with extensive exploration",
    },
}


def get_orchestrator_config(profile: str = "kestrel") -> OrchestratorConfig:
    """Get orchestrator configuration for a specific profile"""
    if profile not in ORCHESTRATOR_PROFILES:
        profile = "kestrel"

    config = ORCHESTRATOR_PROFILES[profile]
    return OrchestratorConfig(
        slice_minutes=config["slice_minutes"],
        max_iterations_per_subtask=config["max_iterations_per_subtask"],
        max_total_iterations=config["max_total_iterations"],
    )


def get_research_agent_config() -> ResearchAgentConfig:
    """Get default research agent configuration"""
    return ResearchAgentConfig()


def get_system_config() -> SystemConfig:
    """Get system configuration"""
    return SystemConfig()
