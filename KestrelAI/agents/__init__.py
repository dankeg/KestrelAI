"""LLM Agents package.

This module uses lazy imports to avoid pulling in heavy dependencies
unnecessarily (e.g., vector store / chromadb) when only lightweight
utilities like the multi-level summarizer are needed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaseAgent",
    "ResearchAgent",
    "OrchestratorAgent",
    "AgentState",
    "WebResearchAgent",
    "ResearchConfig",
    "ResearchOrchestrator",
    "LlmWrapper",
    "get_orchestrator_config",
    "OrchestratorConfig",
]


def __getattr__(name: str) -> Any:
    """
    Lazily import heavy agent classes and utilities on first access.

    This keeps `import KestrelAI.agents.multi_level_summarizer` lightweight
    and avoids importing the vector store / chromadb stack unless callers
    actually need full agent orchestration.
    """
    if name in {"BaseAgent", "ResearchAgent", "OrchestratorAgent", "AgentState"}:
        from .base_agent import AgentState, BaseAgent, OrchestratorAgent, ResearchAgent

        mapping = {
            "BaseAgent": BaseAgent,
            "ResearchAgent": ResearchAgent,
            "OrchestratorAgent": OrchestratorAgent,
            "AgentState": AgentState,
        }
        return mapping[name]

    if name in {"WebResearchAgent", "ResearchConfig"}:
        from .web_research_agent import ResearchConfig, WebResearchAgent

        mapping = {
            "WebResearchAgent": WebResearchAgent,
            "ResearchConfig": ResearchConfig,
        }
        return mapping[name]

    if name == "ResearchOrchestrator":
        from .research_orchestrator import ResearchOrchestrator

        return ResearchOrchestrator

    if name in {"LlmWrapper"}:
        from .base import LlmWrapper

        return LlmWrapper

    if name in {"get_orchestrator_config", "OrchestratorConfig"}:
        from .config import OrchestratorConfig, get_orchestrator_config

        mapping = {
            "get_orchestrator_config": get_orchestrator_config,
            "OrchestratorConfig": OrchestratorConfig,
        }
        return mapping[name]

    raise AttributeError(f"module 'KestrelAI.agents' has no attribute {name!r}")
