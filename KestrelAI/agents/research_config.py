"""
Configuration for research agent behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Default configuration values
THINK_LOOPS = 6
SEARCH_RESULTS = 4
FETCH_BYTES = 30_000
DEBUG = True
CONTEXT_WINDOW = 60
CHECKPOINT_FREQ = 5
MAX_SNIPPET_LENGTH = 3000


@dataclass
class ResearchConfig:
    """Configuration for research agent behavior"""

    think_loops: int = THINK_LOOPS
    search_results: int = SEARCH_RESULTS
    fetch_bytes: int = FETCH_BYTES
    context_window: int = CONTEXT_WINDOW
    checkpoint_freq: int = CHECKPOINT_FREQ
    max_snippet_length: int = MAX_SNIPPET_LENGTH
    debug: bool = DEBUG

    # Subtask-specific settings
    is_subtask_agent: bool = False
    subtask_description: str = ""
    success_criteria: str = ""
    previous_findings: str = ""
    previous_reports: list[str] = field(
        default_factory=list
    )  # Previous reports to build upon

    # MCP settings
    use_mcp: bool = False
    mcp_manager: Any | None = None
