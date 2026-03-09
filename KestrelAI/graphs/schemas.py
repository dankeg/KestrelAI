"""
Typed schemas used by LangChain/LangGraph execution.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResearchActionPlan(BaseModel):
    direction: str = Field(default="")
    action: Literal["think", "search", "mcp_tool", "summarize", "complete"] = "think"
    query: str = Field(default="")
    thought: str = Field(default="")
    tool_name: str = Field(default="")
    tool_parameters: dict = Field(default_factory=dict)


class OrchestratorDecisionSchema(BaseModel):
    reasoning: str
    decision: Literal["continue", "switch", "done"]
    feedback: str
    subtask: Literal["stay", "proceed"]
    next_task: str
