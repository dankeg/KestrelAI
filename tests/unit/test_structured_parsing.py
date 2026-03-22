from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from KestrelAI.agents.langchain_orchestrator_chains import (
    OrchestratorLangChainControlChains,
)
from KestrelAI.agents.structured_parsing import parse_to_schema

try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:  # pragma: no cover - dependency-gated path
    ChatPromptTemplate = None


class _DecisionSchema(BaseModel):
    reasoning: str
    decision: Literal["continue", "switch", "done"]
    feedback: str
    subtask: Literal["stay", "proceed"]
    next_task: str


class _PrePlanSchema(BaseModel):
    reasoning: str = ""
    action: Literal["think", "search", "mcp_tool", "done"] = "done"
    query: str = ""
    thought: str = ""
    tool_name: str = ""
    tool_parameters: dict = Field(default_factory=dict)


class _SubtaskSchema(BaseModel):
    order: int
    description: str
    success_criteria: str


class _PlanningSchema(BaseModel):
    restated_task: str
    subtasks: list[_SubtaskSchema]


class _FactsSchema(BaseModel):
    eligibility: list[str] = Field(default_factory=list)
    deadlines: list[str] = Field(default_factory=list)


def test_parse_to_schema_recovers_truncated_json() -> None:
    raw = '{"reasoning":"ok","action":"search","query":"ai scholarships 2026"'
    parsed = parse_to_schema(raw, _PrePlanSchema)
    assert parsed.action == "search"
    assert "scholarships" in parsed.query


def test_parse_to_schema_recovers_json_code_fence() -> None:
    raw = (
        "Here is the result:\n"
        "```json\n"
        '{"reasoning":"clear","decision":"continue","feedback":"search more","subtask":"stay","next_task":"x"}\n'
        "```"
    )
    parsed = parse_to_schema(raw, _DecisionSchema)
    assert parsed.decision == "continue"
    assert parsed.subtask == "stay"


def test_parse_to_schema_recovers_key_value_lines() -> None:
    raw = (
        "reasoning: move forward\n"
        "decision: switch\n"
        "feedback: pivot to eligibility requirements\n"
        "subtask: proceed\n"
        "next_task: eligibility review"
    )
    parsed = parse_to_schema(raw, _DecisionSchema)
    assert parsed.decision == "switch"
    assert parsed.next_task == "eligibility review"


def test_parse_to_schema_normalizes_capitalized_review_literals() -> None:
    raw = {
        "reasoning": "Need more evidence.",
        "decision": "Continue",
        "feedback": "Search official pages.",
        "subtask": "Stay",
        "next_task": "current_task",
    }
    parsed = parse_to_schema(raw, _DecisionSchema)
    assert parsed.decision == "continue"
    assert parsed.subtask == "stay"


def test_parse_to_schema_repairs_partial_review_dict() -> None:
    raw = {"decision": "Continue"}
    parsed = parse_to_schema(raw, _DecisionSchema)
    assert parsed.decision == "continue"
    assert parsed.subtask == "stay"
    assert parsed.next_task == "current_task"
    assert parsed.feedback
    assert parsed.reasoning


def test_parse_to_schema_recovers_preplanning_think_prefix() -> None:
    raw = "think: Focus first on official NSF program pages and recent REU listings."
    parsed = parse_to_schema(raw, _PrePlanSchema)
    assert parsed.action == "think"
    assert "official NSF" in parsed.thought


def test_parse_to_schema_recovers_preplanning_search_prefix() -> None:
    raw = "search: NSF REU AI machine learning undergraduate 2026"
    parsed = parse_to_schema(raw, _PrePlanSchema)
    assert parsed.action == "search"
    assert "NSF REU" in parsed.query


def test_parse_to_schema_recovers_markdown_planning_outline() -> None:
    raw = """
    1. Restatement of Task:
    Identify current NSF REU opportunities in AI/ML for undergraduates and summarize them.

    Subtask 1: Find official NSF REU listings relevant to AI/ML
    Success Criteria: Capture named programs, links, and basic eligibility constraints.

    Subtask 2: Validate deadlines and application details
    Success Criteria: Confirm timeline details from official pages or host institutions.

    Subtask 3: Produce an actionable shortlist
    Success Criteria: Summarize the strongest opportunities and next actions.
    """
    parsed = parse_to_schema(raw, _PlanningSchema)
    assert "NSF REU opportunities" in parsed.restated_task
    assert len(parsed.subtasks) == 3
    assert parsed.subtasks[0].description.startswith("Find official NSF REU")
    assert "eligibility" in parsed.subtasks[0].success_criteria.lower()


def test_orchestrator_prompt_examples_do_not_create_extra_template_variables() -> None:
    if ChatPromptTemplate is None:
        return

    planning_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", OrchestratorLangChainControlChains._planning_system_prompt()),
            ("human", OrchestratorLangChainControlChains._planning_user_prompt()),
        ]
    )
    preplanning_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", OrchestratorLangChainControlChains._preplanning_system_prompt()),
            ("human", OrchestratorLangChainControlChains._preplanning_user_prompt()),
        ]
    )

    assert set(planning_prompt.input_variables) == {
        "task_name",
        "task_description",
        "task_budget_minutes",
        "preplanning_context",
    }
    assert set(preplanning_prompt.input_variables) == {
        "task_name",
        "task_description",
        "task_budget_minutes",
        "mcp_enabled",
        "exploration_log",
    }


def test_parse_to_schema_recovers_review_decision_from_free_text() -> None:
    raw = (
        "Continue gathering evidence for the current subtask using diversified, "
        "success-criteria-aligned queries. Focus on official NSF pages and deadline verification."
    )
    parsed = parse_to_schema(raw, _DecisionSchema)
    assert parsed.decision == "continue"
    assert parsed.subtask == "stay"
    assert "official NSF" in parsed.feedback


def test_parse_to_schema_coerces_string_to_list_fields() -> None:
    raw = {
        "eligibility": "Open to U.S. citizens and permanent residents.",
        "deadlines": "February 1, 2026",
    }
    parsed = parse_to_schema(raw, _FactsSchema)
    assert parsed.eligibility == ["Open to U.S. citizens and permanent residents."]
    assert parsed.deadlines == ["February 1, 2026"]
