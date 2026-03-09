"""
LangChain-native planning/review chains for orchestrator control flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:  # pragma: no cover - dependency-gated path
    StrOutputParser = None
    ChatPromptTemplate = None

from .structured_parsing import parse_to_schema


def _require_langchain() -> None:
    if ChatPromptTemplate is None or StrOutputParser is None:
        raise ImportError("langchain_core is required for orchestrator chains")


@dataclass
class OrchestratorLangChainControlChains:
    """
    Structured chains for orchestrator review and planning outputs.
    """

    model: Any
    review_schema: type[BaseModel]
    planning_schema: type[BaseModel]
    preplanning_schema: type[BaseModel] | None = None

    def __post_init__(self) -> None:
        _require_langchain()
        parser = StrOutputParser()
        self._review_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{review_system_prompt}"),
                ("human", "{review_user_content}"),
            ]
        )
        self._review_text_chain = self._review_prompt | self.model | parser

        self._planning_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._planning_system_prompt()),
                ("human", self._planning_user_prompt()),
            ]
        )
        self._planning_text_chain = self._planning_prompt | self.model | parser

        self._preplanning_text_chain = None
        if self.preplanning_schema is not None:
            self._preplanning_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self._preplanning_system_prompt()),
                    ("human", self._preplanning_user_prompt()),
                ]
            )
            self._preplanning_text_chain = (
                self._preplanning_prompt | self.model | parser
            )

    @staticmethod
    def _validate_schema_result(result: Any, schema: type[BaseModel]) -> BaseModel:
        return parse_to_schema(result, schema)

    def _invoke_with_recovery(
        self,
        *,
        text_chain: Any,
        payload: dict[str, Any],
        schema: type[BaseModel],
    ) -> BaseModel:
        raw = text_chain.invoke(payload)
        return self._validate_schema_result(raw, schema)

    async def _ainvoke_with_recovery(
        self,
        *,
        text_chain: Any,
        payload: dict[str, Any],
        schema: type[BaseModel],
    ) -> BaseModel:
        raw = await text_chain.ainvoke(payload)
        return self._validate_schema_result(raw, schema)

    @staticmethod
    def build_review_system_prompt(*, use_mcp: bool, mcp_connected: bool) -> str:
        """Build the orchestrator review system prompt."""
        current_date = datetime.utcnow().strftime("%B %d, %Y")
        prompt = f"""You are a project manager overseeing a research agent. Your role is to facilitate research on a given topic by directing the agent to explore multiple angles and new avenues of investigation to a medium depth.

You may create new subtasks and branches of exploration as needed, but you must not create wholly new tasks outside the original scope.

Your work is exclusively focused on research and data gathering.
The goal is to find, synthesize, and combine as much relevant and useful information as possible into a clear, organized, and usable format that directly supports the objectives of the current task.

You have access to research agents that can search the web and gather information.
All information gathered should be actionable, clearly tied to the task at hand, and sufficient to enable well-informed decisions.

The date is {current_date}."""

        if use_mcp and mcp_connected:
            prompt += """

You have access to research agents with enhanced capabilities that can:
- Search multiple data sources simultaneously
- Access structured databases and repositories
- Perform data analysis and extraction
- Store and organize findings systematically
- Cross-reference information from multiple sources"""

        prompt += """

IMPORTANT: Research tasks and subtasks require extensive exploration. Do NOT mark a task as "done" or switch to a new subtask unless you have clear evidence that:
1. All aspects of the current subtask have been thoroughly investigated
2. Multiple approaches and angles have been explored
3. The research has reached a natural saturation point where new information is no longer being discovered
4. The subtask objectives have been explicitly and completely fulfilled

Decision quality requirements:
- Base your decision on the provided subtask metrics, guidance history, and recent notes.
- Prefer "continue" when coverage is thin, repetitive, or missing concrete evidence.
- Use "switch" only when there is a specific next angle and include that angle in feedback.
- Use "done" only when success criteria are clearly satisfied with evidence.
- feedback must be an actionable command (not generic) that tells the researcher what to do next.
- feedback should include concrete focus dimensions (sources, constraints, entities, or query pivots).

Guidelines for decision-making:
- "continue": Use this when there are still unexplored aspects, unanswered questions, or the research is yielding new insights
- "switch": Use this ONLY when the current approach has been exhausted AND you have a specific new angle to explore within the same subtask
- "done": Use this ONLY when you have concrete evidence that ALL subtask objectives have been met comprehensively

If the focus needs to shift or the current angles produce roadblocks, utilize the feedback field to issue a verbal command to the researcher.
"""
        return prompt

    @staticmethod
    def _planning_system_prompt() -> str:
        return """Planning Phase – Task Analysis and Decomposition

You have received the primary research task from the orchestrator.
Your goal is to analyze the task and produce a clear, actionable plan for the research agent to follow.

Your responsibilities in this phase:
1. Understand the task fully – restate it in your own words to ensure clarity.
2. Identify the key objectives and constraints – determine what information is required and what the outputs should look like.
3. Decompose the task into a sequential set of subtasks – each subtask should:
   - Be directly relevant to the main task.
   - Be small enough for focused research.
   - Allow for medium-depth exploration.
4. Define success criteria – state what constitutes a complete and useful result for each subtask.

Important:
- Subtasks must remain within scope — they should extend or branch from the original task, not introduce wholly new research goals.
- Keep in mind: the research agent can search the web but cannot send emails, execute code, or access APIs.
- Plan for progress across the entire scope — avoid tunnel vision on one sub-area.
- Prefer subtasks that target authoritative evidence collection and verification.
- For discovery subtasks, describe the evidence class to collect, not a single page or named search hit to inspect.
- Do not create subtasks that investigate one specific page, article title, or incidental lead unless the task explicitly names that source as required.
- Discovery subtasks should emphasize breadth, diversity of institutions/programs, and deduplication of overlapping leads.
- Do NOT create subtasks centered on low-signal sources such as generic blogs, explainers, forums, or commercial content unless the task explicitly asks for them.
- Do NOT anchor the plan on a specific non-authoritative page just because it appeared in a search result.
- Use subtasks like discovery, authoritative verification, comparison, and synthesis.
- Return ONLY a valid JSON object. Do not include markdown, code fences, commentary, or a preamble.
- Use exactly this shape:
  {{"restated_task":"...","subtasks":[{{"order":1,"description":"...","success_criteria":"..."}}]}}"""

    @staticmethod
    def _planning_user_prompt() -> str:
        return (
            "Task Name: {task_name}\n"
            "Task description: {task_description}\n"
            "Budget: {task_budget_minutes} minutes\n"
            "Pre-planning exploration findings:\n{preplanning_context}"
        )

    @staticmethod
    def _preplanning_system_prompt() -> str:
        return """You are preparing to generate a high-quality research plan.
Run a short bounded exploration before planning.

Choose exactly one action:
- think: reason about scope, constraints, and unknowns
- search: perform one targeted web query
- mcp_tool: call one MCP tool if available
- done: stop exploration and proceed to planning

Guidelines:
- Prefer specific, high-signal actions over generic steps.
- Avoid repeating the same search/tool action.
- Keep each step concise and decision-oriented.
- Choose done once you have enough context for a strong plan.
- Return ONLY a valid JSON object with exactly these keys:
  {{"reasoning":"...","action":"think|search|mcp_tool|done","query":"","thought":"","tool_name":"","tool_parameters":{{}}}}"""

    @staticmethod
    def _preplanning_user_prompt() -> str:
        return (
            "Task Name: {task_name}\n"
            "Task description: {task_description}\n"
            "Budget: {task_budget_minutes} minutes\n"
            "MCP enabled: {mcp_enabled}\n"
            "Exploration log so far:\n{exploration_log}"
        )

    def review_decision(
        self,
        *,
        review_system_prompt: str,
        review_user_content: str,
    ) -> BaseModel:
        return self._invoke_with_recovery(
            text_chain=self._review_text_chain,
            payload={
                "review_system_prompt": review_system_prompt,
                "review_user_content": review_user_content,
            },
            schema=self.review_schema,
        )

    async def areview_decision(
        self,
        *,
        review_system_prompt: str,
        review_user_content: str,
    ) -> BaseModel:
        return await self._ainvoke_with_recovery(
            text_chain=self._review_text_chain,
            payload={
                "review_system_prompt": review_system_prompt,
                "review_user_content": review_user_content,
            },
            schema=self.review_schema,
        )

    def planning_plan(
        self,
        *,
        task_name: str,
        task_description: str,
        task_budget_minutes: int,
        preplanning_context: str = "",
    ) -> BaseModel:
        return self._invoke_with_recovery(
            text_chain=self._planning_text_chain,
            payload={
                "task_name": task_name,
                "task_description": task_description,
                "task_budget_minutes": task_budget_minutes,
                "preplanning_context": preplanning_context or "None",
            },
            schema=self.planning_schema,
        )

    async def aplanning_plan(
        self,
        *,
        task_name: str,
        task_description: str,
        task_budget_minutes: int,
        preplanning_context: str = "",
    ) -> BaseModel:
        return await self._ainvoke_with_recovery(
            text_chain=self._planning_text_chain,
            payload={
                "task_name": task_name,
                "task_description": task_description,
                "task_budget_minutes": task_budget_minutes,
                "preplanning_context": preplanning_context or "None",
            },
            schema=self.planning_schema,
        )

    def preplanning_action(
        self,
        *,
        task_name: str,
        task_description: str,
        task_budget_minutes: int,
        mcp_enabled: bool,
        exploration_log: str,
    ) -> BaseModel:
        if self._preplanning_text_chain is None:
            raise RuntimeError("Pre-planning chain schema is not configured.")
        return self._invoke_with_recovery(
            text_chain=self._preplanning_text_chain,
            payload={
                "task_name": task_name,
                "task_description": task_description,
                "task_budget_minutes": task_budget_minutes,
                "mcp_enabled": "yes" if mcp_enabled else "no",
                "exploration_log": exploration_log or "(none yet)",
            },
            schema=self.preplanning_schema,
        )

    async def apreplanning_action(
        self,
        *,
        task_name: str,
        task_description: str,
        task_budget_minutes: int,
        mcp_enabled: bool,
        exploration_log: str,
    ) -> BaseModel:
        if self._preplanning_text_chain is None:
            raise RuntimeError("Pre-planning chain schema is not configured.")
        return await self._ainvoke_with_recovery(
            text_chain=self._preplanning_text_chain,
            payload={
                "task_name": task_name,
                "task_description": task_description,
                "task_budget_minutes": task_budget_minutes,
                "mcp_enabled": "yes" if mcp_enabled else "no",
                "exploration_log": exploration_log or "(none yet)",
            },
            schema=self.preplanning_schema,
        )
