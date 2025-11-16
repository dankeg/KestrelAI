"""
Consolidated Orchestrator for KestrelAI
Replaces all duplicate orchestrator implementations with a single, configurable orchestrator
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel

from .base_agent import OrchestratorAgent
from .web_research_agent import ResearchConfig, WebResearchAgent

try:
    from memory.vector_store import MemoryStore
    from shared.models import Task, TaskStatus

    from .base import LlmWrapper
    from .config import get_orchestrator_config
    from .context_manager import ContextManager, TokenBudget, TokenCounter
    from .multi_level_summarizer import MultiLevelSummarizer
except ImportError:
    from KestrelAI.agents.base import LlmWrapper
    from KestrelAI.agents.config import get_orchestrator_config
    from KestrelAI.agents.context_manager import (
        ContextManager,
        TokenBudget,
        TokenCounter,
    )
    from KestrelAI.agents.multi_level_summarizer import MultiLevelSummarizer
    from KestrelAI.memory.vector_store import MemoryStore
    from KestrelAI.shared.models import Task, TaskStatus

logger = logging.getLogger(__name__)

# Context management constants
MAX_CONTEXT_CHARS = (
    80_000  # Maximum characters for recent notes in review (legacy fallback)
)
MAX_FEEDBACK_HISTORY = 20  # Maximum feedback entries to keep for loop detection


class OrchestratorDecision(BaseModel):
    reasoning: str
    decision: Literal["continue", "switch", "done"]
    feedback: str
    subtask: Literal["stay", "proceed"]
    next_task: str


class Subtask(BaseModel):
    order: int
    description: str
    success_criteria: str


class PlanningPlan(BaseModel):
    restated_task: str
    subtasks: list[Subtask]


class TaskState:
    """Enhanced task state tracking with subtask-specific agents"""

    def __init__(self, task: Task):
        self.task = task
        self.subtask_index = 0
        self.completed_subtasks: set[int] = set()
        self.notes_history: list[str] = []
        self.last_decision = None
        self.decision_count = 0
        self.stuck_count = 0
        self.last_progress_time = datetime.now()
        self.research_plan = None
        self.feedback_history: list[str] = []
        self.search_history: set[str] = set()
        self.repeated_actions: dict[str, int] = {}

        # Subtask-specific tracking
        self.subtask_agents: dict[int, WebResearchAgent] = {}
        self.subtask_findings: dict[int, list[str]] = {}
        self.subtask_reports: dict[int, str] = {}
        self.current_subtask_agent: WebResearchAgent | None = None
        self.all_findings: list[str] = []
        self.all_reports: list[str] = []  # Track all reports for accumulation

    def is_stuck(self, max_stuck_count: int = 3) -> bool:
        """Check if task is stuck in a loop"""
        return self.stuck_count >= max_stuck_count

    def record_decision(self, decision: str, feedback: str):
        """Record orchestrator decision for loop detection"""
        self.decision_count += 1
        self.last_decision = decision
        self.feedback_history.append(feedback)

        # Keep only last MAX_FEEDBACK_HISTORY feedback entries for loop detection
        if len(self.feedback_history) > MAX_FEEDBACK_HISTORY:
            self.feedback_history = self.feedback_history[-MAX_FEEDBACK_HISTORY:]

        # Check for repeated decisions
        if len(self.feedback_history) >= 3:
            recent_feedback = self.feedback_history[-3:]
            if len(set(recent_feedback)) == 1:  # All same feedback
                self.stuck_count += 1
            else:
                self.stuck_count = max(0, self.stuck_count - 1)

    def mark_subtask_complete(self, subtask_index: int):
        """Mark a subtask as completed"""
        self.completed_subtasks.add(subtask_index)
        self.subtask_index = max(self.completed_subtasks) + 1
        self.stuck_count = 0  # Reset stuck count on progress
        # Collect findings and reports from completed subtask
        if subtask_index in self.subtask_reports:
            report = self.subtask_reports[subtask_index]
            self.all_findings.append(f"Subtask {subtask_index + 1} Report:\n{report}")
            self.all_reports.append(report)  # Track report for accumulation

    def create_subtask_agent(
        self, subtask_index: int, llm: LlmWrapper, memory: MemoryStore, mcp_manager=None
    ) -> WebResearchAgent:
        """Create a new subtask-specific research agent"""
        if not self.research_plan or subtask_index >= len(self.research_plan.subtasks):
            raise ValueError(f"Invalid subtask index: {subtask_index}")

        subtask = self.research_plan.subtasks[subtask_index]
        subtask_id = f"{self.task.name}-subtask-{subtask_index}"

        # Collect previous findings from completed subtasks
        previous_findings = "\n\n".join(self.all_findings) if self.all_findings else ""

        # Collect previous reports for accumulation (pass actual report content, not summaries)
        previous_reports = self.all_reports.copy() if self.all_reports else []

        # Create research config for subtask agent
        config = ResearchConfig(
            is_subtask_agent=True,
            subtask_description=subtask.description,
            success_criteria=subtask.success_criteria,
            previous_findings=previous_findings,
            previous_reports=previous_reports,  # Pass previous reports for accumulation
            use_mcp=mcp_manager is not None,
            mcp_manager=mcp_manager,
        )

        agent = WebResearchAgent(
            agent_id=subtask_id, llm=llm, memory=memory, config=config
        )

        self.subtask_agents[subtask_index] = agent
        self.subtask_findings[subtask_index] = []
        return agent

    def get_current_subtask_agent(self) -> WebResearchAgent | None:
        """Get the current subtask agent"""
        if self.subtask_index in self.subtask_agents:
            return self.subtask_agents[self.subtask_index]
        return None

    def get_progress_percentage(self) -> float:
        """Calculate task progress based on completed subtasks"""
        if not self.research_plan or not self.research_plan.subtasks:
            return 0.0
        return (len(self.completed_subtasks) / len(self.research_plan.subtasks)) * 100.0


class ResearchOrchestrator(OrchestratorAgent):
    """Research orchestrator that manages research tasks and subtasks"""

    def __init__(
        self,
        tasks: list[Task],
        llm: LlmWrapper,
        profile: str = "kestrel",
        mcp_manager=None,
        use_mcp: bool = False,
    ):
        super().__init__("research-orchestrator", llm, MemoryStore())

        self.tasks = {t.name: t for t in tasks}
        self.current = tasks[0].name if tasks else None
        self.task_states: dict[str, TaskState] = {}

        # Initialize memory store for subtask agents
        self.memory = MemoryStore()

        # Load configuration
        self.config = get_orchestrator_config(profile)
        self.slice = timedelta(minutes=self.config.slice_minutes)

        # Initialize task states
        for task in tasks:
            self.task_states[task.name] = TaskState(task)

        # Loop prevention settings from config
        self.max_iterations_per_subtask = self.config.max_iterations_per_subtask
        self.max_total_iterations = self.config.max_total_iterations
        self.total_iterations = 0

        # MCP configuration
        self.use_mcp = use_mcp
        self.mcp_manager = mcp_manager
        self.mcp_connected = False

        # Initialize context management and summarization for orchestrator
        try:
            # Get model name from LLM wrapper if available (for TokenCounter)
            # Note: We pass llm object (not model_name string) to MultiLevelSummarizer
            model_name = getattr(llm, "model", "gemma3:27b")
            self.token_counter = TokenCounter(model_name=model_name)
            self.token_budget = TokenBudget(max_context=32768)  # Adjust based on model
            self.summarizer = MultiLevelSummarizer(
                llm=llm,  # Pass the actual llm object, not model_name string
                token_counter=self.token_counter,
                extract_facts=True,
            )
            self.context_manager = ContextManager(
                self.token_counter,
                self.token_budget,
                llm=llm,
                summarizer=self.summarizer,  # Pass summarizer to context manager
            )
            self.context_management_enabled = True
            logger.info(
                f"Orchestrator context management enabled with model: {model_name}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize orchestrator context management: {e}. Continuing without it."
            )
            self.token_counter = None
            self.token_budget = None
            self.context_manager = None
            self.summarizer = None
            self.context_management_enabled = False

        logger.info(
            f"Initialized research orchestrator with profile '{profile}': {self.config}"
        )

    async def initialize_mcp(self) -> bool:
        """Initialize MCP manager if configured"""
        if not self.use_mcp or not self.mcp_manager:
            return False

        try:
            if not self.mcp_manager.is_initialized:
                self.mcp_connected = await self.mcp_manager.initialize()
            else:
                self.mcp_connected = self.mcp_manager.is_initialized

            if self.mcp_connected:
                logger.info("MCP manager initialized successfully for orchestrator")
                tools = self.mcp_manager.get_available_tools()
                logger.info(f"Available MCP tools: {tools}")
            else:
                logger.error(
                    "MCP manager failed to initialize - no MCP capabilities available"
                )
            return self.mcp_connected
        except Exception as e:
            logger.error(f"Failed to initialize MCP manager: {e}")
            self.mcp_connected = False
            return False

    async def cleanup_mcp(self):
        """Cleanup MCP manager"""
        if self.mcp_manager:
            await self.mcp_manager.cleanup()
            self.mcp_connected = False

    async def _review(self, task: Task, latest_notes: str) -> OrchestratorDecision:
        """Enhanced review with loop prevention and better state tracking"""
        task_state = self.task_states[task.name]
        self.total_iterations += 1

        # Check for stuck state
        if task_state.is_stuck():
            logger.warning(f"Task {task.name} appears stuck, forcing progression")
            return OrchestratorDecision(
                reasoning="Task appears stuck in a loop, forcing progression to next subtask",
                decision="switch",
                feedback="Move to next subtask to break the loop",
                subtask="proceed",
                next_task=task.name,
            )

        # Check total iteration limit
        if self.total_iterations >= self.max_total_iterations:
            logger.warning("Maximum iterations reached, completing task")
            return OrchestratorDecision(
                reasoning="Maximum iterations reached, task should be completed",
                decision="done",
                feedback="Research completed within iteration limits",
                subtask="proceed",
                next_task="Not Applicable",
            )

        response = None
        retry_count = 0
        max_retries = 3

        while response is None and retry_count < max_retries:
            try:
                # Get current subtask info
                current_subtask_info = ""
                if task_state.research_plan and task_state.research_plan.subtasks:
                    if task_state.subtask_index < len(
                        task_state.research_plan.subtasks
                    ):
                        current_subtask = task_state.research_plan.subtasks[
                            task_state.subtask_index
                        ]
                        current_subtask_info = f"Current subtask: {current_subtask.description}\nSuccess criteria: {current_subtask.success_criteria}\nSubtask {task_state.subtask_index + 1} of {len(task_state.research_plan.subtasks)}"
                    else:
                        current_subtask_info = "All subtasks completed"

                # Build context with enhanced information (token-aware if enabled)
                if (
                    self.context_management_enabled
                    and self.token_counter
                    and self.token_budget
                ):
                    try:
                        # Build orchestrator-specific metadata (always include, small)
                        metadata_parts = [
                            f"Current time: {datetime.now()}",
                            f"Progress: {task_state.get_progress_percentage():.1f}%",
                            f"Completed subtasks: {len(task_state.completed_subtasks)}/{len(task_state.research_plan.subtasks) if task_state.research_plan else 0}",
                            f"Decision count: {task_state.decision_count}",
                            f"Stuck count: {task_state.stuck_count}",
                            current_subtask_info,
                            f"Previous feedback: {task_state.feedback_history[-1] if task_state.feedback_history else 'None'}",
                        ]
                        metadata = "\n".join(metadata_parts)

                        # Build full task description with metadata
                        full_task = f"{task.name} - {task.description}\n\n{metadata}"
                        task_tokens = self.token_counter.count_tokens(full_task)

                        # Calculate available tokens for notes (accounting for system prompt and response reserve)
                        # Get system prompt size (estimate if not available)
                        system_prompt = self._get_system_prompt()
                        system_tokens = self.token_counter.count_tokens(system_prompt)

                        # Available for user content = max_context - system - response_reserve
                        available_for_user = (
                            self.token_budget.max_context
                            - system_tokens
                            - self.token_budget.response_reserve
                        )

                        # Reserve some tokens for task description, then use rest for notes
                        max_notes_tokens = max(
                            100, available_for_user - task_tokens - 100
                        )  # Reserve 100 for formatting

                        # Check if recent notes need summarization
                        notes_tokens = self.token_counter.count_tokens(latest_notes)

                        if (
                            notes_tokens > max_notes_tokens
                            and self.context_management_enabled
                            and self.summarizer
                        ):
                            # Summarize notes to fit within budget
                            (
                                summary,
                                level,
                                facts,
                            ) = self.summarizer.create_summary_on_demand(
                                latest_notes,
                                max_tokens=max_notes_tokens,
                                preserve_facts=True,
                            )
                            logger.debug(
                                f"Summarized orchestrator notes: {notes_tokens} -> {self.token_counter.count_tokens(summary)} tokens (level: {level})"
                            )
                            processed_notes = summary
                        elif notes_tokens > max_notes_tokens:
                            # Fallback: truncate if summarizer not available
                            processed_notes = self.token_counter.truncate_to_tokens(
                                latest_notes, max_notes_tokens
                            )
                            logger.debug(
                                f"Truncated orchestrator notes: {notes_tokens} -> {self.token_counter.count_tokens(processed_notes)} tokens"
                            )
                        else:
                            processed_notes = latest_notes

                        # Build final context
                        context_parts = [
                            f"Task: {full_task}",
                            f"Recent notes:\n{processed_notes}",
                        ]
                        user_content = "\n".join(context_parts)

                        # Verify total doesn't exceed limits
                        total_user_tokens = self.token_counter.count_tokens(
                            user_content
                        )
                        total_with_system = system_tokens + total_user_tokens

                        if (
                            total_with_system
                            > self.token_budget.max_context
                            - self.token_budget.response_reserve
                        ):
                            # Emergency truncation if still too large
                            excess = total_with_system - (
                                self.token_budget.max_context
                                - self.token_budget.response_reserve
                            )
                            processed_notes = self.token_counter.truncate_to_tokens(
                                processed_notes,
                                self.token_counter.count_tokens(processed_notes)
                                - excess,
                            )
                            context_parts = [
                                f"Task: {full_task}",
                                f"Recent notes:\n{processed_notes}",
                            ]
                            user_content = "\n".join(context_parts)
                            logger.warning(
                                "Emergency truncation applied: reduced notes to fit context limit"
                            )

                        # Log token usage
                        final_tokens = self.token_counter.count_tokens(user_content)
                        logger.debug(
                            f"Orchestrator context: {final_tokens} tokens (system: {system_tokens}, task: {task_tokens}, notes: {self.token_counter.count_tokens(processed_notes)}, total: {system_tokens + final_tokens})"
                        )

                    except Exception as e:
                        logger.warning(
                            f"Error in orchestrator context management, using legacy: {e}",
                            exc_info=True,
                        )
                        # Fall back to legacy method
                        context_parts = [
                            f"Current time: {datetime.now()}",
                            f"Task: {task.name} - {task.description}",
                            f"Progress: {task_state.get_progress_percentage():.1f}%",
                            f"Completed subtasks: {len(task_state.completed_subtasks)}/{len(task_state.research_plan.subtasks) if task_state.research_plan else 0}",
                            f"Decision count: {task_state.decision_count}",
                            f"Stuck count: {task_state.stuck_count}",
                            current_subtask_info,
                            f"Recent notes: {latest_notes[:MAX_CONTEXT_CHARS]}...",  # Limit context size
                            f"Previous feedback: {task_state.feedback_history[-1] if task_state.feedback_history else 'None'}",
                        ]
                        user_content = "\n".join(context_parts)
                else:
                    # Legacy context building
                    context_parts = [
                        f"Current time: {datetime.now()}",
                        f"Task: {task.name} - {task.description}",
                        f"Progress: {task_state.get_progress_percentage():.1f}%",
                        f"Completed subtasks: {len(task_state.completed_subtasks)}/{len(task_state.research_plan.subtasks) if task_state.research_plan else 0}",
                        f"Decision count: {task_state.decision_count}",
                        f"Stuck count: {task_state.stuck_count}",
                        current_subtask_info,
                        f"Recent notes: {latest_notes[:MAX_CONTEXT_CHARS]}...",  # Limit context size
                        f"Previous feedback: {task_state.feedback_history[-1] if task_state.feedback_history else 'None'}",
                    ]
                    user_content = "\n".join(context_parts)

                # Get system prompt based on configuration
                system_prompt = self._get_system_prompt()

                msg = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]

                raw = self.llm.chat(msg)
                logger.debug(f"Orchestrator response: {raw}")
                response = OrchestratorDecision.model_validate_json(
                    self._clean_markdown_json(raw)
                )

            except Exception as e:
                retry_count += 1
                logger.error(f"Error during review phase (attempt {retry_count}): {e}")
                if retry_count >= max_retries:
                    # Fallback decision
                    response = OrchestratorDecision(
                        reasoning="Failed to get valid response from LLM, defaulting to continue",
                        decision="continue",
                        feedback="Continue with current research direction",
                        subtask="stay",
                        next_task=task.name,
                    )

        return response

    def _get_system_prompt(self) -> str:
        """Get system prompt based on configuration"""
        current_date = datetime.utcnow().strftime("%B %d, %Y")
        base_prompt = f"""You are a project manager overseeing a research agent. Your role is to facilitate research on a given topic by directing the agent to explore multiple angles and new avenues of investigation to a medium depth.

You may create new subtasks and branches of exploration as needed, but you must not create wholly new tasks outside the original scope.

Your work is exclusively focused on research and data gathering.
The goal is to find, synthesize, and combine as much relevant and useful information as possible into a clear, organized, and usable format that directly supports the objectives of the current task.

You have access to research agents that can search the web and gather information.
All information gathered should be actionable, clearly tied to the task at hand, and sufficient to enable well-informed decisions.

The date is {current_date}."""

        if self.use_mcp and self.mcp_connected:
            base_prompt += """

You have access to research agents with enhanced capabilities that can:
- Search multiple data sources simultaneously
- Access structured databases and repositories
- Perform data analysis and extraction
- Store and organize findings systematically
- Cross-reference information from multiple sources"""

        base_prompt += """

IMPORTANT: Research tasks and subtasks require extensive exploration. Do NOT mark a task as "done" or switch to a new subtask unless you have clear evidence that:
1. All aspects of the current subtask have been thoroughly investigated
2. Multiple approaches and angles have been explored
3. The research has reached a natural saturation point where new information is no longer being discovered
4. The subtask objectives have been explicitly and completely fulfilled

Guidelines for decision-making:
- "continue": Use this when there are still unexplored aspects, unanswered questions, or the research is yielding new insights
- "switch": Use this ONLY when the current approach has been exhausted AND you have a specific new angle to explore within the same subtask
- "done": Use this ONLY when you have concrete evidence that ALL subtask objectives have been met comprehensively

If the focus needs to shift or the current angles produce roadblocks, utilize the feedback field to issue a verbal command to the researcher.

Return ONLY JSON with in the following format, and NOTHING ELSE. RETURN ONLY this JSON:
{
  "reasoning": "Perform detailed reasoning about the current state of research. List what has been accomplished and what remains to be explored. Be specific about why you're making your decision.",
  "decision": "continue" | "switch" | "done",
  "feedback": "What specific aspects need more investigation? What questions remain unanswered? How should the research focus shift if needed?",
  "subtask": "stay" | "proceed",
  "next_task": "name_of_task or "Not Applicable" if staying on current task"
}

REMINDER: Err on the side of thoroughness. It's better to continue exploring than to prematurely conclude a research task or subtask."""

        return base_prompt

    def _clean_markdown_json(self, md_str: str) -> str:
        """Strips ```json ... ``` or ``` ... ``` from a Markdown block."""
        return re.sub(
            r"^```(?:json)?\s*|\s*```$", "", md_str.strip(), flags=re.MULTILINE
        )

    async def _planning_phase(self, task: Task) -> None:
        """Enhanced planning phase with better error handling and validation"""
        task_state = self.task_states[task.name]
        research_plan = None
        retry_count = 0
        max_retries = 5

        while research_plan is None and retry_count < max_retries:
            try:
                planning_prompt = """Planning Phase – Task Analysis and Decomposition

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

JSON Output Format
You must return your plan as valid JSON following this schema:

{
  "restated_task": "A clear restatement of the primary task in your own words.",
  "subtasks": [
    {
      "order": 1,
      "description": "Detailed explanation of what this subtask involves.",
      "success_criteria": "Description of what constitutes a complete and useful result for this subtask."
    }
  ]
}

Formatting rules:
- Output must be valid JSON — no extra commentary outside the JSON block.
- 'order' must start at 1 and increment sequentially.
- Be explicit in 'success_criteria' so the research agent knows exactly when the subtask is done."""

                msg = [
                    {"role": "system", "content": planning_prompt},
                    {
                        "role": "user",
                        "content": f"Task Name: {task.name}\nTask description: {task.description}\nBudget: {task.budgetMinutes} minutes",
                    },
                ]
                raw = self.llm.chat(msg)
                logger.debug(f"Planning response: {raw}")
                research_plan = PlanningPlan.model_validate_json(
                    self._clean_markdown_json(raw)
                )

                # Validate the plan
                if not research_plan.subtasks:
                    raise ValueError("No subtasks generated")

                # Ensure subtasks are properly ordered
                for i, subtask in enumerate(research_plan.subtasks):
                    subtask.order = i + 1

                logger.info(
                    f"Generated plan with {len(research_plan.subtasks)} subtasks for task {task.name}"
                )

            except Exception as e:
                retry_count += 1
                logger.error(
                    f"Error during planning phase (attempt {retry_count}): {e}"
                )
                if retry_count >= max_retries:
                    # Create a fallback plan
                    research_plan = PlanningPlan(
                        restated_task=f"Research task: {task.description}",
                        subtasks=[
                            Subtask(
                                order=1,
                                description=f"Conduct initial research on {task.name}",
                                success_criteria="Gather comprehensive information about the topic",
                            ),
                            Subtask(
                                order=2,
                                description="Analyze and synthesize findings",
                                success_criteria="Create a detailed analysis of the research findings",
                            ),
                            Subtask(
                                order=3,
                                description="Generate final report",
                                success_criteria="Produce a comprehensive final report with all findings",
                            ),
                        ],
                    )
                    logger.warning(f"Using fallback plan for task {task.name}")

        task_state.research_plan = research_plan

        # Create initial subtask agent
        if research_plan.subtasks:
            task_state.create_subtask_agent(
                0, self.llm, self.memory, self.mcp_manager if self.use_mcp else None
            )

    async def run_subtask_research(self, task: Task) -> dict[str, any]:
        """Run research for the current subtask using a dedicated agent"""
        task_state = self.task_states[task.name]

        # Get or create current subtask agent
        current_agent = task_state.get_current_subtask_agent()
        if not current_agent:
            # If we've already advanced past the last subtask, signal completion
            if not task_state.research_plan or task_state.subtask_index >= len(
                task_state.research_plan.subtasks
            ):
                return {
                    "status": "no_more_subtasks",
                    "message": "All subtasks completed",
                }

            current_agent = task_state.create_subtask_agent(
                task_state.subtask_index,
                self.llm,
                self.memory,
                self.mcp_manager if self.use_mcp else None,
            )

        # Run one research step for the current subtask agent
        result = await current_agent.run_step(task)

        # Always append the latest research output to notes history
        task_state.notes_history.append(result)

        # Best‑effort: pull structured findings from the agent's per‑task state
        try:
            if hasattr(current_agent, "_state") and task.name in current_agent._state:
                agent_state = current_agent._state[task.name]
                # Use checkpoints as a proxy for key findings for this subtask
                if hasattr(agent_state, "checkpoints"):
                    task_state.subtask_findings.setdefault(task_state.subtask_index, [])
                    task_state.subtask_findings[task_state.subtask_index].extend(
                        str(cp) for cp in agent_state.checkpoints
                    )
        except Exception as e:
            logger.warning(
                f"Failed to extract findings from subtask agent state: {e}",
                exc_info=True,
            )

        return {"status": "in_progress", "result": result}

    async def run_step(self, task: Task) -> str:
        """Run one step of the orchestrator workflow"""
        return await self.next_action(task)

    async def next_action(self, task: Task, notes: str = "") -> str:
        """
        Enhanced next action using subtask-specific research agents + LLM review loop.

        Flow:
        1. Run one research step for the current subtask via a dedicated WebResearchAgent.
        2. If all subtasks are complete, synthesize a final report and advance to the next task.
        3. Otherwise, call the orchestrator's review LLM (_review) to decide whether to:
           - continue on the current subtask,
           - proceed to the next subtask, or
           - mark the entire task as done.
        4. Update TaskState (subtask_index, completed_subtasks, feedback history).
        """
        task_state = self.task_states[task.name]

        # Run subtask research step
        research_result = await self.run_subtask_research(task)

        if research_result["status"] == "no_more_subtasks":
            # All subtasks completed
            logger.info(f"All subtasks completed for {task.name}")
            task.status = TaskStatus.COMPLETE

            # Generate final report from all subtask reports
            final_report = await self.synthesize_final_report(
                task, task_state.all_findings
            )

            # Save final report
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task.name
            ).strip()
            report_path = f"notes/{safe_name.upper()}_FINAL_REPORT.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(final_report)

            # Find next active task
            remaining = [
                t for t in self.tasks.values() if t.status != TaskStatus.COMPLETE
            ]
            self.current = remaining[0].name if remaining else None

            if self.current:
                logger.info(f"Moving to next task: {self.current}")
            else:
                logger.info("All tasks completed")

            return "Task completed - all subtasks finished"

        # We have new research output for the current subtask
        latest_notes = research_result.get("result", "")

        # If for some reason we don't have a research plan, fall back to simple completion
        if not task_state.research_plan or not task_state.research_plan.subtasks:
            logger.warning(f"No research plan for task {task.name}, marking complete")
            task.status = TaskStatus.COMPLETE
            return latest_notes or "Task completed"

        # Run orchestrator review to decide how to proceed
        decision = await self._review(task, latest_notes)
        task_state.record_decision(decision.decision, decision.feedback)
        logger.info(
            f"Orchestrator decision for task {task.name}: "
            f"decision={decision.decision}, subtask={decision.subtask}, next_task={decision.next_task}"
        )

        # Helper: finalize task when all subtasks are complete
        async def _finalize_task() -> str:
            task.status = TaskStatus.COMPLETE
            logger.info(f"Task {task.name} marked complete by orchestrator")

            # Generate final report from accumulated findings/reports
            final_report_sources = (
                task_state.all_findings if task_state.all_findings else [latest_notes]
            )
            final_report = await self.synthesize_final_report(
                task, final_report_sources
            )

            # Persist final report to notes file (legacy behavior, used by model_loop)
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task.name
            ).strip()
            report_path = f"notes/{safe_name.upper()}_FINAL_REPORT.txt"
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(final_report)
                logger.info(f"Wrote final report for task {task.name} to {report_path}")
            except Exception as e:
                logger.error(f"Failed to write final report for task {task.name}: {e}")

            # Find next active task (if orchestrator was initialized with multiple tasks)
            remaining = [
                t for t in self.tasks.values() if t.status != TaskStatus.COMPLETE
            ]
            self.current = remaining[0].name if remaining else None

            if self.current:
                logger.info(f"Moving to next task: {self.current}")
            else:
                logger.info("All tasks completed")

            return "Task completed - orchestrator marked task as done"

        # 1) If orchestrator decides the entire task is done, finalize
        if decision.decision == "done":
            # Ensure the current subtask is counted as complete
            if task_state.subtask_index not in task_state.completed_subtasks:
                # Store latest_notes as the report for this subtask
                task_state.subtask_reports[task_state.subtask_index] = latest_notes
                task_state.mark_subtask_complete(task_state.subtask_index)

            # Mark any remaining subtasks as completed for progress reporting
            if task_state.research_plan and task_state.research_plan.subtasks:
                for idx in range(len(task_state.research_plan.subtasks)):
                    task_state.completed_subtasks.add(idx)

            return await _finalize_task()

        # 2) If orchestrator decides to proceed to the next subtask, advance subtask_index
        if decision.subtask == "proceed" or decision.decision == "switch":
            # Store the latest research output as the report for the current subtask
            task_state.subtask_reports[task_state.subtask_index] = latest_notes

            # Mark current subtask complete and advance
            task_state.mark_subtask_complete(task_state.subtask_index)

            # If there is another subtask, create its agent; otherwise we are done with subtasks
            if task_state.research_plan and task_state.subtask_index < len(
                task_state.research_plan.subtasks
            ):
                logger.info(
                    f"Proceeding to subtask {task_state.subtask_index + 1} "
                    f"for task {task.name}"
                )
                task_state.create_subtask_agent(
                    task_state.subtask_index,
                    self.llm,
                    self.memory,
                    self.mcp_manager if self.use_mcp else None,
                )
            else:
                # All subtasks completed after this progression
                logger.info(
                    f"All subtasks completed for {task.name} after orchestrator decision"
                )
                return await _finalize_task()

        # 3) Otherwise, stay on the current subtask and continue researching next step
        return latest_notes or "Continuing research"

    def get_current_subtask(self, task_name: str) -> str | None:
        """Get current subtask description for a task"""
        if task_name not in self.task_states:
            return None

        task_state = self.task_states[task_name]
        if not task_state.research_plan or not task_state.research_plan.subtasks:
            return None

        if task_state.subtask_index < len(task_state.research_plan.subtasks):
            subtask = task_state.research_plan.subtasks[task_state.subtask_index]
            return f"{subtask.description} (Success: {subtask.success_criteria})"

        return "All subtasks completed"

    def get_task_progress(self, task_name: str) -> dict[str, any]:
        """Get detailed progress information for a task with subtask agent details"""
        if task_name not in self.task_states:
            return {"progress": 0.0, "subtasks": [], "completed": 0, "total": 0}

        task_state = self.task_states[task_name]
        if not task_state.research_plan:
            return {"progress": 0.0, "subtasks": [], "completed": 0, "total": 0}

        subtasks = []
        for i, subtask in enumerate(task_state.research_plan.subtasks):
            subtask_info = {
                "order": subtask.order,
                "description": subtask.description,
                "success_criteria": subtask.success_criteria,
                "completed": i in task_state.completed_subtasks,
                "current": i == task_state.subtask_index,
                "findings_count": len(task_state.subtask_findings.get(i, [])),
                "has_report": i in task_state.subtask_reports,
            }

            # Add agent metrics if available
            if i in task_state.subtask_agents:
                agent = task_state.subtask_agents[i]
                agent_metrics = agent.get_metrics()
                subtask_info.update(
                    {
                        "agent_metrics": agent_metrics,
                    }
                )

            subtasks.append(subtask_info)

        return {
            "progress": task_state.get_progress_percentage(),
            "subtasks": subtasks,
            "completed": len(task_state.completed_subtasks),
            "total": len(task_state.research_plan.subtasks),
            "current_subtask": task_state.subtask_index + 1,
            "decision_count": task_state.decision_count,
            "stuck_count": task_state.stuck_count,
            "total_findings": len(task_state.all_findings),
            "subtask_agents_count": len(task_state.subtask_agents),
        }

    async def synthesize_final_report(
        self, task: Task, research_reports: list[str]
    ) -> str:
        """Combines multiple research reports into one cohesive final report with token-aware summarization"""
        deduplication_prompt = """Extract the most valuable and actionable information from this research report.

Focus on:
- Specific programs, grants, or opportunities with exact details
- Concrete deadlines, requirements, and application processes
- Direct contact information and application links
- Specific eligibility criteria and requirements
- Current opportunities (not generic database descriptions)

Remove:
- Generic advice that applies to any research topic
- Placeholder text and template content
- Meta-commentary about the research process
- Vague recommendations without specific details
- Information that appears verbatim in other reports

Prioritize actionable, specific information over comprehensive archival content."""

        final_synthesis_prompt = f"""Create a focused, actionable final report from these research findings.

Context:
Task: {task.name}
Description: {task.description}

CRITICAL REQUIREMENTS:
- Focus on SPECIFIC, ACTIONABLE opportunities the user can apply to
- Include concrete details: exact deadlines, specific requirements, contact information, application links
- Prioritize CURRENT opportunities (not generic database descriptions)
- Remove generic advice and focus on specific programs, grants, or opportunities
- Include exact eligibility requirements, application processes, and deadlines
- Provide direct links and contact information where available

Structure the report to be:
- Fact-heavy with specific details and numbers
- Actionable with clear next steps
- Well-organized with clear sections
- Professional but concise
- Focused on opportunities the user can actually apply to

Avoid:
- Generic database descriptions
- Vague recommendations
- Placeholder text
- Overly comprehensive archival content
- Generic advice that applies to any research topic

Focus on: Specific programs, exact deadlines, concrete requirements, direct application links."""

        # Deduplicate each report while preserving detail
        deduplicated_findings = []
        for i, report in enumerate(research_reports):
            # Use summarization if report is too long and context management is enabled
            if (
                self.context_management_enabled
                and self.summarizer
                and self.token_budget
            ):
                try:
                    report_tokens = self.token_counter.count_tokens(report)
                    max_report_tokens = self.token_budget.previous_findings // len(
                        research_reports
                    )  # Divide budget across reports

                    if report_tokens > max_report_tokens:
                        # Summarize report before deduplication
                        (
                            summary,
                            level,
                            facts,
                        ) = self.summarizer.create_summary_on_demand(
                            report,
                            max_tokens=max_report_tokens,
                            preserve_facts=True,
                        )
                        logger.debug(
                            f"Summarized report {i+1} for synthesis: {report_tokens} -> {self.token_counter.count_tokens(summary)} tokens"
                        )
                        report = summary
                except Exception as e:
                    logger.warning(
                        f"Error summarizing report {i+1}, using full content: {e}"
                    )

            msg = [
                {"role": "system", "content": deduplication_prompt},
                {
                    "role": "user",
                    "content": f"Report {i+1} of {len(research_reports)}:\n\n{report}",
                },
            ]
            deduplicated_findings.append(self.llm.chat(msg))

        # Combine deduplicated findings
        combined_findings = "\n\n---\n\n".join(deduplicated_findings)

        # Summarize combined findings if too long before final synthesis
        if self.context_management_enabled and self.summarizer and self.token_budget:
            try:
                combined_tokens = self.token_counter.count_tokens(combined_findings)
                max_combined_tokens = self.token_budget.previous_findings

                if combined_tokens > max_combined_tokens:
                    # Summarize combined findings
                    summary, level, facts = self.summarizer.create_summary_on_demand(
                        combined_findings,
                        max_tokens=max_combined_tokens,
                        preserve_facts=True,
                    )
                    logger.debug(
                        f"Summarized combined findings for final synthesis: {combined_tokens} -> {self.token_counter.count_tokens(summary)} tokens"
                    )
                    combined_findings = summary
            except Exception as e:
                logger.warning(
                    f"Error summarizing combined findings, using full content: {e}"
                )

        # Create final synthesis with full context
        final_msg = [
            {"role": "system", "content": final_synthesis_prompt},
            {
                "role": "user",
                "content": "Deduplicated findings from all research:\n\n"
                + combined_findings,
            },
        ]

        return self.llm.chat(final_msg)

    async def __aenter__(self):
        """Async context manager entry"""
        if self.use_mcp:
            await self.initialize_mcp()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.use_mcp:
            await self.cleanup_mcp()
