"""
Consolidated Orchestrator for KestrelAI
Replaces all duplicate orchestrator implementations with a single, configurable orchestrator
"""

from datetime import datetime, timedelta
from typing import List, Literal, Dict, Optional, Set
from pydantic import BaseModel
import re
import json
import logging
import asyncio

from .base_agent import OrchestratorAgent
from .web_research_agent import WebResearchAgent, ResearchConfig
try:
    from .base import LlmWrapper
    from .config import get_orchestrator_config, OrchestratorConfig
    from memory.vector_store import MemoryStore
    from shared.models import Task, TaskStatus
except ImportError:
    from KestrelAI.agents.base import LlmWrapper
    from KestrelAI.agents.config import get_orchestrator_config, OrchestratorConfig
    from KestrelAI.memory.vector_store import MemoryStore
    from KestrelAI.shared.models import Task, TaskStatus

logger = logging.getLogger(__name__)


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
    subtasks: List[Subtask]


class TaskState:
    """Enhanced task state tracking with subtask-specific agents"""
    def __init__(self, task: Task):
        self.task = task
        self.subtask_index = 0
        self.completed_subtasks: Set[int] = set()
        self.notes_history: List[str] = []
        self.last_decision = None
        self.decision_count = 0
        self.stuck_count = 0
        self.last_progress_time = datetime.now()
        self.research_plan = None
        self.feedback_history: List[str] = []
        self.search_history: Set[str] = set()
        self.repeated_actions: Dict[str, int] = {}
        
        # Subtask-specific tracking
        self.subtask_agents: Dict[int, WebResearchAgent] = {}
        self.subtask_findings: Dict[int, List[str]] = {}
        self.subtask_reports: Dict[int, str] = {}
        self.current_subtask_agent: Optional[WebResearchAgent] = None
        self.all_findings: List[str] = []
        
    def is_stuck(self, max_stuck_count: int = 3) -> bool:
        """Check if task is stuck in a loop"""
        return self.stuck_count >= max_stuck_count
    
    def record_decision(self, decision: str, feedback: str):
        """Record orchestrator decision for loop detection"""
        self.decision_count += 1
        self.last_decision = decision
        self.feedback_history.append(feedback)
        
        # Keep only last 5 feedback entries for loop detection
        if len(self.feedback_history) > 5:
            self.feedback_history = self.feedback_history[-5:]
        
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
        
        # Collect findings from completed subtask
        if subtask_index in self.subtask_agents:
            agent = self.subtask_agents[subtask_index]
            if subtask_index in self.subtask_reports:
                self.all_findings.append(f"Subtask {subtask_index + 1} Report:\n{self.subtask_reports[subtask_index]}")
    
    def create_subtask_agent(self, subtask_index: int, llm: LlmWrapper, memory: MemoryStore, 
                           mcp_manager=None) -> WebResearchAgent:
        """Create a new subtask-specific research agent"""
        if not self.research_plan or subtask_index >= len(self.research_plan.subtasks):
            raise ValueError(f"Invalid subtask index: {subtask_index}")
        
        subtask = self.research_plan.subtasks[subtask_index]
        subtask_id = f"{self.task.name}-subtask-{subtask_index}"
        
        # Collect previous findings from completed subtasks
        previous_findings = "\n\n".join(self.all_findings) if self.all_findings else ""
        
        # Create research config for subtask agent
        config = ResearchConfig(
            is_subtask_agent=True,
            subtask_description=subtask.description,
            success_criteria=subtask.success_criteria,
            previous_findings=previous_findings,
            use_mcp=mcp_manager is not None,
            mcp_manager=mcp_manager
        )
        
        agent = WebResearchAgent(
            agent_id=subtask_id,
            llm=llm,
            memory=memory,
            config=config
        )
        
        self.subtask_agents[subtask_index] = agent
        self.subtask_findings[subtask_index] = []
        return agent
    
    def get_current_subtask_agent(self) -> Optional[WebResearchAgent]:
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
    
    def __init__(self, tasks: List[Task], llm: LlmWrapper, profile: str = "kestrel", 
                 mcp_manager=None, use_mcp: bool = False):
        super().__init__("research-orchestrator", llm, MemoryStore())
        
        self.tasks = {t.name: t for t in tasks}
        self.current = tasks[0].name if tasks else None
        self.task_states: Dict[str, TaskState] = {}
        
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
        
        logger.info(f"Initialized research orchestrator with profile '{profile}': {self.config}")
    
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
                logger.error("MCP manager failed to initialize - no MCP capabilities available")
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
        task_state.notes_history.append(latest_notes)
        self.total_iterations += 1
        
        # Check for stuck state
        if task_state.is_stuck():
            logger.warning(f"Task {task.name} appears stuck, forcing progression")
            return OrchestratorDecision(
                reasoning="Task appears stuck in a loop, forcing progression to next subtask",
                decision="switch",
                feedback="Move to next subtask to break the loop",
                subtask="proceed",
                next_task=task.name
            )
        
        # Check total iteration limit
        if self.total_iterations >= self.max_total_iterations:
            logger.warning("Maximum iterations reached, completing task")
            return OrchestratorDecision(
                reasoning="Maximum iterations reached, task should be completed",
                decision="done",
                feedback="Research completed within iteration limits",
                subtask="proceed",
                next_task="Not Applicable"
            )

        response = None
        retry_count = 0
        max_retries = 3
        
        while response is None and retry_count < max_retries:
            try:
                # Get current subtask info
                current_subtask_info = ""
                if task_state.research_plan and task_state.research_plan.subtasks:
                    if task_state.subtask_index < len(task_state.research_plan.subtasks):
                        current_subtask = task_state.research_plan.subtasks[task_state.subtask_index]
                        current_subtask_info = f"Current subtask: {current_subtask.description}\nSuccess criteria: {current_subtask.success_criteria}\nSubtask {task_state.subtask_index + 1} of {len(task_state.research_plan.subtasks)}"
                    else:
                        current_subtask_info = "All subtasks completed"
                
                # Build context with enhanced information
                context_parts = [
                    f"Current time: {datetime.now()}",
                    f"Task: {task.name} - {task.description}",
                    f"Progress: {task_state.get_progress_percentage():.1f}%",
                    f"Completed subtasks: {len(task_state.completed_subtasks)}/{len(task_state.research_plan.subtasks) if task_state.research_plan else 0}",
                    f"Decision count: {task_state.decision_count}",
                    f"Stuck count: {task_state.stuck_count}",
                    current_subtask_info,
                    f"Recent notes: {latest_notes[:1000]}...",  # Limit context size
                    f"Previous feedback: {task_state.feedback_history[-1] if task_state.feedback_history else 'None'}"
                ]

                # Get system prompt based on configuration
                system_prompt = self._get_system_prompt()
                
                msg = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "\n".join(context_parts)}
                ]
                
                raw = self.llm.chat(msg)
                logger.debug(f"Orchestrator response: {raw}")
                response = OrchestratorDecision.model_validate_json(self._clean_markdown_json(raw))
                
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
                        next_task=task.name
                    )

        return response
    
    def _get_system_prompt(self) -> str:
        """Get system prompt based on configuration"""
        base_prompt = """You are a project manager overseeing a research agent. Your role is to facilitate research on a given topic by directing the agent to explore multiple angles and new avenues of investigation to a medium depth.

You may create new subtasks and branches of exploration as needed, but you must not create wholly new tasks outside the original scope.

Your work is exclusively focused on research and data gathering.
The goal is to find, synthesize, and combine as much relevant and useful information as possible into a clear, organized, and usable format that directly supports the objectives of the current task.

You have access to research agents that can search the web and gather information.
All information gathered should be actionable, clearly tied to the task at hand, and sufficient to enable well-informed decisions."""
        
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
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", md_str.strip(), flags=re.MULTILINE)
    
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
                research_plan = PlanningPlan.model_validate_json(self._clean_markdown_json(raw))
                
                # Validate the plan
                if not research_plan.subtasks:
                    raise ValueError("No subtasks generated")
                
                # Ensure subtasks are properly ordered
                for i, subtask in enumerate(research_plan.subtasks):
                    subtask.order = i + 1
                
                logger.info(f"Generated plan with {len(research_plan.subtasks)} subtasks for task {task.name}")
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Error during planning phase (attempt {retry_count}): {e}")
                if retry_count >= max_retries:
                    # Create a fallback plan
                    research_plan = PlanningPlan(
                        restated_task=f"Research task: {task.description}",
                        subtasks=[
                            Subtask(
                                order=1,
                                description=f"Conduct initial research on {task.name}",
                                success_criteria="Gather comprehensive information about the topic"
                            ),
                            Subtask(
                                order=2,
                                description="Analyze and synthesize findings",
                                success_criteria="Create a detailed analysis of the research findings"
                            ),
                            Subtask(
                                order=3,
                                description="Generate final report",
                                success_criteria="Produce a comprehensive final report with all findings"
                            )
                        ]
                    )
                    logger.warning(f"Using fallback plan for task {task.name}")

        task_state.research_plan = research_plan
        
        # Create initial subtask agent
        if research_plan.subtasks:
            task_state.create_subtask_agent(0, self.llm, self.memory, self.mcp_manager if self.use_mcp else None)
    
    async def run_subtask_research(self, task: Task) -> Dict[str, any]:
        """Run research for the current subtask using a dedicated agent"""
        task_state = self.task_states[task.name]
        
        # Get or create current subtask agent
        current_agent = task_state.get_current_subtask_agent()
        if not current_agent:
            if task_state.subtask_index < len(task_state.research_plan.subtasks):
                current_agent = task_state.create_subtask_agent(task_state.subtask_index, self.llm, self.memory, self.mcp_manager if self.use_mcp else None)
            else:
                return {"status": "no_more_subtasks", "message": "All subtasks completed"}
        
        # Run research step
        result = await current_agent.run_step(task)
        
        # Update task state with findings
        if hasattr(current_agent, 'state') and current_agent.state:
            # Add findings to subtask findings
            if hasattr(current_agent.state, 'findings'):
                task_state.subtask_findings[task_state.subtask_index].extend(current_agent.state.findings)
            
            # Update notes history
            task_state.notes_history.append(result)
                
            # Check if subtask is complete (for subtask agents)
            if hasattr(current_agent.config, 'is_subtask_agent') and current_agent.config.is_subtask_agent:
                # Store final report
                task_state.subtask_reports[task_state.subtask_index] = result
                
                # Mark subtask as complete
                task_state.mark_subtask_complete(task_state.subtask_index)
                
                # Create next subtask agent if available
                if task_state.subtask_index < len(task_state.research_plan.subtasks):
                    task_state.create_subtask_agent(task_state.subtask_index, self.llm, self.memory, self.mcp_manager if self.use_mcp else None)
        
        return {"status": "in_progress", "result": result}
    
    async def run_step(self, task: Task) -> str:
        """Run one step of the orchestrator workflow"""
        return await self.next_action(task)
    
    async def next_action(self, task: Task, notes: str = "") -> str:
        """Enhanced next action using subtask-specific research agents"""
        task_state = self.task_states[task.name]
        
        # Run subtask research
        research_result = await self.run_subtask_research(task)
        
        if research_result["status"] == "no_more_subtasks":
            # All subtasks completed
            logger.info(f"All subtasks completed for {task.name}")
            task.status = TaskStatus.COMPLETE
            
            # Generate final report from all subtask reports
            final_report = await self.synthesize_final_report(task, task_state.all_findings)
            
            # Save final report
            safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task.name).strip()
            report_path = f"notes/{safe_name.upper()}_FINAL_REPORT.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(final_report)

            # Find next active task
            remaining = [t for t in self.tasks.values() if t.status != TaskStatus.COMPLETE]
            self.current = remaining[0].name if remaining else None

            if self.current:
                logger.info(f"Moving to next task: {self.current}")
            else:
                logger.info("All tasks completed")
            
            return "Task completed - all subtasks finished"
        
        return research_result.get("result", "Continuing research")
    
    def get_current_subtask(self, task_name: str) -> Optional[str]:
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
    
    def get_task_progress(self, task_name: str) -> Dict[str, any]:
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
                "has_report": i in task_state.subtask_reports
            }
            
            # Add agent metrics if available
            if i in task_state.subtask_agents:
                agent = task_state.subtask_agents[i]
                agent_metrics = agent.get_metrics()
                subtask_info.update({
                    "agent_metrics": agent_metrics,
                })
            
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
            "subtask_agents_count": len(task_state.subtask_agents)
        }
    
    async def synthesize_final_report(self, task: Task, research_reports: List[str]) -> str:
        """Combines multiple research reports into one cohesive final report"""
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
            msg = [
                {"role": "system", "content": deduplication_prompt},
                {"role": "user", "content": f"Report {i+1} of {len(research_reports)}:\n\n{report}"}
            ]
            deduplicated_findings.append(self.llm.chat(msg))
        
        # Create final synthesis with full context
        final_msg = [
            {"role": "system", "content": final_synthesis_prompt},
            {"role": "user", "content": "Deduplicated findings from all research:\n\n" + 
            "\n\n---\n\n".join(deduplicated_findings)}
        ]
        
        return self.llm.chat(final_msg)
    
    def get_current_subtask(self, task_name: str) -> Optional[str]:
        """Get current subtask description for a task"""
        if task_name not in self.task_states:
            return None
            
        task_state = self.task_states[task_name]
        if not task_state.research_plan or not task_state.research_plan.subtasks:
            return None
            
        if task_state.subtask_index < len(task_state.research_plan.subtasks):
            return task_state.research_plan.subtasks[task_state.subtask_index].description
        else:
            return "All subtasks completed"
    
    async def __aenter__(self):
        """Async context manager entry"""
        if self.use_mcp:
            await self.initialize_mcp()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.use_mcp:
            await self.cleanup_mcp()
