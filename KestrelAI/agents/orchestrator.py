from datetime import datetime, timedelta



try:
    from agents.base import LlmWrapper
    from shared.models import Task
    from shared.models import TaskStatus
except ImportError:
    from KestrelAI.agents.base import LlmWrapper
    from KestrelAI.shared.models import Task
    from KestrelAI.shared.models import TaskStatus

from typing import List, Literal
from pydantic import BaseModel
import re

ORCH_INSTRUCTIONS = """
You are a project manager overseeing a research agent. Your role is to facilitate research on a given topic by directing the agent to explore multiple angles and new avenues of investigation to a medium depth.

You may create new subtasks and branches of exploration as needed, but you must not create wholly new tasks outside the original scope. For example, if tasked with performing a trade study, you might explore real-world studies, academic papers, and practical use cases of relevant components in addition to reviewing data sheets. However, it would be 
unreasonable to begin evaluating unrelated technologies or industries with no bearing on the trade study’s objectives, as that would constitute a new task rather than an extension of the current one.

Your work is exclusively focused on research and data gathering.
The goal is to find, synthesize, and combine as much relevant and useful information as possible into a clear, organized, and usable format that directly supports the objectives of the current task. All information gathered should be actionable, clearly tied to the task at hand, and sufficient to enable well-informed decisions.

You have access only to the notes provided by the research agent.
The research agent has web search capability but cannot send emails, execute code, or access APIs.
}}
"""



EVAL_PHASE_PROMPT = """
Provided with the history of notes for the current task, determine how the current task is progressing.

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
For instance, if the researcher seems stuck or updates seem very similar, suggest a new direction to explore.

If good progress is being made, don't interrupt it. Default to "continue" unless you have strong reasons to change course.

Return ONLY JSON with in the following format, and NOTHING ELSE. RETURN ONLY this JSON:
{{
  "reasoning": "Perform detailed reasoning about the current state of research. List what has been accomplished and what remains to be explored. Be specific about why you're making your decision.",
  "decision": "continue" | "switch" | "done",
  "feedback": "What specific aspects need more investigation? What questions remain unanswered? How should the research focus shift if needed?",
  "subtask": "stay" | "proceed",
  "next_task": "name_of_task or "Not Applicable" if staying on current task"
}}

REMINDER: Err on the side of thoroughness. It's better to continue exploring than to prematurely conclude a research task or subtask.
"""

PLANNING_PHASE_PROMPT = """
Planning Phase – Task Analysis and Decomposition

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
- Be explicit in 'success_criteria' so the research agent knows exactly when the subtask is done.
"""

DEDUPLICATION_PROMPT = """Extract ALL unique information from this research report.
Your goal: Preserve ABSOLUTELY EVERYTHING valuable while removing only exact duplicates.

Include EVERY SINGLE:
- Finding, conclusion, insight, observation, and hypothesis
- Data point, number, percentage, date, deadline, and statistic  
- Link, URL, reference, citation, and source
- Name of programs, organizations, companies, people, and locations
- Eligibility requirement, application detail, and program specification
- Recommendation, next step, and future consideration
- Contact information, email addresses, and phone numbers
- Technical specification, requirement, and constraint
- Quote, snippet, and relevant excerpt
- Unique detail, no matter how minor it seems

Remove ONLY:
- Information that appears VERBATIM in other reports (exact duplicates only)
- Meta-commentary about the research process itself ("I searched for...", "Next I will...")

CRITICAL: If you're unsure whether to include something, INCLUDE IT. 
No summarization. No paraphrasing. No condensing. Preserve exact wording and full detail."""

FINAL_SYNTHESIS_PROMPT = """Create an EXHAUSTIVELY COMPREHENSIVE final report from these research findings.

Context:
Task: {task_name}
Description: {task_description}

CRITICAL REQUIREMENTS:
- This report must contain EVERY SINGLE PIECE of information discovered during research
- NO detail is too small to include - every date, link, number, name, requirement must appear
- Organize for readability but NEVER at the expense of completeness
- Include FULL details: complete eligibility requirements, all deadlines, every link, all contact info
- If multiple sources provided different information, include ALL variations
- Preserve exact names, titles, and specifications as discovered

Structure the report to be:
- Exhaustively complete (every detail from the research MUST appear in the final report)
- Well-organized with clear sections and subsections
- Professional with proper formatting
- Scannable despite its comprehensive nature (use bold, headers, lists)
- Actionable with all information needed to act on any finding

Remember: This is an archival document. Someone should be able to reconstruct the ENTIRE research findings from this report alone. If a detail was discovered, it MUST be in this final report."""

# ORCH_INSTRUCTIONS
class OrchestratorDecision(BaseModel):
    reasoning: str
    decision: Literal["continue", "switch", "done"]
    feedback: str
    subtask: Literal["stay", "proceed"]
    next_task: str

# PLANNING_PHASE_PROMPT
class Subtask(BaseModel):
    order: int
    description: str
    success_criteria: str

class PlanningPlan(BaseModel):
    restated_task: str
    subtasks: List[Subtask]


def concat_with_newlines(*args: str) -> str:
    return "\n".join(args)

def clean_markdown_json(md_str: str) -> str:
    """
    Strips ```json ... ``` or ``` ... ``` from a Markdown block.
    """
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", md_str.strip(), flags=re.MULTILINE)

class Orchestrator:
    def __init__(self, tasks: list[Task], llm: LlmWrapper, slice_minutes: int = 15):
        self.tasks = {t.name: t for t in tasks}
        self.llm = llm
        self.slice = timedelta(minutes=slice_minutes)
        self.current = tasks[0].name
        self.notes_history = []
        self.research_plan = None
        self.current_subtask_index = 0

    def _review(self, task: Task, latest_notes: str) -> OrchestratorDecision:
        self.notes_history.append(latest_notes)

        response = None
        while response is None:
            try:
                current_subtask = f"Subtask Description {self.research_plan.subtasks[self.current_subtask_index].description}, Task Description {self.research_plan.subtasks[self.current_subtask_index].success_criteria}"

                msg = [
                    {"role": "system", "content": concat_with_newlines(ORCH_INSTRUCTIONS, EVAL_PHASE_PROMPT)},
                    {
                        "role": "user",
                        "content": f"Current time: {datetime.now()}\n"
                        f"Task statuses: {[ (t.name, t.status) for t in self.tasks.values() ]}\n"
                        f"Update history for {task.name}, {task.description}:\n{' '.join(self.notes_history[-2:])}\n"
                        f"Current subtask: {current_subtask}, Subtask number {self.current_subtask_index + 1} of {len(self.research_plan.subtasks)}\n"
                        f"All subtasks: {self.research_plan.subtasks}",
                    },
                ]
                raw = self.llm.chat(msg)
                print(raw)
                response = OrchestratorDecision.model_validate_json(clean_markdown_json(raw))
            except Exception as e:
                print(f"Error during review phase: {e}")
                print(f"Context Size: {len(raw)} chars")
                continue

        return response

    def _planning_phase(self, task: Task) -> None:
        research_plan = None
        while research_plan is None:
            try: 
                msg = [
                    {"role": "system", "content": PLANNING_PHASE_PROMPT},
                    {
                        "role": "user",
                        "content": f"Task Name: {task.name}\nTask description: {task.description}",
                    },
                ]
                raw = self.llm.chat(msg)
                research_plan = PlanningPlan.model_validate_json(clean_markdown_json(raw))
                print(raw)
            except Exception as e:
                print(f"Error during planning phase: {e}")
                continue

        self.research_plan = research_plan

    def next_action(self, task: Task, notes: str):
        review = self._review(task, notes)
        decision = review.decision

        print(f"The decision is {review}")

        if decision == "switch":
            self.current = review.next_task
        elif decision == "done":

            final_report = self.synthesize_final_report(task, self.notes_history)
            with open(f"{task.name}_final_report.txt", "w") as f:
                f.write(final_report)

            task.status = TaskStatus.COMPLETE
            remaining = [t for t in self.tasks.values() if t.status != TaskStatus.COMPLETE]
            self.current = remaining[0].name if remaining else None

        if review.subtask == "proceed":
            self.current_subtask_index += 1

        return review.feedback
    
    def synthesize_final_report(self, task: Task, research_reports: List[str]) -> str:
        """
        Combines multiple research reports into one cohesive final report.
        Preserves important details like links and data while removing redundancy.
        
        Args:
            task: Current Task object with name and description
            research_reports: List of research report strings to synthesize

        Returns:
            A single string containing the synthesized final report.
        """
    
        # Deduplicate each report while preserving detail
        deduplicated_findings = []
        for i, report in enumerate(research_reports):
            msg = [
                {"role": "system", "content": DEDUPLICATION_PROMPT},
                {"role": "user", "content": f"Report {i+1} of {len(research_reports)}:\n\n{report}"}
            ]
            deduplicated_findings.append(self.llm.chat(msg))
        
        # Create final synthesis with full context
        final_msg = [
            {"role": "system", "content": FINAL_SYNTHESIS_PROMPT.format(
                task_name=task.name,
                task_description=task.description
            )},
            {"role": "user", "content": "Deduplicated findings from all research:\n\n" + 
            "\n\n---\n\n".join(deduplicated_findings)}
        ]
        
        return self.llm.chat(final_msg)


if __name__ == "__main__":
    from KestrelAI.shared.models import Task

    ML_TASK = """Find currently open grants, programs, fellowships, or funding opportunities that
support AI/ML research and are available to senior undergraduate students in the
United States. Include name, eligibility, what it funds, deadline, and link. Focus
on fresh, prestigious, student-accessible opportunities not locked to one college.
Include opportunities from companies like Google, Microsoft, Meta, OpenAI, Anthropic, Cohere.
The current date is August 2025, find programs open for application."""

    tasks = [
        Task(
            name="Research on AI Fellowships",
            description=ML_TASK,
        ),
        Task(name="Task B", description="Description B"),
        Task(name="Task C", description="Description C"),
    ]
    llm = LlmWrapper(temperature=0.7)
    orch = Orchestrator(tasks, llm)
    report = """FINAL REPORT: AI/ML Funding Opportunities for Senior Undergraduates (US) - October 26, 2023
Focus: Identifying and cataloging AI/ML research funding opportunities for senior undergraduate students in the US. Tracking spreadsheet is being populated with data extracted from REU Finder and NSF website.

What Has Been Discovered So Far:

Several organizations are actively funding AI/ML research in the US, including the National Science Foundation (NSF), various university AI labs (Wharton, Eller, etc.), and initiatives like the National AI Research Institutes. The NSF appears to be the largest single source of funding, with REU programs representing a significant portion of this funding. Key programs include:

REU Sites: Various REU sites across the US focus on AI/ML topics (e.g., Computational Creativity, Edge Computing). Eligibility generally requires US citizenship or permanent residency.
AI Institutes: NSF-funded AI Institutes (e.g., AI-EDGE, AI Planning Institute, National AI Research Institutes) offer unique summer research programs, often with specific research focuses. Eligibility varies per institute.
Artificial Intelligence Scholarship for Service Initiative: Focuses on AI applications with societal benefit; offers scholarship and research opportunities.
Beyond NSF, opportunities exist through university AI labs and industry partnerships. REU Finder is a useful starting point, but not exhaustive, requiring cross-referencing with program websites and proactive searches. Eligibility requirements (citizenship, GPA, institutional affiliation) vary significantly between programs.

Key Sources and Evidence:

NSF Website: https://www.nsf.gov/ – Primary source for official program descriptions and deadlines.
REU Finder: https://www.nsf.gov/oia/reu/ – Useful starting point for identifying REU sites, but requires verification.
AI Institute Websites: Direct links to individual AI Institute websites (e.g., AI-EDGE) are essential.
University Research Offices: Local university research offices can provide information on faculty-led research and potential funding.
Current Research Focus (Prioritized):

Populating tracking spreadsheet: Systematically extracting key details (deadlines, eligibility, research areas, application requirements) for each identified program into a tracking spreadsheet.
AI Institute Programs: Prioritize investigating summer research programs offered by NSF AI Institutes due to their unique research opportunities.
Faculty-Led Research: Identify faculty at target institutions conducting AI/ML research and explore potential research assistant opportunities (indirect funding – funding awarded to faculty that may support undergraduate research assistants).
Unexplored Areas/Questions:

Industry Sponsorship: Investigate potential funding opportunities from AI/ML companies (e.g., Google, Microsoft, Amazon) that support undergraduate research.
International Opportunities: Explore AI/ML research opportunities abroad, considering potential visa requirements and program eligibility.
Indirect Funding Sources: Identify alternative funding sources beyond NSF and industry sponsorship (e.g., departmental grants, philanthropic foundations).
Reddit (r/REU): Provides anecdotal information and discussions about REU experiences and eligibility (use with caution and verify information).
Immediate (Next 1-2 Weeks):

Proactively monitor the NSF website and AI Institute websites for new program announcements and deadline updates.
Begin populating tracking spreadsheet with data from REU Finder and NSF website.
Identify 3-5 faculty at target institutions conducting AI/ML research and explore potential research assistant opportunities.
Mid-Term (Next 1-2 Months):

Contact identified faculty to inquire about research opportunities and funding possibilities.
Research industry sponsorship opportunities and submit inquiries.
Refine tracking spreadsheet based on gathered information.
Long-Term (Ongoing):

Maintain ongoing monitoring of funding sources.
Network with researchers and students in the AI/ML field.
Explore international research opportunities.
        """
    orch._planning_phase(tasks[0])
    orch._review(tasks[0], report)