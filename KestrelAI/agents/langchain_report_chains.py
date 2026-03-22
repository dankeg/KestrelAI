"""
Reusable LangChain chains for report-oriented generation flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:  # pragma: no cover - dependency-gated path
    ChatPromptTemplate = None
    StrOutputParser = None


def _require_langchain():
    if ChatPromptTemplate is None or StrOutputParser is None:
        raise ImportError("langchain_core is required for report chains")


@dataclass
class WebResearchLangChainChains:
    """
    LangChain chains for summarize/checkpoint/final report generation.
    """

    model: Any

    def __post_init__(self) -> None:
        _require_langchain()
        parser = StrOutputParser()

        summarize_system_prompt = """Create concise intermediate research notes from the provided material.

HARD REQUIREMENTS:
- Output only compact evidence notes for the current step.
- Do NOT write a report title, executive summary, introduction, conclusion, or recommendation section.
- Do NOT use headings like "Actionable Shortlist", "Final Report", or similar report framing.
- Do NOT restate the task.
- Prefer 3-6 short bullet points.
- Each bullet should contain one concrete fact, entity, date, requirement, or source cue.
- If evidence is weak, label it as tentative.
- No commentary or questions.

When referencing URLs, use the URL flags (e.g., [URL_1], [URL_2]) provided in the URL reference table.
Do NOT write out full URLs - use the flags instead."""

        summarize_user_prompt = """Task: {task_description}

Material:
{material_with_flags}

{url_reference_table}"""

        checkpoint_system_prompt = """Create a focused checkpoint summarizing actionable research findings.

HARD REQUIREMENTS:
- Output only a checkpoint note, not a polished report.
- Do NOT include a title, preamble, executive summary, conclusion, or recommendation section.
- Do NOT produce a shortlist-style report header.
- Keep the output compact and evidence-oriented.
- Prefer short bullets grouped around concrete findings.

Focus on:
- Specific entities, findings, or sources discovered
- Concrete details such as dates, requirements, constraints, quantities, costs, contacts, links, or status
- Exact facts that a later synthesis step can rely on
- Direct source cues and URLs where available

When referencing URLs, use the URL flags (e.g., [URL_1], [URL_2]) provided in the URL reference table.
Do NOT write out full URLs - use the flags instead.

Avoid:
- Generic advice or recommendations
- Vague descriptions of databases or search engines
- Meta-commentary about the research process
- Placeholder text or template content

Be concise but include all actionable information the user can immediately use.

For each important claim:
- Prefer authoritative sources first (.gov, .edu, official program/organization pages)
- If a detail is only weakly supported, label it as tentative
- Preserve source cues, named entities, and dates"""

        checkpoint_user_prompt = """Task: {task_description}

Recent research:
{recent_context_with_flags}

Previous checkpoint:
{previous_checkpoint_with_flags}

{url_reference_table}"""

        final_system_prompt = """Create a focused, actionable research report from these findings.

CRITICAL REQUIREMENTS:
- Focus on SPECIFIC, ACTIONABLE information that the user can immediately use
- Include concrete details: exact dates, requirements, constraints, quantities, costs, contact information, access steps, or links when available
- Prioritize CURRENT and source-backed findings over generic descriptions
- Remove generic advice and focus on specific entities, claims, findings, or opportunities
- Preserve exact facts and source-backed qualifiers

URL REFERENCING (CRITICAL):
- When referencing URLs, use the URL flags (e.g., [URL_1], [URL_2]) provided in the URL reference table
- Do NOT write out full URLs - use the flags instead
- Format as markdown links: [Link Text]([URL_1]) or just [URL_1] for bare references
- The URL reference table shows which flag corresponds to which URL
- This prevents URL corruption and ensures accuracy

CRITICAL: If previous research reports are provided:
- BUILD UPON the information in previous reports by adding NEW findings, details, or opportunities
- PRESERVE all specific details from previous reports (deadlines, contact info, requirements, links)
- EXPAND on previous findings with additional context, related opportunities, or deeper details
- DO NOT comment on, evaluate, or praise previous reports
- DO NOT provide feedback or suggestions about the format or quality of previous reports
- DO NOT repeat information verbatim unless adding new context
- SYNTHESIZE previous findings with new findings into a cohesive, comprehensive report
- Focus on ADDING VALUE, not evaluating previous work

Avoid:
- Generic database descriptions
- Vague recommendations
- Placeholder text
- Overly comprehensive archival content
- Generic advice that applies to any research topic
- Meta-commentary about previous reports
- Writing out full URLs (use flags instead)

Focus on: Specific findings, concrete facts, direct source grounding, and synthesis.

HARD QUALITY RULES:
- Distinguish verified findings from tentative findings when evidence quality differs
- Prefer primary/authoritative sources over secondary summaries
- Do not present weakly supported claims as confirmed facts
- If evidence is mixed, say what is confirmed and what still needs verification
- Use the claim verification brief as the primary authority on whether a claim is verified, tentative, or unsupported
- When in doubt, downgrade the claim rather than upgrading it
- Start directly with concrete findings, not an evaluation of the report quality
- Never write sections like "Strengths", "Overall Assessment", "Suggestions for Improvement", or "Minor Suggestions"
- Never praise, critique, or grade the report or the research process"""

        final_user_prompt = """Task: {task_description}

{previous_reports_with_flags}

Current research checkpoints:
{checkpoints_with_flags}

Additional findings:
{rag_with_flags}

{url_reference_table}"""

        self._summarize_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", summarize_system_prompt),
                    ("human", summarize_user_prompt),
                ]
            )
            | self.model
            | parser
        )

        self._checkpoint_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", checkpoint_system_prompt),
                    ("human", checkpoint_user_prompt),
                ]
            )
            | self.model
            | parser
        )

        self._final_report_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", final_system_prompt),
                    ("human", final_user_prompt),
                ]
            )
            | self.model
            | parser
        )

    def summarize_notes(
        self,
        *,
        task_description: str,
        material_with_flags: str,
        url_reference_table: str = "",
    ) -> str:
        return self._summarize_chain.invoke(
            {
                "task_description": task_description,
                "material_with_flags": material_with_flags,
                "url_reference_table": url_reference_table or "",
            }
        )

    async def asummarize_notes(
        self,
        *,
        task_description: str,
        material_with_flags: str,
        url_reference_table: str = "",
    ) -> str:
        return await self._summarize_chain.ainvoke(
            {
                "task_description": task_description,
                "material_with_flags": material_with_flags,
                "url_reference_table": url_reference_table or "",
            }
        )

    def create_checkpoint(
        self,
        *,
        task_description: str,
        recent_context_with_flags: str,
        previous_checkpoint_with_flags: str,
        url_reference_table: str = "",
    ) -> str:
        return self._checkpoint_chain.invoke(
            {
                "task_description": task_description,
                "recent_context_with_flags": recent_context_with_flags,
                "previous_checkpoint_with_flags": previous_checkpoint_with_flags
                or "None",
                "url_reference_table": url_reference_table or "",
            }
        )

    async def acreate_checkpoint(
        self,
        *,
        task_description: str,
        recent_context_with_flags: str,
        previous_checkpoint_with_flags: str,
        url_reference_table: str = "",
    ) -> str:
        return await self._checkpoint_chain.ainvoke(
            {
                "task_description": task_description,
                "recent_context_with_flags": recent_context_with_flags,
                "previous_checkpoint_with_flags": previous_checkpoint_with_flags
                or "None",
                "url_reference_table": url_reference_table or "",
            }
        )

    def generate_final_report(
        self,
        *,
        task_description: str,
        previous_reports_with_flags: str,
        checkpoints_with_flags: str,
        rag_with_flags: str,
        url_reference_table: str = "",
    ) -> str:
        return self._final_report_chain.invoke(
            {
                "task_description": task_description,
                "previous_reports_with_flags": previous_reports_with_flags or "",
                "checkpoints_with_flags": checkpoints_with_flags,
                "rag_with_flags": rag_with_flags,
                "url_reference_table": url_reference_table or "",
            }
        )

    async def agenerate_final_report(
        self,
        *,
        task_description: str,
        previous_reports_with_flags: str,
        checkpoints_with_flags: str,
        rag_with_flags: str,
        url_reference_table: str = "",
    ) -> str:
        return await self._final_report_chain.ainvoke(
            {
                "task_description": task_description,
                "previous_reports_with_flags": previous_reports_with_flags or "",
                "checkpoints_with_flags": checkpoints_with_flags,
                "rag_with_flags": rag_with_flags,
                "url_reference_table": url_reference_table or "",
            }
        )


@dataclass
class OrchestratorLangChainChains:
    """LangChain chains for orchestrator report deduplication/synthesis."""

    model: Any

    def __post_init__(self) -> None:
        _require_langchain()
        parser = StrOutputParser()

        dedupe_system_prompt = """Extract the most valuable and actionable information from this research report.

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

        dedupe_user_prompt = """Report {index} of {total}:

{report}"""

        synthesis_system_prompt = """Create a comprehensive final research report from these research findings.

Context:
Task: {task_name}
Description: {task_description}

CRITICAL REQUIREMENTS:
- This is a final report, not a status brief.
- Explain the main answer in a readable narrative, not just a stack of status bullets.
- Include concrete details: dates, requirements, constraints, quantities, costs, contacts, application/access steps, or links when available.
- Prioritize current, actionable findings over generic descriptions.
- Group related findings and compare them when useful to the reader.
- Preserve exact qualifiers where evidence is mixed or incomplete.

Structure the report to be:
- well-organized with clear sections
- comprehensive enough to stand on its own
- factual and source-aware
- practical, but still recognizably a report rather than a checkpoint memo

Avoid:
- Generic database descriptions
- Vague recommendations
- Placeholder text
- Overly comprehensive archival content
- Generic advice that applies to any research topic

Focus on: specific findings, concrete facts, comparisons, and direct source-grounded conclusions.

HARD QUALITY RULES:
- Separate verified findings from tentative or partially verified findings
- Prefer authoritative sources (.gov, .edu, official organizations, primary program pages)
- Do not convert weak evidence into confident conclusions
- If an item lacks strong evidence, keep it brief and label the uncertainty
- End with practical next steps based only on the strongest evidence
- Use the claim verification brief as the source of truth for what belongs in verified vs tentative sections
- Use the structured evidence packet as the primary factual basis for source names, links, and concrete details
- Start with the literal heading `## Executive Summary`
- Never write sections like "Strengths", "Overall Assessment", "Suggestions for Improvement", or "Minor Suggestions"
- Never evaluate the quality of the report or the research process

URL REFERENCING (CRITICAL):
- When referencing URLs, use only the URL flags (for example [URL_1], [URL_2]) from the URL reference table
- Do NOT write out full URLs
- Do NOT invent, normalize, or guess URLs that are not present in the evidence packet or URL table
- If the exact URL is not provided, name the source and domain without adding a link
- Prefer markdown links like [Source Name]([URL_1]) when the source/title is known

REQUIRED OUTPUT STRUCTURE:
## Executive Summary
- 1-3 paragraphs summarizing the strongest findings and practical takeaway

## Scope and Method
- Briefly explain what was investigated, what evidence types were used, and major evidence constraints

## Findings
- Present the main findings in a coherent, topic-centered structure
- Use subsections if that improves clarity

## Comparative Analysis
- Compare the most relevant entities, opportunities, approaches, or findings
- If comparison is weakly supported, say so briefly instead of inventing one

## Limitations and Open Questions
- State important unresolved evidence gaps, ambiguities, or conflicts

## Recommended Next Steps
- Provide practical, evidence-grounded next actions for the user

## Evidence Status Appendix
### Verified Findings
- Only claims supported by the verification brief and authoritative evidence

### Tentative Findings
- Claims with partial support or unresolved ambiguity

### Open Uncertainties
- Missing dates, eligibility gaps, or conflicts still needing verification"""

        synthesis_user_prompt = """Structured evidence packet (primary factual basis):
{structured_evidence_packet_with_flags}

Evidence quality brief:
{evidence_brief_with_flags}

Claim verification brief:
{verification_brief_with_flags}

Clean deduplicated findings from all research (secondary context):

{combined_findings_with_flags}

{url_reference_table}"""

        repair_system_prompt = """Rewrite an INVALID draft into a valid comprehensive final research report.

The draft is invalid because it contains meta-critique/review language (for example: "this report is strong", "overall assessment", "would you like me to...").

HARD REQUIREMENTS:
- Output only the final report content in markdown.
- Do NOT evaluate, praise, critique, or grade any report.
- Do NOT include follow-up questions, offers, or conversational fillers.
- Do NOT include sections like "Strengths", "Overall Assessment", or "Suggestions for Improvement" unless the task explicitly asks for those as research findings.
- Use the evidence provided to produce factual, task-aligned findings.
- Prefer concrete entities, dates, requirements, links, and actionable next steps.
- Respect the claim verification brief. Unsupported claims should not appear as verified findings.
- Use the structured evidence packet as the primary factual basis for source names, links, and concrete details.
- Use only URL flags from the URL reference table; never write full URLs or invent links.
- The result must read like a complete report, not a short status brief.
- Include an evidence-status appendix.

If information is incomplete, state uncertainty briefly and continue with available facts.
Start directly with `## Executive Summary` (no preamble)."""

        repair_user_prompt = """Task: {task_name}
Description: {task_description}

Invalid draft to repair:
{invalid_draft}

Evidence quality brief:
{evidence_brief_with_flags}

Claim verification brief:
{verification_brief_with_flags}

Structured evidence packet:
{structured_evidence_packet_with_flags}

Evidence to ground the corrected report:
{combined_findings_with_flags}

{url_reference_table}"""

        self._dedupe_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", dedupe_system_prompt),
                    ("human", dedupe_user_prompt),
                ]
            )
            | self.model
            | parser
        )

        self._synthesis_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", synthesis_system_prompt),
                    ("human", synthesis_user_prompt),
                ]
            )
            | self.model
            | parser
        )

        self._repair_synthesis_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", repair_system_prompt),
                    ("human", repair_user_prompt),
                ]
            )
            | self.model
            | parser
        )

    def dedupe_report(self, *, report: str, index: int, total: int) -> str:
        return self._dedupe_chain.invoke(
            {"report": report, "index": index, "total": total}
        )

    async def adedupe_report(self, *, report: str, index: int, total: int) -> str:
        return await self._dedupe_chain.ainvoke(
            {"report": report, "index": index, "total": total}
        )

    def synthesize(
        self,
        *,
        task_name: str,
        task_description: str,
        combined_findings_with_flags: str,
        evidence_brief_with_flags: str = "",
        verification_brief_with_flags: str = "",
        structured_evidence_packet_with_flags: str = "",
        url_reference_table: str = "",
    ) -> str:
        return self._synthesis_chain.invoke(
            {
                "task_name": task_name,
                "task_description": task_description,
                "combined_findings_with_flags": combined_findings_with_flags,
                "evidence_brief_with_flags": evidence_brief_with_flags
                or "No evidence-quality brief available.",
                "verification_brief_with_flags": verification_brief_with_flags
                or "No claim verification brief available.",
                "structured_evidence_packet_with_flags": structured_evidence_packet_with_flags
                or "No structured evidence packet available.",
                "url_reference_table": url_reference_table or "",
            }
        )

    async def asynthesize(
        self,
        *,
        task_name: str,
        task_description: str,
        combined_findings_with_flags: str,
        evidence_brief_with_flags: str = "",
        verification_brief_with_flags: str = "",
        structured_evidence_packet_with_flags: str = "",
        url_reference_table: str = "",
    ) -> str:
        return await self._synthesis_chain.ainvoke(
            {
                "task_name": task_name,
                "task_description": task_description,
                "combined_findings_with_flags": combined_findings_with_flags,
                "evidence_brief_with_flags": evidence_brief_with_flags
                or "No evidence-quality brief available.",
                "verification_brief_with_flags": verification_brief_with_flags
                or "No claim verification brief available.",
                "structured_evidence_packet_with_flags": structured_evidence_packet_with_flags
                or "No structured evidence packet available.",
                "url_reference_table": url_reference_table or "",
            }
        )

    def repair_synthesis(
        self,
        *,
        task_name: str,
        task_description: str,
        invalid_draft: str,
        combined_findings_with_flags: str,
        evidence_brief_with_flags: str = "",
        verification_brief_with_flags: str = "",
        structured_evidence_packet_with_flags: str = "",
        url_reference_table: str = "",
    ) -> str:
        return self._repair_synthesis_chain.invoke(
            {
                "task_name": task_name,
                "task_description": task_description,
                "invalid_draft": invalid_draft,
                "combined_findings_with_flags": combined_findings_with_flags,
                "evidence_brief_with_flags": evidence_brief_with_flags
                or "No evidence-quality brief available.",
                "verification_brief_with_flags": verification_brief_with_flags
                or "No claim verification brief available.",
                "structured_evidence_packet_with_flags": structured_evidence_packet_with_flags
                or "No structured evidence packet available.",
                "url_reference_table": url_reference_table or "",
            }
        )

    async def arepair_synthesis(
        self,
        *,
        task_name: str,
        task_description: str,
        invalid_draft: str,
        combined_findings_with_flags: str,
        evidence_brief_with_flags: str = "",
        verification_brief_with_flags: str = "",
        structured_evidence_packet_with_flags: str = "",
        url_reference_table: str = "",
    ) -> str:
        return await self._repair_synthesis_chain.ainvoke(
            {
                "task_name": task_name,
                "task_description": task_description,
                "invalid_draft": invalid_draft,
                "combined_findings_with_flags": combined_findings_with_flags,
                "evidence_brief_with_flags": evidence_brief_with_flags
                or "No evidence-quality brief available.",
                "verification_brief_with_flags": verification_brief_with_flags
                or "No claim verification brief available.",
                "structured_evidence_packet_with_flags": structured_evidence_packet_with_flags
                or "No structured evidence packet available.",
                "url_reference_table": url_reference_table or "",
            }
        )
