"""
Multi-Level Summarization for KestrelAI
Provides hierarchical summarization at multiple granularity levels with
quality-focused fact preservation for research information.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from KestrelAI.shared.research_utils import timeouts_disabled

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:  # pragma: no cover - dependency-gated path
    ChatPromptTemplate = None
    StrOutputParser = None

try:
    from .context_manager import TokenCounter
    from .langchain_adapter import LangChainChatAdapter
    from .structured_parsing import parse_to_schema
except ImportError:
    from KestrelAI.agents.context_manager import TokenCounter
    from KestrelAI.agents.langchain_adapter import LangChainChatAdapter
    from KestrelAI.agents.structured_parsing import parse_to_schema

logger = logging.getLogger(__name__)


@dataclass
class SummaryLevel:
    """Configuration for a summary level."""

    name: str
    compression_ratio: float  # Target compression ratio (0.0 to 1.0)
    description: str  # Description of what this level preserves


@dataclass
class ExtractedFacts:
    """Extracted key facts from research content."""

    deadlines: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    contact_info: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    amounts: list[str] = field(default_factory=list)
    eligibility: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Convert facts to formatted text for inclusion in summaries."""
        parts = []
        if self.deadlines:
            parts.append(f"Deadlines: {', '.join(self.deadlines)}")
        if self.dates:
            parts.append(f"Important Dates: {', '.join(self.dates)}")
        if self.requirements:
            parts.append(f"Requirements: {', '.join(self.requirements)}")
        if self.contact_info:
            parts.append(f"Contact: {', '.join(self.contact_info)}")
        if self.urls:
            parts.append(f"Links: {', '.join(self.urls)}")
        if self.programs:
            parts.append(f"Programs: {', '.join(self.programs)}")
        if self.amounts:
            parts.append(f"Funding/Amounts: {', '.join(self.amounts)}")
        if self.eligibility:
            parts.append(f"Eligibility: {', '.join(self.eligibility)}")

        return "\n".join(parts) if parts else ""

    def merge(self, other: ExtractedFacts) -> ExtractedFacts:
        """Merge facts from another ExtractedFacts instance."""
        merged = ExtractedFacts()
        merged.deadlines = list(set(self.deadlines + other.deadlines))
        merged.dates = list(set(self.dates + other.dates))
        merged.requirements = list(set(self.requirements + other.requirements))
        merged.contact_info = list(set(self.contact_info + other.contact_info))
        merged.urls = list(set(self.urls + other.urls))
        merged.programs = list(set(self.programs + other.programs))
        merged.amounts = list(set(self.amounts + other.amounts))
        merged.eligibility = list(set(self.eligibility + other.eligibility))
        return merged


class _ExtractedFactsSchema(BaseModel):
    """Structured output schema for fact extraction."""

    deadlines: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    contact_info: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    programs: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)
    eligibility: list[str] = Field(default_factory=list)


class MultiLevelSummarizer:
    """
    Creates summaries at multiple levels of granularity with quality-focused
    fact preservation for research information.

    Key features:
    - Extracts and preserves critical facts (deadlines, requirements, URLs, etc.)
    - Research-specific summarization prompts
    - Fact validation to ensure important information isn't lost
    - Supports URL flag system for proper URL handling
    """

    DEFAULT_LEVELS = [
        SummaryLevel(
            name="detailed",
            compression_ratio=1.0,
            description="Full content, no compression",
        ),
        SummaryLevel(
            name="medium",
            compression_ratio=0.5,
            description="50% compression, key facts preserved",
        ),
        SummaryLevel(
            name="summary",
            compression_ratio=0.2,
            description="20% compression, essential information only",
        ),
        SummaryLevel(
            name="executive",
            compression_ratio=0.1,
            description="10% compression, high-level overview",
        ),
    ]

    def __init__(
        self,
        llm,  # LLM wrapper for summarization
        token_counter: TokenCounter,
        summary_levels: list[SummaryLevel] | None = None,
        extract_facts: bool = True,  # Whether to extract and preserve facts
    ):
        """
        Initialize multi-level summarizer.

        Args:
            llm: LLM wrapper with chat() method for summarization.
            token_counter: TokenCounter instance for token counting.
            summary_levels: Optional list of SummaryLevel configs.
                          Defaults to DEFAULT_LEVELS if not provided.
            extract_facts: If True, extract and preserve key facts before summarization.
        """
        # Validate that llm is an object with a chat method, not a string
        if isinstance(llm, str):
            raise TypeError(
                f"Expected LLM wrapper object with chat() method, but got string: {llm}. "
                f"Did you accidentally pass model_name instead of llm?"
            )
        if not hasattr(llm, "chat") or not callable(getattr(llm, "chat", None)):
            raise TypeError(
                f"Expected LLM wrapper object with chat() method, but got {type(llm).__name__}. "
                f"The llm parameter must be an object with a chat() method."
            )

        self.llm = llm
        self.counter = token_counter
        self.levels = summary_levels or self.DEFAULT_LEVELS
        self.extract_facts = extract_facts
        self.langchain_adapter: LangChainChatAdapter | None = None
        self._fact_extraction_chain = None
        self._summary_chain = None

        # Sort levels by compression ratio (most detailed first)
        self.levels.sort(key=lambda x: x.compression_ratio, reverse=True)

        if (
            ChatPromptTemplate is not None
            and StrOutputParser is not None
            and self._supports_langchain_adapter(llm)
        ):
            try:
                self.langchain_adapter = LangChainChatAdapter.from_llm(llm)
                parser = StrOutputParser()
                self._fact_extraction_chain = (
                    ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You are a fact extraction system. Extract key facts accurately.\n"
                                "Return facts as structured JSON-compatible content.\n"
                                "Every field must be a list of strings, even if there is only one item.\n"
                                "Never return bare strings for list fields.",
                            ),
                            ("human", "{content}"),
                        ]
                    )
                    | self.langchain_adapter.client
                    | parser
                )
                self._summary_chain = (
                    ChatPromptTemplate.from_messages(
                        [
                            ("system", "{system_prompt}"),
                            ("human", "{user_prompt}"),
                        ]
                    )
                    | self.langchain_adapter.client
                    | parser
                )
            except Exception as e:
                logger.debug(
                    "LangChain summarizer chains unavailable: %s",
                    e,
                )
                self.langchain_adapter = None
                self._fact_extraction_chain = None
                self._summary_chain = None

    @staticmethod
    def _supports_langchain_adapter(llm: Any) -> bool:
        """Only enable LangChain chains for actual configured LLM wrappers.

        Bare mocks and minimal test doubles often expose a ``chat`` method but not a
        real provider/model/host configuration. Treat those as non-LangChain inputs
        so summarization stays fully local and deterministic in tests/fallback paths.
        """
        raw_model = getattr(llm, "model", None)
        raw_provider = getattr(llm, "provider", None)
        raw_host = getattr(llm, "host", None)
        if not isinstance(raw_host, str):
            client = getattr(llm, "client", None)
            client_host = getattr(client, "host", None) if client is not None else None
            raw_host = client_host if isinstance(client_host, str) else None
        return isinstance(raw_model, str) and (
            isinstance(raw_provider, str) or isinstance(raw_host, str)
        )

    def create_summary_hierarchy(
        self, content: str, preserve_facts: bool = True
    ) -> dict[str, Any]:
        """
        Create summaries at multiple levels for given content.

        Args:
            content: Content to summarize.
            preserve_facts: If True, extract and preserve key facts in all summaries.

        Returns:
            Dictionary with:
            - "summaries": Dict mapping level names to summaries
            - "facts": ExtractedFacts object with preserved facts
        """
        if not content:
            return {"summaries": {}, "facts": ExtractedFacts()}

        # Extract key facts before summarization
        extracted_facts = ExtractedFacts()
        if self.extract_facts and preserve_facts:
            extracted_facts = self._extract_facts(content)
            logger.debug(
                f"Extracted {len(extracted_facts.deadlines)} deadlines, "
                f"{len(extracted_facts.urls)} URLs, "
                f"{len(extracted_facts.programs)} programs"
            )

        summaries = {}
        current_content = content

        # Start with detailed (full content)
        summaries["detailed"] = content

        # Create progressively more compressed summaries
        for level in self.levels[1:]:  # Skip "detailed" level
            if level.compression_ratio <= 0:
                continue

            target_tokens = int(
                self.counter.count_tokens(current_content) * level.compression_ratio
            )

            if target_tokens < 10:  # Minimum viable summary
                # Use previous level if target is too small
                summaries[level.name] = current_content
                continue

            # Summarize with fact preservation
            summary = self._summarize_with_facts(
                current_content, target_tokens, level, extracted_facts
            )
            summaries[level.name] = summary
            current_content = summary  # Use summary as base for next level

        return {"summaries": summaries, "facts": extracted_facts}

    async def create_summary_hierarchy_async(
        self, content: str, preserve_facts: bool = True
    ) -> dict[str, Any]:
        """
        Async variant of create_summary_hierarchy.
        Uses async chain invocations so upstream cancellation can propagate.
        """
        if not content:
            return {"summaries": {}, "facts": ExtractedFacts()}

        extracted_facts = ExtractedFacts()
        if self.extract_facts and preserve_facts:
            extracted_facts = await self._extract_facts_async(content)
            logger.debug(
                "Extracted %s deadlines, %s URLs, %s programs",
                len(extracted_facts.deadlines),
                len(extracted_facts.urls),
                len(extracted_facts.programs),
            )

        summaries: dict[str, str] = {}
        current_content = content
        summaries["detailed"] = content

        for level in self.levels[1:]:
            if level.compression_ratio <= 0:
                continue

            target_tokens = int(
                self.counter.count_tokens(current_content) * level.compression_ratio
            )

            if target_tokens < 10:
                summaries[level.name] = current_content
                continue

            summary = await self._summarize_with_facts_async(
                current_content, target_tokens, level, extracted_facts
            )
            summaries[level.name] = summary
            current_content = summary

        return {"summaries": summaries, "facts": extracted_facts}

    def _extract_facts(self, content: str) -> ExtractedFacts:
        """
        Extract key facts from research content using LLM.

        Args:
            content: Content to extract facts from.

        Returns:
            ExtractedFacts object with preserved facts.
        """
        if not content:
            return ExtractedFacts()

        if self._fact_extraction_chain is None:
            return self._extract_facts_regex(content)

        extraction_input = (
            "Extract all key facts from this research content.\n"
            "Be thorough and preserve exact wording for critical details.\n\n"
            "Return the following fields as lists of strings: "
            "deadlines, dates, requirements, contact_info, urls, programs, amounts, eligibility.\n"
            f"Content:\n{content[:4000]}"
        )
        try:
            extracted = self._fact_extraction_chain.invoke(
                {"content": extraction_input}
            )
            extracted = parse_to_schema(extracted, _ExtractedFactsSchema)

            return ExtractedFacts(
                deadlines=extracted.deadlines,
                dates=extracted.dates,
                requirements=extracted.requirements,
                contact_info=extracted.contact_info,
                urls=extracted.urls,
                programs=extracted.programs,
                amounts=extracted.amounts,
                eligibility=extracted.eligibility,
            )
        except Exception as e:
            logger.warning(
                "Error extracting facts with LangChain structured output, using regex fallback: %s",
                e,
            )

        # Fallback: regex-based extraction
        return self._extract_facts_regex(content)

    async def _extract_facts_async(self, content: str) -> ExtractedFacts:
        """Async variant of _extract_facts with regex fallback."""
        if not content:
            return ExtractedFacts()

        if self._fact_extraction_chain is None:
            return self._extract_facts_regex(content)

        extraction_input = (
            "Extract all key facts from this research content.\n"
            "Be thorough and preserve exact wording for critical details.\n\n"
            "Return the following fields as lists of strings: "
            "deadlines, dates, requirements, contact_info, urls, programs, amounts, eligibility.\n"
            f"Content:\n{content[:4000]}"
        )
        raw_timeout = float(os.getenv("AGENT_FACT_EXTRACTION_TIMEOUT_SECONDS", "20"))
        if timeouts_disabled() or raw_timeout <= 0:
            fact_extraction_timeout_seconds: float | None = None
        else:
            fact_extraction_timeout_seconds = max(1.0, raw_timeout)
        try:
            extraction_call = self._fact_extraction_chain.ainvoke(
                {"content": extraction_input}
            )
            if fact_extraction_timeout_seconds is None:
                extracted = await extraction_call
            else:
                extracted = await asyncio.wait_for(
                    extraction_call,
                    timeout=fact_extraction_timeout_seconds,
                )
            extracted = parse_to_schema(extracted, _ExtractedFactsSchema)

            return ExtractedFacts(
                deadlines=extracted.deadlines,
                dates=extracted.dates,
                requirements=extracted.requirements,
                contact_info=extracted.contact_info,
                urls=extracted.urls,
                programs=extracted.programs,
                amounts=extracted.amounts,
                eligibility=extracted.eligibility,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Async fact extraction timed out after %.1fs; using regex fallback",
                fact_extraction_timeout_seconds,
            )
            return self._extract_facts_regex(content)
        except Exception as e:
            logger.warning(
                "Error extracting facts with async LangChain structured output, using regex fallback: %s",
                e,
            )
            return self._extract_facts_regex(content)

    def _extract_facts_regex(self, content: str) -> ExtractedFacts:
        """Fallback regex-based fact extraction."""
        facts = ExtractedFacts()

        # Extract URLs
        url_pattern = r"https?://[^\s\)\]\.,;:!?<>\n]+"
        facts.urls = re.findall(url_pattern, content)

        # Extract dates (basic patterns)
        date_patterns = [
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        ]
        for pattern in date_patterns:
            facts.dates.extend(re.findall(pattern, content, re.IGNORECASE))

        # Extract deadlines (mentions of "deadline")
        deadline_pattern = r"deadline[:\s]+([^\.\n]+)"
        facts.deadlines = re.findall(deadline_pattern, content, re.IGNORECASE)

        # Extract email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        facts.contact_info.extend(re.findall(email_pattern, content))

        # Extract dollar amounts
        amount_pattern = r"\$[\d,]+(?:\.\d{2})?"
        facts.amounts = re.findall(amount_pattern, content)

        return facts

    def _summarize_with_facts(
        self,
        content: str,
        target_tokens: int,
        level: SummaryLevel,
        facts: ExtractedFacts,
    ) -> str:
        """
        Summarize content while preserving extracted facts.

        Args:
            content: Content to summarize.
            target_tokens: Target number of tokens.
            level: Summary level configuration.
            facts: ExtractedFacts to preserve.

        Returns:
            Summarized content with facts preserved.
        """
        # Reserve tokens for facts (about 20% of target)
        facts_tokens = int(target_tokens * 0.2)
        summary_target = target_tokens - facts_tokens

        # Build prompt with facts
        prompt = self._build_summarization_prompt_with_facts(
            content, summary_target, level, facts
        )

        try:
            summary = self._generate_summary(
                system_prompt=self._get_system_prompt(level),
                user_prompt=prompt,
            )

            # Verify summary is within reasonable bounds
            summary_tokens = self.counter.count_tokens(summary)

            # If summary is way too long, truncate
            if summary_tokens > summary_target * 1.5:
                summary = self.counter.truncate_to_tokens(summary, summary_target)

            # Append facts to summary
            facts_text = facts.to_text()
            if facts_text:
                summary = f"{summary.strip()}\n\n--- Key Facts ---\n{facts_text}"

            return summary.strip()

        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            # Fallback: truncate and append facts
            truncated = self.counter.truncate_to_tokens(content, summary_target)
            facts_text = facts.to_text()
            if facts_text:
                truncated = f"{truncated}\n\n--- Key Facts ---\n{facts_text}"
            return truncated

    async def _summarize_with_facts_async(
        self,
        content: str,
        target_tokens: int,
        level: SummaryLevel,
        facts: ExtractedFacts,
    ) -> str:
        """Async variant of _summarize_with_facts."""
        facts_tokens = int(target_tokens * 0.2)
        summary_target = target_tokens - facts_tokens

        prompt = self._build_summarization_prompt_with_facts(
            content, summary_target, level, facts
        )

        try:
            summary = await self._generate_summary_async(
                system_prompt=self._get_system_prompt(level),
                user_prompt=prompt,
            )
            summary_tokens = self.counter.count_tokens(summary)
            if summary_tokens > summary_target * 1.5:
                summary = self.counter.truncate_to_tokens(summary, summary_target)

            facts_text = facts.to_text()
            if facts_text:
                summary = f"{summary.strip()}\n\n--- Key Facts ---\n{facts_text}"
            return summary.strip()
        except Exception as e:
            logger.error("Error during async summarization: %s", e)
            truncated = self.counter.truncate_to_tokens(content, summary_target)
            facts_text = facts.to_text()
            if facts_text:
                truncated = f"{truncated}\n\n--- Key Facts ---\n{facts_text}"
            return truncated

    def _summarize(self, content: str, target_tokens: int, level: SummaryLevel) -> str:
        """
        Summarize content to target token count (without fact extraction).

        Args:
            content: Content to summarize.
            target_tokens: Target number of tokens.
            level: Summary level configuration.

        Returns:
            Summarized content.
        """
        prompt = self._build_summarization_prompt(content, target_tokens, level)

        try:
            summary = self._generate_summary(
                system_prompt=self._get_system_prompt(level),
                user_prompt=prompt,
            )

            # Verify summary is within reasonable bounds
            summary_tokens = self.counter.count_tokens(summary)

            # If summary is way too long, truncate
            if summary_tokens > target_tokens * 1.5:
                summary = self.counter.truncate_to_tokens(summary, target_tokens)

            return summary.strip()

        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            # Fallback: truncate to target tokens
            return self.counter.truncate_to_tokens(content, target_tokens)

    def _get_system_prompt(self, level: SummaryLevel) -> str:
        """Get system prompt for summarization level with research focus."""
        base_prompt = """You are an expert at creating concise, accurate summaries of research information for academic and professional research tasks.

CRITICAL: Your summaries will be used for research decision-making. Accuracy and completeness of key facts is more important than brevity.

Your goal is to preserve ALL actionable information while reducing length."""

        level_specific = {
            "medium": """
Focus on preserving:
- ALL deadlines, dates, and time-sensitive information
- ALL requirements, eligibility criteria, and qualifications
- ALL contact information (emails, phone numbers, addresses)
- ALL URLs and application links
- ALL funding amounts, costs, and financial information
- Program names, grant names, and opportunity titles
- Specific details: GPA requirements, application processes, submission methods

You may remove:
- Redundant explanations
- Background context that's not actionable
- Examples (unless they illustrate requirements)
- Meta-commentary about the research process
""",
            "summary": """
Focus on preserving:
- Critical deadlines and dates
- Essential requirements and eligibility
- Contact information
- URLs and links
- Key program/grant names
- Important amounts or funding levels

You may remove:
- Detailed explanations
- Examples
- Background information
- Non-essential context
""",
            "executive": """
Focus on preserving:
- Most critical deadlines
- Key program/grant names
- Essential eligibility requirements
- Important URLs

You may remove:
- All detailed explanations
- All examples
- All background information
- Most specific details (keep only most important)
""",
        }

        return base_prompt + level_specific.get(level.name, "")

    def _build_summarization_prompt(
        self, content: str, target_tokens: int, level: SummaryLevel
    ) -> str:
        """Build user prompt for research-focused summarization."""
        return f"""Summarize the following research content to approximately {target_tokens} tokens.

{level.description}

CRITICAL PRESERVATION REQUIREMENTS:
- ALL deadlines and dates (exact dates, not approximations)
- ALL requirements and eligibility criteria (exact wording for critical requirements)
- ALL contact information (emails, phone numbers, addresses)
- ALL URLs and application links (complete URLs)
- ALL funding amounts and costs (exact numbers)
- Program names, grant names, opportunity titles (exact names)
- Application processes and submission methods
- GPA requirements, academic requirements (exact numbers)

PRESERVE ACCURACY:
- Copy exact dates, numbers, and requirements - do not paraphrase critical details
- Maintain exact wording for eligibility criteria
- Keep all actionable information that a researcher would need to apply

You may remove:
- Redundant explanations and background context
- Examples (unless they illustrate a requirement)
- Meta-commentary about the research process
- Generic advice that applies to any research topic
- Vague descriptions of databases or search engines

Content to summarize:
{content}

Summary (preserve ALL critical facts):"""

    def _build_summarization_prompt_with_facts(
        self,
        content: str,
        target_tokens: int,
        level: SummaryLevel,
        facts: ExtractedFacts,
    ) -> str:
        """Build user prompt with extracted facts for preservation."""
        facts_text = facts.to_text()

        prompt = f"""Summarize the following research content to approximately {target_tokens} tokens.

{level.description}

CRITICAL: The following facts MUST be preserved in your summary:
{facts_text if facts_text else "No specific facts extracted - preserve all important facts from content."}

PRESERVATION REQUIREMENTS:
- Include ALL facts listed above in your summary
- Preserve exact dates, deadlines, requirements, and contact information
- Maintain accuracy - do not paraphrase critical details
- Keep all actionable information

You may remove:
- Redundant explanations
- Examples (unless critical)
- Background context
- Meta-commentary

Content to summarize:
{content}

Summary (MUST include all facts listed above):"""

        return prompt

    def _generate_summary(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate summary text through LangChain."""
        if self._summary_chain is None:
            raise RuntimeError("LangChain summary chain unavailable")

        return self._summary_chain.invoke(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

    async def _generate_summary_async(
        self, *, system_prompt: str, user_prompt: str
    ) -> str:
        """Async variant of _generate_summary."""
        if self._summary_chain is None:
            raise RuntimeError("LangChain summary chain unavailable")
        return await self._summary_chain.ainvoke(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

    def retrieve_adaptive(
        self,
        summaries: dict[str, str],
        max_tokens: int,
        facts: ExtractedFacts | None = None,
    ) -> tuple[str, str]:
        """
        Retrieve content at appropriate level based on token budget.

        Args:
            summaries: Dictionary of summaries by level name.
            max_tokens: Maximum tokens available.
            facts: Optional ExtractedFacts to append if space allows.

        Returns:
            Tuple of (selected_content, level_name)
        """
        if not summaries:
            return "", "none"

        # Try to get most detailed level that fits
        for level in self.levels:
            if level.name not in summaries:
                continue

            content = summaries[level.name]
            tokens = self.counter.count_tokens(content)

            # If facts provided, check if we can include them
            if facts and facts.to_text():
                facts_text = facts.to_text()
                facts_tokens = self.counter.count_tokens(facts_text)
                if tokens + facts_tokens <= max_tokens:
                    content = f"{content}\n\n--- Key Facts ---\n{facts_text}"
                    tokens += facts_tokens

            if tokens <= max_tokens:
                return content, level.name

        # If nothing fits, truncate the most detailed available
        most_detailed = summaries.get("detailed", "")
        if not most_detailed:
            # Get first available summary
            most_detailed = next(iter(summaries.values()))

        truncated = self.counter.truncate_to_tokens(most_detailed, max_tokens)

        # Try to append facts if space allows
        if facts and facts.to_text():
            facts_text = facts.to_text()
            remaining = max_tokens - self.counter.count_tokens(truncated)
            facts_tokens = self.counter.count_tokens(facts_text)
            if facts_tokens <= remaining:
                truncated = f"{truncated}\n\n--- Key Facts ---\n{facts_text}"

        return truncated, "truncated"

    def get_summary_at_level(
        self, summaries: dict[str, str], level_name: str
    ) -> str | None:
        """
        Get summary at specific level.

        Args:
            summaries: Dictionary of summaries by level name.
            level_name: Name of level to retrieve.

        Returns:
            Summary at requested level, or None if not available.
        """
        return summaries.get(level_name)

    def estimate_tokens_for_level(self, original_tokens: int, level_name: str) -> int:
        """
        Estimate token count for a given level.

        Args:
            original_tokens: Original token count.
            level_name: Name of summary level.

        Returns:
            Estimated token count.
        """
        level_cfg = next(
            (level for level in self.levels if level.name == level_name), None
        )
        if not level_cfg:
            return original_tokens

        return int(original_tokens * level_cfg.compression_ratio)

    def create_summary_on_demand(
        self, content: str, max_tokens: int, preserve_facts: bool = True
    ) -> tuple[str, str, ExtractedFacts | None]:
        """
        Create a summary on-demand that fits within token budget (lazy evaluation).

        This is more efficient than creating all levels upfront when you only need one.

        Args:
            content: Content to summarize.
            max_tokens: Maximum tokens available.
            preserve_facts: If True, extract and preserve facts.

        Returns:
            Tuple of (summary_content, level_name, extracted_facts)
        """
        if not content:
            return "", "none", None

        original_tokens = self.counter.count_tokens(content)

        # If content already fits, return as-is
        if original_tokens <= max_tokens:
            facts = self._extract_facts(content) if preserve_facts else None
            return content, "detailed", facts

        # Extract facts first
        facts = ExtractedFacts()
        if preserve_facts:
            facts = self._extract_facts(content)

        # Find appropriate level based on token budget
        for level in self.levels:
            if level.name == "detailed":
                continue

            estimated_tokens = self.estimate_tokens_for_level(
                original_tokens, level.name
            )

            # If this level should fit, create it
            if estimated_tokens <= max_tokens * 0.9:  # 90% threshold for safety
                summary = self._summarize_with_facts(
                    content, estimated_tokens, level, facts
                )
                actual_tokens = self.counter.count_tokens(summary)

                # If it fits, return it
                if actual_tokens <= max_tokens:
                    return summary, level.name, facts

        # If no level fits, create the most compressed level
        most_compressed = self.levels[-1]  # Last level is most compressed
        target_tokens = int(max_tokens * 0.8)  # Use 80% of budget
        summary = self._summarize_with_facts(
            content, target_tokens, most_compressed, facts
        )

        return summary, most_compressed.name, facts

    def validate_summary_quality(
        self, original: str, summary: str, facts: ExtractedFacts | None = None
    ) -> dict[str, Any]:
        """
        Validate that summary preserves important information.

        Args:
            original: Original content.
            summary: Summary to validate.
            facts: Optional extracted facts to check for.

        Returns:
            Dictionary with quality metrics and missing information.
        """
        quality = {
            "has_deadlines": False,
            "has_dates": False,
            "has_urls": False,
            "has_contact": False,
            "has_requirements": False,
            "missing_facts": [],
            "compression_ratio": 0.0,
        }

        original_tokens = self.counter.count_tokens(original)
        summary_tokens = self.counter.count_tokens(summary)
        quality["compression_ratio"] = (
            summary_tokens / original_tokens if original_tokens > 0 else 0.0
        )

        # Check if facts are preserved
        if facts:
            # Check deadlines
            for deadline in facts.deadlines:
                if deadline.lower() in summary.lower():
                    quality["has_deadlines"] = True
                else:
                    quality["missing_facts"].append(f"Deadline: {deadline}")

            # Check dates
            for date in facts.dates:
                if date in summary:
                    quality["has_dates"] = True
                else:
                    quality["missing_facts"].append(f"Date: {date}")

            # Check URLs
            for url in facts.urls:
                if url in summary or any(
                    part in summary for part in url.split("/")[-2:]
                ):
                    quality["has_urls"] = True
                else:
                    quality["missing_facts"].append(f"URL: {url}")

            # Check contact info
            for contact in facts.contact_info:
                if contact in summary:
                    quality["has_contact"] = True
                else:
                    quality["missing_facts"].append(f"Contact: {contact}")

            # Check requirements
            if facts.requirements:
                for req in facts.requirements[:3]:  # Check first 3
                    if any(word in summary.lower() for word in req.lower().split()[:5]):
                        quality["has_requirements"] = True
                        break

        return quality
