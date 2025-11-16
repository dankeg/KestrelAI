"""
Context Management and Token Counting for KestrelAI
Provides token counting, budget management, and intelligent context building.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Token budget allocation for different context components."""

    max_context: int = 32768  # Model's context window (adjust per model)
    system_prompt: int = 500
    task_description: int = 200
    previous_findings: int = 2000
    checkpoints: int = 3000
    history: int = 2000
    rag_content: int = 2000
    url_reference: int = 500
    response_reserve: int = 2000  # Reserve for LLM response

    @property
    def available_for_context(self) -> int:
        """Calculate available tokens for context after system prompt and response reserve."""
        return self.max_context - self.system_prompt - self.response_reserve

    def validate(self) -> bool:
        """Validate that budget allocations are reasonable."""
        total_allocated = (
            self.system_prompt
            + self.task_description
            + self.previous_findings
            + self.checkpoints
            + self.history
            + self.rag_content
            + self.url_reference
            + self.response_reserve
        )
        return total_allocated <= self.max_context


class TokenCounter:
    """Token counter using tiktoken for accurate token counting."""

    # Model encoding mappings for common models
    MODEL_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gemma": "cl100k_base",  # Gemma models typically use cl100k_base
        "gemma2": "cl100k_base",
        "gemma3": "cl100k_base",
        "llama": "cl100k_base",  # Many LLaMA-based models use cl100k_base
        "default": "cl100k_base",  # Default fallback
    }

    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize token counter.

        Args:
            model_name: Name of the model to use for tokenization.
                       Will map to appropriate tiktoken encoding.
        """
        if tiktoken is None:
            raise ImportError(
                "tiktoken is required for token counting. "
                "Install it with: pip install tiktoken"
            )

        self.model_name = model_name
        encoding_name = self._get_encoding_name(model_name)

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except KeyError:
            logger.warning(
                f"Encoding {encoding_name} not found, falling back to cl100k_base"
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def _get_encoding_name(self, model_name: str) -> str:
        """Get encoding name for a given model."""
        model_lower = model_name.lower()

        # Check for exact match
        if model_lower in self.MODEL_ENCODINGS:
            return self.MODEL_ENCODINGS[model_lower]

        # Check for partial match (e.g., "gemma3:27b" -> "gemma3")
        for key, encoding in self.MODEL_ENCODINGS.items():
            if key in model_lower:
                return encoding

        # Default fallback
        return self.MODEL_ENCODINGS["default"]

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0

        try:
            return len(self.encoding.encode(str(text)))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback: rough estimate (1 token ≈ 4 characters for English)
            return len(str(text)) // 4

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """
        Count tokens in a list of messages (for chat API).

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            Total number of tokens including message formatting overhead.
        """
        total = 0

        for msg in messages:
            content = msg.get("content", "")
            if content:
                total += self.count_tokens(content)

            # Add overhead for message formatting (typically 4 tokens per message)
            # This includes role tags and formatting
            total += 4

        return total

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token budget.

        Args:
            text: Text to truncate.
            max_tokens: Maximum number of tokens allowed.

        Returns:
            Truncated text.
        """
        if not text:
            return text

        tokens = self.encoding.encode(text)

        if len(tokens) <= max_tokens:
            return text

        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.encoding.decode(truncated_tokens)

        # Add ellipsis if truncated
        if len(tokens) > max_tokens:
            truncated_text += "..."

        return truncated_text


class ContextManager:
    """Manages context building within token budget constraints with full context summarization."""

    def __init__(
        self,
        token_counter: TokenCounter,
        token_budget: TokenBudget,
        llm=None,  # Optional LLM for summarization
        summarizer=None,  # Optional MultiLevelSummarizer for full context summarization
    ):
        """
        Initialize context manager.

        Args:
            token_counter: TokenCounter instance for counting tokens.
            token_budget: TokenBudget instance defining token allocations.
            llm: Optional LLM wrapper for summarization (if needed).
            summarizer: Optional MultiLevelSummarizer for full context summarization.
        """
        self.counter = token_counter
        self.budget = token_budget
        self.llm = llm
        self.summarizer = summarizer

    def build_context(
        self, components: dict[str, Any], prioritize_recent: bool = True
    ) -> tuple[str, dict[str, int]]:
        """
        Build context within token budget.

        Args:
            components: Dictionary with context components:
                - task: Task description
                - subtask: Subtask description (optional)
                - previous_findings: Previous findings text (optional)
                - checkpoints: List of checkpoint strings (optional)
                - history: List of history entry strings (optional)
                - rag_content: RAG retrieved content (optional)
                - url_reference: URL reference table (optional)
            prioritize_recent: If True, prioritize recent items in lists.

        Returns:
            Tuple of (context_string, token_usage_dict)
        """
        context_parts = []
        token_usage = {
            "task": 0,
            "subtask": 0,
            "previous_findings": 0,
            "checkpoints": 0,
            "history": 0,
            "rag_content": 0,
            "url_reference": 0,
            "total": 0,
        }

        # 1. Always include task description
        task = components.get("task", "")
        if task:
            task_tokens = self.counter.count_tokens(task)
            if task_tokens <= self.budget.task_description:
                context_parts.append(f"Task: {task}")
                token_usage["task"] = task_tokens
            else:
                # Truncate if too long
                truncated = self.counter.truncate_to_tokens(
                    task, self.budget.task_description
                )
                context_parts.append(f"Task: {truncated}")
                token_usage["task"] = self.counter.count_tokens(truncated)

        # 2. Add subtask if available
        subtask = components.get("subtask", "")
        if subtask:
            subtask_tokens = self.counter.count_tokens(subtask)
            if subtask_tokens <= 200:  # Small budget for subtask
                context_parts.append(f"Subtask: {subtask}")
                token_usage["subtask"] = subtask_tokens

        # 3. Add previous findings (summarized if too long)
        previous_findings = components.get("previous_findings", "")
        if previous_findings:
            findings_tokens = self.counter.count_tokens(previous_findings)
            if findings_tokens <= self.budget.previous_findings:
                context_parts.append(f"Previous findings: {previous_findings}")
                token_usage["previous_findings"] = findings_tokens
            else:
                # Truncate or summarize
                truncated = self.counter.truncate_to_tokens(
                    previous_findings, self.budget.previous_findings
                )
                context_parts.append(f"Previous findings: {truncated}")
                token_usage["previous_findings"] = self.counter.count_tokens(truncated)

        # 4. Add checkpoints (most recent first if prioritize_recent)
        checkpoints = components.get("checkpoints", [])
        if checkpoints:
            checkpoint_list = list(checkpoints)
            if prioritize_recent:
                checkpoint_list = reversed(checkpoint_list)  # Most recent first

            checkpoint_tokens_used = 0
            included_checkpoints = []

            for checkpoint in checkpoint_list:
                checkpoint_tokens = self.counter.count_tokens(checkpoint)

                if (
                    checkpoint_tokens_used + checkpoint_tokens
                    <= self.budget.checkpoints
                ):
                    included_checkpoints.append(checkpoint)
                    checkpoint_tokens_used += checkpoint_tokens
                else:
                    # Try to fit partial checkpoint
                    remaining = self.budget.checkpoints - checkpoint_tokens_used
                    if remaining > 100:  # Minimum viable chunk
                        truncated = self.counter.truncate_to_tokens(
                            checkpoint, remaining
                        )
                        included_checkpoints.append(truncated)
                        checkpoint_tokens_used += self.counter.count_tokens(truncated)
                    break

            if included_checkpoints:
                if prioritize_recent:
                    included_checkpoints = list(reversed(included_checkpoints))
                checkpoint_text = "\n\n---\n\n".join(included_checkpoints)
                context_parts.append(f"Checkpoints:\n{checkpoint_text}")
                token_usage["checkpoints"] = checkpoint_tokens_used

        # 5. Add history (most recent first if prioritize_recent)
        history = components.get("history", [])
        if history:
            history_list = list(history)
            if prioritize_recent:
                history_list = reversed(history_list)  # Most recent first

            history_tokens_used = 0
            included_history = []

            for entry in history_list:
                entry_tokens = self.counter.count_tokens(entry)

                if history_tokens_used + entry_tokens <= self.budget.history:
                    included_history.append(entry)
                    history_tokens_used += entry_tokens
                else:
                    break

            if included_history:
                if prioritize_recent:
                    included_history = list(reversed(included_history))
                history_text = "\n".join(included_history)
                context_parts.append(f"Recent actions: {history_text}")
                token_usage["history"] = history_tokens_used

        # 6. Add RAG content
        rag_content = components.get("rag_content", "")
        if rag_content:
            rag_tokens = self.counter.count_tokens(rag_content)
            if rag_tokens <= self.budget.rag_content:
                context_parts.append(f"Additional findings: {rag_content}")
                token_usage["rag_content"] = rag_tokens
            else:
                # Truncate
                truncated = self.counter.truncate_to_tokens(
                    rag_content, self.budget.rag_content
                )
                context_parts.append(f"Additional findings: {truncated}")
                token_usage["rag_content"] = self.counter.count_tokens(truncated)

        # 7. Add URL reference table
        url_reference = components.get("url_reference", "")
        if url_reference:
            url_tokens = self.counter.count_tokens(url_reference)
            if url_tokens <= self.budget.url_reference:
                context_parts.append(url_reference)
                token_usage["url_reference"] = url_tokens
            else:
                # Truncate
                truncated = self.counter.truncate_to_tokens(
                    url_reference, self.budget.url_reference
                )
                context_parts.append(truncated)
                token_usage["url_reference"] = self.counter.count_tokens(truncated)

        # Build final context
        context = "\n\n".join(context_parts)
        total_tokens = self.counter.count_tokens(context)

        # Calculate available budget (accounting for system prompt and response reserve)
        available_budget = (
            self.budget.max_context
            - self.budget.system_prompt
            - self.budget.response_reserve
        )

        # If context exceeds available budget, use full context summarization
        if total_tokens > available_budget and self.summarizer:
            try:
                logger.debug(
                    f"Context exceeds budget ({total_tokens} > {available_budget}), creating full context summary"
                )

                # Create a coherent summary of the entire context
                summary, level, facts = self.summarizer.create_summary_on_demand(
                    context, max_tokens=available_budget, preserve_facts=True
                )

                summary_tokens = self.counter.count_tokens(summary)
                logger.debug(
                    f"Full context summary created: {total_tokens} -> {summary_tokens} tokens (level: {level})"
                )

                # Update token usage to reflect summary
                token_usage["total"] = summary_tokens
                token_usage["summarized"] = True
                token_usage["original_tokens"] = total_tokens
                token_usage["summary_level"] = level

                return summary, token_usage
            except Exception as e:
                logger.warning(
                    f"Error in full context summarization, falling back to truncation: {e}"
                )
                # Fall through to truncation

        # If still too large (or summarization failed), truncate as last resort
        if total_tokens > available_budget:
            logger.warning(
                "Context still exceeds budget after summarization attempt, truncating"
            )
            context = self.counter.truncate_to_tokens(context, available_budget)
            total_tokens = self.counter.count_tokens(context)

        # Track total token usage as the sum of individual components for easier testing
        token_usage["total"] = sum(v for k, v in token_usage.items() if k != "total")
        token_usage["summarized"] = False

        return context, token_usage
