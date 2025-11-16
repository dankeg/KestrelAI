"""
Context builder for web research agent.
Handles building context for LLM with token-aware management.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.models import Task

    from .base_agent import AgentState
    from .context_manager import ContextManager, TokenBudget
    from .research_config import ResearchConfig
    from .url_utils import URLFlagManager

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds context for the research agent with token-aware management"""

    def __init__(
        self,
        config: "ResearchConfig",
        url_flag_manager: "URLFlagManager",
        context_manager: "ContextManager" = None,
        token_budget: "TokenBudget" = None,
        retrieve_from_rag_func=None,
    ):
        self.config = config
        self.url_flag_manager = url_flag_manager
        self.context_manager = context_manager
        self.token_budget = token_budget
        self.retrieve_from_rag_func = retrieve_from_rag_func
        self.context_management_enabled = context_manager is not None

    def build_context(self, task: "Task", state: "AgentState") -> str:
        """Build context for the agent with token-aware management"""
        # Use new context manager if available, otherwise fall back to old method
        if not self.context_management_enabled:
            return self._build_context_legacy(task, state)

        try:
            # Prepare components for context manager
            components = {"task": task.description}

            # Add subtask-specific context
            if self.config.is_subtask_agent:
                components[
                    "subtask"
                ] = f"{self.config.subtask_description}\nSuccess Criteria: {self.config.success_criteria}"
                if self.config.previous_findings:
                    # Replace URLs with flags in previous findings
                    (
                        findings_with_flags,
                        _,
                    ) = self.url_flag_manager.replace_urls_with_flags(
                        self.config.previous_findings
                    )
                    components["previous_findings"] = findings_with_flags

            # Add checkpoints (use summarization if needed)
            if state.checkpoints:
                checkpoint_list = []
                for checkpoint in state.checkpoints:
                    # Replace URLs with flags
                    (
                        checkpoint_with_flags,
                        _,
                    ) = self.url_flag_manager.replace_urls_with_flags(checkpoint)
                    checkpoint_list.append(checkpoint_with_flags)
                components["checkpoints"] = checkpoint_list

            # Add last checkpoint separately if different from checkpoints list
            if state.last_checkpoint and state.last_checkpoint not in state.checkpoints:
                (
                    checkpoint_with_flags,
                    _,
                ) = self.url_flag_manager.replace_urls_with_flags(state.last_checkpoint)
                if "checkpoints" not in components:
                    components["checkpoints"] = []
                components["checkpoints"].insert(0, checkpoint_with_flags)

            # Add history
            if state.history:
                history_list = []
                for entry in state.history:
                    # Replace URLs with flags
                    entry_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(
                        entry
                    )
                    history_list.append(entry_with_flags)
                components["history"] = history_list

            # Add current focus
            if state.current_focus:
                components["current_focus"] = state.current_focus

            # Add RAG content (retrieved using semantic search with hierarchical summaries)
            # Use current focus or task description as query for better relevance
            if self.retrieve_from_rag_func:
                rag_query = state.current_focus or task.description
                rag_content = self.retrieve_from_rag_func(
                    task,
                    query=rag_query,
                    max_tokens=self.token_budget.rag_content
                    if self.token_budget
                    else None,
                )
                if rag_content and rag_content != "(No previous findings)":
                    # Replace URLs with flags
                    rag_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(
                        rag_content
                    )
                    components["rag_content"] = rag_with_flags

            # Add URL reference table
            url_table = self.url_flag_manager.get_url_reference_table()
            if url_table:
                components["url_reference"] = url_table

            # Build context using context manager
            context, token_usage = self.context_manager.build_context(
                components, prioritize_recent=True
            )

            # Log token usage for debugging
            if self.config.debug:
                logger.debug(f"Context token usage: {token_usage}")
                logger.debug(f"Total context tokens: {token_usage['total']}")

            return context

        except Exception as e:
            logger.error(
                f"Error in context management, falling back to legacy: {e}",
                exc_info=True,
            )
            return self._build_context_legacy(task, state)

    def _build_context_legacy(self, task: "Task", state: "AgentState") -> str:
        """Legacy context building method (fallback)"""
        context_parts = [f"Task: {task.description}"]

        # Add subtask-specific context
        if self.config.is_subtask_agent:
            context_parts.extend(
                [
                    f"Subtask: {self.config.subtask_description}",
                    f"Success Criteria: {self.config.success_criteria}",
                ]
            )
            if self.config.previous_findings:
                # Replace URLs with flags in previous findings
                findings_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(
                    self.config.previous_findings
                )
                context_parts.append(f"Previous findings: {findings_with_flags}")

        # Add last checkpoint if available
        if state.last_checkpoint:
            # Replace URLs with flags in checkpoint
            checkpoint_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(
                state.last_checkpoint
            )
            context_parts.append(f"Previous checkpoint: {checkpoint_with_flags}")

        # Add current focus if set
        if state.current_focus:
            context_parts.append(f"Current focus: {state.current_focus}")

        # Add sliding window of recent history
        if state.history:
            recent_history = "\n".join(state.history)
            # Replace URLs with flags in history
            history_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(
                recent_history
            )
            context_parts.append(f"Recent actions: {history_with_flags}")
        else:
            context_parts.append("[No actions yet]")

        # Add URL reference table if there are any URLs
        url_table = self.url_flag_manager.get_url_reference_table()
        if url_table:
            context_parts.append(url_table)

        return "\n".join(context_parts)
