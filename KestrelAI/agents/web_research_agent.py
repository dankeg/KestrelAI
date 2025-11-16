"""
Consolidated Research Agent for KestrelAI
Replaces all duplicate research agent implementations with a single, configurable agent
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import deque
from datetime import datetime
from typing import Any, Literal, Optional

from .base_agent import AgentState
from .base_agent import ResearchAgent as BaseResearchAgent
from .context_builder import ContextBuilder
from .prompt_builder import PromptBuilder
from .research_config import ResearchConfig
from .searxng_service import SearXNGService
from .url_utils import URLFlagManager, clean_url

try:
    from memory.hybrid_retriever import HybridRetriever
    from memory.vector_store import MemoryStore
    from shared.models import Task

    from .context_manager import ContextManager, TokenBudget, TokenCounter
    from .multi_level_summarizer import MultiLevelSummarizer
except ImportError:
    from KestrelAI.agents.context_manager import (
        ContextManager,
        TokenBudget,
        TokenCounter,
    )
    from KestrelAI.agents.multi_level_summarizer import MultiLevelSummarizer
    from KestrelAI.memory.hybrid_retriever import HybridRetriever
    from KestrelAI.memory.vector_store import MemoryStore
    from KestrelAI.shared.models import Task

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class WebResearchAgent(BaseResearchAgent):
    """Web research agent that handles search, analysis, and reporting"""

    def __init__(
        self, agent_id: str, llm, memory: MemoryStore, config: ResearchConfig = None
    ):
        super().__init__(agent_id, llm, memory)
        self.config = config or ResearchConfig()
        self.scratchpad = []

        # Initialize hybrid retriever
        try:
            self.hybrid_retriever = HybridRetriever(memory, enable_bm25=True)
            self.hybrid_retrieval_enabled = True
            logger.info("Hybrid retrieval enabled (vector + BM25)")
        except Exception as e:
            logger.warning(
                f"Failed to initialize hybrid retriever: {e}. Using vector search only."
            )
            self.hybrid_retriever = None
            self.hybrid_retrieval_enabled = False
        # Initialize URL flag manager
        self.url_flag_manager = URLFlagManager()

        # Initialize SearXNG service
        self.searxng_service = SearXNGService(
            searxng_url=None,  # Use default from env
            search_results=self.config.search_results,
            debug=self.config.debug,
        )

        # Initialize prompt builder
        self.prompt_builder = PromptBuilder(self.config)

        # Initialize context management and summarization
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
            logger.info(f"Context management enabled with model: {model_name}")
        except Exception as e:
            logger.warning(
                f"Failed to initialize context management: {e}. Continuing without it."
            )
            self.token_counter = None
            self.token_budget = None
            self.context_manager = None
            self.summarizer = None
            self.context_management_enabled = False

        # Initialize context builder (lazy initialization to avoid forward reference)
        self._context_builder_initialized = False
        self.context_builder = None

        # Initialize MCP if configured
        self.mcp_connected = False
        if self.config.use_mcp and self.config.mcp_manager:
            asyncio.create_task(self._initialize_mcp())

    async def _initialize_mcp(self):
        """Initialize MCP manager if configured"""
        try:
            if not self.config.mcp_manager.is_initialized:
                self.mcp_connected = await self.config.mcp_manager.initialize()
            else:
                self.mcp_connected = self.config.mcp_manager.is_initialized
        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}")
            self.mcp_connected = False

    async def run_step(self, task: Task) -> str:
        """Run one research step"""
        # Ensure context builder is initialized (lazy initialization)
        if not self._context_builder_initialized:
            self._initialize_context_builder()

        state = self._state.setdefault(task.name, AgentState(task_id=task.name))

        for loop_idx in range(self.config.think_loops):
            # Check for loops before proceeding
            if state.is_in_loop():
                if self.config.debug:
                    print(f"[{task.name}] Detected loop, forcing summarize action")
                action: Literal[
                    "think", "search", "mcp_tool", "summarize", "complete"
                ] = "summarize"
                plan = {"action": "summarize"}
            else:
                # Build context
                context = self.context_builder.build_context(task, state)

                if self.config.debug:
                    print(f"\n[Loop {loop_idx+1}/{self.config.think_loops}]")
                    if self.context_management_enabled and self.token_counter:
                        context_tokens = self.token_counter.count_tokens(context)
                        print(f"Context: {len(context)} chars, {context_tokens} tokens")
                    else:
                        print(f"Context length: {len(context)} chars")

                # Get system prompt based on configuration
                system_prompt = self.prompt_builder.get_system_prompt()

                plan_raw = self._chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context},
                    ]
                )

                plan = self._json_from(plan_raw)
                if not plan:
                    if self.config.debug:
                        logger.warning(
                            f"[WARN] Invalid JSON response, defaulting to think. Response text: {plan_raw}"
                        )
                        print("[WARN] Invalid JSON response, defaulting to think")
                    plan = {"action": "think", "thought": "Processing..."}

                action: Literal[
                    "think", "search", "mcp_tool", "summarize", "complete"
                ] = plan.get("action", "think")

            state.action_count += 1
            state.record_action(action, plan.get("query", ""))

            if self.config.debug:
                print(f"[{task.name}] Action {state.action_count}: {action}")

            if action == "think":
                await self._handle_think_action(plan, state)
            elif action == "search":
                await self._handle_search_action(plan, state)
            elif action == "mcp_tool":
                await self._handle_mcp_tool_action(plan, state)
            elif action == "summarize":
                await self._handle_summarize_action(task, state)
            elif action == "complete":
                return await self._handle_complete_action(task, state)

            # Create checkpoint periodically
            if state.action_count % self.config.checkpoint_freq == 0:
                await self._create_checkpoint(task, state)

        # Final checkpoint and report
        if state.action_count % self.config.checkpoint_freq != 0:
            await self._create_checkpoint(task, state)

        final_report = await self._generate_final_report(task, state)

        # Store final report in RAG with summaries (same logic as subtask completion)
        report_id = self._add_to_rag(
            task,
            final_report,
            "final_report",
            metadata={
                "layer": "episodic",
                "action_count": state.action_count,
                "importance_score": self._calculate_importance(final_report),
                "is_final": True,
            },
        )

        # Create and store summary hierarchy for final report
        if self.context_management_enabled and self.summarizer:
            try:
                hierarchy = self.summarizer.create_summary_hierarchy(
                    final_report, preserve_facts=True
                )
                summaries = hierarchy.get("summaries", {})

                for level_name, summary_text in summaries.items():
                    if level_name == "detailed":
                        continue

                    layer = "semantic" if level_name in ["medium"] else "summary"
                    self._add_to_rag(
                        task,
                        summary_text,
                        f"final_report_{level_name}",
                        metadata={
                            "report_id": report_id,
                            "layer": layer,
                            "summary_level": level_name,
                            "is_final": True,
                        },
                    )
            except Exception as e:
                logger.warning(f"Failed to create/store final report summaries: {e}")

        return final_report

    def _initialize_context_builder(self):
        """Initialize context builder after all methods are defined (lazy initialization)"""
        if not self._context_builder_initialized:
            self.context_builder = ContextBuilder(
                config=self.config,
                url_flag_manager=self.url_flag_manager,
                context_manager=self.context_manager,
                token_budget=self.token_budget,
                retrieve_from_rag_func=self._retrieve_from_rag,
            )
            self._context_builder_initialized = True

    async def _handle_think_action(self, plan: dict, state: AgentState):
        """Handle think action"""
        thought = plan.get("thought", "").strip()
        if not thought:
            thought = "Analyzing current information..."

        if self.config.debug:
            print(f"  Thinking: {thought[:100]}...")

        state.history.append(f"[THOUGHT] {thought}")
        state.current_focus = plan.get("direction", state.current_focus)

        # Track metrics
        state.think_count += 1
        self.metrics["total_thoughts"] += 1

    async def _handle_search_action(self, plan: dict, state: AgentState):
        """Handle search action"""
        query = plan.get("query", "").strip()
        if not query:
            state.history.append("[SKIP] Empty search query")
            return

        # Enhanced duplicate detection
        if query in state.queries or state.repeated_queries.get(query, 0) >= 2:
            state.history.append(f"[SKIP] Already searched: {query}")
            return

        # Execute search
        search_start = datetime.now()
        hits = self.searxng_service.search(query)
        search_time = (datetime.now() - search_start).total_seconds()

        if not hits:
            state.history.append(f"[NO RESULTS] {query}")
            return

        state.queries.add(query)
        if self.config.debug:
            print(f"  Searching: {query}")

        # Track search in detailed history
        search_entry = {
            "timestamp": _now_iso(),
            "query": query,
            "results_count": len(hits),
            "search_time": search_time,
            "results": [],
        }

        search_results = []
        for hit in hits:
            # Validate URL before processing
            clean_href = clean_url(hit["href"])
            if clean_href is None:
                # Skip invalid URLs
                logger.warning(
                    f"Skipping invalid URL from search results: {hit['href'][:100]}"
                )
                continue

            body = self.searxng_service.extract_text(clean_href)
            if body:
                self.metrics["total_web_fetches"] += 1

            # Get/create flag for valid URL
            url_flag = self.url_flag_manager.get_or_create_flag(clean_href)
            if url_flag is None:
                # Should not happen if clean_href is valid, but handle it anyway
                logger.warning(
                    f"Failed to create flag for valid URL: {clean_href[:100]}"
                )
                continue

            snippet = (
                f"Title: {hit['title']}\n"
                f"URL: {url_flag} (see URL reference table)\n"
                f"Summary: {hit['body'][:200]}\n"
                f"Content: {body[:500]}"
            )
            # Store in a temporary location for processing
            if not hasattr(state, "snips"):
                state.snips = deque(maxlen=10)
            state.snips.append(snippet)
            search_results.append(hit["title"])

            # Add to search entry (use cleaned URL)
            search_entry["results"].append(
                {
                    "title": hit["title"],
                    "url": clean_href,  # Use cleaned URL
                    "fetched": bool(body),
                }
            )

        state.search_history.append(search_entry)
        state.history.append(f"[SEARCH] {query}\n  Found: {', '.join(search_results)}")

        # Track metrics
        state.search_count += 1
        self.metrics["total_searches"] += 1
        self.metrics["total_search_results"] += len(hits)

    async def _handle_mcp_tool_action(self, plan: dict, state: AgentState):
        """Handle MCP tool action"""
        if not self.config.use_mcp or not self.mcp_connected:
            state.history.append("[SKIP] MCP not available")
            return

        tool_name = plan.get("tool_name", "").strip()
        tool_parameters = plan.get("tool_parameters", {})

        if not tool_name:
            state.history.append("[SKIP] No tool name specified")
            return

        if self.config.debug:
            print(f"  Using MCP tool: {tool_name}")

        # Execute MCP tool
        try:
            result = await self.config.mcp_manager.call_tool(tool_name, tool_parameters)

            if result.success:
                state.history.append(f"[MCP_TOOL] {tool_name}: Success")
                # Store tool result
                if not hasattr(state, "snips"):
                    state.snips = deque(maxlen=10)
                if result.data:
                    state.snips.append(
                        f"MCP Tool Result ({tool_name}):\n{json.dumps(result.data, indent=2)}"
                    )
            else:
                state.history.append(f"[MCP_TOOL] {tool_name}: Failed - {result.error}")
        except Exception as e:
            state.history.append(f"[MCP_TOOL] {tool_name}: Error - {str(e)}")

    async def _handle_summarize_action(self, task: Task, state: AgentState):
        """Handle summarize action"""
        snips = getattr(state, "snips", deque())
        material = "\n---\n".join(snips) if snips else "(no new material)"

        # Replace URLs with flags in material before sending to LLM
        material_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(material)

        summary_prompt = """Create concise research notes from the provided material.
Focus on key findings, data, and sources.
No commentary or questions.
When referencing URLs, use the URL flags (e.g., [URL_1], [URL_2]) provided in the URL reference table.
Do NOT write out full URLs - use the flags instead."""

        # Build user content with URL reference table
        user_content = f"Task: {task.description}\n\nMaterial:\n{material_with_flags}"
        url_table = self.url_flag_manager.get_url_reference_table()
        if url_table:
            user_content += "\n\n" + url_table

        notes = self._chat(
            [
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": user_content},
            ]
        )

        # Replace flags back with URLs using the complete flag mapping
        flag_mapping = self.url_flag_manager.flag_to_url.copy()
        notes = self.url_flag_manager.replace_flags_with_urls(notes, flag_mapping)

        state.history.append(f"[SUMMARY] {notes[:200]}...")

        # Store summary in RAG with metadata
        self._add_to_rag(
            task,
            notes,
            "summary",
            metadata={
                "layer": "semantic",  # Summaries go to semantic layer
                "action_count": state.action_count,
                "importance_score": self._calculate_importance(notes),
            },
        )
        snips.clear()

        # Track metrics
        state.summary_count += 1
        self.metrics["total_summaries"] += 1

        # Reset loop counters on summarize
        state.consecutive_thinks = 0
        state.consecutive_searches = 0

    async def _handle_complete_action(self, task: Task, state: AgentState) -> str:
        """Handle complete action (for subtask agents)"""
        if not self.config.is_subtask_agent:
            state.history.append(
                "[SKIP] Complete action only available for subtask agents"
            )
            return await self._generate_final_report(task, state)

        completion_reason = "Subtask objectives met"
        state.history.append(f"[COMPLETE] {completion_reason}")

        # Generate final report for this subtask
        final_report = await self._generate_final_report(task, state)

        # Store final report in RAG with summaries
        report_id = self._add_to_rag(
            task,
            final_report,
            "final_report",
            metadata={
                "layer": "episodic",  # Final reports are detailed
                "action_count": state.action_count,
                "importance_score": self._calculate_importance(final_report),
                "is_final": True,
            },
        )

        # Create and store summary hierarchy for final report
        if self.context_management_enabled and self.summarizer:
            try:
                hierarchy = self.summarizer.create_summary_hierarchy(
                    final_report, preserve_facts=True
                )
                summaries = hierarchy.get("summaries", {})

                for level_name, summary_text in summaries.items():
                    if level_name == "detailed":
                        continue

                    layer = "semantic" if level_name in ["medium"] else "summary"
                    self._add_to_rag(
                        task,
                        summary_text,
                        f"final_report_{level_name}",
                        metadata={
                            "report_id": report_id,
                            "layer": layer,
                            "summary_level": level_name,
                            "is_final": True,
                        },
                    )
            except Exception as e:
                logger.warning(f"Failed to create/store final report summaries: {e}")

        return final_report

    async def _create_checkpoint(self, task: Task, state: AgentState):
        """Create a checkpoint summarizing current progress"""
        if self.config.debug:
            print(f"[CHECKPOINT] Creating checkpoint at action {state.action_count}")

        # Gather recent context
        recent_context = "\n".join(state.history)

        # Replace URLs with flags in context
        # Process all texts to build up the flag manager's complete mapping
        recent_context_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(
            recent_context
        )
        previous_checkpoint_with_flags = ""
        if state.last_checkpoint:
            prev_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(
                state.last_checkpoint
            )
            previous_checkpoint_with_flags = prev_with_flags

        # Get the complete flag mapping after processing all texts
        flag_mapping = self.url_flag_manager.flag_to_url.copy()

        checkpoint_prompt = """Create a focused checkpoint summarizing actionable research findings.

Focus on:
- Specific opportunities, programs, or grants discovered
- Concrete details: deadlines, requirements, contact information, application links
- Exact eligibility criteria and application processes
- Direct links and contact information

When referencing URLs, use the URL flags (e.g., [URL_1], [URL_2]) provided in the URL reference table.
Do NOT write out full URLs - use the flags instead.

Avoid:
- Generic advice or recommendations
- Vague descriptions of databases or search engines
- Meta-commentary about the research process
- Placeholder text or template content

Be concise but include all actionable information the user can immediately use."""

        # Build user content with URL reference table
        user_content = (
            f"Task: {task.description}\n\n"
            f"Recent research:\n{recent_context_with_flags}\n\n"
            f"Previous checkpoint:\n{previous_checkpoint_with_flags or 'None'}"
        )
        url_table = self.url_flag_manager.get_url_reference_table()
        if url_table:
            user_content += "\n\n" + url_table

        checkpoint = self._chat(
            [
                {"role": "system", "content": checkpoint_prompt},
                {"role": "user", "content": user_content},
            ]
        )

        # Replace flags back with URLs
        checkpoint = self.url_flag_manager.replace_flags_with_urls(
            checkpoint, flag_mapping
        )

        # Store checkpoint in state
        state.last_checkpoint = checkpoint
        state.checkpoints.append(checkpoint)

        # Store checkpoint in RAG (detailed/episodic layer)
        checkpoint_id = self._add_to_rag(
            task,
            checkpoint,
            "checkpoint",
            metadata={
                "checkpoint_index": state.checkpoint_count,
                "layer": "episodic",  # Detailed layer
                "action_count": state.action_count,
                "importance_score": self._calculate_importance(
                    checkpoint
                ),  # Based on fact count
            },
        )

        # Create and store summary hierarchy if context management is enabled
        if self.context_management_enabled and self.summarizer:
            try:
                hierarchy = self.summarizer.create_summary_hierarchy(
                    checkpoint, preserve_facts=True
                )
                summaries = hierarchy.get("summaries", {})
                facts = hierarchy.get("facts")

                # Store summaries at different levels in RAG
                for level_name, summary_text in summaries.items():
                    if level_name == "detailed":
                        continue  # Already stored as checkpoint

                    # Determine layer based on compression level
                    if level_name in ["medium"]:
                        layer = "semantic"
                    elif level_name in ["summary", "executive"]:
                        layer = "summary"
                    else:
                        layer = "semantic"

                    self._add_to_rag(
                        task,
                        summary_text,
                        f"checkpoint_{level_name}",
                        metadata={
                            "checkpoint_index": state.checkpoint_count,
                            "checkpoint_id": checkpoint_id,  # Link to detailed version
                            "layer": layer,
                            "summary_level": level_name,
                            "original_length": len(checkpoint),
                            "compressed_length": len(summary_text),
                            "action_count": state.action_count,
                            "importance_score": self._calculate_importance(checkpoint),
                        },
                    )

                # Store facts separately for quick access
                if facts and facts.to_text():
                    self._add_to_rag(
                        task,
                        facts.to_text(),
                        "checkpoint_facts",
                        metadata={
                            "checkpoint_index": state.checkpoint_count,
                            "checkpoint_id": checkpoint_id,
                            "layer": "facts",
                            "type": "extracted_facts",
                        },
                    )

                # Store in memory for quick access
                if hasattr(state, "checkpoint_summaries"):
                    state.checkpoint_summaries[state.checkpoint_count] = {
                        "checkpoint_id": checkpoint_id,
                        "hierarchy": hierarchy,
                    }
                else:
                    state.checkpoint_summaries = {
                        state.checkpoint_count: {
                            "checkpoint_id": checkpoint_id,
                            "hierarchy": hierarchy,
                        }
                    }

                if self.config.debug:
                    logger.debug(
                        f"Stored checkpoint {state.checkpoint_count} with {len(summaries)} summary levels in RAG"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to create/store checkpoint summaries: {e}", exc_info=True
                )

        # Track metrics
        state.checkpoint_count += 1
        self.metrics["total_checkpoints"] += 1

        # Progressive compression: Compress old checkpoints periodically
        # Run compression every 3 checkpoints to manage memory growth
        # FIXED: Add error handling for background async task
        if state.checkpoint_count % 3 == 0:

            async def compress_with_error_handling():
                try:
                    await self._compress_old_checkpoints(task, state)
                except Exception as e:
                    logger.error(
                        f"Error in background checkpoint compression for task {task.name}: {e}",
                        exc_info=True,
                    )

            asyncio.create_task(compress_with_error_handling())

        # Clear old history after checkpoint
        state.history.clear()
        state.history.append(
            f"[CHECKPOINT CREATED] Focus: {state.current_focus or 'General research'}"
        )

    async def _generate_final_report(self, task: Task, state: AgentState) -> str:
        """Generate final report from all checkpoints and findings"""
        # Retrieve relevant content from RAG using semantic search
        # Use task description as query to get most relevant information
        rag_content = self._retrieve_from_rag(
            task,
            query=task.description,
            max_tokens=self.token_budget.rag_content if self.token_budget else None,
        )

        # Combine all checkpoints
        all_checkpoints = (
            "\n\n---\n\n".join(state.checkpoints) if state.checkpoints else ""
        )

        # Build context from previous reports if available
        previous_reports_context = ""
        if self.config.previous_reports:
            previous_reports_context = (
                "\n\n--- Previous Research Reports ---\n\n"
                + "\n\n---\n\n".join(self.config.previous_reports)
            )

        # If context management is enabled, use summarization for long content
        if self.context_management_enabled and self.summarizer and self.token_budget:
            try:
                # Summarize checkpoints if too long
                if all_checkpoints:
                    checkpoint_tokens = self.token_counter.count_tokens(all_checkpoints)
                    max_checkpoint_tokens = self.token_budget.checkpoints
                    if checkpoint_tokens > max_checkpoint_tokens:
                        (
                            summary,
                            level,
                            facts,
                        ) = self.summarizer.create_summary_on_demand(
                            all_checkpoints,
                            max_tokens=max_checkpoint_tokens,
                            preserve_facts=True,
                        )
                        all_checkpoints = summary
                        if self.config.debug:
                            logger.debug(
                                f"Summarized checkpoints: {checkpoint_tokens} -> {self.token_counter.count_tokens(summary)} tokens"
                            )

                # Summarize previous reports if too long
                if previous_reports_context:
                    reports_tokens = self.token_counter.count_tokens(
                        previous_reports_context
                    )
                    max_reports_tokens = self.token_budget.previous_findings
                    if reports_tokens > max_reports_tokens:
                        (
                            summary,
                            level,
                            facts,
                        ) = self.summarizer.create_summary_on_demand(
                            previous_reports_context,
                            max_tokens=max_reports_tokens,
                            preserve_facts=True,
                        )
                        previous_reports_context = summary
                        if self.config.debug:
                            logger.debug(
                                f"Summarized previous reports: {reports_tokens} -> {self.token_counter.count_tokens(summary)} tokens"
                            )
            except Exception as e:
                logger.warning(
                    f"Error in report summarization, using full content: {e}"
                )

        # Replace URLs with flags in all content before sending to LLM
        # Process all texts to build up the flag manager's complete mapping
        checkpoints_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(
            all_checkpoints
        )
        rag_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(rag_content)
        previous_reports_with_flags = ""
        if previous_reports_context:
            prev_with_flags, _ = self.url_flag_manager.replace_urls_with_flags(
                previous_reports_context
            )
            previous_reports_with_flags = prev_with_flags

        # Get the complete flag mapping after processing all texts
        # This ensures we have all flags that were created during processing
        combined_flag_mapping = self.url_flag_manager.flag_to_url.copy()

        final_report_prompt = """Create a focused, actionable research report from these findings.

CRITICAL REQUIREMENTS:
- Focus on SPECIFIC, ACTIONABLE information that the user can immediately use
- Include concrete details: exact deadlines, specific requirements, contact information, application links
- Prioritize CURRENT opportunities (not generic database descriptions)
- Remove generic advice and focus on specific programs, grants, or opportunities
- Include exact eligibility requirements, application processes, and deadlines
- Provide direct links and contact information where available

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
- DO NOT comment on, evaluate, or praise previous reports (e.g., avoid phrases like "excellent list", "great job", "well done")
- DO NOT provide feedback or suggestions about the format or quality of previous reports
- DO NOT repeat information verbatim unless adding new context
- SYNTHESIZE previous findings with new findings into a cohesive, comprehensive report
- Focus on ADDING VALUE, not evaluating previous work

Structure the report to be:
- Fact-heavy with specific details and numbers
- Actionable with clear next steps
- Well-organized with clear sections
- Professional but concise
- Focused on opportunities the user can actually apply to
- A comprehensive synthesis that builds upon all previous research

Avoid:
- Generic database descriptions
- Vague recommendations
- Placeholder text
- Overly comprehensive archival content
- Generic advice that applies to any research topic
- Meta-commentary about previous reports (e.g., "This is an excellent list", "You've done a great job")
- Evaluation or feedback about previous reports
- Repeating previous reports without adding new information
- Writing out full URLs (use flags instead)

Focus on: Specific programs, exact deadlines, concrete requirements, direct application links, and building upon previous research with new findings."""

        user_content = f"Task: {task.description}\n\n"

        if previous_reports_with_flags:
            user_content += previous_reports_with_flags + "\n\n"

        user_content += f"Current research checkpoints:\n{checkpoints_with_flags}\n\n"
        user_content += f"Additional findings:\n{rag_with_flags}"

        # Add URL reference table
        url_table = self.url_flag_manager.get_url_reference_table()
        if url_table:
            user_content += "\n\n" + url_table

        final_report = self._chat(
            [
                {"role": "system", "content": final_report_prompt},
                {"role": "user", "content": user_content},
            ]
        )

        # Replace flags back with actual URLs
        final_report = self.url_flag_manager.replace_flags_with_urls(
            final_report, combined_flag_mapping
        )

        return final_report

    def _retrieve_from_rag(
        self, task: Task, query: str = None, max_tokens: int = None
    ) -> str:
        """
        Retrieve relevant content from RAG using semantic search with hierarchical summary levels.

        Args:
            task: Task to retrieve content for
            query: Optional query string for semantic search (defaults to task description)
            max_tokens: Optional token budget for retrieved content

        Returns:
            Retrieved content, using appropriate summary level based on token budget
        """
        if max_tokens is None and self.token_budget:
            max_tokens = self.token_budget.rag_content

        # Use task description as query if not provided
        search_query = query or task.description

        try:
            # Use hybrid retrieval if available, otherwise fall back to vector search
            if self.hybrid_retrieval_enabled and self.hybrid_retriever:
                # Use hybrid retrieval (vector + BM25)
                hybrid_results = self.hybrid_retriever.retrieve(
                    search_query,
                    k=20,  # Get more results for better filtering
                    task_name=task.name,
                    use_hybrid=True,
                )

                # Convert hybrid results to our format
                task_docs = []
                for result in hybrid_results:
                    meta = result.get("metadata", {})
                    fused_score = result.get("fused_score", result.get("score", 0.0))

                    # FIXED: Don't convert fused_score to distance incorrectly
                    # fused_score is already a similarity score (higher = better)
                    # For RRF scores (when only one method finds doc), scores are in [0, ~0.033]
                    # For weighted scores (both methods), scores are in [0, 1]
                    #
                    # The issue: RRF scores are on a different scale than weighted scores
                    # Converting with 1.0 - score would make RRF results appear as poor matches
                    #
                    # Solution: Since we sort by fused_score directly (primary key), we don't
                    # need to convert to distance for ranking. However, distance is still used
                    # as a tertiary tie-breaker, so we calculate it in a way that doesn't
                    # penalize RRF scores unfairly.
                    #
                    # For hybrid results, we use fused_score directly for sorting, and set
                    # distance to a value that reflects the score but doesn't affect primary ranking.
                    # Since fused_score is the primary sort key, distance only matters for ties.

                    # Calculate distance for tie-breaking only (not for primary ranking)
                    # Use inverse of normalized score, but since we sort by fused_score first,
                    # this is only used when fused_scores are equal (rare)
                    # For RRF scores, we use a normalized version to avoid extreme distances
                    if fused_score > 1.0:
                        # Shouldn't happen, but handle edge case
                        normalized_score = 1.0
                    elif fused_score > 0.1:
                        # Likely a weighted score [0.1, 1.0] - use directly
                        normalized_score = fused_score
                    else:
                        # Likely an RRF score [0, 0.1] - normalize to comparable scale
                        # Map [0, 0.1] to [0, 0.5] so it's not completely dominated
                        # This is a heuristic - the real fix is sorting by fused_score first
                        normalized_score = min(0.5, fused_score * 5.0)

                    distance = 1.0 - normalized_score

                    task_docs.append(
                        {
                            "content": result["content"],
                            "metadata": meta,
                            "distance": distance,  # For backward compatibility and tie-breaking
                            "layer": meta.get("layer", "episodic"),
                            "checkpoint_index": meta.get("checkpoint_index", -1),
                            "fused_score": fused_score,  # PRIMARY: Use this for sorting
                        }
                    )
            else:
                # Fall back to vector search only
                results = self.memory.search(
                    search_query, k=20
                )  # Get more results, we'll filter by layer

                if (
                    not results
                    or not results.get("documents")
                    or not results["documents"][0]
                ):
                    # Fallback to scratchpad if no RAG results
                    if self.scratchpad:
                        recent_entries = (
                            self.scratchpad[-5:]
                            if len(self.scratchpad) > 5
                            else self.scratchpad
                        )
                        return "\n\n".join(recent_entries)
                    return "(No previous findings)"

                # Organize results by layer and relevance
                documents = results["documents"][0]
                metadatas = (
                    results["metadatas"][0]
                    if results.get("metadatas")
                    else [{}] * len(documents)
                )
                distances = (
                    results["distances"][0]
                    if results.get("distances")
                    else [0.0] * len(documents)
                )

                # Filter to task-specific content
                task_docs = []
                for doc, meta, dist in zip(documents, metadatas, distances):
                    if meta.get("task") == task.name:
                        task_docs.append(
                            {
                                "content": doc,
                                "metadata": meta,
                                "distance": dist,
                                "layer": meta.get("layer", "episodic"),
                                "checkpoint_index": meta.get("checkpoint_index", -1),
                            }
                        )

            if not task_docs:
                # Fallback to scratchpad
                if self.scratchpad:
                    recent_entries = (
                        self.scratchpad[-5:]
                        if len(self.scratchpad) > 5
                        else self.scratchpad
                    )
                    return "\n\n".join(recent_entries)
                return "(No previous findings)"

            # Prioritize by: 1) fused_score (if available), 2) checkpoint_index, 3) distance
            # Higher fused_score = better, higher checkpoint_index = more recent, lower distance = better
            # FIXED: Sort by fused_score first (primary), which correctly handles both weighted and RRF scores
            # fused_score is already a similarity score where higher = better
            task_docs.sort(
                key=lambda x: (
                    x.get(
                        "fused_score", 0.0
                    ),  # Primary: use fused_score directly (higher = better)
                    x["checkpoint_index"],  # Secondary: more recent checkpoints first
                    -x[
                        "distance"
                    ],  # Tertiary: lower distance = better (for tie-breaking)
                ),
                reverse=True,  # Sort descending: highest fused_score first
            )

            # Select appropriate layer based on token budget
            if max_tokens and self.context_management_enabled:
                # Try to retrieve from appropriate summary level
                selected_docs = self._select_documents_by_budget(task_docs, max_tokens)
            else:
                # No budget constraint, use most recent detailed checkpoints
                selected_docs = [d for d in task_docs if d["layer"] == "episodic"][:5]

            if not selected_docs:
                selected_docs = task_docs[:5]  # Fallback to top 5

            # Combine retrieved content
            retrieved_content = "\n\n---\n\n".join(
                [doc["content"] for doc in selected_docs]
            )

            # If still too long and we have summarization, summarize
            if max_tokens and self.context_management_enabled and self.summarizer:
                content_tokens = self.token_counter.count_tokens(retrieved_content)
                if content_tokens > max_tokens:
                    summary, level, facts = self.summarizer.create_summary_on_demand(
                        retrieved_content, max_tokens=max_tokens, preserve_facts=True
                    )
                    if self.config.debug:
                        logger.debug(
                            f"Summarized retrieved RAG content: {content_tokens} -> {self.token_counter.count_tokens(summary)} tokens (level: {level})"
                        )
                    return summary

            return retrieved_content

        except Exception as e:
            logger.warning(
                f"Error retrieving from RAG, falling back to scratchpad: {e}",
                exc_info=True,
            )
            # Fallback to scratchpad
            if self.scratchpad:
                recent_entries = (
                    self.scratchpad[-5:]
                    if len(self.scratchpad) > 5
                    else self.scratchpad
                )
                return "\n\n".join(recent_entries)
            return "(No previous findings)"

    def _select_documents_by_budget(
        self, documents: list[dict], max_tokens: int
    ) -> list[dict]:
        """
        Select documents from appropriate summary levels to fit within token budget.
        Prioritizes recent detailed content, falls back to summaries if needed.
        """
        selected = []
        tokens_used = 0

        # Group by checkpoint_index to avoid duplicates
        checkpoint_groups = {}
        for doc in documents:
            idx = doc["checkpoint_index"]
            if idx not in checkpoint_groups:
                checkpoint_groups[idx] = []
            checkpoint_groups[idx].append(doc)

        # Sort checkpoints by index (most recent first)
        sorted_checkpoints = sorted(checkpoint_groups.keys(), reverse=True)

        for checkpoint_idx in sorted_checkpoints:
            group = checkpoint_groups[checkpoint_idx]

            # Try to get detailed version first
            detailed = [d for d in group if d["layer"] == "episodic"]
            if detailed:
                doc = detailed[0]
                doc_tokens = self.token_counter.count_tokens(doc["content"])
                if tokens_used + doc_tokens <= max_tokens:
                    selected.append(doc)
                    tokens_used += doc_tokens
                    continue

            # Try semantic layer (medium summary)
            semantic = [d for d in group if d["layer"] == "semantic"]
            if semantic:
                doc = semantic[0]
                doc_tokens = self.token_counter.count_tokens(doc["content"])
                if tokens_used + doc_tokens <= max_tokens:
                    selected.append(doc)
                    tokens_used += doc_tokens
                    continue

            # Try summary layer (compressed)
            summary = [d for d in group if d["layer"] == "summary"]
            if summary:
                doc = summary[0]
                doc_tokens = self.token_counter.count_tokens(doc["content"])
                if tokens_used + doc_tokens <= max_tokens:
                    selected.append(doc)
                    tokens_used += doc_tokens
                    continue

            # If we can't fit even the summary, break
            break

        return selected

    def _calculate_importance(self, content: str) -> float:
        """
        Calculate importance score for content based on fact density.
        Higher score = more important (more facts = more important).
        """
        if not content:
            return 0.0

        # Count facts (deadlines, URLs, requirements, etc.)
        fact_indicators = [
            r"\bdeadline\b",
            r"\bdue\b",
            r"\bapplication\b",
            r"https?://",
            r"www\.",
            r"\.edu",
            r"\.gov",
            r"\$\d+",
            r"\d+%",
            r"GPA",
            r"eligibility",
            r"@\w+",
            r"contact",
            r"email",
            r"phone",
        ]

        fact_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in fact_indicators
        )

        # Normalize by content length (facts per 1000 chars)
        content_length = len(content)
        if content_length == 0:
            return 0.0

        importance = (fact_count / content_length) * 1000
        return min(importance, 10.0)  # Cap at 10.0

    async def _compress_old_checkpoints(self, task: Task, state: AgentState):
        """
        Progressive compression: Move old checkpoints to summary layers.
        This is called periodically to manage memory growth.
        """
        if not self.context_management_enabled or not self.summarizer:
            return

        # Compression thresholds
        EPISODIC_LIMIT = 5  # Keep last 5 checkpoints in episodic layer
        SEMANTIC_LIMIT = 10  # Keep checkpoints 6-10 in semantic layer

        # Only compress if we have more than EPISODIC_LIMIT checkpoints
        if state.checkpoint_count <= EPISODIC_LIMIT:
            return

        try:
            # Get old checkpoints that should be compressed
            # Checkpoints are indexed from 0, so older ones have lower indices
            old_checkpoint_indices = list(
                range(
                    max(0, state.checkpoint_count - SEMANTIC_LIMIT),
                    state.checkpoint_count - EPISODIC_LIMIT,
                )
            )

            if not old_checkpoint_indices:
                return

            # Search for old checkpoints in RAG
            results = self.memory.search(f"checkpoint task:{task.name}", k=50)

            if not results or not results.get("documents"):
                return

            documents = results["documents"][0]
            metadatas = (
                results["metadatas"][0]
                if results.get("metadatas")
                else [{}] * len(documents)
            )
            ids = results["ids"][0] if results.get("ids") else []

            # Find checkpoints that need compression
            for doc_id, doc, meta in zip(ids, documents, metadatas):
                if meta.get("task") != task.name:
                    continue

                checkpoint_idx = meta.get("checkpoint_index", -1)
                current_layer = meta.get("layer", "episodic")

                # Skip if already compressed or not in target range
                if checkpoint_idx not in old_checkpoint_indices:
                    continue

                if current_layer == "episodic":
                    # Move from episodic to semantic layer
                    # Check if semantic version already exists
                    has_semantic = any(
                        m.get("checkpoint_index") == checkpoint_idx
                        and m.get("layer") == "semantic"
                        for m in metadatas
                    )

                    if (
                        not has_semantic
                        and checkpoint_idx in old_checkpoint_indices[:5]
                    ):
                        # Create semantic summary if it doesn't exist
                        # The summary should already exist from checkpoint creation, but verify
                        logger.debug(
                            f"Checkpoint {checkpoint_idx} should be in semantic layer"
                        )

                elif current_layer == "semantic":
                    # Move from semantic to summary layer if very old
                    if checkpoint_idx < state.checkpoint_count - SEMANTIC_LIMIT:
                        # Check if summary version exists
                        has_summary = any(
                            m.get("checkpoint_index") == checkpoint_idx
                            and m.get("layer") == "summary"
                            for m in metadatas
                        )

                        if not has_summary:
                            # Create summary version
                            # Get the semantic version to compress further
                            semantic_doc = doc
                            (
                                summary,
                                level,
                                facts,
                            ) = self.summarizer.create_summary_on_demand(
                                semantic_doc,
                                max_tokens=500,  # Very compressed
                                preserve_facts=True,
                            )

                            # Store summary version
                            checkpoint_id = meta.get("checkpoint_id", doc_id)
                            self._add_to_rag(
                                task,
                                summary,
                                "checkpoint_summary_compressed",
                                metadata={
                                    "checkpoint_index": checkpoint_idx,
                                    "checkpoint_id": checkpoint_id,
                                    "layer": "summary",
                                    "summary_level": "executive",
                                    "compressed_from": "semantic",
                                    "action_count": meta.get("action_count", 0),
                                    "importance_score": meta.get(
                                        "importance_score", 0.0
                                    ),
                                },
                            )

                            logger.debug(
                                f"Compressed checkpoint {checkpoint_idx} from semantic to summary layer"
                            )

        except Exception as e:
            logger.warning(f"Error in progressive compression: {e}", exc_info=True)
