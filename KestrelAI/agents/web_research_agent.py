"""
Consolidated Research Agent for KestrelAI
Replaces all duplicate research agent implementations with a single, configurable agent
"""

from __future__ import annotations
import json
import os
import re
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Set, Deque, Optional, Any, Union
from collections import deque
from uuid import uuid4

import requests
from bs4 import BeautifulSoup
import logging

from .base_agent import ResearchAgent as BaseResearchAgent, AgentState
try:
    from memory.vector_store import MemoryStore
    from shared.models import Task
except ImportError:
    from KestrelAI.memory.vector_store import MemoryStore
    from KestrelAI.shared.models import Task

logger = logging.getLogger(__name__)

# Configuration
THINK_LOOPS = 6
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080/search")
SEARCH_RESULTS = 4
FETCH_BYTES = 30_000
DEBUG = True
CONTEXT_WINDOW = 60
CHECKPOINT_FREQ = 5
MAX_SNIPPET_LENGTH = 3000


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _extract_text(url: str) -> str:
    """Download page & return readable text (best-effort)."""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text[:FETCH_BYTES]
        soup = BeautifulSoup(html, "html.parser")

        # Remove non-content elements
        for t in soup(["script", "style", "noscript", "meta", "link"]):
            t.extract()

        text = " ".join(soup.get_text(" ", strip=True).split())
        return text[:MAX_SNIPPET_LENGTH]
    except Exception as e:
        if DEBUG:
            print(f"Failed to extract from {url}: {e}")
        return ""


@dataclass
class ResearchConfig:
    """Configuration for research agent behavior"""
    think_loops: int = THINK_LOOPS
    search_results: int = SEARCH_RESULTS
    fetch_bytes: int = FETCH_BYTES
    context_window: int = CONTEXT_WINDOW
    checkpoint_freq: int = CHECKPOINT_FREQ
    max_snippet_length: int = MAX_SNIPPET_LENGTH
    debug: bool = DEBUG
    
    # Subtask-specific settings
    is_subtask_agent: bool = False
    subtask_description: str = ""
    success_criteria: str = ""
    previous_findings: str = ""
    
    # MCP settings
    use_mcp: bool = False
    mcp_manager: Optional[Any] = None


class WebResearchAgent(BaseResearchAgent):
    """Web research agent that handles search, analysis, and reporting"""
    
    def __init__(self, agent_id: str, llm, memory: MemoryStore, config: ResearchConfig = None):
        super().__init__(agent_id, llm, memory)
        self.config = config or ResearchConfig()
        self.scratchpad = []
        
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
        state = self._state.setdefault(task.name, AgentState(task_id=task.name))
        
        for loop_idx in range(self.config.think_loops):
            # Check for loops before proceeding
            if state.is_in_loop() and False:
                if self.config.debug:
                    print(f"[{task.name}] Detected loop, forcing summarize action")
                action = "summarize"
                plan = {"action": "summarize"}
            else:
                # Build context
                context = self._build_context(task, state)
                
                if self.config.debug:
                    print(f"\n[Loop {loop_idx+1}/{self.config.think_loops}]")
                    print(f"Context length: {len(context)} chars")
                
                # Get system prompt based on configuration
                system_prompt = self._get_system_prompt()
                
                plan_raw = self._chat([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ])
                
                plan = self._json_from(plan_raw)
                if not plan:
                    if self.config.debug:
                        print("[WARN] Invalid JSON response, defaulting to think")
                    plan = {"action": "think", "thought": "Processing..."}
                
                action: Literal["think", "search", "mcp_tool", "summarize", "complete"] = plan.get("action", "think")
            
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
        self._add_to_rag(task, final_report, "final_report")
        
        return final_report
    
    def _get_system_prompt(self) -> str:
        """Get appropriate system prompt based on configuration"""
        if self.config.is_subtask_agent:
            return self._get_subtask_system_prompt()
        elif self.config.use_mcp:
            return self._get_mcp_system_prompt()
        else:
            return self._get_standard_system_prompt()
    
    def _get_standard_system_prompt(self) -> str:
        """Standard research agent system prompt"""
        return """You are an autonomous research agent conducting focused investigations to find specific, actionable information.

Your goal is to find concrete details that the user can immediately act upon:
- Specific programs, grants, or opportunities with exact details
- Concrete deadlines, requirements, and application processes  
- Direct contact information and application links
- Exact eligibility criteria and requirements
- Current opportunities (not generic database descriptions)

CORE RULES:
- Focus on ACTIONABLE information the user can apply to or use immediately
- Prioritize specific programs over generic database descriptions
- Find concrete details: exact deadlines, specific requirements, contact information
- Avoid generic advice that applies to any research topic
- Do not make up information
- Do not have conversations
- All context is either your own words, or results from tools you called
- Do not retry failed searches or use the same search terms repeatedly. Move on.
- The date is September 20th, 2025

OUTPUT FORMAT (JSON only):
{
  "direction": "Your reasoning for the next action (1-2 sentences)",
  "action": "think" | "search" | "summarize",
  "query": "search terms (if action is 'search', else empty string)",
  "thought": "detailed planning and brainstorming (if action is 'think', else empty string)"
}

ACTIONS:
- think: Reason about findings and plan next steps to find specific opportunities
- search: Targeted queries for specific programs, grants, or opportunities
- summarize: Checkpoint actionable findings with concrete details"""
    
    def _get_subtask_system_prompt(self) -> str:
        """Subtask-specific system prompt"""
        return f"""You are a specialized research agent focused on finding specific, actionable information for your assigned subtask.

Your goal is to find concrete details that the user can immediately act upon:
- Specific programs, grants, or opportunities with exact details
- Concrete deadlines, requirements, and application processes  
- Direct contact information and application links
- Exact eligibility criteria and requirements
- Current opportunities (not generic database descriptions)

SUBTASK CONTEXT:
- Subtask: {self.config.subtask_description}
- Success Criteria: {self.config.success_criteria}
- Previous Findings: {self.config.previous_findings}

CORE RULES:
- Focus on ACTIONABLE information the user can apply to or use immediately
- Prioritize specific programs over generic database descriptions
- Find concrete details: exact deadlines, specific requirements, contact information
- Avoid generic advice that applies to any research topic
- Stay strictly focused on your assigned subtask
- Conduct thorough research to meet the success criteria
- Build upon any previous findings provided to you
- Do not make up information
- Do not have conversations
- All context is either your own words, or results from tools you called
- Do not retry failed searches or use the same search terms repeatedly
- The date is September 20th, 2025

OUTPUT FORMAT (JSON only):
{{
  "direction": "Your reasoning for the next action (1-2 sentences)",
  "action": "think" | "search" | "summarize" | "complete",
  "query": "search terms (if action is 'search', else empty string)",
  "thought": "detailed planning and brainstorming (if action is 'think', else empty string)"
}}

ACTIONS:
- think: Reason about findings and plan next steps
- search: Targeted, human readable queries for new information
- summarize: Checkpoint important findings
- complete: Indicate that the subtask success criteria have been met"""
    
    def _get_mcp_system_prompt(self) -> str:
        """MCP-enhanced system prompt"""
        return """You are an autonomous research agent conducting deep investigations with access to powerful tools and data sources.

Your goal is to gather accurate, relevant, and up-to-date information on the assigned topic using all available resources.

AVAILABLE TOOLS:
You have access to powerful research tools through the MCP (Model Context Protocol) system:

DATA SOURCES:
- search_web: Enhanced web search with multiple engines and filtering
- query_database: Execute SQL queries on structured databases
- search_repositories: Search GitHub repositories for code and documentation
- read_file: Read contents of files from the filesystem
- search_files: Search for files by name or content

ANALYSIS TOOLS:
- analyze_data: Perform statistical analysis on data
- extract_text: Extract and clean text from web pages or documents

RESEARCH TOOLS:
- get_repository_info: Get detailed information about a GitHub repository
- navigate_to_page: Navigate to a web page and extract information

AUTOMATION TOOLS:
- write_file: Write data to files
- create_table: Create database tables for structured data storage

CORE RULES:
- Stay strictly on topic for the given task
- Probe deeply - don't settle for surface-level information
- If blocked, try alternative research paths and tools
- If information is already known, find new information and pathways
- Build upon previous findings iteratively
- Do not make up information
- Do not have conversations
- All context is either your own words, or results from tools you called
- Do not retry failed searches or use the same search terms repeatedly. Move on.
- The date is September 20th, 2025

OUTPUT FORMAT (JSON only):
{
  "direction": "Your reasoning for the next action (1-2 sentences)",
  "action": "think" | "search" | "mcp_tool" | "summarize",
  "query": "search terms (if action is 'search', else empty string)",
  "tool_name": "tool name (if action is 'mcp_tool', else empty string)",
  "tool_parameters": {"param": "value"} (if action is 'mcp_tool', else empty object),
  "thought": "detailed planning and brainstorming (if action is 'think', else empty string)"
}

ACTIONS:
- think: Reason about findings and plan next steps
- search: Traditional web search via SearXNG (fallback)
- mcp_tool: Use MCP tools for enhanced research capabilities
- summarize: Checkpoint important findings"""
    
    def _build_context(self, task: Task, state: AgentState) -> str:
        """Build context for the agent"""
        context_parts = [f"Task: {task.description}"]
        
        # Add subtask-specific context
        if self.config.is_subtask_agent:
            context_parts.extend([
                f"Subtask: {self.config.subtask_description}",
                f"Success Criteria: {self.config.success_criteria}"
            ])
            if self.config.previous_findings:
                context_parts.append(f"Previous findings: {self.config.previous_findings}")
        
        # Add last checkpoint if available
        if state.last_checkpoint:
            context_parts.append(f"Previous checkpoint: {state.last_checkpoint}")
        
        # Add current focus if set
        if state.current_focus:
            context_parts.append(f"Current focus: {state.current_focus}")
        
        # Add sliding window of recent history
        if state.history:
            recent_history = "\n".join(state.history)
            context_parts.append(f"Recent actions: {recent_history}")
        else:
            context_parts.append("[No actions yet]")
        
        return "\n".join(context_parts)
    
    async def _handle_think_action(self, plan: Dict, state: AgentState):
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
    
    async def _handle_search_action(self, plan: Dict, state: AgentState):
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
        hits = self._searx_search(query)
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
            body = _extract_text(hit["href"])
            if body:
                self.metrics["total_web_fetches"] += 1
            
            snippet = (
                f"Title: {hit['title']}\n"
                f"URL: {hit['href']}\n"
                f"Summary: {hit['body'][:200]}\n"
                f"Content: {body[:500]}"
            )
            # Store in a temporary location for processing
            if not hasattr(state, 'snips'):
                state.snips = deque(maxlen=10)
            state.snips.append(snippet)
            search_results.append(hit["title"])
            
            # Add to search entry
            search_entry["results"].append({
                "title": hit["title"],
                "url": hit["href"],
                "fetched": bool(body),
            })
        
        state.search_history.append(search_entry)
        state.history.append(f"[SEARCH] {query}\n  Found: {', '.join(search_results)}")
        
        # Track metrics
        state.search_count += 1
        self.metrics["total_searches"] += 1
        self.metrics["total_search_results"] += len(hits)
    
    async def _handle_mcp_tool_action(self, plan: Dict, state: AgentState):
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
                if not hasattr(state, 'snips'):
                    state.snips = deque(maxlen=10)
                if result.data:
                    state.snips.append(f"MCP Tool Result ({tool_name}):\n{json.dumps(result.data, indent=2)}")
            else:
                state.history.append(f"[MCP_TOOL] {tool_name}: Failed - {result.error}")
        except Exception as e:
            state.history.append(f"[MCP_TOOL] {tool_name}: Error - {str(e)}")
    
    async def _handle_summarize_action(self, task: Task, state: AgentState):
        """Handle summarize action"""
        snips = getattr(state, 'snips', deque())
        material = "\n---\n".join(snips) if snips else "(no new material)"
        
        summary_prompt = """Create concise research notes from the provided material.
Focus on key findings, data, and sources.
No commentary or questions."""
        
        notes = self._chat([
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": f"Task: {task.description}\n\nMaterial:\n{material}"},
        ])
        
        state.history.append(f"[SUMMARY] {notes[:200]}...")
        self._add_to_rag(task, notes, "summary")
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
            state.history.append("[SKIP] Complete action only available for subtask agents")
            return await self._generate_final_report(task, state)
        
        completion_reason = "Subtask objectives met"
        state.history.append(f"[COMPLETE] {completion_reason}")
        
        # Generate final report for this subtask
        final_report = await self._generate_final_report(task, state)
        self._add_to_rag(task, final_report, "final_report")
        
        return final_report
    
    async def _create_checkpoint(self, task: Task, state: AgentState):
        """Create a checkpoint summarizing current progress"""
        if self.config.debug:
            print(f"[CHECKPOINT] Creating checkpoint at action {state.action_count}")
        
        # Gather recent context
        recent_context = "\n".join(state.history)
        
        checkpoint_prompt = """Create a focused checkpoint summarizing actionable research findings.

Focus on:
- Specific opportunities, programs, or grants discovered
- Concrete details: deadlines, requirements, contact information, application links
- Exact eligibility criteria and application processes
- Direct links and contact information

Avoid:
- Generic advice or recommendations
- Vague descriptions of databases or search engines
- Meta-commentary about the research process
- Placeholder text or template content

Be concise but include all actionable information the user can immediately use."""
        
        checkpoint = self._chat([
            {"role": "system", "content": checkpoint_prompt},
            {"role": "user", "content": (
                f"Task: {task.description}\n\n"
                f"Recent research:\n{recent_context}\n\n"
                f"Previous checkpoint:\n{state.last_checkpoint or 'None'}"
            )},
        ])
        
        state.last_checkpoint = checkpoint
        state.checkpoints.append(checkpoint)
        self._add_to_rag(task, checkpoint, "checkpoint")
        
        # Track metrics
        state.checkpoint_count += 1
        self.metrics["total_checkpoints"] += 1
        
        # Clear old history after checkpoint
        state.history.clear()
        state.history.append(f"[CHECKPOINT CREATED] Focus: {state.current_focus or 'General research'}")
    
    async def _generate_final_report(self, task: Task, state: AgentState) -> str:
        """Generate final report from all checkpoints and findings"""
        # Retrieve relevant content from RAG
        rag_content = self._retrieve_from_rag(task)
        
        # Combine all checkpoints
        all_checkpoints = "\n\n---\n\n".join(state.checkpoints) if state.checkpoints else ""
        
        final_report_prompt = """Create a focused, actionable research report from these findings.

CRITICAL REQUIREMENTS:
- Focus on SPECIFIC, ACTIONABLE information that the user can immediately use
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
        
        final_report = self._chat([
            {"role": "system", "content": final_report_prompt},
            {"role": "user", "content": (
                f"Task: {task.description}\n\n"
                f"Research checkpoints:\n{all_checkpoints}\n\n"
                f"Additional findings:\n{rag_content}"
            )},
        ])
        
        return final_report
    
    def _retrieve_from_rag(self, task: Task) -> str:
        """Retrieve relevant content from RAG for the task"""
        if self.scratchpad:
            return "\n\n".join(self.scratchpad[-5:])  # Last 5 entries
        return "(No previous findings)"
    
    def _searx_search(self, query: str) -> List[Dict]:
        """Execute web search via SearXNG"""
        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "safesearch": 1,
            "engines": "google",
            "categories": "general",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        
        try:
            res = requests.get(SEARXNG_URL, params=params, headers=headers, timeout=15)
            res.raise_for_status()
            data = res.json()
            
            return [
                {
                    "title": r.get("title", "")[:100],
                    "href": r.get("url", ""),
                    "body": r.get("content", "")[:300],
                }
                for r in data.get("results", [])[:self.config.search_results]
                if r.get("url")
            ]
        except Exception as e:
            if self.config.debug:
                print(f"Search error: {e}")
            return []
