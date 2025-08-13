from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Set, Deque
from collections import deque
from uuid import uuid4

import requests
from bs4 import BeautifulSoup

from .base import LlmWrapper
from memory.vector_store import MemoryStore
from dataclass.tasks import Task

# --------------------------------------------------------------------------- #
# 💬 Prompt fragments
# --------------------------------------------------------------------------- #

_PLANNER_SYS = """You are an autonomous research agent conducting deep investigations.

CORE RULES:
- Stay strictly on topic for the given task
- Probe deeply - don't settle for surface-level information
- If blocked, try alternative research paths
- If information is already known, find new information and pathways
- Build upon previous findings iteratively
- Do not make up information
- Do not have conversations. 
- All context is either your own words, or results from tools you called. 
- The date is August 10th, 2025

OUTPUT FORMAT (JSON only):
{
  "direction": "Your reasoning for the next action (1-2 sentences)",
  "action": "think" | "search" | "summarize",
  "query": "search terms (if action is 'search', else empty string)",
  "thought": "brief reflection (if action is 'think', else empty string)"
}

ACTIONS:
- think: Reason about findings and plan next steps
- search: Targeted queries for new information
- summarize: Checkpoint important findings"""

_SUMMARY_SYS = """Create concise research notes from the provided material.
Focus on key findings, data, and sources.
No commentary or questions."""

_CHECKPOINT_SYS = """Create a knowledge checkpoint summarizing current research progress.

Include:
- What has been discovered so far
- Key sources and evidence
- Current research focus
- Unexplored areas or questions

Be concise but comprehensive. This will serve as context for your future research steps.
This checkpoint is crucial to ensure that you do not repeat research and searches."""

_FINAL_REPORT_SYS = """Create a concise, fact-based technical report consolidating all research notes.  
Only include verifiable, relevant information. Do not add praise, opinions, speculative business logic, or self-referential commentary.

STRUCTURE:
1. Executive Summary — 2–3 sentences summarizing the most critical findings.  
2. Key Findings — Bullet points listing each major finding, written clearly and without interpretation beyond what the data supports.  
3. Supporting Evidence & Sources — Include links, citations, or exact references for each finding when available.  
4. Technical Details — Precise, relevant specifications, parameters, metrics, or methodologies.  
5. Next Steps / Gaps — Concrete unanswered questions or missing data points that should be addressed.

Rules:
- Be direct, concise, and data-focused.  
- Do not invent information or extrapolate beyond the provided evidence.  
- Avoid conversational tone, filler language, or subjective commentary.
- Add information, rather than removing it.
"""

INSTRUCTIONS = """
Write a highly detailed FINAL REPORT using ONLY the CONTEXT above. 
Revise and merge prior reports to incorporate new findings. 
Ensure that you add new information while revising old information. 
Do not remove useful and relevant information. Stay concise, but include all the information of previous reports in extreme detail.
No feedback, no praise, no questions, no requests for more info.
Do not preface with explanations. Output the report only.
"""

# --------------------------------------------------------------------------- #
# 🔧 Config
# --------------------------------------------------------------------------- #

THINK_LOOPS        = 6
SEARXNG_URL        = os.getenv("SEARXNG_URL", "http://localhost:8888/search")
SEARCH_RESULTS     = 8           # results to keep per query
FETCH_BYTES        = 30_000      # max bytes per fetched web page
DEBUG              = True       # set True for verbose output
CONTEXT_WINDOW     = 10          # max history entries in sliding window
CHECKPOINT_FREQ    = 5           # create checkpoint every N actions
MAX_SNIPPET_LENGTH = 2000        # max chars per web snippet

# --------------------------------------------------------------------------- #
# 🛠️  Helper functions
# --------------------------------------------------------------------------- #

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _json_from(text: str) -> Dict | None:
    """Return dict or None on failure; log the raw reply so we can debug."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if DEBUG:
            print("JSON decode error:", e)
            print("Raw reply:\n", text[:800])

        # Prefer a fenced ```json block if one exists
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # Non-greedy first-brace fallback
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    
    if DEBUG:
        print("No valid JSON found!")
    return None   

def _extract_text(url: str) -> str:
    """Download page & return readable text (best-effort)."""
    try:
        resp = requests.get(
            url, 
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
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

# --------------------------------------------------------------------------- #
# 📦  Per-task state container
# --------------------------------------------------------------------------- #

@dataclass
class TaskState:
    queries: Set[str]           = field(default_factory=set)
    snips: Deque[str]           = field(default_factory=lambda: deque(maxlen=10))
    history: Deque[str]         = field(default_factory=lambda: deque(maxlen=CONTEXT_WINDOW))
    action_count: int           = 0
    think_count: int            = 0  # Track think actions
    search_count: int           = 0  # Track searches
    summary_count: int          = 0  # Track summaries
    checkpoint_count: int       = 0  # Track checkpoints
    last_checkpoint: str        = ""
    checkpoints: List[str]      = field(default_factory=list)
    current_focus: str          = ""
    search_history: List[Dict]  = field(default_factory=list)  # Detailed search history

# --------------------------------------------------------------------------- #
# 🤖  ResearchAgent
# --------------------------------------------------------------------------- #

class ResearchAgent:
    """LLM-powered researcher with sliding window context and checkpoint-based memory."""

    def __init__(self, memory: MemoryStore, llm: LlmWrapper) -> None:
        self.memory = memory
        self.llm = llm
        self._state: Dict[str, TaskState] = {}
        
        # Public metrics for dashboard access
        self.metrics = {
            "total_llm_calls": 0,
            "total_searches": 0,
            "total_summaries": 0,
            "total_checkpoints": 0,
            "total_thoughts": 0,
            "total_web_fetches": 0,
            "total_search_results": 0
        }
    
    def get_task_metrics(self, task_name: str) -> Dict:
        """Get detailed metrics for a specific task."""
        if task_name not in self._state:
            return {
                "searches": [],
                "search_history": [],
                "action_count": 0,
                "search_count": 0,
                "think_count": 0,
                "summary_count": 0,
                "checkpoint_count": 0,
                "current_focus": ""
            }
        
        st = self._state[task_name]
        return {
            "searches": list(st.queries),
            "search_history": st.search_history,
            "action_count": st.action_count,
            "search_count": st.search_count,
            "think_count": st.think_count,
            "summary_count": st.summary_count,
            "checkpoint_count": st.checkpoint_count,
            "current_focus": st.current_focus
        }

    def run_step(self, task: Task) -> str:
        st = self._state.setdefault(task.name, TaskState())

        for loop_idx in range(THINK_LOOPS):
            # Build context with sliding window and checkpoint
            context = self._build_context(task, st)
            
            if DEBUG:
                print(f"\n[Loop {loop_idx+1}/{THINK_LOOPS}]")
                print(f"Context length: {len(context)} chars")

            plan_raw = self._chat(
                [{"role": "system", "content": _PLANNER_SYS},
                 {"role": "user", "content": context}]
            )
            
            # Track LLM call
            self.metrics["total_llm_calls"] += 1
            
            plan = _json_from(plan_raw)
            if not plan:
                print("[WARN] Invalid JSON response, defaulting to think")
                plan = {"action": "think", "thought": "Processing..."}
            
            action: Literal["think", "search", "summarize"] = plan.get("action", "think")
            st.action_count += 1

            print(f"[{task.name}] Action {st.action_count}: {action}")

            if action == "think":
                thought = plan.get("thought", "").strip()
                if not thought:
                    thought = "Analyzing current information..."
                print(f"  Thinking: {thought[:100]}...")
                st.history.append(f"[THOUGHT] {thought}")
                st.current_focus = plan.get("direction", st.current_focus)
                
                # Track metrics
                st.think_count += 1
                self.metrics["total_thoughts"] += 1
                
            elif action == "search":
                query = plan.get("query", "").strip()
                if not query:
                    st.history.append("[SKIP] Empty search query")
                    continue
                    
                if query in st.queries:
                    st.history.append(f"[SKIP] Already searched: {query}")
                    continue
                
                # Execute search
                search_start = datetime.now()
                hits = self._searx_search(query)
                search_time = (datetime.now() - search_start).total_seconds()
                
                if not hits:
                    st.history.append(f"[NO RESULTS] {query}")
                    continue

                st.queries.add(query)
                print(f"  Searching: {query}")
                
                # Track search in detailed history
                search_entry = {
                    "timestamp": _now_iso(),
                    "query": query,
                    "results_count": len(hits),
                    "search_time": search_time,
                    "results": []
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
                    st.snips.append(snippet)
                    search_results.append(hit['title'])
                    
                    # Add to search entry
                    search_entry["results"].append({
                        "title": hit['title'],
                        "url": hit['href'],
                        "fetched": bool(body)
                    })
                
                st.search_history.append(search_entry)
                st.history.append(
                    f"[SEARCH] {query}\n"
                    f"  Found: {', '.join(search_results)}"
                )
                
                # Track metrics
                st.search_count += 1
                self.metrics["total_searches"] += 1
                self.metrics["total_search_results"] += len(hits)
                
            elif action == "summarize":
                material = "\n---\n".join(st.snips) if st.snips else "(no new material)"
                
                notes = self._chat(
                    [{"role": "system", "content": _SUMMARY_SYS},
                     {"role": "user", "content": f"Task: {task.user_prompt}\n\nMaterial:\n{material}"}]
                )
                
                # Track LLM call for summary
                self.metrics["total_llm_calls"] += 1
                
                st.history.append(f"[SUMMARY] {notes[:200]}...")
                self._add_to_rag(task, notes, "summary")
                st.snips.clear()
                
                # Track metrics
                st.summary_count += 1
                self.metrics["total_summaries"] += 1

            # Create checkpoint periodically
            if st.action_count % CHECKPOINT_FREQ == 0:
                self._create_checkpoint(task, st)

        # Final checkpoint and report
        if st.action_count % CHECKPOINT_FREQ != 0:
            self._create_checkpoint(task, st)
        
        final_report = self._generate_final_report(task, st)
        self._add_to_rag(task, final_report, "final_report")
        
        return final_report

    def _build_context(self, task: Task, state: TaskState) -> str:
        """Build context with checkpoint + sliding window."""
        context_parts = [f"Task: {task.user_prompt}"]
        
        # Add last checkpoint if available
        if state.last_checkpoint:
            context_parts.append(f"\nPrevious checkpoint:\n{state.last_checkpoint}")
        
        # Add current focus if set
        if state.current_focus:
            context_parts.append(f"\nCurrent focus: {state.current_focus}")
        
        # Add sliding window of recent history
        if state.history:
            recent_history = "\n".join(state.history)
            context_parts.append(f"\nRecent actions:\n{recent_history}")
        else:
            context_parts.append("\n[No actions yet]")
        
        return "\n".join(context_parts)

    def _create_checkpoint(self, task: Task, state: TaskState) -> None:
        """Create a checkpoint summarizing current progress."""
        print(f"[CHECKPOINT] Creating checkpoint at action {state.action_count}")
        
        # Gather recent context
        recent_context = "\n".join(state.history)
        
        checkpoint = self._chat(
            [{"role": "system", "content": _CHECKPOINT_SYS},
             {"role": "user", "content": (
                 f"Task: {task.user_prompt}\n\n"
                 f"Recent research:\n{recent_context}\n\n"
                 f"Previous checkpoint:\n{state.last_checkpoint or 'None'}"
             )}]
        )
        
        # Track LLM call
        self.metrics["total_llm_calls"] += 1
        
        state.last_checkpoint = checkpoint
        state.checkpoints.append(checkpoint)
        self._add_to_rag(task, checkpoint, "checkpoint")
        
        # Track metrics
        state.checkpoint_count += 1
        self.metrics["total_checkpoints"] += 1
        
        # Clear old history after checkpoint
        state.history.clear()
        state.history.append(f"[CHECKPOINT CREATED] Focus: {state.current_focus or 'General research'}")

    def _generate_final_report(self, task: Task, state: TaskState) -> str:
        """Generate final report from all checkpoints and RAG content."""
        # Retrieve relevant content from RAG
        rag_content = self._retrieve_from_rag(task)
        
        # Combine all checkpoints
        all_checkpoints = "\n\n---\n\n".join(state.checkpoints) if state.checkpoints else ""
        
        final_report = self._chat(
            [{"role": "system", "content": _FINAL_REPORT_SYS},
             {"role": "user", "content": (
                 f"Task: {task.user_prompt}\n\n"
                 f"Research checkpoints:\n{all_checkpoints}\n\n"
                 f"Additional findings:\n{rag_content}"
                 f"Instructions:\n{INSTRUCTIONS}"
             )},
             ]
        )
        
        # Track LLM call
        self.metrics["total_llm_calls"] += 1
        
        return final_report

    def _add_to_rag(self, task: Task, text: str, doc_type: str) -> None:
        """Add document to RAG with improved metadata."""
        doc_id = f"{task.name}-{doc_type}-{uuid4().hex[:8]}"
        metadata = {
            "task": task.name,
            "type": doc_type,
            "timestamp": _now_iso(),
            "length": len(text)
        }
        
        self.memory.add(doc_id, text, metadata)
        
        # Only add summaries and final reports to scratchpad
        if doc_type in ["summary", "final_report", "checkpoint"]:
            task.scratchpad.append(f"[{doc_type.upper()}] {text[:500]}...")

    def _retrieve_from_rag(self, task: Task) -> str:
        """Retrieve relevant content from RAG for the task."""
        # This would ideally use semantic search, but for now we'll use the scratchpad
        if task.scratchpad:
            return "\n\n".join(task.scratchpad[-5:])  # Last 5 entries
        return "(No previous findings)"

    def _searx_search(self, query: str) -> List[Dict]:
        """Execute web search via SearXNG."""
        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "safesearch": 1,
            "engines": "google",
            "categories": "general",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"
        }
        
        try:
            res = requests.get(
                SEARXNG_URL, 
                params=params,
                headers=headers, 
                timeout=15
            )
            res.raise_for_status()
            data = res.json()
            
            return [
                {
                    "title": r.get("title", "")[:100],
                    "href": r.get("url", ""),
                    "body": r.get("content", "")[:300],
                }
                for r in data.get("results", [])[:SEARCH_RESULTS]
                if r.get("url")
            ]
        except Exception as e:
            if DEBUG:
                print(f"Search error: {e}")
            return []

    def _chat(self, messages: List[Dict]) -> str:
        """Send chat request to LLM."""
        return self.llm.chat(messages)
    
    def get_global_metrics(self) -> Dict:
        """Get all global metrics for the dashboard."""
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all metrics (useful for new research sessions)."""
        self.metrics = {
            "total_llm_calls": 0,
            "total_searches": 0,
            "total_summaries": 0,
            "total_checkpoints": 0,
            "total_thoughts": 0,
            "total_web_fetches": 0,
            "total_search_results": 0
        }
        for state in self._state.values():
            state.action_count = 0
            state.think_count = 0
            state.search_count = 0
            state.summary_count = 0
            state.checkpoint_count = 0
            state.search_history.clear()