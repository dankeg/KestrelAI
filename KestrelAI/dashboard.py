"""
dashboard.py – Comprehensive Testing Dashboard for KestrelAI Research System

A standalone testing tool for the research orchestrator and model loop that provides
complete visibility into all system components without requiring the backend or frontend.

Run with:
  panel serve dashboard.py --autoreload --show

Features:
  • Real-time research plan display with subtask status and progress
  • Orchestrator state visibility (decisions, stuck detection, feedback history)
  • Per-subtask metrics (searches, thoughts, summaries, checkpoints, LLM calls)
  • Real-time metrics from all subtask agents
  • Beautiful kestrel-inspired design
  • Search history with details
  • Export functionality (includes research plans and orchestrator state)
  • Full report version browser (filter + picker + prev/next)
  • Error handling and display
  • Task lifecycle tracking
  • Activity feed with meaningful activity detection

Testing Capabilities:
  • Test orchestrator decision-making and state transitions
  • Monitor subtask progression and completion
  • Track agent metrics per subtask
  • Observe research plan generation and execution
  • Debug stuck states and loop detection
  • Verify search query collection and execution
  • Monitor LLM call patterns and efficiency

IMPORTANT: This is a Panel application. Run it with:
  panel serve dashboard.py --autoreload --show

Do NOT run it directly with python dashboard.py
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import html
import json
import logging
import os
import pathlib
import re
import threading
import time
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

# Disable ChromaDB telemetry BEFORE importing any KestrelAI modules
# This prevents telemetry errors from appearing in the logs
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

# -----------------------------------------------------------------------------
# Third-party deps
# -----------------------------------------------------------------------------
import panel as pn
from panel.widgets import Button

# -----------------------------------------------------------------------------
# KestrelAI modules
# -----------------------------------------------------------------------------
try:
    from agents.base import LlmWrapper
    from agents.research_orchestrator import ResearchOrchestrator, TaskState
    from memory.vector_store import MemoryStore
    from shared.models import Task, TaskStatus
except ImportError:
    from KestrelAI.agents.base import LlmWrapper
    from KestrelAI.agents.research_orchestrator import ResearchOrchestrator, TaskState
    from KestrelAI.memory.vector_store import MemoryStore
    from KestrelAI.shared.models import Task, TaskStatus

# -----------------------------------------------------------------------------
# Beautiful Kestrel-inspired CSS
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Kestrel-inspired color palette */
    :root {
        --feather-brown: #8B6F47;
        --feather-cream: #F5E6D3;
        --rust-orange: #D2691E;
        --slate-gray: #4A5568;
        --sky-blue: #87CEEB;
        --warm-white: #FAF8F3;
        --shadow-brown: #5C4A3A;
        --accent-gold: #DAA520;
        --soft-black: #2D3436;
        --text-dark: #1A202C;
        --text-medium: #4A5568;
    }

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    body {
        background: linear-gradient(135deg, #2D3436 0%, #4A5568 100%);
    }

    /* Beautiful card design with depth */
    .kestrel-card {
        background: linear-gradient(145deg, rgba(250, 248, 243, 0.98), rgba(250, 248, 243, 0.95));
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 20px;
        box-shadow: 
            0 20px 40px rgba(92, 74, 58, 0.15),
            0 10px 20px rgba(92, 74, 58, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(139, 111, 71, 0.1);
        position: relative;
        overflow: hidden;
    }

    .kestrel-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, 
            var(--rust-orange), 
            var(--accent-gold) 50%, 
            var(--sky-blue));
        opacity: 0.6;
    }

    .kestrel-card:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 25px 50px rgba(92, 74, 58, 0.2),
            0 15px 30px rgba(92, 74, 58, 0.15);
    }

    /* Typography with character - darker text */
    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: var(--text-dark);
        margin: 0 0 20px 0;
        letter-spacing: -0.5px;
    }

    /* Active task animation */
    @keyframes soar {
        0%, 100% { 
            transform: translateY(0px);
            box-shadow: 
                0 20px 40px rgba(210, 105, 30, 0.15),
                0 10px 20px rgba(210, 105, 30, 0.1);
        }
        50% { 
            transform: translateY(-3px);
            box-shadow: 
                0 25px 50px rgba(210, 105, 30, 0.2),
                0 15px 30px rgba(210, 105, 30, 0.15);
        }
    }

    .task-active {
        animation: soar 3s ease-in-out infinite;
        border: 2px solid rgba(210, 105, 30, 0.3);
        background: linear-gradient(145deg, 
            rgba(250, 248, 243, 1), 
            rgba(250, 248, 243, 0.98));
    }

    /* Beautiful status badges - softer gradients */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .status-active {
        background: linear-gradient(135deg, var(--rust-orange), rgba(218, 165, 32, 0.8));
        color: black;
        box-shadow: 0 3px 10px rgba(210, 105, 30, 0.25);
    }

    .status-active::before {
        content: '';
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10b981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(0.8); }
    }

    .status-pending {
        background: linear-gradient(135deg, #94a3b8, #cbd5e1);
        color: var(--text-dark);
    }

    .status-complete {
        background: linear-gradient(135deg, #87CEEB, #a8d8ea);
        color: var(--text-dark);
        box-shadow: 0 3px 10px rgba(135, 206, 235, 0.25);
    }

    .status-paused {
        background: linear-gradient(135deg, var(--feather-brown), rgba(139, 111, 71, 0.8));
        color: var(--warm-white);
    }

    /* Elegant progress bars - softer gradients */
    .progress-container {
        background: rgba(74, 85, 104, 0.08);
        border-radius: 12px;
        height: 10px;
        overflow: hidden;
        margin: 20px 0;
        position: relative;
    }

    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, 
            rgba(210, 105, 30, 0.9), 
            rgba(218, 165, 32, 0.7));
        border-radius: 12px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(210, 105, 30, 0.2);
    }

    .progress-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.2),
            transparent
        );
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 13px;
        color: var(--text-medium);
        font-weight: 500;
    }

    /* Beautiful metrics - better contrast */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 16px;
        margin-top: 24px;
    }

    .metric-card {
        background: linear-gradient(145deg, white, rgba(250, 248, 243, 0.5));
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        box-shadow: 
            0 4px 6px rgba(92, 74, 58, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(139, 111, 71, 0.05);
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: var(--rust-orange);
        margin-bottom: 4px;
    }

    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-medium);
        font-weight: 600;
    }

    /* Activity feed styling - darker text */
    .activity-item {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        padding: 14px;
        margin-bottom: 10px;
        background: linear-gradient(145deg, 
            rgba(255, 255, 255, 0.95), 
            rgba(250, 248, 243, 0.3));
        border-radius: 12px;
        border-left: 3px solid rgba(210, 105, 30, 0.5);
        box-shadow: 0 2px 4px rgba(92, 74, 58, 0.08);
        font-size: 14px;
        color: var(--text-dark);
        animation: slideIn 0.4s ease;
    }

    @keyframes slideIn {
        from { 
            opacity: 0;
            transform: translateX(-20px);
        }
        to { 
            opacity: 1;
            transform: translateX(0);
        }
    }

    .activity-time {
        color: var(--rust-orange);
        font-family: 'SF Mono', 'Monaco', monospace;
        font-size: 12px;
        font-weight: 600;
        min-width: 65px;
    }

    /* Search history cards - better contrast */
    .search-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.9), 
            rgba(250, 248, 243, 0.4));
        border: 1px solid rgba(135, 206, 235, 0.15);
        border-radius: 12px;
        font-size: 13px;
        color: var(--text-dark);
        transition: all 0.3s;
    }

    .search-item:hover {
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.95), 
            rgba(250, 248, 243, 0.5));
        transform: translateX(4px);
    }

    .search-task {
        color: #5090c0;
        font-weight: 600;
        min-width: 80px;
    }

    .search-results {
        color: var(--accent-gold);
        font-weight: 600;
        font-size: 11px;
    }

    /* Beautiful buttons - softer gradients */
    .kestrel-btn {
        padding: 10px 20px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 600;
        border: none;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .btn-pause {
        background: linear-gradient(135deg, var(--feather-brown), rgba(139, 111, 71, 0.85));
        color: var(--warm-white);
    }

    .btn-pause:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(139, 111, 71, 0.25);
    }

    .btn-export {
        background: linear-gradient(135deg, #87CEEB, #a8d8ea);
        color: var(--text-dark);
    }

    .btn-export:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(135, 206, 235, 0.25);
    }

    /* Header section - softer gradient */
    .header-section {
        background: linear-gradient(135deg, 
            rgba(139, 111, 71, 0.95), 
            rgba(210, 105, 30, 0.85));
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 
            0 20px 40px rgba(92, 74, 58, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .header-section::after {
        content: '🦅';
        position: absolute;
        right: 30px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 48px;
        opacity: 0.2;
    }

    .header-title {
        font-size: 32px;
        font-weight: 800;
        color: var(--warm-white);
        margin: 0 0 8px 0;
        letter-spacing: -1px;
    }

    .header-subtitle {
        color: rgba(250, 248, 243, 0.95);
        font-size: 16px;
        font-weight: 500;
    }

    /* Runtime display */
    .runtime-display {
        font-family: 'SF Mono', 'Monaco', monospace;
        font-size: 24px;
        font-weight: 700;
        color: var(--rust-orange);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--feather-cream);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, 
            rgba(139, 111, 71, 0.6), 
            rgba(210, 105, 30, 0.4));
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(210, 105, 30, 0.6);
    }

    /* Research Plan Widget Styles */
    .research-plan-card {
        background: linear-gradient(145deg, rgba(250, 248, 243, 0.98), rgba(250, 248, 243, 0.95));
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 
            0 20px 40px rgba(92, 74, 58, 0.15),
            0 10px 20px rgba(92, 74, 58, 0.1);
        border: 1px solid rgba(139, 111, 71, 0.1);
    }

    .subtask-item {
        padding: 16px;
        margin-bottom: 12px;
        border-radius: 12px;
        border-left: 4px solid;
        transition: all 0.3s;
    }

    .subtask-pending {
        background: linear-gradient(135deg, rgba(148, 163, 184, 0.1), rgba(203, 213, 225, 0.05));
        border-left-color: #94a3b8;
    }

    .subtask-in-progress {
        background: linear-gradient(135deg, rgba(210, 105, 30, 0.15), rgba(218, 165, 32, 0.1));
        border-left-color: var(--rust-orange);
        box-shadow: 0 4px 12px rgba(210, 105, 30, 0.2);
        animation: pulse-border 2s infinite;
    }

    @keyframes pulse-border {
        0%, 100% { border-left-width: 4px; }
        50% { border-left-width: 6px; }
    }

    .subtask-completed {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.05));
        border-left-color: #10b981;
    }

    .subtask-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        font-weight: 700;
        font-size: 14px;
        margin-right: 12px;
    }

    .subtask-number-pending {
        background: #94a3b8;
        color: white;
    }

    .subtask-number-in-progress {
        background: var(--rust-orange);
        color: white;
        animation: pulse 2s infinite;
    }

    .subtask-number-completed {
        background: #10b981;
        color: white;
    }

    .subtask-number-completed::before {
        content: '✓';
    }

    .subtask-description {
        font-weight: 600;
        color: var(--text-dark);
        margin-bottom: 8px;
        font-size: 14px;
    }

    .subtask-criteria {
        font-size: 12px;
        color: var(--text-medium);
        margin-top: 6px;
        padding-left: 40px;
        font-style: italic;
    }

    .subtask-metrics {
        display: flex;
        gap: 12px;
        margin-top: 8px;
        padding-left: 40px;
        flex-wrap: wrap;
    }

    .subtask-metric {
        font-size: 11px;
        color: var(--text-medium);
        background: rgba(255, 255, 255, 0.5);
        padding: 4px 8px;
        border-radius: 6px;
    }

    .orchestrator-state {
        background: linear-gradient(135deg, rgba(135, 206, 235, 0.1), rgba(168, 216, 234, 0.05));
        border: 1px solid rgba(135, 206, 235, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
    }

    .orchestrator-state-item {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        font-size: 13px;
        border-bottom: 1px solid rgba(139, 111, 71, 0.1);
    }

    .orchestrator-state-item:last-child {
        border-bottom: none;
    }

    .orchestrator-state-label {
        color: var(--text-medium);
        font-weight: 500;
    }

    .orchestrator-state-value {
        color: var(--text-dark);
        font-weight: 600;
    }

    .stuck-warning {
        color: #ef4444;
        font-weight: 700;
    }

    .feedback-item {
        font-size: 11px;
        color: var(--text-medium);
        padding: 4px 8px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 6px;
        margin: 2px 0;
    }
</style>
"""

# -----------------------------------------------------------------------------
# Task configuration
# -----------------------------------------------------------------------------
ML_TASK = """Find currently open grants, programs, fellowships, or funding opportunities that
support AI/ML research and are available to senior undergraduate students in the
United States. Include name, eligibility, what it funds, deadline, and link. Focus
on fresh, prestigious, student-accessible opportunities not locked to one college.
Include opportunities from companies like Google, Microsoft, Meta, OpenAI, Anthropic, Cohere.
The current date is August 2025, find programs open for application."""

AI_CONFERENCES_CALL_FOR_ABSTRACTS_TASK = """Find AI/ML conferences, symposia, workshops, or student research programs that are currently accepting abstract submissions (not full-paper–only). For each opportunity, include: conference name, organizer/society, location & event dates, scope/topics, submission type (talk/poster/workshop) and abstract length/format, eligibility (explicitly note if senior undergraduates can submit), 
important dates (abstract deadline with timezone, notification date, and any later full/extended-abstract deadline), review model (single/double-blind), submission/CFP link, template/guidelines link, student registration fee (if published), whether accepted abstracts are published/archived (e.g., proceedings/indexing), any travel support, and key restrictions.
Prioritize reputable, broadly accessible venues and student programs tied to major conferences/societies (AAAI, NeurIPS, ICML, ICLR, KDD, ACL/EMNLP, CVPR/ICCV workshops & student programs, MICCAI, IEEE/ACM technical societies) plus interdisciplinary venues that welcome AI work. Exclude calls that are closed, invitation-only, or restricted to a single institution. Return only calls that are still open.
"""

ML_COMPETITIONS_TASK = """Find active AI/ML student competitions or challenges suitable for senior undergraduates in the United States. Include organizer, problem theme, eligibility, prize(s) or publication opportunities, important dates (registration close, submission due), compute support (if any), and link. Prioritize well-known organizers (Kaggle, Google, Microsoft, Meta, OpenAI, Anthropic) and academic conferences. Exclude archived or invitation-only events."""

# Start with empty tasks - user will create tasks via the UI
tasks: list[Task] = []

# -----------------------------------------------------------------------------
# Initialize backend
# -----------------------------------------------------------------------------
mem = MemoryStore()
llm = LlmWrapper(model="gemma3:27b")
# Note: We don't need a separate research agent - the orchestrator handles research internally
# Initialize orchestrator with empty tasks, will be updated when tasks are added
orch = ResearchOrchestrator([], llm, profile="kestrel")


# -----------------------------------------------------------------------------
# State management
# -----------------------------------------------------------------------------
class DashboardState:
    def __init__(self):
        self.current_task: str = "No task selected"
        self.current_task_obj: Task = None
        self.task_start: float = time.time()
        self.latest_notes: str = ""
        self.latest_feedback: str = "Create a task to begin research"
        self.latest_subtask: str = ""
        self.is_paused: bool = False
        self.start_time: float = time.time()
        self.task_running: bool = False  # Track if a task is actively running

        # Store all report versions
        self.report_history: list[dict] = []  # List of {timestamp, task, content}
        self.current_report_index: int = -1  # Index of currently viewed report

        # Store research plans per task
        self.research_plans: dict[str, dict] = {}  # task_name -> research plan data

        # Store orchestrator decisions and state
        self.orchestrator_state: dict[str, dict] = {}  # task_name -> orchestrator state

        # Task history will be populated as tasks are created
        self.task_history: dict[str, dict] = {}

        self.activity_log: deque = deque(maxlen=50)
        self.search_history: deque = deque(maxlen=100)

        # Real metrics from agent
        self.total_llm_calls: int = 0
        self.total_searches: int = 0
        self.total_summaries: int = 0
        self.total_checkpoints: int = 0
        self.total_web_fetches: int = 0

        self.last_update: float = time.time()

    def add_report(self, task_name: str, content: str):
        """Add a new report to history"""
        self.report_history.append(
            {"timestamp": datetime.now(), "task": task_name, "content": content}
        )
        self.current_report_index = len(self.report_history) - 1

    def get_current_report(self) -> dict:
        """Get the currently selected report"""
        # Ensure current_report_index is an integer
        idx = self.current_report_index
        if isinstance(idx, tuple):
            idx = idx[1] if len(idx) > 1 else 0

        if 0 <= idx < len(self.report_history):
            return self.report_history[idx]
        elif self.report_history:
            return self.report_history[-1]
        return None

    def navigate_report(self, direction: int):
        """Navigate through report history (-1 for previous, 1 for next)"""
        if not self.report_history:
            return

        new_index = self.current_report_index + direction
        if 0 <= new_index < len(self.report_history):
            self.current_report_index = new_index

    def get_runtime(self) -> str:
        """Get total runtime formatted as HH:MM:SS"""
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def export_state(self, include_full_reports: bool = False) -> dict:
        """Export current state as JSON"""
        return {
            "timestamp": datetime.now().isoformat(),
            "runtime": self.get_runtime(),
            "current_task": self.current_task,
            "metrics": {
                "total_llm_calls": self.total_llm_calls,
                "total_searches": self.total_searches,
                "total_summaries": self.total_summaries,
                "total_checkpoints": self.total_checkpoints,
                "total_web_fetches": self.total_web_fetches,
            },
            "tasks": {
                name: {
                    "status": info["status"],
                    "progress": info["progress"],
                    "elapsed": info["elapsed"],
                    "searches": info["searches"],
                    "action_count": info["action_count"],
                    "think_count": info["think_count"],
                    "search_count": info["search_count"],
                    "summary_count": info["summary_count"],
                    "subtask_metrics": info.get("subtask_metrics", {}),
                }
                for name, info in self.task_history.items()
            },
            "research_plans": self.research_plans,
            "orchestrator_state": self.orchestrator_state,
            "search_history": list(self.search_history)[-50:],
            "report_history": [
                {
                    "timestamp": r["timestamp"].isoformat(),
                    "task": r["task"],
                    "content": (
                        r["content"] if include_full_reports else r["content"][:1000]
                    ),
                }
                for r in self.report_history
            ],
        }


state = DashboardState()


# -----------------------------------------------------------------------------
# Orchestration loop (CORRECTED FOR STRING-BASED TASK TRACKING)
# -----------------------------------------------------------------------------
def run_async_in_thread(coro):
    """Run async function in a thread with its own event loop"""
    import asyncio

    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            loop.close()
            return result
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def orchestration_loop():
    pathlib.Path("notes").mkdir(exist_ok=True)

    while True:
        try:
            # Wait for a task to be started
            if not state.task_running or state.is_paused:
                time.sleep(0.5)
                continue

            # Initialize current task if not set
            if not hasattr(orch, "current") or orch.current is None:
                if tasks and len(tasks) > 0:
                    orch.current = tasks[0].name
                else:
                    time.sleep(1)
                    continue

            # Check if current task exists
            if orch.current not in orch.tasks:
                time.sleep(1)
                continue

            # Do planning phase if not already done
            if orch.current and orch.current in orch.tasks:
                first_task = orch.tasks[orch.current]

                # Ensure TaskState exists
                if orch.current not in orch.task_states:
                    orch.task_states[orch.current] = TaskState(first_task)

                # Check if planning is already done
                task_state = orch.task_states[orch.current]
                if task_state.research_plan:
                    # Planning already done, skip
                    pass
                else:
                    # Do planning - wait for it to complete
                    try:
                        state.latest_feedback = (
                            f"🧠 Generating research plan for: {first_task.name}..."
                        )
                        state.activity_log.append(
                            {
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "analysis",
                                "message": f"🧠 Planning research strategy for: {first_task.name}",
                            }
                        )
                        # Run planning and wait for completion
                        # This is a blocking call that waits for the async function to complete
                        run_async_in_thread(orch._planning_phase(first_task))

                        # Verify planning completed successfully
                        if task_state.research_plan:
                            logger.info(f"Planning completed for {first_task.name}")
                        else:
                            logger.warning(
                                f"Planning may not have completed for {first_task.name}"
                            )
                            # Give it another moment and check again
                            time.sleep(1)
                    except Exception as e:
                        error_msg = f"Error in planning phase: {e}"
                        print(error_msg)
                        import traceback

                        traceback.print_exc()
                        state.activity_log.append(
                            {
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": "error",
                                "message": f"❌ Planning error: {str(e)[:100]}",
                            }
                        )
                        # Continue anyway - the loop will retry
                        time.sleep(1)
                        continue

                # Capture research plan after planning
                if (
                    hasattr(orch, "task_states")
                    and orch.task_states
                    and orch.current in orch.task_states
                ):
                    task_state = orch.task_states[orch.current]
                    if task_state.research_plan and task_state.research_plan.subtasks:
                        state.latest_subtask = task_state.research_plan.subtasks[
                            0
                        ].description

                        # Check if this is the first time we're capturing the plan
                        plan_already_captured = orch.current in state.research_plans

                        # Capture initial research plan
                        plan_data = {
                            "restated_task": task_state.research_plan.restated_task,
                            "subtasks": [
                                {
                                    "order": st.order,
                                    "description": st.description,
                                    "success_criteria": st.success_criteria,
                                    "status": "pending" if i > 0 else "in_progress",
                                }
                                for i, st in enumerate(
                                    task_state.research_plan.subtasks
                                )
                            ],
                            "current_subtask_index": 0,
                        }
                        state.research_plans[orch.current] = plan_data

                        # Update feedback to show planning is complete (only first time)
                        if not plan_already_captured:
                            subtask_count = len(task_state.research_plan.subtasks)
                            state.latest_feedback = (
                                f"✅ Research plan generated!\n\n"
                                f"Task: {task_state.research_plan.restated_task}\n\n"
                                f"Plan includes {subtask_count} research steps.\n"
                                f"Starting with: {state.latest_subtask}"
                            )
                            state.activity_log.append(
                                {
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "type": "analysis",
                                    "message": f"📋 Research plan generated with {subtask_count} steps",
                                }
                            )
                    else:
                        state.latest_subtask = "Initial research"

            # Main execution loop - process research steps
            if state.is_paused:
                time.sleep(0.5)
                continue

            # Check if current is valid (it's a string task name)
            if orch.current is None or orch.current not in orch.tasks:
                time.sleep(1)
                continue

            # Check if planning is complete before executing research steps
            if (
                orch.current in orch.task_states
                and orch.task_states[orch.current].research_plan is None
            ):
                # Planning not complete yet, wait a bit and continue
                time.sleep(0.5)
                continue

            task = orch.tasks[orch.current]

            # Update task status
            task.status = TaskStatus.ACTIVE

            # Handle task switching
            if state.current_task != task.name:
                if (
                    state.current_task != "No task selected"
                    and state.current_task != "Initializing..."
                    and state.current_task in state.task_history
                ):
                    state.task_history[state.current_task]["status"] = "complete"
                    state.task_history[state.current_task]["end_time"] = time.time()

                state.current_task = task.name
                state.current_task_obj = task
                state.task_start = time.time()

                # Initialize task history if not exists
                if task.name not in state.task_history:
                    state.task_history[task.name] = {
                        "status": "active",
                        "elapsed": 0,
                        "progress": 0,
                        "notes": "",
                        "searches": [],
                        "action_count": 0,
                        "think_count": 0,
                        "search_count": 0,
                        "summary_count": 0,
                        "checkpoint_count": 0,
                        "last_action": "",
                        "start_time": time.time(),
                        "end_time": None,
                        "subtask_metrics": {},
                    }
                else:
                    state.task_history[task.name]["status"] = "active"
                    if state.task_history[task.name]["start_time"] is None:
                        state.task_history[task.name]["start_time"] = time.time()

                # Get current subtask from orchestrator
                try:
                    if task.name in orch.task_states:
                        task_state = orch.task_states[task.name]
                        if (
                            task_state.research_plan
                            and task_state.research_plan.subtasks
                        ):
                            if task_state.subtask_index < len(
                                task_state.research_plan.subtasks
                            ):
                                current_subtask = task_state.research_plan.subtasks[
                                    task_state.subtask_index
                                ]
                                state.latest_subtask = current_subtask.description
                            else:
                                state.latest_subtask = "Finalizing research"
                        else:
                            state.latest_subtask = "Initial research"
                    else:
                        state.latest_subtask = "Initial research"
                except Exception:
                    state.latest_subtask = "Initial research"

                state.activity_log.append(
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": "task_start",
                        "message": f"🦅 Started: {str(task.name).title()}",
                    }
                )
                state.last_update = time.time()

            # Execute research step using orchestrator
            # The orchestrator handles research internally with subtask-specific agents
            try:
                research_result = run_async_in_thread(orch.next_action(task))
                if research_result:
                    state.latest_feedback = research_result
                else:
                    research_result = "No result from orchestrator"
                    state.latest_feedback = research_result
            except Exception as e:
                error_msg = f"Error executing orchestrator step: {e}"
                print(error_msg)
                import traceback

                traceback.print_exc()
                research_result = f"Processing research step - {str(e)[:100]}"
                state.latest_feedback = research_result
                state.activity_log.append(
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": "error",
                        "message": f"❌ Orchestrator error: {str(e)[:100]}",
                    }
                )

            # Get the latest notes from the orchestrator's current subtask
            try:
                if task.name in orch.task_states:
                    task_state = orch.task_states[task.name]
                    if task_state.research_plan and task_state.research_plan.subtasks:
                        if task_state.subtask_index < len(
                            task_state.research_plan.subtasks
                        ):
                            current_subtask = task_state.research_plan.subtasks[
                                task_state.subtask_index
                            ]
                            state.latest_subtask = current_subtask.description
                        else:
                            state.latest_subtask = "Finalizing research"
            except Exception:
                # If we can't get current subtask, continue with previous value
                pass

            # The research_result contains the actual research output with [SEARCH], [THOUGHT], etc.
            notes = research_result if research_result else ""

            if notes and len(notes) > 10:
                state.latest_notes = notes

                # Store report in history
                state.add_report(task.name, notes)

                # Get REAL metrics from orchestrator's subtask agents
                try:
                    progress_info = orch.get_task_progress(task.name)
                    if not progress_info:
                        progress_info = {
                            "progress": 0.0,
                            "subtasks": [],
                            "completed": 0,
                            "total": 0,
                        }
                except Exception as e:
                    print(f"Error getting task progress: {e}")
                    progress_info = {
                        "progress": 0.0,
                        "subtasks": [],
                        "completed": 0,
                        "total": 0,
                    }

                # Capture research plan if available
                try:
                    if (
                        hasattr(orch, "task_states")
                        and orch.task_states
                        and task.name in orch.task_states
                    ):
                        task_state = orch.task_states[task.name]
                        if task_state and task_state.research_plan:
                            # Store research plan data
                            plan_data = {
                                "restated_task": task_state.research_plan.restated_task,
                                "subtasks": [
                                    {
                                        "order": st.order,
                                        "description": st.description,
                                        "success_criteria": st.success_criteria,
                                        "status": (
                                            "completed"
                                            if i in task_state.completed_subtasks
                                            else (
                                                "in_progress"
                                                if i == task_state.subtask_index
                                                else "pending"
                                            )
                                        ),
                                    }
                                    for i, st in enumerate(
                                        task_state.research_plan.subtasks
                                    )
                                ],
                                "current_subtask_index": task_state.subtask_index,
                            }
                            state.research_plans[task.name] = plan_data

                            # Store orchestrator state
                            state.orchestrator_state[task.name] = {
                                "decision_count": task_state.decision_count,
                                "stuck_count": task_state.stuck_count,
                                "last_decision": task_state.last_decision,
                                "feedback_history": task_state.feedback_history[
                                    -5:
                                ],  # Last 5
                                "completed_subtasks": list(
                                    task_state.completed_subtasks
                                ),
                                "subtask_index": task_state.subtask_index,
                                "total_findings": (
                                    len(task_state.all_findings)
                                    if hasattr(task_state, "all_findings")
                                    else 0
                                ),
                            }
                except Exception:
                    # Silently continue if we can't capture orchestrator state
                    pass

                # Aggregate metrics from all subtask agents
                total_searches = 0
                total_thinks = 0
                total_summaries = 0
                total_checkpoints = 0
                total_actions = 0
                search_queries = []
                subtask_metrics_dict = {}

                for subtask_info in progress_info.get("subtasks", []):
                    # Safely get subtask index (order is 1-based, convert to 0-based)
                    order = subtask_info.get("order", 0)
                    if order < 1:
                        continue  # Skip invalid orders
                    subtask_idx = order - 1  # Convert to 0-based

                    if "agent_metrics" in subtask_info:
                        metrics = subtask_info["agent_metrics"]
                        if metrics:  # Ensure metrics is not None
                            total_searches += metrics.get("total_searches", 0)
                            total_thinks += metrics.get("total_thoughts", 0)
                            total_summaries += metrics.get("total_summaries", 0)
                            total_checkpoints += metrics.get("total_checkpoints", 0)
                            total_actions += metrics.get("total_llm_calls", 0)

                            # Store per-subtask metrics
                            subtask_metrics_dict[subtask_idx] = {
                                "searches": metrics.get("total_searches", 0),
                                "thoughts": metrics.get("total_thoughts", 0),
                                "summaries": metrics.get("total_summaries", 0),
                                "checkpoints": metrics.get("total_checkpoints", 0),
                                "llm_calls": metrics.get("total_llm_calls", 0),
                            }

                    # Get search queries from subtask agent's task metrics (not global metrics)
                    try:
                        if (
                            hasattr(orch, "task_states")
                            and orch.task_states
                            and task.name in orch.task_states
                        ):
                            task_state = orch.task_states[task.name]
                            if (
                                task_state
                                and hasattr(task_state, "subtask_agents")
                                and subtask_idx in task_state.subtask_agents
                            ):
                                agent = task_state.subtask_agents[subtask_idx]
                                if agent and hasattr(agent, "get_task_metrics"):
                                    task_metrics = agent.get_task_metrics(task.name)
                                    if task_metrics:
                                        agent_search_queries = task_metrics.get(
                                            "searches", []
                                        )
                                        if agent_search_queries:
                                            # Only add unique queries
                                            for query in agent_search_queries:
                                                if (
                                                    query
                                                    and query not in search_queries
                                                ):
                                                    search_queries.append(query)
                    except Exception:
                        # Silently continue if we can't get search queries
                        pass

                # Update global metrics
                state.total_llm_calls = total_actions
                state.total_searches = total_searches
                state.total_summaries = total_summaries
                state.total_checkpoints = total_checkpoints
                state.total_web_fetches = progress_info.get("web_fetches", 0)

                # Update task-specific metrics (with safety check)
                if task.name in state.task_history:
                    task_info = state.task_history[task.name]
                    task_info["searches"] = search_queries
                    task_info["search_count"] = total_searches
                    task_info["think_count"] = total_thinks
                    task_info["summary_count"] = total_summaries
                    task_info["checkpoint_count"] = total_checkpoints
                    task_info["action_count"] = total_actions
                    task_info["subtask_metrics"] = subtask_metrics_dict

                # Update search history with real search data (avoid duplicates)
                for search_query in search_queries:
                    if search_query and isinstance(search_query, str):
                        # Check if this query already exists for this task
                        if not any(
                            s.get("query") == search_query
                            and s.get("task") == task.name
                            for s in state.search_history
                        ):
                            state.search_history.append(
                                {
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "task": task.name,
                                    "query": search_query,
                                    "results_count": 4,
                                }
                            )

                # Check for meaningful activity by examining the orchestrator's current subtask agent
                has_meaningful_activity = False
                activity_type = "analysis"
                message = f"⚙️ Working on: {current_subtask or 'research'}"

                # Get task_info for activity checking (must be after metrics update)
                task_info_for_activity = state.task_history.get(task.name, {})

                try:
                    if (
                        hasattr(orch, "task_states")
                        and orch.task_states
                        and task.name in orch.task_states
                    ):
                        task_state = orch.task_states[task.name]
                        if (
                            task_state
                            and hasattr(task_state, "subtask_agents")
                            and hasattr(task_state, "subtask_index")
                        ):
                            if task_state.subtask_index in task_state.subtask_agents:
                                current_agent = task_state.subtask_agents[
                                    task_state.subtask_index
                                ]
                                if (
                                    current_agent
                                    and hasattr(current_agent, "_state")
                                    and task.name in current_agent._state
                                ):
                                    agent_state = current_agent._state[task.name]
                                    if agent_state:
                                        # Check if there's new activity since last check
                                        last_action_count = task_info_for_activity.get(
                                            "last_agent_action_count", 0
                                        )
                                        if agent_state.action_count > last_action_count:
                                            has_meaningful_activity = True

                                            # Determine activity type based on recent actions
                                            if (
                                                agent_state.search_count
                                                > task_info_for_activity.get(
                                                    "last_search_count", 0
                                                )
                                            ):
                                                activity_type = "search"
                                                message = f"🔍 Searching: {current_subtask or 'research'}"
                                            elif (
                                                agent_state.think_count
                                                > task_info_for_activity.get(
                                                    "last_think_count", 0
                                                )
                                            ):
                                                activity_type = "thinking"
                                                message = f"🤔 Analyzing: {current_subtask or 'research'}"
                                            elif (
                                                agent_state.summary_count
                                                > task_info_for_activity.get(
                                                    "last_summary_count", 0
                                                )
                                            ):
                                                activity_type = "summary"
                                                message = "📝 Summarizing findings"
                                            elif (
                                                agent_state.checkpoint_count
                                                > task_info_for_activity.get(
                                                    "last_checkpoint_count", 0
                                                )
                                            ):
                                                activity_type = "checkpoint"
                                                message = "💾 Saving progress"

                                            # Update tracking (use the actual task_info from state)
                                            if task.name in state.task_history:
                                                task_info = state.task_history[
                                                    task.name
                                                ]
                                                task_info[
                                                    "last_agent_action_count"
                                                ] = agent_state.action_count
                                                task_info[
                                                    "last_search_count"
                                                ] = agent_state.search_count
                                                task_info[
                                                    "last_think_count"
                                                ] = agent_state.think_count
                                                task_info[
                                                    "last_summary_count"
                                                ] = agent_state.summary_count
                                                task_info[
                                                    "last_checkpoint_count"
                                                ] = agent_state.checkpoint_count
                except Exception:
                    # Silently continue if we can't check activity
                    pass

                # Update common fields (always update progress)
                elapsed = time.time() - state.task_start
                progress = min(100, (elapsed / (task.budgetMinutes * 60)) * 100)

                # Safely update task info
                if task.name in state.task_history:
                    task_info = state.task_history[task.name]
                    task_info.update(
                        {
                            "elapsed": elapsed,
                            "progress": progress,
                            "notes": notes[:1000] if notes else "",
                            "last_action": datetime.now().strftime("%H:%M:%S"),
                        }
                    )

                    # Only send activity if there's actual meaningful activity
                    if has_meaningful_activity:
                        task_info["action_count"] = task_info.get("action_count", 0) + 1

                    # Log activity with meaningful activity
                    if has_meaningful_activity:
                        state.activity_log.append(
                            {
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "type": activity_type,
                                "message": message,
                            }
                        )

                        # Only save notes and generate reports for meaningful research findings
                        if activity_type in ["summary", "checkpoint"] or (
                            notes and len(notes) > 200
                        ):
                            # Save notes
                            safe_name = "".join(
                                c if c.isalnum() or c in (" ", "-", "_") else "_"
                                for c in task.name
                            ).strip()
                            try:
                                with open(
                                    f"notes/{safe_name.upper()}.txt",
                                    "w",
                                    encoding="utf-8",
                                ) as fh:
                                    fh.write(notes)
                            except Exception as e:
                                print(f"Error saving notes: {e}")

                    state.last_update = time.time()

                # Check if task is complete based on orchestrator state or time budget
                orchestrator_progress = (
                    progress_info.get("progress", 0) if progress_info else 0
                )
                if (
                    task.status == TaskStatus.COMPLETE
                    or progress >= 100.0
                    or orchestrator_progress >= 100.0
                ):
                    task.status = TaskStatus.COMPLETE
                    if task.name in state.task_history:
                        state.task_history[task.name]["status"] = "complete"
                        state.task_history[task.name]["end_time"] = time.time()

                    # Move to next task (orch.current is a string task name)
                    # Find current task index and move to next
                    task_names = [t.name for t in tasks]
                    current_idx = (
                        task_names.index(orch.current)
                        if orch.current in task_names
                        else -1
                    )

                    if current_idx >= 0 and current_idx < len(task_names) - 1:
                        orch.current = task_names[current_idx + 1]
                    else:
                        orch.current = None

                    # Plan for next task if available
                    if orch.current is not None and orch.current in orch.tasks:
                        next_task = orch.tasks[orch.current]
                        try:
                            run_async_in_thread(orch._planning_phase(next_task))
                        except Exception as e:
                            print(f"Error planning next task: {e}")
                            import traceback

                            traceback.print_exc()
                            state.activity_log.append(
                                {
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "type": "error",
                                    "message": f"❌ Planning error: {str(e)[:100]}",
                                }
                            )
                        try:
                            if (
                                hasattr(orch, "task_states")
                                and orch.task_states
                                and orch.current in orch.task_states
                            ):
                                task_state = orch.task_states[orch.current]
                                if task_state and (
                                    task_state.research_plan
                                    and task_state.research_plan.subtasks
                                ):
                                    state.latest_subtask = (
                                        task_state.research_plan.subtasks[0].description
                                    )
                                    # Capture research plan for next task
                                    plan_data = {
                                        "restated_task": task_state.research_plan.restated_task,
                                        "subtasks": [
                                            {
                                                "order": st.order,
                                                "description": st.description,
                                                "success_criteria": st.success_criteria,
                                                "status": (
                                                    "pending"
                                                    if i > 0
                                                    else "in_progress"
                                                ),
                                            }
                                            for i, st in enumerate(
                                                task_state.research_plan.subtasks
                                            )
                                        ],
                                        "current_subtask_index": 0,
                                    }
                                    state.research_plans[orch.current] = plan_data
                        except Exception as e:
                            print(f"Error capturing research plan for next task: {e}")

            time.sleep(0.5)

        except KeyboardInterrupt:
            print("Orchestration loop interrupted")
            break
        except Exception as e:
            error_msg = f"Error in orchestration loop: {e}"
            print(error_msg)
            import traceback

            traceback.print_exc()
            try:
                state.activity_log.append(
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": "error",
                        "message": f"❌ Error: {str(e)[:100]}",
                    }
                )
                state.last_update = time.time()
            except Exception:
                # If we can't log, just continue
                pass
            time.sleep(2)  # Wait before retrying to avoid tight error loops


# Start background thread
thread = threading.Thread(
    target=orchestration_loop, daemon=True, name="OrchestratorThread"
)
thread.start()

# -----------------------------------------------------------------------------
# Panel UI
# -----------------------------------------------------------------------------
pn.extension(notifications=True)

TEMPLATE = pn.template.FastListTemplate(
    title="🦅 KestrelAI Research System",
    theme="default",
    accent_base_color="#D2691E",
    header_background="#8B6F47",
    sidebar_width=500,
)

pn.config.raw_css.append(CUSTOM_CSS)

# Header
header_pane = pn.pane.HTML(
    """
    <div class="header-section">
        <h1 class="header-title">KestrelAI Research Center</h1>
        <p class="header-subtitle">Autonomous Multi-Agent Intelligence System</p>
    </div>
    """,
    sizing_mode="stretch_width",
)

# Task creation form widgets (defined before functions that use them)
task_name_input = pn.widgets.TextInput(
    name="Task Name",
    placeholder="e.g., AI Research Grants",
    width=400,
    value="",  # Explicitly set empty string as default
)

task_description_input = pn.widgets.TextAreaInput(
    name="Task Description",
    placeholder="Describe what you want to research...",
    height=150,
    width=400,
)

task_budget_input = pn.widgets.IntInput(
    name="Budget (minutes)", value=180, start=1, end=1440, width=200
)


# Task creation and management
def create_task():
    """Create a new task from the form inputs"""
    # Get values and handle None cases
    task_name_raw = task_name_input.value
    task_description_raw = task_description_input.value
    budget_minutes = task_budget_input.value if task_budget_input.value else 180

    # Handle None values
    if task_name_raw is None:
        task_name_raw = ""
    if task_description_raw is None:
        task_description_raw = ""

    task_name = task_name_raw.strip() if isinstance(task_name_raw, str) else ""
    task_description = (
        task_description_raw.strip() if isinstance(task_description_raw, str) else ""
    )

    if not task_name:
        pn.state.notifications.error(
            "Task name is required. Please enter a task name.", duration=3000
        )
        return

    if not task_description:
        pn.state.notifications.error(
            "Task description is required. Please enter a task description.",
            duration=3000,
        )
        return

    # Create new task with validation
    try:
        new_task = Task(
            name=task_name, description=task_description, budgetMinutes=budget_minutes
        )
    except Exception as e:
        error_msg = f"Error creating task: {str(e)}"
        print(error_msg)
        pn.state.notifications.error(
            f"Failed to create task: {str(e)[:100]}", duration=5000
        )
        return

    # Add to tasks list
    tasks.append(new_task)

    # Update orchestrator with new task
    orch.tasks[new_task.name] = new_task

    # Initialize TaskState for the new task in orchestrator
    if new_task.name not in orch.task_states:
        orch.task_states[new_task.name] = TaskState(new_task)

    # Initialize task history
    state.task_history[new_task.name] = {
        "status": "pending",
        "elapsed": 0,
        "progress": 0,
        "notes": "",
        "searches": [],
        "action_count": 0,
        "think_count": 0,
        "search_count": 0,
        "summary_count": 0,
        "checkpoint_count": 0,
        "last_action": "",
        "start_time": None,
        "end_time": None,
        "subtask_metrics": {},
    }

    # Clear form
    task_name_input.value = ""
    task_description_input.value = ""
    task_budget_input.value = 180

    pn.state.notifications.success(
        f"Task '{task_name}' created successfully!", duration=3000
    )
    update_dashboard()


def start_task():
    """Start the selected task"""
    if not tasks:
        pn.state.notifications.error(
            "No tasks available. Please create a task first.", duration=3000
        )
        return

    # Get first task if none selected
    if not orch.current or orch.current not in orch.tasks:
        orch.current = tasks[0].name

    # Get the task object
    task = orch.tasks.get(orch.current)
    if not task:
        # Fallback to finding in tasks list
        task = next((t for t in tasks if t.name == orch.current), None)
        if not task:
            pn.state.notifications.error(
                f"Task '{orch.current}' not found", duration=3000
            )
            return

    # Update state immediately so dashboard shows the task
    state.current_task = task.name
    state.current_task_obj = task
    state.task_start = time.time()

    # Initialize task history if not exists
    if task.name not in state.task_history:
        state.task_history[task.name] = {
            "status": "active",
            "elapsed": 0,
            "progress": 0,
            "notes": "",
            "searches": [],
            "action_count": 0,
            "think_count": 0,
            "search_count": 0,
            "summary_count": 0,
            "checkpoint_count": 0,
            "last_action": "",
            "start_time": time.time(),
            "end_time": None,
            "subtask_metrics": {},
        }
    else:
        state.task_history[task.name]["status"] = "active"
        if state.task_history[task.name]["start_time"] is None:
            state.task_history[task.name]["start_time"] = time.time()

    # Start the task
    state.task_running = True
    state.is_paused = False

    # Set initial feedback to show task is starting
    state.latest_feedback = (
        f"🚀 Starting task: {task.name}\n\nInitializing research plan..."
    )
    state.latest_subtask = "Initializing research plan..."

    # Update UI buttons
    start_btn.disabled = True
    pause_btn.disabled = False

    state.activity_log.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": "control",
            "message": f"🚀 Started task: {orch.current}",
        }
    )

    pn.state.notifications.success(f"Task '{orch.current}' started!", duration=3000)
    update_dashboard()


# Control buttons
def pause_resume():
    state.is_paused = not state.is_paused
    pause_btn.name = "▶️ Resume" if state.is_paused else "⏸️ Pause"

    # Update task status when pausing/resuming
    if state.current_task_obj:
        if state.is_paused:
            state.current_task_obj.status = TaskStatus.PAUSED
        else:
            state.current_task_obj.status = TaskStatus.ACTIVE

    state.activity_log.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": "control",
            "message": "⏸️ Paused" if state.is_paused else "▶️ Resumed",
        }
    )


def export_results(include_full: bool = True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kestrel_export_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(state.export_state(include_full_reports=include_full), f, indent=2)
    pn.state.notifications.success(f"📊 Exported to {filename}", duration=3000)


def prev_report():
    state.navigate_report(-1)
    update_dashboard()


def next_report():
    state.navigate_report(1)
    update_dashboard()


# Task creation form UI
create_task_btn = Button(name="➕ Create Task", button_type="primary", width=150)
create_task_btn.on_click(lambda e: create_task())

start_btn = Button(name="🚀 Start Task", button_type="success", width=150)
start_btn.on_click(lambda e: start_task())
start_btn.disabled = False

pause_btn = Button(name="⏸️ Pause", button_type="warning", width=120)
pause_btn.on_click(lambda e: pause_resume())
pause_btn.disabled = True  # Disabled until task starts

export_btn = Button(name="📊 Export", button_type="success", width=120)
export_btn.on_click(lambda e: export_results(True))

task_form = pn.Column(
    pn.pane.HTML(
        "<h3 style='margin-bottom: 16px; color: var(--text-dark); font-weight: 600;'>Create New Research Task</h3>"
    ),
    task_name_input,
    task_description_input,
    pn.Row(task_budget_input, create_task_btn, sizing_mode="stretch_width"),
    pn.pane.HTML(
        "<hr style='margin: 20px 0; border: 1px solid rgba(139, 111, 71, 0.2);'>"
    ),
    sizing_mode="stretch_width",
)

control_row = pn.Row(start_btn, pause_btn, export_btn, sizing_mode="stretch_width")

# Report navigation buttons
prev_btn = Button(name="◀ Previous", button_type="default", width=100)
prev_btn.on_click(lambda e: prev_report())

next_btn = Button(name="Next ▶", button_type="default", width=100)
next_btn.on_click(lambda e: next_report())

report_nav_row = pn.Row(prev_btn, next_btn, sizing_mode="stretch_width")


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def format_report_content(content: str) -> str:
    """Format report content by replacing tags with markdown headers and formatting links"""
    if not content:
        return ""
    formatted = content
    formatted = re.sub(r"\[SEARCH\]", "\n\n### 🔍 Search:", formatted)
    formatted = re.sub(r"\[THOUGHT\]", "\n\n### 🤔 Analysis:", formatted)
    formatted = re.sub(r"\[THINKING\]", "\n\n### 🤔 Analysis:", formatted)
    formatted = re.sub(r"\[SUMMARY\]", "\n\n### 📝 Summary:", formatted)
    formatted = re.sub(r"\[CHECKPOINT.*?\]", "\n\n### 💾 Checkpoint:", formatted)
    formatted = re.sub(r"(https?://[^\s]+)", r"[🔗 Link](\\1)", formatted)
    return formatted


# Dashboard panes
status_pane = pn.pane.HTML(
    "<div class='kestrel-card'>Loading...</div>", sizing_mode="stretch_width"
)
task_cards_pane = pn.pane.HTML("<div>Loading...</div>", sizing_mode="stretch_width")
metrics_pane = pn.pane.HTML(
    "<div class='kestrel-card'>Loading...</div>", sizing_mode="stretch_width"
)
activity_pane = pn.pane.HTML(
    "<div class='kestrel-card'>Loading...</div>",
    min_height=250,
    max_height=350,
    sizing_mode="stretch_width",
    styles={"overflow-y": "auto"},
)
notes_pane = pn.pane.Markdown(
    "## 📝 Research Notes\n\n_Waiting..._",
    min_height=500,
    sizing_mode="stretch_width",
    styles={
        "background": "rgba(245, 230, 211, 0.3)",
        "padding": "24px",
        "border-radius": "20px",
        "overflow-y": "auto",
    },
)
search_pane = pn.pane.HTML(
    "<div class='kestrel-card'>Loading...</div>",
    min_height=250,
    max_height=350,
    sizing_mode="stretch_width",
    styles={"overflow-y": "auto"},
)
report_info_pane = pn.pane.HTML(
    "<div style='text-align: center; color: #8B6F47; font-size: 14px;'>No reports yet</div>",
    sizing_mode="stretch_width",
)
research_plan_pane = pn.pane.HTML(
    "<div class='research-plan-card'><h3 class='card-title'>Research Plan</h3><p>Planning phase in progress...</p></div>",
    sizing_mode="stretch_width",
    min_height=400,
    max_height=600,
    styles={"overflow-y": "auto"},
)
orchestrator_state_pane = pn.pane.HTML(
    "<div class='kestrel-card'><h3 class='card-title'>Orchestrator State</h3><p>Waiting for orchestrator data...</p></div>",
    sizing_mode="stretch_width",
)

# Report history browser controls
task_filter = pn.widgets.Select(
    name="Task", options=["All"] + [t.name for t in tasks], value="All", width=160
)
search_filter = pn.widgets.TextInput(
    name="Search text", placeholder="filter report contents…", width=220
)
version_select = pn.widgets.Select(name="Version", options=[], width=320)


def _format_option(i, r):
    ts = r["timestamp"].strftime("%H:%M:%S")
    return f"{i+1}. [{r['task']}] @ {ts} — {len(r['content'])} chars"


def rebuild_version_options(
    update_value: bool = False, triggered_by_user: bool = False
):
    opts = []
    q = (search_filter.value or "").lower()
    for i, r in enumerate(state.report_history):
        if task_filter.value != "All" and r["task"] != task_filter.value:
            continue
        if q and q not in r["content"].lower():
            continue
        opts.append((_format_option(i, r), i))

    # FIX: Normalize options for comparison
    current_options = version_select.options

    # Convert current_options to list of tuples for comparison
    if current_options is None:
        normalized_current = []
    elif isinstance(current_options, dict):
        normalized_current = list(current_options.items())
    elif isinstance(current_options, list):
        # Could be list of tuples or list of values
        if current_options and isinstance(current_options[0], tuple):
            normalized_current = current_options
        else:
            # Just values, no labels
            normalized_current = [(str(v), v) for v in current_options]
    else:
        normalized_current = []

    # Only compare the values (indices), not the labels which can change
    current_values = (
        sorted([v for _, v in normalized_current]) if normalized_current else []
    )
    new_values = sorted([v for _, v in opts]) if opts else []

    # Update only if the actual values have changed
    if current_values != new_values:
        version_select.options = opts
        if update_value and opts:
            # Prefer to keep current index if present; otherwise, select latest
            existing_indices = [val for _, val in opts]
            if state.current_report_index in existing_indices:
                version_select.value = state.current_report_index
            else:
                version_select.value = opts[-1][1]
            if triggered_by_user:
                update_dashboard()


def on_select_version(event):
    if event.new is not None:
        # Extract value if it's a tuple (label, value)
        new_val = event.new
        if isinstance(new_val, tuple):
            new_val = new_val[1] if len(new_val) > 1 else new_val[0]

        if new_val != state.current_report_index:
            state.current_report_index = new_val
            update_dashboard()


# Watchers
task_filter.param.watch(
    lambda e: rebuild_version_options(update_value=True, triggered_by_user=True),
    "value",
)
search_filter.param.watch(
    lambda e: rebuild_version_options(update_value=True, triggered_by_user=True),
    "value",
)
version_select.param.watch(on_select_version, "value")

history_controls = pn.Row(
    task_filter, search_filter, version_select, sizing_mode="stretch_width"
)

# Initial populate of version options
rebuild_version_options(update_value=True, triggered_by_user=False)


# Update function
def update_dashboard():
    try:
        # Status with runtime
        runtime = state.get_runtime()
        status_html = f"""
        <div class='kestrel-card'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <h2 class='card-title'>⚡ {html.escape(str(state.current_task).title())}</h2>
                </div>
                <div class='runtime-display'>{runtime}</div>
            </div>
        </div>
        """
        status_pane.object = status_html

        # Task cards with beautiful design
        cards_html = ""
        for task in tasks:
            info = state.task_history[task.name]
            status = (
                "paused"
                if state.is_paused and info["status"] == "active"
                else info["status"]
            )
            active_class = "task-active" if status == "active" else ""
            elapsed_str = (
                f"{int(info['elapsed']//60):02d}:{int(info['elapsed']%60):02d}"
            )

            cards_html += f"""
            <div class='kestrel-card {active_class}'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;'>
                    <h3 class='card-title'>{html.escape(str(task.name).title())}</h3>
                    <span class='status-badge status-{status}'>{status}</span>
                </div>
                <div class='progress-wrapper'>
                    <div class='progress-label'>
                        <span>Progress</span>
                        <span style='font-weight: 700;'>{info["progress"]:.1f}%</span>
                    </div>
                    <div class='progress-container'>
                        <div class='progress-bar' style='width: {info["progress"]:.1f}%;'></div>
                    </div>
                </div>
                <div class='metrics-grid'>
                    <div class='metric-card'>
                        <div class='metric-value'>{elapsed_str}</div>
                        <div class='metric-label'>Time</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-value'>{info.get("search_count", 0)}</div>
                        <div class='metric-label'>Searches</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-value'>{info.get("think_count", 0)}</div>
                        <div class='metric-label'>Analysis</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-value'>{info.get("checkpoint_count", 0)}</div>
                        <div class='metric-label'>Saves</div>
                    </div>
                </div>
            </div>
            """

        task_cards_pane.object = cards_html

        # Global metrics with REAL data
        metrics_html = f"""
        <div class='kestrel-card'>
            <h3 class='card-title'>System Intelligence Metrics</h3>
            <div class='metrics-grid'>
                <div class='metric-card'>
                    <div class='metric-value'>{state.total_llm_calls}</div>
                    <div class='metric-label'>LLM Calls</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value'>{state.total_searches}</div>
                    <div class='metric-label'>Searches</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value'>{state.total_web_fetches}</div>
                    <div class='metric-label'>Pages</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value'>{state.total_summaries}</div>
                    <div class='metric-label'>Summaries</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value'>{state.total_checkpoints}</div>
                    <div class='metric-label'>Checkpoints</div>
                </div>
            </div>
        </div>
        """
        metrics_pane.object = metrics_html

        # Activity feed
        if state.activity_log:
            activity_html = (
                "<div class='kestrel-card'><h3 class='card-title'>Live Activity</h3>"
            )
            for entry in list(state.activity_log)[-10:][::-1]:
                activity_html += f"""
                <div class='activity-item'>
                    <span class='activity-time'>{entry['time']}</span>
                    <span>{entry['message']}</span>
                </div>
                """
            activity_html += "</div>"
            activity_pane.object = activity_html

        # Search history with results count
        if state.search_history:
            search_html = "<div class='kestrel-card'><h3 class='card-title'>Search Intelligence</h3>"
            for search in list(state.search_history)[-8:][::-1]:
                results_text = (
                    f"<span class='search-results'>({search.get('results_count', 0)} hits)</span>"
                    if "results_count" in search
                    else ""
                )
                search_html += f"""
                <div class='search-item'>
                    <span class='activity-time'>{search['time']}</span>
                    <span class='search-task'>[{search['task']}]</span>
                    <span style='flex: 1;'>{search['query']}</span>
                    {results_text}
                </div>
                """
            search_html += "</div>"
            search_pane.object = search_html

        # Ensure the version selector stays in sync (no dashboard recursion)
        rebuild_version_options(update_value=False, triggered_by_user=False)

        # Research Plan Display
        if state.current_task in state.research_plans:
            plan = state.research_plans[state.current_task]
            completed_count = sum(
                1 for st in plan["subtasks"] if st["status"] == "completed"
            )
            total_count = len(plan["subtasks"])
            progress_pct = (
                (completed_count / total_count * 100) if total_count > 0 else 0
            )

            plan_html = f"""
            <div class='research-plan-card'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
                    <h3 class='card-title'>🧠 Research Plan</h3>
                    <div style='font-size: 14px; color: var(--text-medium); font-weight: 600;'>
                        {completed_count}/{total_count} completed
                    </div>
                </div>
                <div style='margin-bottom: 20px;'>
                    <div style='display: flex; justify-content: space-between; font-size: 12px; color: var(--text-medium); margin-bottom: 8px;'>
                        <span>Progress</span>
                        <span style='font-weight: 700;'>{progress_pct:.1f}%</span>
                    </div>
                    <div class='progress-container'>
                        <div class='progress-bar' style='width: {progress_pct:.1f}%;'></div>
                    </div>
                </div>
                <div style='margin-bottom: 20px;'>
                    <h4 style='font-size: 14px; font-weight: 600; color: var(--text-dark); margin-bottom: 12px;'>Task Restatement</h4>
                    <p style='font-size: 13px; color: var(--text-medium); background: rgba(255, 255, 255, 0.5); padding: 12px; border-radius: 8px; border: 1px solid rgba(139, 111, 71, 0.1);'>
                        {html.escape(str(plan.get("restated_task", "")))}
                    </p>
                </div>
                <div>
                    <h4 style='font-size: 14px; font-weight: 600; color: var(--text-dark); margin-bottom: 12px;'>Research Steps</h4>
            """

            for subtask in plan["subtasks"]:
                status = subtask["status"]
                status_class = f"subtask-{status}"
                number_class = f"subtask-number-{status}"
                subtask_idx = subtask["order"] - 1  # 0-based index

                # Get metrics for this subtask
                metrics_html = ""
                if state.current_task in state.task_history:
                    task_info = state.task_history[state.current_task]
                    if (
                        "subtask_metrics" in task_info
                        and subtask_idx in task_info["subtask_metrics"]
                    ):
                        metrics = task_info["subtask_metrics"][subtask_idx]
                        metrics_html = f"""
                        <div class='subtask-metrics'>
                            <span class='subtask-metric'>🔍 {metrics.get('searches', 0)} searches</span>
                            <span class='subtask-metric'>🤔 {metrics.get('thoughts', 0)} thoughts</span>
                            <span class='subtask-metric'>📝 {metrics.get('summaries', 0)} summaries</span>
                            <span class='subtask-metric'>💾 {metrics.get('checkpoints', 0)} checkpoints</span>
                            <span class='subtask-metric'>⚡ {metrics.get('llm_calls', 0)} LLM calls</span>
                        </div>
                        """

                plan_html += f"""
                    <div class='subtask-item {status_class}'>
                        <div style='display: flex; align-items: flex-start;'>
                            <div class='subtask-number {number_class}'>{subtask["order"] if status != "completed" else ""}</div>
                            <div style='flex: 1;'>
                                <div class='subtask-description'>
                                    Step {subtask.get("order", 0)}: {html.escape(str(subtask.get("description", "")))}
                                    {'<span style="margin-left: 8px; font-size: 10px; color: var(--rust-orange);">🔄 In Progress</span>' if status == "in_progress" else ""}
                                    {'<span style="margin-left: 8px; font-size: 10px; color: #10b981;">✓ Completed</span>' if status == "completed" else ""}
                                </div>
                                <div class='subtask-criteria'>
                                    <strong>Success Criteria:</strong> {html.escape(str(subtask.get("success_criteria", "")))}
                                </div>
                                {metrics_html}
                            </div>
                        </div>
                    </div>
                """

            plan_html += """
                </div>
            </div>
            """
            research_plan_pane.object = plan_html
        else:
            research_plan_pane.object = """
            <div class='research-plan-card'>
                <h3 class='card-title'>🧠 Research Plan</h3>
                <p style='text-align: center; color: var(--text-medium); padding: 40px 20px;'>
                    Research plan will be generated when the task starts...
                </p>
            </div>
            """

        # Orchestrator State Display
        if state.current_task in state.orchestrator_state:
            orch_state = state.orchestrator_state[state.current_task]
            stuck_class = (
                "stuck-warning" if orch_state.get("stuck_count", 0) > 0 else ""
            )

            orch_html = f"""
            <div class='kestrel-card'>
                <h3 class='card-title'>🎯 Orchestrator State</h3>
                <div class='orchestrator-state'>
                    <div class='orchestrator-state-item'>
                        <span class='orchestrator-state-label'>Decision Count</span>
                        <span class='orchestrator-state-value'>{orch_state.get("decision_count", 0)}</span>
                    </div>
                    <div class='orchestrator-state-item'>
                        <span class='orchestrator-state-label'>Stuck Count</span>
                        <span class='orchestrator-state-value {stuck_class}'>{orch_state.get("stuck_count", 0)}</span>
                    </div>
                    <div class='orchestrator-state-item'>
                        <span class='orchestrator-state-label'>Current Subtask Index</span>
                        <span class='orchestrator-state-value'>{orch_state.get("subtask_index", 0)}</span>
                    </div>
                    <div class='orchestrator-state-item'>
                        <span class='orchestrator-state-label'>Completed Subtasks</span>
                        <span class='orchestrator-state-value'>{len(orch_state.get("completed_subtasks", []))}</span>
                    </div>
                    <div class='orchestrator-state-item'>
                        <span class='orchestrator-state-label'>Total Findings</span>
                        <span class='orchestrator-state-value'>{orch_state.get("total_findings", 0)}</span>
                    </div>
                    <div class='orchestrator-state-item'>
                        <span class='orchestrator-state-label'>Last Decision</span>
                        <span class='orchestrator-state-value'>{html.escape(str(orch_state.get("last_decision", "N/A")))}</span>
                    </div>
            """

            feedback_history = orch_state.get("feedback_history", [])
            if feedback_history:
                orch_html += """
                    <div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(139, 111, 71, 0.1);'>
                        <div style='font-size: 12px; font-weight: 600; color: var(--text-medium); margin-bottom: 8px;'>Recent Feedback:</div>
                """
                for feedback in feedback_history[-3:]:  # Last 3
                    feedback_text = html.escape(str(feedback))[:100]
                    orch_html += f'<div class="feedback-item">{feedback_text}{"..." if len(str(feedback)) > 100 else ""}</div>'
                orch_html += "</div>"

            orch_html += """
                </div>
            </div>
            """
            orchestrator_state_pane.object = orch_html
        else:
            orchestrator_state_pane.object = """
            <div class='kestrel-card'>
                <h3 class='card-title'>🎯 Orchestrator State</h3>
                <p style='text-align: center; color: var(--text-medium); padding: 20px;'>
                    Waiting for orchestrator data...
                </p>
            </div>
            """

        # Research notes with navigation
        current_report = state.get_current_report()
        if current_report:
            # Format report info
            report_num = state.current_report_index + 1
            total_reports = len(state.report_history)
            report_time = current_report["timestamp"].strftime("%H:%M:%S")
            report_task = current_report["task"]

            report_info_html = f"""
            <div style='display: flex; justify-content: space-between; align-items: center; padding: 12px;
                        background: rgba(139, 111, 71, 0.1); border-radius: 10px; margin-bottom: 10px;'>
                <span style='color: #8B6F47; font-weight: 600;'>
                    Report {report_num}/{total_reports}
                </span>
                <span style='color: #D2691E; font-weight: 500;'>
                    [{report_task}] @ {report_time}
                </span>
            </div>
            """
            report_info_pane.object = report_info_html

            # Display the FULL report content (no truncation)
            formatted = format_report_content(current_report["content"])
            notes_pane.object = f"## 📝 Research Intelligence Notes\n\n{formatted}"

            # Enable/disable navigation buttons
            prev_btn.disabled = state.current_report_index <= 0
            next_btn.disabled = (
                state.current_report_index >= len(state.report_history) - 1
            )
        else:
            # Show latest notes if no reports yet
            if state.latest_notes:
                formatted = format_report_content(state.latest_notes)
                notes_pane.object = f"## 📝 Research Intelligence Notes\n\n{formatted}"

            prev_btn.disabled = True
            next_btn.disabled = True

    except Exception as e:
        error_msg = f"Dashboard update error: {e}"
        print(error_msg)
        import traceback

        traceback.print_exc()

        # Show error in UI
        status_pane.object = f"""
        <div class='kestrel-card' style='border: 2px solid #ef4444;'>
            <h2 class='card-title' style='color: #ef4444;'>⚠️ Dashboard Error</h2>
            <p style='color: var(--text-dark);'>{str(e)[:200]}</p>
            <p style='font-size: 12px; color: var(--text-medium); margin-top: 12px;'>
                Check console for full traceback
            </p>
        </div>
        """


# Periodic updates
pn.state.add_periodic_callback(update_dashboard, period=750)

# Optional: periodically refresh version options so new reports appear in the picker
pn.state.add_periodic_callback(
    lambda: rebuild_version_options(update_value=False, triggered_by_user=False),
    period=2000,
)

# Build layout
sidebar = pn.Column(
    header_pane,
    task_form,
    control_row,
    status_pane,
    task_cards_pane,
    metrics_pane,
    sizing_mode="stretch_width",
)

main = pn.Column(
    pn.Row(activity_pane, search_pane, sizing_mode="stretch_width"),
    research_plan_pane,  # Research plan with subtasks
    pn.Row(orchestrator_state_pane, sizing_mode="stretch_width"),  # Orchestrator state
    history_controls,  # filter + version picker
    report_info_pane,  # shows current report index/time/task
    report_nav_row,  # ◀ Previous / Next ▶ navigation
    notes_pane,
    sizing_mode="stretch_width",
)

TEMPLATE.sidebar.append(sidebar)
TEMPLATE.main.append(main)

# Initial update
update_dashboard()

# Serve
TEMPLATE.servable()
