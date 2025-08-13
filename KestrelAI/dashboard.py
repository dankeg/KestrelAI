"""
dashboard.py – Beautiful Panel UI for KestrelAI Research System

Run with:
  panel serve dashboard.py --autoreload --show

Features:
  • Real-time metrics from agent internals
  • Beautiful kestrel-inspired design
  • Rich, modern interface
  • Search history with details
  • Export functionality
  • Full report version browser (filter + picker + prev/next)
"""

# -----------------------------------------------------------------------------
# Standard library
# -----------------------------------------------------------------------------
import pathlib
import threading
import time
import re
from datetime import datetime
from collections import deque
from typing import Dict, List
import json

# -----------------------------------------------------------------------------
# Third-party deps
# -----------------------------------------------------------------------------
import panel as pn
from panel.widgets import Button

# -----------------------------------------------------------------------------
# KestrelAI modules
# -----------------------------------------------------------------------------
from agents.orchestrator import Orchestrator
from agents.base import LlmWrapper
from agents.research_agents import ResearchAgent
from memory.vector_store import MemoryStore
from dataclass.tasks import Task

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
</style>
"""

# -----------------------------------------------------------------------------
# Task configuration
# -----------------------------------------------------------------------------
ML_TASK = """Find currently open grants, programs, fellowships, or funding opportunities that
support AI/ML research and are available to senior undergraduate students in the
United States. Include name, eligibility, what it funds, deadline, and link. Focus
on fresh, prestigious, student-accessible opportunities not locked to one college.
Include opportunities from companies like Google, Microsoft, Meta, OpenAI, Anthropic, Cohere."""

AI_CONFERENCES_CALL_FOR_ABSTRACTS_TASK = """Find AI/ML conferences, symposia, workshops, or student research programs that are currently accepting abstract submissions (not full-paper–only). For each opportunity, include: conference name, organizer/society, location & event dates, scope/topics, submission type (talk/poster/workshop) and abstract length/format, eligibility (explicitly note if senior undergraduates can submit), 
important dates (abstract deadline with timezone, notification date, and any later full/extended-abstract deadline), review model (single/double-blind), submission/CFP link, template/guidelines link, student registration fee (if published), whether accepted abstracts are published/archived (e.g., proceedings/indexing), any travel support, and key restrictions.
Prioritize reputable, broadly accessible venues and student programs tied to major conferences/societies (AAAI, NeurIPS, ICML, ICLR, KDD, ACL/EMNLP, CVPR/ICCV workshops & student programs, MICCAI, IEEE/ACM technical societies) plus interdisciplinary venues that welcome AI work. Exclude calls that are closed, invitation-only, or restricted to a single institution. Return only calls that are still open.
"""

ML_COMPETITIONS_TASK = """Find active AI/ML student competitions or challenges suitable for senior undergraduates in the United States. Include organizer, problem theme, eligibility, prize(s) or publication opportunities, important dates (registration close, submission due), compute support (if any), and link. Prioritize well-known organizers (Kaggle, Google, Microsoft, Meta, OpenAI, Anthropic) and academic conferences. Exclude archived or invitation-only events."""

tasks = [
    Task("ML Fellowships", ML_TASK, budget_minutes=180),
    Task("Conferences", AI_CONFERENCES_CALL_FOR_ABSTRACTS_TASK, budget_minutes=180),
    Task("Competitions", ML_COMPETITIONS_TASK, budget_minutes=180),
]

# -----------------------------------------------------------------------------
# Initialize backend
# -----------------------------------------------------------------------------
mem = MemoryStore()
llm = LlmWrapper(model="gemma3:12b")
agent = ResearchAgent(mem, llm)
orch = Orchestrator(tasks, llm)

# -----------------------------------------------------------------------------
# State management
# -----------------------------------------------------------------------------
class DashboardState:
    def __init__(self):
        self.current_task: str = "Initializing..."
        self.task_start: float = time.time()
        self.latest_notes: str = ""
        self.is_paused: bool = False
        self.start_time: float = time.time()
        
        # Store all report versions
        self.report_history: List[Dict] = []  # List of {timestamp, task, content}
        self.current_report_index: int = -1  # Index of currently viewed report
        
        self.task_history: Dict[str, Dict] = {
            task.name: {
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
                "end_time": None
            } for task in tasks
        }
        
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
        self.report_history.append({
            "timestamp": datetime.now(),
            "task": task_name,
            "content": content
        })
        self.current_report_index = len(self.report_history) - 1
    
    def get_current_report(self) -> Dict:
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
    
    def export_state(self, include_full_reports: bool = False) -> Dict:
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
                "total_web_fetches": self.total_web_fetches
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
                    "summary_count": info["summary_count"]
                }
                for name, info in self.task_history.items()
            },
            "search_history": list(self.search_history)[-50:],
            "report_history": [
                {
                    "timestamp": r["timestamp"].isoformat(),
                    "task": r["task"],
                    "content": r["content"] if include_full_reports else r["content"][:1000]
                }
                for r in self.report_history
            ]
        }

state = DashboardState()

# -----------------------------------------------------------------------------
# Orchestration loop
# -----------------------------------------------------------------------------
def orchestration_loop():
    pathlib.Path("notes").mkdir(exist_ok=True)
    
    while True:
        if state.is_paused:
            time.sleep(0.5)
            continue
            
        if not orch.current:
            time.sleep(1)
            continue
            
        try:
            task = orch.tasks[orch.current]
            
            # Handle task switching
            if state.current_task != task.name:
                if state.current_task != "Initializing..." and state.current_task in state.task_history:
                    state.task_history[state.current_task]["status"] = "complete"
                    state.task_history[state.current_task]["end_time"] = time.time()
                
                state.current_task = task.name
                state.task_start = time.time()
                state.task_history[task.name]["status"] = "active"
                if state.task_history[task.name]["start_time"] is None:
                    state.task_history[task.name]["start_time"] = time.time()
                
                state.activity_log.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "task_start",
                    "message": f"🦅 Started: {task.name.title()}"
                })
                state.last_update = time.time()

            # Execute research step
            notes = agent.run_step(task)
            
            if notes and len(notes) > 10:
                state.latest_notes = notes
                
                # Store report in history
                state.add_report(task.name, notes)
                
                # Get REAL metrics from agent
                if hasattr(agent, 'get_global_metrics'):
                    metrics = agent.get_global_metrics()
                    state.total_llm_calls = metrics.get("total_llm_calls", 0)
                    state.total_searches = metrics.get("total_searches", 0)
                    state.total_summaries = metrics.get("total_summaries", 0)
                    state.total_checkpoints = metrics.get("total_checkpoints", 0)
                    state.total_web_fetches = metrics.get("total_web_fetches", 0)
                
                # Get task-specific metrics
                if hasattr(agent, 'get_task_metrics'):
                    task_metrics = agent.get_task_metrics(task.name)
                    task_info = state.task_history[task.name]
                    
                    # Update with REAL data from agent
                    task_info["searches"] = task_metrics.get("searches", [])
                    task_info["search_count"] = task_metrics.get("search_count", 0)
                    task_info["think_count"] = task_metrics.get("think_count", 0)
                    task_info["summary_count"] = task_metrics.get("summary_count", 0)
                    task_info["checkpoint_count"] = task_metrics.get("checkpoint_count", 0)
                    task_info["action_count"] = task_metrics.get("action_count", 0)
                    
                    # Update search history with real search data
                    for search_entry in task_metrics.get("search_history", []):
                        if isinstance(search_entry, dict) and "query" in search_entry:
                            if not any(s.get("query") == search_entry["query"] for s in state.search_history):
                                state.search_history.append({
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "task": task.name,
                                    "query": search_entry["query"],
                                    "results_count": search_entry.get("results_count", 0)
                                })
                
                # Update common fields
                elapsed = time.time() - state.task_start
                progress = min(100, (elapsed / (task.budget_minutes * 60)) * 100)
                
                task_info = state.task_history[task.name]
                task_info.update({
                    "elapsed": elapsed,
                    "progress": progress,
                    "notes": notes[:1000],
                    "last_action": datetime.now().strftime("%H:%M:%S")
                })
                
                # Save notes
                with open(f"notes/{task.name.upper()}.txt", "w", encoding="utf-8") as fh:
                    fh.write(notes)
                
                # Log activity with emojis
                if "[SEARCH]" in notes:
                    message = "🔍 Searching for information"
                elif "[THOUGHT]" in notes:
                    message = "🤔 Analyzing findings"
                elif "[SUMMARY]" in notes:
                    message = "📝 Creating summary"
                elif "[CHECKPOINT" in notes:
                    message = "💾 Saving checkpoint"
                else:
                    message = "⚙️ Processing"
                
                state.activity_log.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "action",
                    "message": message
                })
                
                state.last_update = time.time()
            
            # Let orchestrator decide next
            orch.next_action(task, notes)
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error in orchestration loop: {e}")
            state.activity_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": "error",
                "message": f"❌ Error: {str(e)[:100]}"
            })
            state.last_update = time.time()
            time.sleep(2)

# Start background thread
thread = threading.Thread(target=orchestration_loop, daemon=True, name="OrchestratorThread")
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
    sidebar_width=500
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
    sizing_mode="stretch_width"
)

# Control buttons
def pause_resume():
    state.is_paused = not state.is_paused
    pause_btn.name = "▶️ Resume" if state.is_paused else "⏸️ Pause"
    state.activity_log.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "type": "control",
        "message": "⏸️ Paused" if state.is_paused else "▶️ Resumed"
    })

def export_results(include_full: bool = True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kestrel_export_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(state.export_state(include_full_reports=include_full), f, indent=2)
    pn.state.notifications.success(f'📊 Exported to {filename}', duration=3000)

def prev_report():
    state.navigate_report(-1)
    update_dashboard()

def next_report():
    state.navigate_report(1)
    update_dashboard()

pause_btn = Button(name="⏸️ Pause", button_type="warning", width=120)
pause_btn.on_click(lambda e: pause_resume())

export_btn = Button(name="📊 Export", button_type="success", width=120)
export_btn.on_click(lambda e: export_results(True))

control_row = pn.Row(pause_btn, export_btn, sizing_mode="stretch_width")

# --- Report navigation buttons (shown in layout) ---
prev_btn = Button(name="◀ Previous", button_type="default", width=100)
prev_btn.on_click(lambda e: prev_report())

next_btn = Button(name="Next ▶", button_type="default", width=100)
next_btn.on_click(lambda e: next_report())

report_nav_row = pn.Row(prev_btn, next_btn, sizing_mode="stretch_width")

# Dashboard panes
status_pane = pn.pane.HTML("<div class='kestrel-card'>Loading...</div>", sizing_mode="stretch_width")
task_cards_pane = pn.pane.HTML("<div>Loading...</div>", sizing_mode="stretch_width")
metrics_pane = pn.pane.HTML("<div class='kestrel-card'>Loading...</div>", sizing_mode="stretch_width")
activity_pane = pn.pane.HTML("<div class='kestrel-card'>Loading...</div>", min_height=250, max_height=350, sizing_mode="stretch_width",
    styles={"overflow-y": "auto"})
notes_pane = pn.pane.Markdown("## 📝 Research Notes\n\n_Waiting..._", min_height=500, sizing_mode="stretch_width",
    styles={"background": "rgba(245, 230, 211, 0.3)", "padding": "24px", "border-radius": "20px", "overflow-y": "auto"})
search_pane = pn.pane.HTML("<div class='kestrel-card'>Loading...</div>", min_height=250, max_height=350, sizing_mode="stretch_width",
    styles={"overflow-y": "auto"})
report_info_pane = pn.pane.HTML("<div style='text-align: center; color: #8B6F47; font-size: 14px;'>No reports yet</div>", 
    sizing_mode="stretch_width")

# --- Report history browser controls (Task filter + search + version select) ---
task_filter = pn.widgets.Select(name="Task", options=["All"] + [t.name for t in tasks], value="All", width=160)
search_filter = pn.widgets.TextInput(name="Search text", placeholder="filter report contents…", width=220)
version_select = pn.widgets.Select(name="Version", options=[], width=320)

def _format_option(i, r):
    ts = r["timestamp"].strftime("%H:%M:%S")
    return f"{i+1}. [{r['task']}] @ {ts} — {len(r['content'])} chars"

def rebuild_version_options(update_value: bool = False, triggered_by_user: bool = False):
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
    current_values = sorted([v for _, v in normalized_current]) if normalized_current else []
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
task_filter.param.watch(lambda e: rebuild_version_options(update_value=True, triggered_by_user=True), "value")
search_filter.param.watch(lambda e: rebuild_version_options(update_value=True, triggered_by_user=True), "value")
version_select.param.watch(on_select_version, "value")

history_controls = pn.Row(task_filter, search_filter, version_select, sizing_mode="stretch_width")

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
                    <h2 class='card-title'>⚡ {state.current_task.title()}</h2>
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
            status = "paused" if state.is_paused and info["status"] == "active" else info["status"]
            active_class = "task-active" if status == "active" else ""
            elapsed_str = f"{int(info['elapsed']//60):02d}:{int(info['elapsed']%60):02d}"
            
            cards_html += f"""
            <div class='kestrel-card {active_class}'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;'>
                    <h3 class='card-title'>{task.name.title()}</h3>
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
            activity_html = "<div class='kestrel-card'><h3 class='card-title'>Live Activity</h3>"
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
                results_text = f"<span class='search-results'>({search.get('results_count', 0)} hits)</span>" if 'results_count' in search else ""
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
            formatted = current_report["content"]
            formatted = re.sub(r'\[SEARCH\]', '\n\n### 🔍 Search:', formatted)
            formatted = re.sub(r'\[THOUGHT\]', '\n\n### 🤔 Analysis:', formatted)
            formatted = re.sub(r'\[SUMMARY\]', '\n\n### 📝 Summary:', formatted)
            formatted = re.sub(r'\[CHECKPOINT.*?\]', '\n\n### 💾 Checkpoint:', formatted)
            formatted = re.sub(r'(https?://[^\s]+)', r'[🔗 Link](\\1)', formatted)
            
            notes_pane.object = f"## 📝 Research Intelligence Notes\n\n{formatted}"
            
            # Enable/disable navigation buttons
            prev_btn.disabled = state.current_report_index <= 0
            next_btn.disabled = state.current_report_index >= len(state.report_history) - 1
        else:
            # Show latest notes if no reports yet
            if state.latest_notes:
                formatted = state.latest_notes
                formatted = re.sub(r'\[SEARCH\]', '\n\n### 🔍 Search:', formatted)
                formatted = re.sub(r'\[THOUGHT\]', '\n\n### 🤔 Analysis:', formatted)
                formatted = re.sub(r'\[SUMMARY\]', '\n\n### 📝 Summary:', formatted)
                formatted = re.sub(r'\[CHECKPOINT.*?\]', '\n\n### 💾 Checkpoint:', formatted)
                formatted = re.sub(r'(https?://[^\s]+)', r'[🔗 Link](\\1)', formatted)
                
                notes_pane.object = f"## 📝 Research Intelligence Notes\n\n{formatted}"
            
            prev_btn.disabled = True
            next_btn.disabled = True
            
    except Exception as e:
        print(f"Dashboard update error: {e}")

# Periodic updates
cb = pn.state.add_periodic_callback(update_dashboard, period=750)

# Optional: periodically refresh version options so new reports appear in the picker
pn.state.add_periodic_callback(lambda: rebuild_version_options(update_value=False, triggered_by_user=False), period=2000)

# Build layout
sidebar = pn.Column(
    header_pane,
    control_row,
    status_pane,
    task_cards_pane,
    metrics_pane,
    sizing_mode="stretch_width"
)

main = pn.Column(
    pn.Row(activity_pane, search_pane, sizing_mode="stretch_width"),
    history_controls,     # ← new: filter + version picker
    report_info_pane,     # ← shows current report index/time/task
    report_nav_row,       # ← ◀ Previous / Next ▶ navigation
    notes_pane,
    sizing_mode="stretch_width"
)

TEMPLATE.sidebar.append(sidebar)
TEMPLATE.main.append(main)

# Initial update
update_dashboard()

# Serve
TEMPLATE.servable()