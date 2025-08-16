import React, { useEffect, useRef, useState } from "react";
import {
  Plus,
  Search,
  Send,
  Play,
  Pause,
  Clock,
  Activity,
  FileText,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
  Settings,
  Download,
  Brain,
  Globe,
  Save,
  Zap,
  MoreHorizontal,
  Edit3,
  Trash2,
  Timer,
  Target,
  CheckCircle,
  AlertCircle,
  BookOpen,
  BarChart3,
} from "lucide-react";

// API Configuration
const API_BASE_URL = "http://localhost:8000/api/v1";

// Types
interface Metrics {
  searchCount: number;
  thinkCount: number;
  summaryCount: number;
  checkpointCount: number;
}

interface Task {
  id: string;
  name: string;
  description: string;
  user_prompt?: string;
  deadline?: string | null;
  budget_minutes: number;
  status: "configuring" | "pending" | "active" | "complete" | "paused";
  progress?: number;
  elapsed?: number;
  metrics?: Metrics;
  scratchpad?: string[];
  result?: string | null;
  createdAt?: number;
  updatedAt?: number;

  /** Frontend-only: true until we POST this task on Start Research */
  isDraft?: boolean;
}

interface ActivityEntry {
  id: string;
  taskId: string;
  time: string;
  type: "task_start" | "search" | "analysis" | "summary" | "checkpoint" | "error";
  message: string;
  timestamp: number;
}

interface SearchEntry {
  id: string;
  taskId: string;
  time: string;
  query: string;
  results: number;
  timestamp: number;
}

interface Report {
  id: string;
  taskId: string;
  timestamp: number;
  title: string;
  content: string;
}

interface SystemMetrics {
  llmCalls: number;
  searches: number;
  pagesAnalyzed: number;
  summaries: number;
  checkpoints: number;
}

// API Client
class KestrelAPI {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  private async fetch(endpoint: string, options?: RequestInit) {
    const url = `${this.baseURL}${endpoint}`;
    console.log("API Request:", options?.method || "GET", url); // Debug log

    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API Error:", response.status, errorText); // Debug log
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    // Some endpoints may return 204; guard that.
    const text = await response.text();
    return text ? JSON.parse(text) : null;
  }

  // Task endpoints
  async getTasks(): Promise<Task[]> {
    return this.fetch("/tasks");
  }

  async getTask(id: string): Promise<Task> {
    return this.fetch(`/tasks/${id}`);
  }

  async createTask(task: Partial<Task>): Promise<Task> {
    return this.fetch("/tasks", {
      method: "POST",
      body: JSON.stringify(task),
    });
  }

  async updateTask(id: string, updates: Partial<Task>): Promise<Task> {
    console.log("Updating task:", id, updates); // Debug log
    return this.fetch(`/tasks/${id}`, {
      method: "PATCH",
      body: JSON.stringify(updates),
    });
  }

  async deleteTask(id: string): Promise<void> {
    return this.fetch(`/tasks/${id}`, {
      method: "DELETE",
    });
  }

  async startTask(id: string): Promise<Task> {
    return this.fetch(`/tasks/${id}/start`, {
      method: "POST",
    });
  }

  async pauseTask(id: string): Promise<Task> {
    return this.fetch(`/tasks/${id}/pause`, {
      method: "POST",
    });
  }

  async resumeTask(id: string): Promise<Task> {
    return this.fetch(`/tasks/${id}/resume`, {
      method: "POST",
    });
  }

  // Activity endpoints
  async getTaskActivity(taskId: string, limit: number = 50): Promise<ActivityEntry[]> {
    return this.fetch(`/tasks/${taskId}/activity?limit=${limit}`);
  }

  // Search history endpoints
  async getTaskSearchHistory(taskId: string, limit: number = 50): Promise<SearchEntry[]> {
    return this.fetch(`/tasks/${taskId}/searches?limit=${limit}`);
  }

  // Reports endpoints
  async getTaskReports(taskId: string): Promise<Report[]> {
    return this.fetch(`/tasks/${taskId}/reports`);
  }

  async getReport(taskId: string, reportId: string): Promise<Report> {
    return this.fetch(`/tasks/${taskId}/reports/${reportId}`);
  }

  // Metrics endpoints
  async getTaskMetrics(taskId: string): Promise<SystemMetrics> {
    return this.fetch(`/tasks/${taskId}/metrics`);
  }

  // Export endpoints
  async exportTask(taskId: string, format: "json" | "pdf" | "markdown" = "json"): Promise<Blob> {
    const response = await fetch(`${this.baseURL}/tasks/${taskId}/export?format=${format}`);
    if (!response.ok) {
      throw new Error(`Export failed: ${response.status}`);
    }
    return response.blob();
  }
}

// Initialize API client
const api = new KestrelAPI();

// Utilities
const uid = () => Math.random().toString(36).slice(2, 10);

const defaultMetrics = (): Metrics => ({
  searchCount: 0,
  thinkCount: 0,
  summaryCount: 0,
  checkpointCount: 0,
});

const formatElapsed = (seconds?: number): string => {
  if (seconds === undefined) return "--:--";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return hours > 0
    ? `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${secs
        .toString()
        .padStart(2, "0")}`
    : `${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
};

const formatDate = (timestamp?: number): string => {
  if (!timestamp) return "Unknown";
  const date = new Date(timestamp);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  if (date.toDateString() === today.toDateString()) {
    return "Today";
  } else if (date.toDateString() === yesterday.toDateString()) {
    return "Yesterday";
  } else {
    return date.toLocaleDateString([], { month: "short", day: "numeric" });
  }
};

// Custom hook for managing tasks with API
function useTaskManager() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load tasks from API + merge local drafts
  useEffect(() => {
    loadTasks();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadTasks = async () => {
    try {
      setIsLoading(true);
      const serverTasks = (await api.getTasks()) || [];
      const serverNormalized: Task[] = serverTasks.map((t) => ({ ...t, isDraft: false }));
      // Merge local drafts from localStorage
      let drafts: Task[] = [];
      try {
        const saved = localStorage.getItem("kestrel.tasks");
        if (saved) {
          const parsed = JSON.parse(saved);
          if (Array.isArray(parsed)) {
            drafts = parsed.filter((t: Task) => t.isDraft === true);
          }
        }
      } catch {
        // ignore parse errors
      }
      const combined = [...drafts, ...serverNormalized];
      setTasks(combined);
      if (combined.length > 0 && !selectedTaskId) {
        setSelectedTaskId(combined[0].id);
      }
    } catch (err) {
      console.error("Failed to load tasks from API:", err);
      setError("Failed to load tasks from API. Using local data.");
      // Fallback to localStorage
      try {
        const saved = localStorage.getItem("kestrel.tasks");
        if (saved) {
          const parsed = JSON.parse(saved);
          if (Array.isArray(parsed)) {
            setTasks(parsed);
            if (parsed.length > 0 && parsed[0].id) {
              setSelectedTaskId(parsed[0].id);
            }
          }
        }
      } catch (e) {
        console.error("Failed to load from localStorage:", e);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Create a LOCAL draft task only. We will POST on Start.
  const createTask = async (): Promise<string> => {
    const newTask: Task = {
      id: `draft-${uid()}`,
      name: "New Research Task",
      description: "",
      budget_minutes: 180,
      status: "configuring",
      progress: 0,
      elapsed: 0,
      metrics: defaultMetrics(),
      createdAt: Date.now(),
      updatedAt: Date.now(),
      isDraft: true,
    };

    setTasks((prev) => {
      const next = [newTask, ...prev];
      localStorage.setItem("kestrel.tasks", JSON.stringify(next));
      return next;
    });
    setSelectedTaskId(newTask.id);
    return newTask.id;
  };

  // Local update, and only PATCH server for non-drafts
  const updateTask = async (id: string, updates: Partial<Task>) => {
    setTasks((prev) => {
      const next = prev.map((t) =>
        t.id === id ? { ...t, ...updates, updatedAt: Date.now() } : t
      );
      localStorage.setItem("kestrel.tasks", JSON.stringify(next));
      return next;
    });

    // If the task exists on the server, sync PATCH (best-effort)
    const existing = tasks.find((t) => t.id === id);
    if (existing && !existing.isDraft) {
      try {
        await api.updateTask(id, updates);
      } catch (err) {
        console.error("Failed to update task on server (kept local changes):", err);
      }
    }
  };

  const deleteTask = async (id: string) => {
    const task = tasks.find((t) => t.id === id);
    if (task && !task.isDraft) {
      try {
        await api.deleteTask(id);
      } catch (err) {
        console.error("Failed to delete task on server, removing locally:", err);
      }
    }
    setTasks((prev) => {
      const next = prev.filter((t) => t.id !== id);
      // adjust selection if needed
      setSelectedTaskId((curr) => (curr === id ? next[0]?.id || null : curr));
      localStorage.setItem("kestrel.tasks", JSON.stringify(next));
      return next;
    });
  };

  /**
   * Persist a draft to the backend and start it.
   * If already persisted, just start it.
   */
  const persistAndStartTask = async (id: string) => {
    const t = tasks.find((tk) => tk.id === id);
    if (!t) return;

    if (t.isDraft) {
      try {
        // POST only now (single POST with the task)
        const created = await api.createTask({
          name: t.name,
          description: t.description,
          budget_minutes: t.budget_minutes,
          user_prompt: t.user_prompt,
          deadline: t.deadline ?? null,
          status: "configuring", // backend may ignore/override; harmless to include
        });

        // Replace draft with server task (preserve some local fields if useful)
        setTasks((prev) => {
          const next = prev.map((x) =>
            x.id === id
              ? {
                  ...created,
                  isDraft: false,
                  progress: x.progress ?? 0,
                  elapsed: x.elapsed ?? 0,
                  metrics: x.metrics ?? defaultMetrics(),
                  createdAt: x.createdAt ?? Date.now(),
                  updatedAt: Date.now(),
                  status: "active",
                }
              : x
          );
          localStorage.setItem("kestrel.tasks", JSON.stringify(next));
          return next;
        });
        setSelectedTaskId(created.id);

        // Start the task (second POST, command-style; not a duplicate "task create" POST)
        try {
          await api.startTask(created.id);
        } catch (err) {
          console.error("Failed to start task on server; keeping local state as active:", err);
        }
      } catch (err) {
        console.error("Failed to persist draft task:", err);
        throw err;
      }
    } else {
      // Already on server: just start it
      try {
        await api.startTask(id);
      } catch (err) {
        console.error("Failed to start task:", err);
      } finally {
        // Ensure local status is active regardless (optimistic)
        setTasks((prev) => {
          const next = prev.map((x) => (x.id === id ? { ...x, status: "active", updatedAt: Date.now() } : x));
          localStorage.setItem("kestrel.tasks", JSON.stringify(next));
          return next;
        });
      }
    }
  };

  const selectedTask = tasks.find((t) => t.id === selectedTaskId) || null;

  return {
    tasks,
    selectedTask,
    selectedTaskId,
    setSelectedTaskId,
    createTask,
    updateTask,
    deleteTask,
    persistAndStartTask,
    isLoading,
    error,
    reloadTasks: loadTasks,
  };
}

// Hook for task activity with API
function useTaskActivity(taskId: string | null) {
  const [activity, setActivity] = useState<ActivityEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!taskId) return;

    const loadActivity = async () => {
      try {
        setIsLoading(true);
        const data = await api.getTaskActivity(taskId);
        setActivity(data);
      } catch (err) {
        console.error("Failed to load activity:", err);
        // Use mock data as fallback
        setActivity([
          { id: "1", taskId, time: "14:23:12", type: "search", message: "🔍 Searching for information", timestamp: Date.now() },
          { id: "2", taskId, time: "14:22:45", type: "analysis", message: "🤔 Analyzing findings", timestamp: Date.now() - 27000 },
          { id: "3", taskId, time: "14:22:10", type: "summary", message: "📝 Creating summary", timestamp: Date.now() - 62000 },
          { id: "4", taskId, time: "14:21:30", type: "checkpoint", message: "💾 Saving checkpoint", timestamp: Date.now() - 102000 },
        ]);
      } finally {
        setIsLoading(false);
      }
    };

    loadActivity();

    // Poll for updates every 5 seconds when task is active
    const interval = setInterval(loadActivity, 5000);
    return () => clearInterval(interval);
  }, [taskId]);

  return { activity, isLoading };
}

// Hook for search history with API
function useSearchHistory(taskId: string | null) {
  const [searches, setSearches] = useState<SearchEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!taskId) return;

    const loadSearches = async () => {
      try {
        setIsLoading(true);
        const data = await api.getTaskSearchHistory(taskId);
        setSearches(data);
      } catch (err) {
        console.error("Failed to load search history:", err);
        // Use mock data as fallback
        setSearches([
          { id: "1", taskId, time: "14:23:12", query: "Anthropic research grants undergraduate", results: 8, timestamp: Date.now() },
          { id: "2", taskId, time: "14:22:30", query: "AAAI undergraduate fellowships 2025", results: 12, timestamp: Date.now() - 42000 },
          { id: "3", taskId, time: "14:21:15", query: "NSF REU AI programs deadline", results: 15, timestamp: Date.now() - 117000 },
        ]);
      } finally {
        setIsLoading(false);
      }
    };

    loadSearches();
  }, [taskId]);

  return { searches, isLoading };
}

// Hook for reports with API
function useReports(taskId: string | null) {
  const [reports, setReports] = useState<Report[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!taskId) return;

    const loadReports = async () => {
      try {
        setIsLoading(true);
        const data = await api.getTaskReports(taskId);
        setReports(data);
      } catch (err) {
        console.error("Failed to load reports:", err);
        // Use mock data as fallback
        setReports([
          {
            id: uid(),
            taskId,
            timestamp: Date.now(),
            title: "Research Summary",
            content: `### 📝 Research Summary

**Google AI Student Research Program**
• Eligibility: Senior undergraduates in CS/AI
• Funding: $10,000 stipend + conference travel
• Deadline: September 15, 2025
• Link: google.com/research/students

**Microsoft Research Fellowship**
• Eligibility: Final year undergraduates
• Funding: Full tuition + $42,000 stipend
• Deadline: October 1, 2025

### 🔍 Search Activity
Searched 15 sources across major tech companies and research institutions.

### 🤔 Analysis
The most prestigious opportunities have September-October deadlines.`,
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    };

    loadReports();
  }, [taskId]);

  return { reports, isLoading };
}

// Hook for metrics with API
function useMetrics(taskId: string | null) {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    llmCalls: 0,
    searches: 0,
    pagesAnalyzed: 0,
    summaries: 0,
    checkpoints: 0,
  });
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!taskId) return;

    const loadMetrics = async () => {
      try {
        setIsLoading(true);
        const data = await api.getTaskMetrics(taskId);
        setMetrics(data);
      } catch (err) {
        console.error("Failed to load metrics:", err);
        // Use mock data as fallback
        setMetrics({
          llmCalls: 128,
          searches: 42,
          pagesAnalyzed: 181,
          summaries: 19,
          checkpoints: 8,
        });
      } finally {
        setIsLoading(false);
      }
    };

    loadMetrics();

    // Poll for updates every 10 seconds
    const interval = setInterval(loadMetrics, 10000);
    return () => clearInterval(interval);
  }, [taskId]);

  return { metrics, isLoading };
}

// Sidebar Task Item
function TaskItem({
  task,
  isSelected,
  onSelect,
  onDelete,
  onRename,
}: {
  task: Task;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onRename: (name: string) => void;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(task.name);
  const [showMenu, setShowMenu] = useState(false);

  const statusIcon: Record<string, string> = {
    configuring: "⚙️",
    pending: "⏳",
    active: "🔍",
    complete: "✅",
    paused: "⏸️",
  };

  return (
    <div
      onClick={onSelect}
      className={`group relative rounded-lg px-3 py-2 cursor-pointer transition-all ${
        isSelected ? "bg-amber-600/30 backdrop-blur border border-amber-500/30" : "hover:bg-amber-700/20 hover:backdrop-blur"
      }`}
    >
      {isEditing ? (
        <input
          value={editName}
          onChange={(e) => setEditName(e.target.value)}
          onBlur={() => {
            onRename(editName);
            setIsEditing(false);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              onRename(editName);
              setIsEditing(false);
            }
            if (e.key === "Escape") {
              setEditName(task.name);
              setIsEditing(false);
            }
          }}
          onClick={(e) => e.stopPropagation()}
          className="w-full px-2 py-1 text-sm bg-amber-50 text-gray-900 rounded border border-amber-400 focus:outline-none focus:ring-2 focus:ring-amber-500"
          autoFocus
        />
      ) : (
        <>
          <div className="flex items-start gap-2">
            <span className="text-lg mt-0.5">{statusIcon[task.status]}</span>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium truncate text-amber-50">
                {task.name}
                {task.isDraft && <span className="ml-1 text-[10px] text-amber-200/80">(draft)</span>}
              </div>
              <div className="text-xs text-amber-200/70">{formatDate(task.updatedAt)}</div>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowMenu(!showMenu);
              }}
              className={`p-1 rounded hover:bg-amber-600/30 ${showMenu || isSelected ? "opacity-100" : "opacity-0 group-hover:opacity-100"} transition-opacity`}
            >
              <MoreHorizontal className="w-4 h-4 text-amber-200" />
            </button>
          </div>

          {showMenu && (
            <div className="absolute right-0 top-8 bg-amber-50 rounded-lg shadow-lg border border-amber-200 py-1 z-10 w-32">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setIsEditing(true);
                  setShowMenu(false);
                }}
                className="w-full px-3 py-1.5 text-left text-sm hover:bg-amber-100 flex items-center gap-2 text-gray-700"
              >
                <Edit3 className="w-3 h-3" /> Rename
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirm("Delete this task?")) {
                    onDelete();
                  }
                  setShowMenu(false);
                }}
                className="w-full px-3 py-1.5 text-left text-sm hover:bg-amber-100 flex items-center gap-2 text-red-600"
              >
                <Trash2 className="w-3 h-3" /> Delete
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// Task Configuration View
function TaskConfiguration({
  task,
  onUpdate,
  onStart, // Will persist draft (single POST) and start
}: {
  task: Task;
  onUpdate: (updates: Partial<Task>) => void;
  onStart: () => Promise<void>;
}) {
  const templates = [
    {
      id: "fellowships",
      icon: "🎓",
      name: "ML Fellowships",
      description:
        "Find currently open grants, programs, fellowships, or funding opportunities that support AI/ML research and are available to senior undergraduate students in the United States.",
      budget: 180,
    },
    {
      id: "conferences",
      icon: "📚",
      name: "AI Conferences",
      description:
        "Find AI/ML conferences, symposia, workshops, or student research programs that are currently accepting abstract submissions.",
      budget: 180,
    },
    {
      id: "competitions",
      icon: "🏆",
      name: "ML Competitions",
      description: "Find active AI/ML student competitions or challenges suitable for senior undergraduates.",
      budget: 180,
    },
  ] as const;

  const handleTemplateSelect = (template: (typeof templates)[number]) => {
    onUpdate({
      name: template.name,
      description: template.description,
      budget_minutes: template.budget,
    });
  };

  const handleStart = async () => {
    await onStart();
  };

  const canStart = !!task.name.trim() && !!task.description.trim();

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-4xl mx-auto p-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Configure Research Task</h1>
          <p className="text-gray-600">Set up your autonomous research objective</p>
        </div>

        {/* Template Selection */}
        <div className="mb-8">
          <h2 className="text-sm font-semibold text-gray-700 mb-4">Quick Start Templates</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {templates.map((template) => (
              <button
                key={template.id}
                onClick={() => handleTemplateSelect(template)}
                className="p-4 bg-white rounded-xl border-2 border-gray-200 hover:border-amber-400 hover:shadow-md transition-all text-left"
              >
                <div className="text-2xl mb-2">{template.icon}</div>
                <h3 className="font-semibold text-gray-900 mb-1">{template.name}</h3>
                <p className="text-xs text-gray-600 line-clamp-2">{template.description}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Task Configuration Form */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Task Name</label>
            <input
              type="text"
              value={task.name}
              onChange={(e) => onUpdate({ name: e.target.value })}
              placeholder="e.g., AI Research Grants"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Research Objective</label>
            <textarea
              value={task.description}
              onChange={(e) => onUpdate({ description: e.target.value })}
              placeholder="Describe what you want Kestrel to research..."
              rows={6}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent resize-none"
            />
            <p className="text-xs text-gray-500 mt-2">Be specific about eligibility criteria, deadlines, and what information to collect.</p>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Time Budget</label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="30"
                max="360"
                step="30"
                value={task.budget_minutes}
                onChange={(e) => onUpdate({ budget_minutes: Number(e.target.value) })}
                className="flex-1"
              />
              <div className="px-4 py-2 bg-amber-50 rounded-lg min-w-[120px] text-center">
                <span className="text-2xl font-bold text-amber-700">{task.budget_minutes}</span>
                <span className="text-sm text-amber-600 ml-1">minutes</span>
              </div>
            </div>
          </div>

          <div className="pt-4 border-t">
            <button
              onClick={handleStart}
              disabled={!canStart}
              className={`w-full py-3 rounded-lg font-semibold transition-all flex items-center justify-center gap-2 ${
                canStart ? "bg-gradient-to-r from-amber-600 to-orange-500 text-white hover:shadow-lg" : "bg-gray-200 text-gray-400 cursor-not-allowed"
              }`}
            >
              <Zap className="w-5 h-5" />
              Start Research
            </button>
            {task.isDraft && <p className="text-xs text-gray-500 mt-2">Note: This draft is only saved locally until you start.</p>}
          </div>
        </div>
      </div>
    </div>
  );
}

// Task Dashboard
function TaskDashboard({ task, onUpdate }: { task: Task; onUpdate: (id: string, updates: Partial<Task>) => void }) {
  const [isPaused, setIsPaused] = useState(task.status === "paused");

  const { activity } = useTaskActivity(task.id);
  const { searches } = useSearchHistory(task.id);
  const { reports } = useReports(task.id);
  const { metrics } = useMetrics(task.id);

  const handlePauseResume = async () => {
    if (task.isDraft) return; // should never happen in dashboard, but guard anyway
    try {
      if (task.status === "paused") {
        await api.resumeTask(task.id);
        onUpdate(task.id, { status: "active" });
      } else {
        await api.pauseTask(task.id);
        onUpdate(task.id, { status: "paused" });
      }
      setIsPaused(!isPaused);
    } catch (err) {
      console.error("Failed to pause/resume task:", err);
      onUpdate(task.id, { status: task.status === "paused" ? "active" : "paused" });
    }
  };

  const handleExport = async (format: "json" | "pdf" | "markdown" = "json") => {
    if (task.isDraft) {
      alert("Please start the task before exporting.");
      return;
    }
    try {
      const blob = await api.exportTask(task.id, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${task.name.replace(/\s+/g, "_")}_${Date.now()}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to export task:", err);
      alert("Failed to export task. Please try again.");
    }
  };

  return (
    <div className="flex-1 overflow-y-auto bg-gradient-to-br from-amber-50/50 via-white to-orange-50/50">
      {/* Header Bar */}
      <div className="bg-white/80 backdrop-blur border-b border-amber-200 sticky top-0 z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div>
                <h1 className="text-xl font-bold text-gray-900">{task.name}</h1>
                <p className="text-sm text-gray-600">{(task.description || "").substring(0, 100)}...</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="px-4 py-2 bg-gradient-to-r from-amber-900 to-orange-900 rounded-lg shadow-inner">
                <div className="text-lg font-mono font-bold text-amber-400">{formatElapsed(task.elapsed)}</div>
              </div>

              <button
                onClick={handlePauseResume}
                className={`px-4 py-2 rounded-lg font-semibold transition-all flex items-center gap-2 ${
                  isPaused ? "bg-green-600 hover:bg-green-700 text-white shadow-lg" : "bg-amber-600 hover:bg-amber-700 text-white shadow-lg"
                }`}
              >
                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                {isPaused ? "Resume" : "Pause"}
              </button>

              <button onClick={() => handleExport("json")} className="p-2 hover:bg-amber-100 rounded-lg transition-colors" title="Export as JSON">
                <Download className="w-5 h-5 text-amber-700" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Progress Overview */}
        <div className="bg-white/90 backdrop-blur rounded-xl border border-amber-200 p-6 shadow-lg">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-gray-900">Task Progress</h2>
            <span
              className={`px-3 py-1 rounded-full text-xs font-bold text-white ${
                task.status === "active"
                  ? "bg-gradient-to-r from-green-500 to-emerald-500"
                  : task.status === "complete"
                  ? "bg-gradient-to-r from-blue-500 to-sky-500"
                  : "bg-gradient-to-r from-amber-500 to-orange-500"
              } shadow-md`}
            >
              {task.status === "active" && <span className="inline-block w-2 h-2 bg-white rounded-full mr-1 animate-pulse" />}
              {task.status.toUpperCase()}
            </span>
          </div>

          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-600 mb-2">
              <span>Overall Progress</span>
              <span className="font-semibold">{task.progress?.toFixed(1) || "0.0"}%</span>
            </div>
            <div className="h-3 bg-amber-100 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-amber-500 to-orange-500 transition-all duration-500 shadow-inner" style={{ width: `${task.progress || 0}%` }} />
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-3 text-center border border-amber-200">
              <div className="text-2xl font-bold text-amber-700">{task.metrics?.searchCount ?? 0}</div>
              <div className="text-xs uppercase text-gray-600 font-semibold">Searches</div>
            </div>
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-3 text-center border border-amber-200">
              <div className="text-2xl font-bold text-amber-700">{task.metrics?.thinkCount ?? 0}</div>
              <div className="text-xs uppercase text-gray-600 font-semibold">Analysis</div>
            </div>
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-3 text-center border-amber-200">
              <div className="text-2xl font-bold text-amber-700">{task.metrics?.summaryCount ?? 0}</div>
              <div className="text-xs uppercase text-gray-600 font-semibold">Summaries</div>
            </div>
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-3 text-center border border-amber-200">
              <div className="text-2xl font-bold text-amber-700">{task.metrics?.checkpointCount ?? 0}</div>
              <div className="text-xs uppercase text-gray-600 font-semibold">Saves</div>
            </div>
          </div>
        </div>

        {/* System Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <div className="bg-white/90 backdrop-blur rounded-lg p-4 border border-amber-200 shadow-md hover:shadow-lg transition-shadow">
            <div className="flex items-center justify-between mb-1">
              <Brain className="w-4 h-4 text-amber-600" />
              <span className="text-2xl font-bold text-amber-700">{metrics.llmCalls}</span>
            </div>
            <div className="text-xs uppercase text-gray-600 font-semibold">LLM Calls</div>
          </div>
          <div className="bg-white/90 backdrop-blur rounded-lg p-4 border border-amber-200 shadow-md hover:shadow-lg transition-shadow">
            <div className="flex items-center justify-between mb-1">
              <Search className="w-4 h-4 text-amber-600" />
              <span className="text-2xl font-bold text-amber-700">{metrics.searches}</span>
            </div>
            <div className="text-xs uppercase text-gray-600 font-semibold">Searches</div>
          </div>
          <div className="bg-white/90 backdrop-blur rounded-lg p-4 border border-amber-200 shadow-md hover:shadow-lg transition-shadow">
            <div className="flex items-center justify-between mb-1">
              <Globe className="w-4 h-4 text-amber-600" />
              <span className="text-2xl font-bold text-amber-700">{metrics.pagesAnalyzed}</span>
            </div>
            <div className="text-xs uppercase text-gray-600 font-semibold">Pages</div>
          </div>
          <div className="bg-white/90 backdrop-blur rounded-lg p-4 border border-amber-200 shadow-md hover:shadow-lg transition-shadow">
            <div className="flex items-center justify-between mb-1">
              <FileText className="w-4 h-4 text-amber-600" />
              <span className="text-2xl font-bold text-amber-700">{metrics.summaries}</span>
            </div>
            <div className="text-xs uppercase text-gray-600 font-semibold">Summaries</div>
          </div>
          <div className="bg-white/90 backdrop-blur rounded-lg p-4 border border-amber-200 shadow-md hover:shadow-lg transition-shadow">
            <div className="flex items-center justify-between mb-1">
              <Save className="w-4 h-4 text-amber-600" />
              <span className="text-2xl font-bold text-amber-700">{metrics.checkpoints}</span>
            </div>
            <div className="text-xs uppercase text-gray-600 font-semibold">Checkpoints</div>
          </div>
        </div>

        {/* Activity and Search */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Activity Feed */}
          <div className="bg-white/90 backdrop-blur rounded-xl border border-amber-200 shadow-lg">
            <div className="p-4 border-b border-amber-100 bg-gradient-to-r from-amber-50 to-orange-50">
              <h3 className="font-bold text-gray-900 flex items-center gap-2">
                <Activity className="w-4 h-4 text-amber-600" />
                Live Activity
              </h3>
            </div>
            <div className="p-4 space-y-2 max-h-64 overflow-y-auto">
              {activity.map((entry) => (
                <div key={entry.id} className="flex items-center gap-3 p-2 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg border border-amber-100">
                  <span className="text-xs font-mono text-amber-700 font-semibold min-w-[60px]">{entry.time}</span>
                  <span className="text-sm text-gray-700">{entry.message}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Search History */}
          <div className="bg-white/90 backdrop-blur rounded-xl border border-amber-200 shadow-lg">
            <div className="p-4 border-b border-amber-100 bg-gradient-to-r from-amber-50 to-orange-50">
              <h3 className="font-bold text-gray-900 flex items-center gap-2">
                <Search className="w-4 h-4 text-amber-600" />
                Search Intelligence
              </h3>
            </div>
            <div className="p-4 space-y-2 max-h-64 overflow-y-auto">
              {searches.map((search) => (
                <div key={search.id} className="flex items-center gap-2 p-2 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg text-xs border border-amber-100">
                  <span className="font-mono text-amber-700 font-semibold min-w-[55px]">{search.time}</span>
                  <span className="flex-1 text-gray-700 truncate">{search.query}</span>
                  <span className="font-bold text-amber-700">({search.results})</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Reports */}
        {reports.length > 0 && (
          <div className="bg-white/90 backdrop-blur rounded-xl border border-amber-200 shadow-lg">
            <div className="p-4 border-b border-amber-100 bg-gradient-to-r from-amber-50 to-orange-50">
              <div className="flex items-center justify-between">
                <h3 className="font-bold text-gray-900">Research Report</h3>
                <Pager total={reports.length} />
              </div>
            </div>
            <div className="p-6 max-h-96 overflow-y-auto">
              <div className="prose prose-sm max-w-none">
                <div className="whitespace-pre-wrap text-gray-700 text-sm">{reports[0]?.content}</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Pager({ total }: { total: number }) {
  const [index, setIndex] = useState(0);
  return (
    <div className="flex items-center gap-2">
      <button onClick={() => setIndex((i) => Math.max(0, i - 1))} disabled={index === 0} className="p-1 hover:bg-amber-100 rounded disabled:opacity-50">
        <ChevronLeft className="w-4 h-4" />
      </button>
      <span className="text-sm text-gray-600 px-2">
        {index + 1} / {total}
      </span>
      <button onClick={() => setIndex((i) => Math.min(total - 1, i + 1))} disabled={index === total - 1} className="p-1 hover:bg-amber-100 rounded disabled:opacity-50">
        <ChevronRight className="w-4 h-4" />
      </button>
    </div>
  );
}

// Main App Component
export default function App() {
  const {
    tasks,
    selectedTask,
    selectedTaskId,
    setSelectedTaskId,
    createTask,
    updateTask,
    deleteTask,
    persistAndStartTask,
    isLoading,
    error,
  } = useTaskManager();

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // Filter tasks based on search
  const filteredTasks = tasks.filter(
    (task) =>
      task.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (task.description || "").toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleStartTask = async () => {
    if (!selectedTask) return;
    await persistAndStartTask(selectedTask.id);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-amber-50 to-orange-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-5xl mb-4 animate-pulse">🦅</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-amber-50 flex">
      {/* Error Toast */}
      {error && (
        <div className="fixed top-4 right-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded z-50">
          <span className="block sm:inline">{error}</span>
        </div>
      )}

      {/* Sidebar */}
      <div
        className={`fixed inset-y-0 left-0 z-40 w-64 bg-gradient-to-b from-amber-900 via-amber-800 to-orange-900 transform transition-transform lg:relative lg:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex flex-col h-full">
          {/* Sidebar Header */}
          <div className="p-4">
            <button
              onClick={async () => {
                const newId = await createTask(); // local draft only
                setSelectedTaskId(newId);
              }}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-amber-700/50 hover:bg-amber-700/70 backdrop-blur text-amber-50 rounded-lg transition-all border border-amber-600/30"
            >
              <Plus className="w-4 h-4" />
              New Task
            </button>
          </div>

          {/* Search */}
          <div className="px-4 pb-2">
            <div className="relative">
              <Search className="absolute left-3 top-2.5 w-4 h-4 text-amber-300" />
              <input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search tasks..."
                className="w-full pl-9 pr-3 py-2 bg-amber-800/30 backdrop-blur text-amber-50 placeholder-amber-300/70 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 border border-amber-700/30"
              />
            </div>
          </div>

          {/* Tasks List */}
          <div className="flex-1 overflow-y-auto px-2">
            <div className="space-y-1">
              {filteredTasks.map((task) => (
                <TaskItem
                  key={task.id}
                  task={task}
                  isSelected={task.id === selectedTaskId}
                  onSelect={() => setSelectedTaskId(task.id)}
                  onDelete={() => deleteTask(task.id)}
                  onRename={(name) => updateTask(task.id, { name })}
                />
              ))}
            </div>

            {filteredTasks.length === 0 && <div className="text-center py-8 text-amber-300/70 text-sm">{searchQuery ? "No tasks found" : "No tasks yet"}</div>}
          </div>

          {/* Sidebar Footer */}
          <div className="p-4 border-t border-amber-700/30">
            <div className="flex items-center gap-3">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <div className="text-2xl">🦅</div>
                  <div>
                    <div className="text-sm font-semibold text-amber-50">KestrelAI</div>
                    <div className="text-xs text-amber-200/70">Research Agent</div>
                  </div>
                </div>
              </div>
              <button className="p-2 hover:bg-amber-700/30 rounded-lg transition-colors">
                <Settings className="w-4 h-4 text-amber-300" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile sidebar toggle */}
      {!sidebarOpen && (
        <button onClick={() => setSidebarOpen(true)} className="fixed top-4 left-4 z-30 p-2 bg-white rounded-lg shadow-lg lg:hidden">
          <Menu className="w-5 h-5" />
        </button>
      )}

      {/* Overlay for mobile */}
      {sidebarOpen && <div className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden" onClick={() => setSidebarOpen(false)} />}

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {selectedTask ? (
          selectedTask.status === "configuring" ? (
            <TaskConfiguration
              task={selectedTask}
              onUpdate={(updates) => updateTask(selectedTask.id, updates)}
              onStart={handleStartTask}
            />
          ) : (
            <TaskDashboard task={selectedTask} onUpdate={updateTask} />
          )
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <div className="text-6xl mb-4">🦅</div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Welcome to KestrelAI</h2>
              <p className="text-gray-600 mb-6">Create a new research task to get started</p>
              <button
                onClick={async () => {
                  const newId = await createTask(); // local draft only
                  setSelectedTaskId(newId);
                }}
                className="px-6 py-3 bg-gradient-to-r from-amber-600 to-orange-500 text-white rounded-lg font-semibold hover:shadow-lg transition-all inline-flex items-center gap-2"
              >
                <Plus className="w-5 h-5" />
                Create Your First Task
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
