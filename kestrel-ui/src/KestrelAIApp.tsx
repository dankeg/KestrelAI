import React, { useEffect, useRef, useState } from "react";
import {
  Plus,
  Search,
  Play,
  Pause,
  FileText,
  ChevronLeft,
  ChevronRight,
  Menu,
  Settings,
  Download,
  Brain,
  Globe,
  Save,
  MoreHorizontal,
  Edit3,
  Trash2,
  Zap,
  Activity,
} from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

// ==========================
// API Configuration
// ==========================
const API_BASE_URL = "http://localhost:8000/api/v1";

// ==========================
// Types
// ==========================
interface Metrics {
  searchCount: number;
  thinkCount: number;
  summaryCount: number;
  checkpointCount: number;
  webFetchCount?: number;
  llmTokensUsed?: number;
  errorCount?: number;
}

type TaskStatus = "configuring" | "pending" | "active" | "complete" | "paused" | "failed";

interface Task {
  id: string;
  name: string;
  description: string;
  budgetMinutes: number;
  status: TaskStatus;
  progress?: number;
  elapsed?: number;
  metrics?: Metrics;
  createdAt?: number;
  updatedAt?: number;

  /** Frontend-only: true until we POST this task on Start Research */
  isDraft?: boolean;
}

interface ActivityEntry {
  id: string;
  taskId: string;
  time: string;
  type:
    | "task_start"
    | "search"
    | "analysis"
    | "summary"
    | "checkpoint"
    | "error"
    | "thinking"
    | "web_fetch";
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
  sources?: string[];
}

interface Report {
  id: string;
  taskId: string;
  timestamp: number;
  title: string;
  content: string;
  format?: "markdown" | "text" | "html";
  metadata?: Record<string, any>;
}

interface SystemMetrics {
  llmCalls: number;
  searches: number;
  pagesAnalyzed: number;
  summaries: number;
  checkpoints: number;
  tokensUsed?: number;
  estimatedCost?: number;
}

// ==========================
// API Client
// ==========================
class KestrelAPI {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  private async fetch(endpoint: string, options?: RequestInit) {
    const url = `${this.baseURL}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => "");
      throw new Error(`API Error: ${response.status} ${response.statusText} ${errorText}`);
    }

    // Some endpoints may return 204; guard that.
    const text = await response.text();
    try {
      return text ? JSON.parse(text) : null;
    } catch {
      return text ?? null;
    }
  }

  // ---- Task endpoints ----
  async getTasks(): Promise<Task[]> {
    const res = await this.fetch("/tasks");
    return (res || []).map(normalizeTask);
  }

  async getTask(id: string): Promise<Task> {
    const res = await this.fetch(`/tasks/${id}`);
    return normalizeTask(res);
  }

  async createTask(task: Partial<Task>): Promise<Task> {
    const body = {
      name: task.name,
      description: task.description,
      budgetMinutes: task.budgetMinutes,
      // config: task.config ?? {}
    };
    const res = await this.fetch("/tasks", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return normalizeTask(res);
  }

  async updateTask(id: string, updates: Partial<Task>): Promise<Task> {
    const body: Record<string, any> = { ...updates };
    delete body.isDraft; // frontend-only
    const res = await this.fetch(`/tasks/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    });
    return normalizeTask(res);
  }

  async deleteTask(id: string): Promise<void> {
    await this.fetch(`/tasks/${id}`, { method: "DELETE" });
  }

  async startTask(id: string): Promise<Task> {
    const res = await this.fetch(`/tasks/${id}/start`, { method: "POST" });
    return normalizeTask(res);
  }

  async pauseTask(id: string): Promise<Task> {
    const res = await this.fetch(`/tasks/${id}/pause`, { method: "POST" });
    return normalizeTask(res);
  }

  async resumeTask(id: string): Promise<Task> {
    const res = await this.fetch(`/tasks/${id}/resume`, { method: "POST" });
    return normalizeTask(res);
  }

  // ---- Activity/Search/Reports/Metrics ----
  async getTaskActivity(taskId: string, limit: number = 50): Promise<ActivityEntry[]> {
    const res = await this.fetch(`/tasks/${taskId}/activity?limit=${limit}`);
    return res || [];
  }

  async getTaskSearchHistory(taskId: string, limit: number = 50): Promise<SearchEntry[]> {
    const res = await this.fetch(`/tasks/${taskId}/searches?limit=${limit}`);
    return res || [];
  }

  async getTaskReports(taskId: string): Promise<Report[]> {
    const res = await this.fetch(`/tasks/${taskId}/reports`);
    return res || [];
  }

  async getTaskMetrics(taskId: string): Promise<SystemMetrics> {
    const res = await this.fetch(`/tasks/${taskId}/metrics`);
    return res || {
      llmCalls: 0,
      searches: 0,
      pagesAnalyzed: 0,
      summaries: 0,
      checkpoints: 0,
    };
  }

  // ---- Export ----
  async exportTask(taskId: string, format: "json" | "pdf" | "markdown" = "json"): Promise<Blob> {
    const response = await fetch(`${this.baseURL}/tasks/${taskId}/export?format=${format}`);
    if (!response.ok) throw new Error(`Export failed: ${response.status}`);
    return response.blob();
  }
}

// Initialize API client
const api = new KestrelAPI();

// ==========================
// Utilities
// ==========================
const uid = () => Math.random().toString(36).slice(2, 10);

const defaultMetrics = (): Metrics => ({
  searchCount: 0,
  thinkCount: 0,
  summaryCount: 0,
  checkpointCount: 0,
});

const normalizeTask = (t: any): Task => {
  const budgetMinutes = t?.budgetMinutes ?? t?.budget_minutes ?? 180;
  return {
    ...t,
    budgetMinutes,
    metrics: t?.metrics ?? defaultMetrics(),
    isDraft: !!t?.isDraft, // default false
  };
};

const formatElapsed = (seconds?: number): string => {
  if (seconds === undefined || seconds === null) return "--:--";
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

  if (date.toDateString() === today.toDateString()) return "Today";
  if (date.toDateString() === yesterday.toDateString()) return "Yesterday";
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
};

const wsUrlForTask = (taskId: string) => {
  const base = API_BASE_URL.replace(/^http/i, "ws").replace(/\/api\/v1$/, "");
  return `${base}/ws/tasks/${taskId}`;
};

// ==========================
// Markdown Renderer Component
// ==========================
const mdComponents: Components = {
  h1: ({ node, ...props }) => (
    <h1 className="text-2xl font-bold text-gray-900 mt-4 mb-2" {...props} />
  ),
  h2: ({ node, ...props }) => (
    <h2 className="text-xl font-bold text-gray-900 mt-4 mb-2" {...props} />
  ),
  h3: ({ node, ...props }) => (
    <h3 className="text-lg font-bold text-gray-900 mt-4 mb-2" {...props} />
  ),
  p: ({ node, ...props }) => <p className="mb-3" {...props} />,

  // Open links in a new tab (replacement for removed linkTarget)
  a: ({ node, ...props }) => (
    <a
      target="_blank"
      rel="noopener noreferrer"
      className="text-amber-600 hover:text-amber-700 underline"
      {...props}
    />
  ),

  ul: ({ node, ...props }) => (
    <ul className="my-2 space-y-1 list-disc ml-6" {...props} />
  ),
  ol: ({ node, ...props }) => (
    <ol className="my-2 space-y-1 list-decimal ml-6" {...props} />
  ),
  li: ({ node, ...props }) => <li className="marker:text-gray-600" {...props} />,
  blockquote: ({ node, ...props }) => (
    <blockquote
      className="border-l-4 border-amber-400 pl-4 my-2 text-gray-700"
      {...props}
    />
  ),
  hr: ({ node, ...props }) => (
    <hr className="my-4 border-t border-gray-300" {...props} />
  ),

  // Properly typed `code` with `inline`
  code({ inline, className, children, ...props }) {
    return inline ? (
      <code className="bg-gray-100 px-1 py-0.5 rounded text-sm" {...props}>
        {children}
      </code>
    ) : (
      <pre className="bg-gray-100 p-3 rounded-lg overflow-x-auto my-2">
        <code className={className} {...props}>
          {children}
        </code>
      </pre>
    );
  },

  // GFM tables
  table: ({ node, ...props }) => (
    <div className="overflow-x-auto my-3">
      <table className="min-w-full border-collapse" {...props} />
    </div>
  ),
  thead: ({ node, ...props }) => <thead className="bg-gray-50" {...props} />,
  th: ({ node, ...props }) => (
    <th className="border px-3 py-1 text-left font-semibold" {...props} />
  ),
  td: ({ node, ...props }) => (
    <td className="border px-3 py-1 align-top" {...props} />
  ),
  tr: ({ node, ...props }) => <tr className="even:bg-gray-50" {...props} />,
};

function MarkdownRenderer({ content }: { content: string }) {
  return (
    <div className="prose prose-sm max-w-none text-gray-700">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
        {content}
      </ReactMarkdown>
    </div>
  );
}


// ==========================
// Hooks: Task Manager
// ==========================
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
      const serverNormalized: Task[] = serverTasks.map(normalizeTask);
      // Merge local drafts from localStorage
      let drafts: Task[] = [];
      try {
        const saved = localStorage.getItem("kestrel.tasks");
        if (saved) {
          const parsed = JSON.parse(saved);
          if (Array.isArray(parsed)) {
            drafts = parsed.filter((t: Task) => t.isDraft === true).map(normalizeTask);
          }
        }
      } catch {
        /* ignore parse errors */
      }
      const combined = [...drafts, ...serverNormalized];
      setTasks(combined);
      if (combined.length > 0 && !selectedTaskId) {
        setSelectedTaskId(combined[0].id);
      }
      setError(null);
    } catch (err) {
      console.error("Failed to load tasks from API:", err);
      setError("Failed to load tasks from API. Using local data.");
      // Fallback to localStorage
      try {
        const saved = localStorage.getItem("kestrel.tasks");
        if (saved) {
          const parsed = JSON.parse(saved);
          if (Array.isArray(parsed)) {
            const normalized = parsed.map(normalizeTask);
            setTasks(normalized);
            if (normalized.length > 0 && normalized[0].id) {
              setSelectedTaskId(normalized[0].id);
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
      budgetMinutes: 180,
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
    // Optimistically update local
    setTasks((prev) => {
      const next = prev.map((t) =>
        t.id === id ? normalizeTask({ ...t, ...updates, updatedAt: Date.now() }) : t
      );
      localStorage.setItem("kestrel.tasks", JSON.stringify(next));
      return next;
    });

    // If the task exists on the server, sync PATCH (best-effort)
    const existing = tasks.find((t) => t.id === id);
    if (existing && !existing.isDraft) {
      try {
        const updated = await api.updateTask(id, updates);
        // Reconcile with server response (authoritative)
        setTasks((prev) => {
          const next = prev.map((t) => (t.id === id ? normalizeTask({ ...t, ...updated }) : t));
          localStorage.setItem("kestrel.tasks", JSON.stringify(next));
          return next;
        });
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
          budgetMinutes: t.budgetMinutes,
        });

        // Replace draft with server task (do NOT force status)
        setTasks((prev) => {
          const next = prev.map((x) => (x.id === id ? { ...created, isDraft: false } : x));
          localStorage.setItem("kestrel.tasks", JSON.stringify(next));
          return next;
        });
        setSelectedTaskId(created.id);

        // Start the task (command-style)
        try {
          await api.startTask(created.id);
          // Fetch fresh server state after start
          const fresh = await api.getTask(created.id);
          setTasks((prev) => {
            const next = prev.map((x) => (x.id === created.id ? normalizeTask(fresh) : x));
            localStorage.setItem("kestrel.tasks", JSON.stringify(next));
            return next;
          });
        } catch (err) {
          console.error("Failed to start task on server:", err);
        }
      } catch (err) {
        console.error("Failed to persist draft task:", err);
        throw err;
      }
    } else {
      // Already on server: just start it
      try {
        await api.startTask(id);
        const fresh = await api.getTask(id);
        setTasks((prev) => {
          const next = prev.map((x) => (x.id === id ? normalizeTask(fresh) : x));
          localStorage.setItem("kestrel.tasks", JSON.stringify(next));
          return next;
        });
      } catch (err) {
        console.error("Failed to start task:", err);
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

// ==========================
// Hooks: Data (Activity/Search/Reports/Metrics)
// ==========================
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
    // Poll safety-net every 30s (WS handles real-time)
    const interval = setInterval(loadActivity, 30000);
    return () => clearInterval(interval);
  }, [taskId]);

  return {
    activity,
    isLoading,
    appendActivity: (entry: ActivityEntry) => setActivity((prev) => [entry, ...prev]),
  };
}

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
    // No polling needed beyond initial load; WS will update
  }, [taskId]);

  return {
    searches,
    isLoading,
    appendSearch: (entry: SearchEntry) => setSearches((prev) => [entry, ...prev]),
  };
}

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
        setReports([
          {
            id: uid(),
            taskId: taskId!,
            timestamp: Date.now(),
            title: "Research Summary",
            content: `# Research Summary

## Key Findings

This is a **placeholder report** with *markdown* formatting.

### Important Points
- First finding with **bold text**
- Second finding with *italic text*
- Third finding with a [link](https://example.com)

### Code Example
\`\`\`python
def example():
    return "Hello World"
\`\`\`

> This is a blockquote with important information

---

### Conclusion
The research has uncovered several important insights that will guide our next steps.`,
            format: "markdown",
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    };

    loadReports();
  }, [taskId]);

  return {
    reports,
    isLoading,
    appendReport: (report: Report) => setReports((prev) => [report, ...prev]),
  };
}

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

    // Poll every 30 seconds (WS will push more frequent updates)
    const interval = setInterval(loadMetrics, 30000);
    return () => clearInterval(interval);
  }, [taskId]);

  return { metrics, isLoading, setMetrics };
}

// ==========================
// Realtime (WebSocket)
// ==========================
function useTaskRealtime(
  task: Task | null,
  handlers: {
    onStatus: (update: any) => void;
    onActivity: (entry: ActivityEntry) => void;
    onSearch: (entry: SearchEntry) => void;
    onReport: (entry: Report) => void;
    onMetrics: (m: SystemMetrics) => void;
  }
) {
  useEffect(() => {
    if (!task) return;
    const url = wsUrlForTask(task.id);
    const ws = new WebSocket(url);

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        const payload = msg.payload ?? msg.data; // backend may use either
        switch (msg.type) {
          case "status":
            handlers.onStatus(payload);
            break;
          case "activity":
            handlers.onActivity(payload as ActivityEntry);
            break;
          case "search":
            handlers.onSearch(payload as SearchEntry);
            break;
          case "report":
            handlers.onReport(payload as Report);
            break;
          case "metrics":
            handlers.onMetrics(payload as SystemMetrics);
            break;
          default:
            break;
        }
      } catch (e) {
        console.error("WS parse error:", e);
      }
    };

    ws.onerror = (e) => console.error("WS error:", e);
    return () => ws.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task?.id]);
}

// ==========================
// Sidebar Task Item
// ==========================
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
    failed: "❌",
  };

  return (
    <div
      onClick={onSelect}
      className={`group relative rounded-lg px-3 py-2 cursor-pointer transition-all ${
        isSelected
          ? "bg-amber-600/30 backdrop-blur border border-amber-500/30"
          : "hover:bg-amber-700/20 hover:backdrop-blur"
      }`}
    >
      {isEditing ? (
        <input
          value={editName}
          onChange={(e) => setEditName(e.target.value)}
          onBlur={() => {
            onRename(editName.trim() || task.name);
            setIsEditing(false);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              onRename(editName.trim() || task.name);
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
                {task.isDraft && (
                  <span className="ml-1 text-[10px] text-amber-200/80">(draft)</span>
                )}
              </div>
              <div className="text-xs text-amber-200/70">{formatDate(task.updatedAt)}</div>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowMenu(!showMenu);
              }}
              className={`p-1 rounded hover:bg-amber-600/30 ${
                showMenu || isSelected ? "opacity-100" : "opacity-0 group-hover:opacity-100"
              } transition-opacity`}
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

// ==========================
// Task Configuration View
// ==========================
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
      description:
        "Find active AI/ML student competitions or challenges suitable for senior undergraduates.",
      budget: 180,
    },
  ] as const;

  const handleTemplateSelect = (template: (typeof templates)[number]) => {
    onUpdate({
      name: template.name,
      description: template.description,
      budgetMinutes: template.budget,
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
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Configure Research Task
          </h1>
          <p className="text-gray-600">Set up your autonomous research objective</p>
        </div>

        {/* Template Selection */}
        <div className="mb-8">
          <h2 className="text-sm font-semibold text-gray-700 mb-4">
            Quick Start Templates
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {templates.map((template) => (
              <button
                key={template.id}
                onClick={() => handleTemplateSelect(template)}
                className="p-4 bg-white rounded-xl border-2 border-gray-200 hover:border-amber-400 hover:shadow-md transition-all text-left"
              >
                <div className="text-2xl mb-2">{template.icon}</div>
                <h3 className="font-semibold text-gray-900 mb-1">{template.name}</h3>
                <p className="text-xs text-gray-600 line-clamp-2">
                  {template.description}
                </p>
              </button>
            ))}
          </div>
        </div>

        {/* Task Configuration Form */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Task Name
            </label>
            <input
              type="text"
              value={task.name}
              onChange={(e) => onUpdate({ name: e.target.value })}
              placeholder="e.g., AI Research Grants"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Research Objective
            </label>
            <textarea
              value={task.description}
              onChange={(e) => onUpdate({ description: e.target.value })}
              placeholder="Describe what you want Kestrel to research..."
              rows={6}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent resize-none"
            />
            <p className="text-xs text-gray-500 mt-2">
              Be specific about eligibility criteria, deadlines, and what information to
              collect.
            </p>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Time Budget
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="30"
                max="360"
                step="30"
                value={task.budgetMinutes}
                onChange={(e) =>
                  onUpdate({ budgetMinutes: Number(e.target.value) })
                }
                className="flex-1"
              />
              <div className="px-4 py-2 bg-amber-50 rounded-lg min-w-[120px] text-center">
                <span className="text-2xl font-bold text-amber-700">
                  {task.budgetMinutes}
                </span>
                <span className="text-sm text-amber-600 ml-1">minutes</span>
              </div>
            </div>
          </div>

          <div className="pt-4 border-t">
            <button
              onClick={handleStart}
              disabled={!canStart}
              className={`w-full py-3 rounded-lg font-semibold transition-all flex items-center justify-center gap-2 ${
                canStart
                  ? "bg-gradient-to-r from-amber-600 to-orange-500 text-white hover:shadow-lg"
                  : "bg-gray-200 text-gray-400 cursor-not-allowed"
              }`}
            >
              <Zap className="w-5 h-5" />
              Start Research
            </button>
            {task.isDraft && (
              <p className="text-xs text-gray-500 mt-2">
                Note: This draft is only saved locally until you start.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ==========================
// Task Dashboard
// ==========================
function TaskDashboard({
  task,
  onUpdate,
}: {
  task: Task;
  onUpdate: (id: string, updates: Partial<Task>) => void;
}) {
  const [isPaused, setIsPaused] = useState(task.status === "paused");
  const [currentReportIndex, setCurrentReportIndex] = useState(0);

  const { activity, appendActivity } = useTaskActivity(task.id);
  const { searches, appendSearch } = useSearchHistory(task.id);
  const { reports, appendReport } = useReports(task.id);
  const { metrics, setMetrics } = useMetrics(task.id);

  // Reset report index when reports change
  useEffect(() => {
    setCurrentReportIndex(0);
  }, [reports.length]);

  // Realtime wiring
  useTaskRealtime(task, {
    onStatus: (u) => {
      const merged: Partial<Task> = {};
      if (u?.status) merged.status = u.status;
      if (typeof u?.progress === "number") merged.progress = u.progress;
      if (typeof u?.elapsed === "number") merged.elapsed = u.elapsed;
      if (u?.metrics) {
        merged.metrics = { ...(task.metrics || {}), ...(u.metrics as Metrics) };
      }
      onUpdate(task.id, merged);
      if (u?.status) setIsPaused(u.status === "paused");
    },
    onActivity: appendActivity,
    onSearch: appendSearch,
    onReport: (report) => {
      appendReport(report);
      // Reset to first report when new report arrives
      setCurrentReportIndex(0);
    },
    onMetrics: setMetrics,
  });

  const handlePauseResume = async () => {
    if (task.isDraft) return;
    try {
      if (task.status === "paused") {
        const updated = await api.resumeTask(task.id);
        onUpdate(task.id, { status: updated.status });
        setIsPaused(false);
      } else {
        const updated = await api.pauseTask(task.id);
        onUpdate(task.id, { status: updated.status });
        setIsPaused(true);
      }
    } catch (err) {
      console.error("Failed to pause/resume task:", err);
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
      a.download = `${task.name.replace(/\s+/g, "_")}_${Date.now()}.${format === "markdown" ? "md" : format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to export task:", err);
      alert("Failed to export task. Please try again.");
    }
  };

  const currentReport = reports[currentReportIndex];

  return (
    <div className="flex-1 overflow-y-auto bg-gradient-to-br from-amber-50/50 via-white to-orange-50/50">
      {/* Header Bar */}
      <div className="bg-white/80 backdrop-blur border-b border-amber-200 sticky top-0 z-10">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div>
                <h1 className="text-xl font-bold text-gray-900">{task.name}</h1>
                <p className="text-sm text-gray-600">
                  {(task.description || "").substring(0, 100)}...
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="px-4 py-2 bg-gradient-to-r from-amber-900 to-orange-900 rounded-lg shadow-inner">
                <div className="text-lg font-mono font-bold text-amber-400">
                  {formatElapsed(task.elapsed)}
                </div>
              </div>

              <button
                onClick={handlePauseResume}
                className={`px-4 py-2 rounded-lg font-semibold transition-all flex items-center gap-2 ${
                  isPaused
                    ? "bg-green-600 hover:bg-green-700 text-white shadow-lg"
                    : "bg-amber-600 hover:bg-amber-700 text-white shadow-lg"
                }`}
              >
                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                {isPaused ? "Resume" : "Pause"}
              </button>

              <button
                onClick={() => handleExport("json")}
                className="p-2 hover:bg-amber-100 rounded-lg transition-colors"
                title="Export as JSON"
              >
                <Download className="w-5 h-5 text-amber-700" />
              </button>
              <button
                onClick={() => handleExport("markdown")}
                className="p-2 hover:bg-amber-100 rounded-lg transition-colors"
                title="Export as Markdown"
              >
                <FileText className="w-5 h-5 text-amber-700" />
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
                  : task.status === "paused"
                  ? "bg-gradient-to-r from-amber-500 to-orange-500"
                  : task.status === "failed"
                  ? "bg-gradient-to-r from-red-500 to-rose-500"
                  : "bg-gradient-to-r from-gray-400 to-gray-500"
              } shadow-md`}
            >
              {task.status === "active" && (
                <span className="inline-block w-2 h-2 bg-white rounded-full mr-1 animate-pulse" />
              )}
              {task.status.toUpperCase()}
            </span>
          </div>

          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-600 mb-2">
              <span>Overall Progress</span>
              <span className="font-semibold">
                {typeof task.progress === "number" ? task.progress.toFixed(1) : "0.0"}%
              </span>
            </div>
            <div className="h-3 bg-amber-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-amber-500 to-orange-500 transition-all duration-500 shadow-inner"
                style={{ width: `${task.progress || 0}%` }}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-3 text-center border border-amber-200">
              <div className="text-2xl font-bold text-amber-700">
                {task.metrics?.searchCount ?? 0}
              </div>
              <div className="text-xs uppercase text-gray-600 font-semibold">
                Searches
              </div>
            </div>
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-3 text-center border border-amber-200">
              <div className="text-2xl font-bold text-amber-700">
                {task.metrics?.thinkCount ?? 0}
              </div>
              <div className="text-xs uppercase text-gray-600 font-semibold">
                Analysis
              </div>
            </div>
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-3 text-center border-amber-200">
              <div className="text-2xl font-bold text-amber-700">
                {task.metrics?.summaryCount ?? 0}
              </div>
              <div className="text-xs uppercase text-gray-600 font-semibold">
                Summaries
              </div>
            </div>
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-3 text-center border border-amber-200">
              <div className="text-2xl font-bold text-amber-700">
                {task.metrics?.checkpointCount ?? 0}
              </div>
              <div className="text-xs uppercase text-gray-600 font-semibold">
                Saves
              </div>
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
              <span className="text-2xl font-bold text-amber-700">
                {metrics.pagesAnalyzed}
              </span>
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
              <span className="text-2xl font-bold text-amber-700">
                {metrics.checkpoints}
              </span>
            </div>
            <div className="text-xs uppercase text-gray-600 font-semibold">
              Checkpoints
            </div>
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
                <div
                  key={entry.id}
                  className="flex items-center gap-3 p-2 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg border border-amber-100"
                >
                  <span className="text-xs font-mono text-amber-700 font-semibold min-w-[60px]">
                    {entry.time}
                  </span>
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
                <div
                  key={search.id}
                  className="flex items-center gap-2 p-2 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg text-xs border border-amber-100"
                >
                  <span className="font-mono text-amber-700 font-semibold min-w-[55px]">
                    {search.time}
                  </span>
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
                <h3 className="font-bold text-gray-900">
                  {currentReport?.title || "Research Report"}
                </h3>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setCurrentReportIndex((i) => Math.max(0, i - 1))}
                    disabled={currentReportIndex === 0}
                    className="p-1 hover:bg-amber-100 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <ChevronLeft className="w-4 h-4 text-amber-700" />
                  </button>
                  <span className="text-sm text-gray-600 px-2 min-w-[60px] text-center">
                    {currentReportIndex + 1} / {reports.length}
                  </span>
                  <button
                    onClick={() => setCurrentReportIndex((i) => Math.min(reports.length - 1, i + 1))}
                    disabled={currentReportIndex === reports.length - 1}
                    className="p-1 hover:bg-amber-100 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <ChevronRight className="w-4 h-4 text-amber-700" />
                  </button>
                </div>
              </div>
            </div>
            <div className="p-6 max-h-96 overflow-y-auto">
              {currentReport && (
                currentReport.format === "markdown" ? (
                  <MarkdownRenderer content={currentReport.content} />
                ) : (
                  <div className="whitespace-pre-wrap text-gray-700 text-sm">
                    {currentReport.content}
                  </div>
                )
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ==========================
// Main App Component
// ==========================
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

            {filteredTasks.length === 0 && (
              <div className="text-center py-8 text-amber-300/70 text-sm">
                {searchQuery ? "No tasks found" : "No tasks yet"}
              </div>
            )}
          </div>

          {/* Sidebar Footer */}
          <div className="p-4 border-t border-amber-700/30">
            <div className="flex items-center gap-3">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <div className="text-2xl">🦅</div>
                  <div>
                    <div className="text-sm font-semibold text-amber-50">
                      KestrelAI
                    </div>
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
        <button
          onClick={() => setSidebarOpen(true)}
          className="fixed top-4 left-4 z-30 p-2 bg-white rounded-lg shadow-lg lg:hidden"
        >
          <Menu className="w-5 h-5" />
        </button>
      )}

      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

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
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Welcome to KestrelAI
              </h2>
              <p className="text-gray-600 mb-6">
                Create a new research task to get started
              </p>
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