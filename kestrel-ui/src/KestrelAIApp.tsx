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

// Types
interface Task {
  id: string;
  name: string;
  description: string;
  budgetMinutes: number;
  status: "configuring" | "pending" | "active" | "complete" | "paused";
  progress: number;
  elapsed: number;
  metrics: {
    searchCount: number;
    thinkCount: number;
    summaryCount: number;
    checkpointCount: number;
  };
  createdAt: number;
  updatedAt: number;
}

interface ActivityEntry {
  time: string;
  type: "task_start" | "search" | "analysis" | "summary" | "checkpoint" | "error";
  message: string;
}

interface SearchEntry {
  time: string;
  task: string;
  query: string;
  results: number;
}

interface Report {
  id: string;
  timestamp: Date;
  task: string;
  content: string;
}

// Utilities
const uid = () => Math.random().toString(36).slice(2, 10);

const formatElapsed = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  if (hours > 0) {
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

const formatDate = (timestamp: number): string => {
  const date = new Date(timestamp);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  
  if (date.toDateString() === today.toDateString()) {
    return "Today";
  } else if (date.toDateString() === yesterday.toDateString()) {
    return "Yesterday";
  } else {
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }
};

// Custom hook for managing tasks
function useTaskManager() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    // Load from localStorage
    try {
      const saved = localStorage.getItem("kestrel.tasks");
      if (saved) {
        const parsed = JSON.parse(saved);
        setTasks(parsed);
        if (parsed.length > 0) {
          setSelectedTaskId(parsed[0].id);
        }
      }
    } catch (e) {
      console.error("Failed to load tasks:", e);
    }
    setIsInitialized(true);
  }, []);

  useEffect(() => {
    if (isInitialized && tasks.length > 0) {
      localStorage.setItem("kestrel.tasks", JSON.stringify(tasks));
    }
  }, [tasks, isInitialized]);

  const createTask = (): string => {
    const newTask: Task = {
      id: uid(),
      name: "New Research Task",
      description: "",
      budgetMinutes: 180,
      status: "configuring",
      progress: 0,
      elapsed: 0,
      metrics: {
        searchCount: 0,
        thinkCount: 0,
        summaryCount: 0,
        checkpointCount: 0,
      },
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    
    setTasks(prev => [newTask, ...prev]);
    setSelectedTaskId(newTask.id);
    return newTask.id;
  };

  const updateTask = (id: string, updates: Partial<Task>) => {
    setTasks(prev => prev.map(task => 
      task.id === id 
        ? { ...task, ...updates, updatedAt: Date.now() }
        : task
    ));
  };

  const deleteTask = (id: string) => {
    setTasks(prev => prev.filter(t => t.id !== id));
    if (selectedTaskId === id) {
      const remaining = tasks.filter(t => t.id !== id);
      setSelectedTaskId(remaining[0]?.id || null);
    }
  };

  const selectedTask = tasks.find(t => t.id === selectedTaskId) || null;

  return {
    tasks,
    selectedTask,
    selectedTaskId,
    setSelectedTaskId,
    createTask,
    updateTask,
    deleteTask,
    isInitialized,
  };
}

// Sidebar Task Item (warm themed)
function TaskItem({ 
  task, 
  isSelected, 
  onSelect, 
  onDelete, 
  onRename 
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

  const statusIcon = {
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
        isSelected
          ? 'bg-amber-600/30 backdrop-blur border border-amber-500/30'
          : 'hover:bg-amber-700/20 hover:backdrop-blur'
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
            if (e.key === 'Enter') {
              onRename(editName);
              setIsEditing(false);
            }
            if (e.key === 'Escape') {
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
              <div className="text-sm font-medium truncate text-amber-50">{task.name}</div>
              <div className="text-xs text-amber-200/70">{formatDate(task.updatedAt)}</div>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowMenu(!showMenu);
              }}
              className={`p-1 rounded hover:bg-amber-600/30 ${
                showMenu || isSelected ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
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
                  if (confirm('Delete this task?')) {
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

// Task Configuration View (shown when creating/configuring a task)
function TaskConfiguration({ 
  task, 
  onUpdate, 
  onStart 
}: { 
  task: Task;
  onUpdate: (updates: Partial<Task>) => void;
  onStart: () => void;
}) {
  const templates = [
    {
      id: "fellowships",
      icon: "🎓",
      name: "ML Fellowships",
      description: "Find currently open grants, programs, fellowships, or funding opportunities that support AI/ML research and are available to senior undergraduate students in the United States.",
      budget: 180,
    },
    {
      id: "conferences",
      icon: "📚",
      name: "AI Conferences",
      description: "Find AI/ML conferences, symposia, workshops, or student research programs that are currently accepting abstract submissions.",
      budget: 180,
    },
    {
      id: "competitions",
      icon: "🏆",
      name: "ML Competitions",
      description: "Find active AI/ML student competitions or challenges suitable for senior undergraduates.",
      budget: 180,
    },
  ];

  const handleTemplateSelect = (template: typeof templates[0]) => {
    onUpdate({
      name: template.name,
      description: template.description,
      budgetMinutes: template.budget,
    });
  };

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
          {/* Task Name */}
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

          {/* Description */}
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
              Be specific about eligibility criteria, deadlines, and what information to collect.
            </p>
          </div>

          {/* Time Budget */}
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
                onChange={(e) => onUpdate({ budgetMinutes: Number(e.target.value) })}
                className="flex-1"
              />
              <div className="px-4 py-2 bg-amber-50 rounded-lg min-w-[120px] text-center">
                <span className="text-2xl font-bold text-amber-700">{task.budgetMinutes}</span>
                <span className="text-sm text-amber-600 ml-1">minutes</span>
              </div>
            </div>
          </div>

          {/* Start Button */}
          <div className="pt-4 border-t">
            <button
              onClick={onStart}
              disabled={!task.name.trim() || !task.description.trim()}
              className={`w-full py-3 rounded-lg font-semibold transition-all flex items-center justify-center gap-2 ${
                task.name.trim() && task.description.trim()
                  ? 'bg-gradient-to-r from-amber-600 to-orange-500 text-white hover:shadow-lg'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed'
              }`}
            >
              <Zap className="w-5 h-5" />
              Start Research
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Task Dashboard (warm themed)
function TaskDashboard({ task }: { task: Task }) {
  const [isPaused, setIsPaused] = useState(task.status === "paused");
  const [currentReportIndex, setCurrentReportIndex] = useState(0);
  
  // Mock data for demonstration
  const [activity] = useState<ActivityEntry[]>([
    { time: "14:23:12", type: "search", message: "🔍 Searching for information" },
    { time: "14:22:45", type: "analysis", message: "🤔 Analyzing findings" },
    { time: "14:22:10", type: "summary", message: "📝 Creating summary" },
    { time: "14:21:30", type: "checkpoint", message: "💾 Saving checkpoint" },
  ]);

  const [searchHistory] = useState<SearchEntry[]>([
    { time: "14:23:12", task: task.name, query: "Anthropic research grants undergraduate", results: 8 },
    { time: "14:22:30", task: task.name, query: "AAAI undergraduate fellowships 2025", results: 12 },
    { time: "14:21:15", task: task.name, query: "NSF REU AI programs deadline", results: 15 },
  ]);

  const [reports] = useState<Report[]>([
    {
      id: uid(),
      timestamp: new Date(),
      task: task.name,
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

  const metrics = {
    llmCalls: 128,
    searches: 42,
    pages: 181,
    summaries: 19,
    checkpoints: 8,
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
                <p className="text-sm text-gray-600">{task.description.substring(0, 100)}...</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <div className="px-4 py-2 bg-gradient-to-r from-amber-900 to-orange-900 rounded-lg shadow-inner">
                <div className="text-lg font-mono font-bold text-amber-400">
                  {formatElapsed(task.elapsed)}
                </div>
              </div>
              
              <button
                onClick={() => setIsPaused(!isPaused)}
                className={`px-4 py-2 rounded-lg font-semibold transition-all flex items-center gap-2 ${
                  isPaused 
                    ? 'bg-green-600 hover:bg-green-700 text-white shadow-lg' 
                    : 'bg-amber-600 hover:bg-amber-700 text-white shadow-lg'
                }`}
              >
                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                {isPaused ? 'Resume' : 'Pause'}
              </button>
              
              <button className="p-2 hover:bg-amber-100 rounded-lg transition-colors">
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
            <span className={`px-3 py-1 rounded-full text-xs font-bold text-white ${
              task.status === 'active' 
                ? 'bg-gradient-to-r from-green-500 to-emerald-500' 
                : task.status === 'complete'
                ? 'bg-gradient-to-r from-blue-500 to-sky-500'
                : 'bg-gradient-to-r from-amber-500 to-orange-500'
            } shadow-md`}>
              {task.status === 'active' && <span className="inline-block w-2 h-2 bg-white rounded-full mr-1 animate-pulse" />}
              {task.status.toUpperCase()}
            </span>
          </div>
          
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-600 mb-2">
              <span>Overall Progress</span>
              <span className="font-semibold">{task.progress.toFixed(1)}%</span>
            </div>
            <div className="h-3 bg-amber-100 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-amber-500 to-orange-500 transition-all duration-500 shadow-inner"
                style={{ width: `${task.progress}%` }}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-3 text-center border border-amber-200">
              <div className="text-2xl font-bold text-amber-700">{task.metrics.searchCount}</div>
              <div className="text-xs uppercase text-gray-600 font-semibold">Searches</div>
            </div>
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-3 text-center border border-amber-200">
              <div className="text-2xl font-bold text-amber-700">{task.metrics.thinkCount}</div>
              <div className="text-xs uppercase text-gray-600 font-semibold">Analysis</div>
            </div>
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-3 text-center border border-amber-200">
              <div className="text-2xl font-bold text-amber-700">{task.metrics.summaryCount}</div>
              <div className="text-xs uppercase text-gray-600 font-semibold">Summaries</div>
            </div>
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-3 text-center border border-amber-200">
              <div className="text-2xl font-bold text-amber-700">{task.metrics.checkpointCount}</div>
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
              <span className="text-2xl font-bold text-amber-700">{metrics.pages}</span>
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
              {activity.map((entry, i) => (
                <div key={i} className="flex items-center gap-3 p-2 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg border border-amber-100">
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
              {searchHistory.map((search, i) => (
                <div key={i} className="flex items-center gap-2 p-2 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg text-xs border border-amber-100">
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
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setCurrentReportIndex(Math.max(0, currentReportIndex - 1))}
                    disabled={currentReportIndex === 0}
                    className="p-1 hover:bg-amber-100 rounded disabled:opacity-50"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <span className="text-sm text-gray-600 px-2">
                    {currentReportIndex + 1} / {reports.length}
                  </span>
                  <button
                    onClick={() => setCurrentReportIndex(Math.min(reports.length - 1, currentReportIndex + 1))}
                    disabled={currentReportIndex === reports.length - 1}
                    className="p-1 hover:bg-amber-100 rounded disabled:opacity-50"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
            <div className="p-6 max-h-96 overflow-y-auto">
              <div className="prose prose-sm max-w-none">
                <div className="whitespace-pre-wrap text-gray-700 text-sm">
                  {reports[currentReportIndex]?.content}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
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
    isInitialized,
  } = useTaskManager();

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // Filter tasks based on search
  const filteredTasks = tasks.filter(task =>
    task.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    task.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleStartTask = () => {
    if (selectedTask) {
      updateTask(selectedTask.id, { status: "active" });
    }
  };

  if (!isInitialized) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-amber-50 to-orange-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-5xl mb-4 animate-pulse">🦅</div>
          <p className="text-amber-700 font-semibold">Loading KestrelAI...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 via-orange-50 to-amber-50 flex">
      {/* Sidebar - Warm Kestrel Theme */}
      <div className={`fixed inset-y-0 left-0 z-40 w-64 bg-gradient-to-b from-amber-900 via-amber-800 to-orange-900 transform transition-transform lg:relative lg:translate-x-0 ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        <div className="flex flex-col h-full">
          {/* Sidebar Header */}
          <div className="p-4">
            <button
              onClick={() => {
                const newId = createTask();
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
            <TaskDashboard task={selectedTask} />
          )
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <div className="text-6xl mb-4">🦅</div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Welcome to KestrelAI</h2>
              <p className="text-gray-600 mb-6">Create a new research task to get started</p>
              <button
                onClick={() => {
                  const newId = createTask();
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