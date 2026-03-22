export const TASK_STATUSES = [
  "configuring",
  "pending",
  "active",
  "complete",
  "paused",
  "failed",
] as const;

export const SUBTASK_STATUSES = [
  "pending",
  "in_progress",
  "completed",
] as const;

export const SUBTASK_TYPES = [
  "general",
  "discovery",
  "verification",
  "comparison",
  "synthesis",
] as const;

export const ACTIVITY_TYPES = [
  "task_start",
  "task_pause",
  "task_resume",
  "task_complete",
  "task_stopped",
  "search",
  "analysis",
  "summary",
  "checkpoint",
  "error",
  "thinking",
  "web_fetch",
] as const;

export type TaskStatus = (typeof TASK_STATUSES)[number];
export type SubtaskStatus = (typeof SUBTASK_STATUSES)[number];
export type SubtaskType = (typeof SUBTASK_TYPES)[number];
export type ActivityType = (typeof ACTIVITY_TYPES)[number];

export interface Metrics {
  searchCount: number;
  thinkCount: number;
  summaryCount: number;
  checkpointCount: number;
  webFetchCount?: number;
  llmTokensUsed?: number;
  errorCount?: number;
}

export interface Task {
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
  isDraft?: boolean;
}

export interface Subtask {
  order: number;
  description: string;
  success_criteria: string;
  subtask_type: SubtaskType;
  status: SubtaskStatus;
  findings?: string[];
}

export interface ResearchPlan {
  restated_task: string;
  subtasks: Subtask[];
  current_subtask_index: number;
  created_at: number;
}

export interface ActivityEntry {
  id: string;
  taskId: string;
  time: string;
  type: ActivityType;
  message: string;
  timestamp: number;
}

const normalizeStringEnum = <T extends string>(
  value: unknown,
  valid: readonly T[],
  fallback: T
): T => {
  if (typeof value !== "string") return fallback;
  const normalized = value.trim().toLowerCase() as T;
  return valid.includes(normalized) ? normalized : fallback;
};

export const defaultMetrics = (): Metrics => ({
  searchCount: 0,
  thinkCount: 0,
  summaryCount: 0,
  checkpointCount: 0,
});

export const normalizeTask = (t: any): Task => {
  const budgetMinutes = t?.budgetMinutes ?? t?.budget_minutes ?? 180;
  return {
    ...t,
    budgetMinutes,
    status: normalizeStringEnum(t?.status, TASK_STATUSES, "configuring"),
    metrics: t?.metrics ?? defaultMetrics(),
    isDraft: !!t?.isDraft,
  };
};

export const normalizeSubtask = (subtask: any): Subtask => ({
  order: Number(subtask?.order ?? 0),
  description: String(subtask?.description ?? ""),
  success_criteria: String(subtask?.success_criteria ?? ""),
  subtask_type: normalizeStringEnum(
    subtask?.subtask_type,
    SUBTASK_TYPES,
    "general"
  ),
  status: normalizeStringEnum(subtask?.status, SUBTASK_STATUSES, "pending"),
  findings: Array.isArray(subtask?.findings)
    ? subtask.findings.map((item: unknown) => String(item))
    : [],
});

export const normalizeResearchPlan = (plan: any): ResearchPlan | null => {
  if (!plan || typeof plan !== "object" || plan.message) return null;
  return {
    restated_task: String(plan.restated_task ?? ""),
    subtasks: Array.isArray(plan.subtasks)
      ? plan.subtasks.map(normalizeSubtask)
      : [],
    current_subtask_index: Number(plan.current_subtask_index ?? 0),
    created_at: Number(plan.created_at ?? Date.now()),
  };
};

export const normalizeActivityEntry = (entry: any): ActivityEntry => ({
  id: String(entry?.id ?? ""),
  taskId: String(entry?.taskId ?? ""),
  time: String(entry?.time ?? ""),
  type: normalizeStringEnum(entry?.type, ACTIVITY_TYPES, "analysis"),
  message: String(entry?.message ?? ""),
  timestamp: Number(entry?.timestamp ?? Date.now()),
});
