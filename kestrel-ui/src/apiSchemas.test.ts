import { describe, expect, it } from "vitest";

import {
  normalizeActivityEntry,
  normalizeResearchPlan,
  normalizeTask,
} from "./apiSchemas";

describe("apiSchemas", () => {
  it("normalizes unknown task status to configuring", () => {
    expect(normalizeTask({ id: "1", name: "x", description: "", status: "RUNNING" }).status)
      .toBe("configuring");
  });

  it("normalizes research plan subtask enums and preserves known subtask type", () => {
    const plan = normalizeResearchPlan({
      restated_task: "test",
      current_subtask_index: 0,
      subtasks: [
        {
          order: 1,
          description: "d",
          success_criteria: "s",
          subtask_type: "DISCOVERY",
          status: "IN_PROGRESS",
        },
        {
          order: 2,
          description: "d2",
          success_criteria: "s2",
          subtask_type: "made_up",
          status: "bogus",
        },
      ],
    });

    expect(plan).not.toBeNull();
    expect(plan!.subtasks[0].subtask_type).toBe("discovery");
    expect(plan!.subtasks[0].status).toBe("in_progress");
    expect(plan!.subtasks[1].subtask_type).toBe("general");
    expect(plan!.subtasks[1].status).toBe("pending");
  });

  it("normalizes unknown activity type to analysis", () => {
    expect(normalizeActivityEntry({ type: "UNUSED_HEARTBEAT" }).type).toBe("analysis");
  });
});
