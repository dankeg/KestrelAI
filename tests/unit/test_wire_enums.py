from __future__ import annotations

import pytest

from KestrelAI.shared.models import (
    ResearchPlan,
    Subtask,
    SubtaskStatus,
    SubtaskType,
    Task,
    TaskStatus,
)
from KestrelAI.shared.wire_models import ActivityEntry, ActivityType, TaskUpdate


def test_task_assignment_validates_status_strings():
    task = Task(name="x", description="y")
    task.status = "ACTIVE"
    assert task.status == TaskStatus.ACTIVE


def test_task_assignment_rejects_invalid_status_strings():
    task = Task(name="x", description="y")
    with pytest.raises(ValueError):
        task.status = "totally_made_up"


def test_subtask_enums_normalize_case():
    subtask = Subtask(
        order=1,
        description="d",
        success_criteria="s",
        subtask_type="DISCOVERY",
        status="IN_PROGRESS",
    )
    assert subtask.subtask_type == SubtaskType.DISCOVERY
    assert subtask.status == SubtaskStatus.IN_PROGRESS


def test_research_plan_rejects_invalid_subtask_status():
    with pytest.raises(ValueError):
        ResearchPlan(
            restated_task="r",
            subtasks=[
                {
                    "order": 1,
                    "description": "d",
                    "success_criteria": "s",
                    "status": "made_up",
                }
            ],
        )


def test_activity_type_normalizes_case():
    activity = ActivityEntry(
        taskId="t1",
        time="12:00:00",
        type="TASK_PAUSE",
        message="paused",
    )
    assert activity.type == ActivityType.TASK_PAUSE


def test_task_update_serialization_excludes_none_fields():
    payload = TaskUpdate(taskId="t1", status=TaskStatus.ACTIVE).model_dump(
        exclude_none=True
    )
    assert payload["status"] == "active"
    assert "progress" not in payload
    assert "research_plan" not in payload
