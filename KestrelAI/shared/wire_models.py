from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from KestrelAI.shared.models import TaskStatus


class ActivityType(str, Enum):
    TASK_START = "task_start"
    TASK_PAUSE = "task_pause"
    TASK_RESUME = "task_resume"
    TASK_COMPLETE = "task_complete"
    TASK_STOPPED = "task_stopped"
    SEARCH = "search"
    ANALYSIS = "analysis"
    SUMMARY = "summary"
    CHECKPOINT = "checkpoint"
    ERROR = "error"
    THINKING = "thinking"
    WEB_FETCH = "web_fetch"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if member.value == value:
                    return member
        return None


class StreamEventType(str, Enum):
    STATUS = "status"
    ACTIVITY = "activity"
    SEARCH = "search"
    REPORT = "report"
    METRICS = "metrics"
    RESEARCH_PLAN = "research_plan"


class TaskUpdate(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    taskId: str
    status: TaskStatus | None = None
    progress: float | None = None
    elapsed: int | None = None
    metrics: dict[str, int] | None = None
    error: str | None = None
    research_plan: dict[str, Any] | None = None
    stopReason: str | None = None
    completed: bool | None = None
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now().timestamp() * 1000)
    )


class ActivityEntry(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    taskId: str
    time: str
    type: ActivityType
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now().timestamp() * 1000)
    )


class SearchEntry(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    taskId: str
    time: str
    query: str
    results: int
    sources: list[str] = Field(default_factory=list)
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now().timestamp() * 1000)
    )


class Report(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    taskId: str
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now().timestamp() * 1000)
    )
    title: str
    content: str
    format: str = "markdown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemMetrics(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    taskId: str | None = None
    llmCalls: int
    searches: int
    pagesAnalyzed: int
    summaries: int
    checkpoints: int
    tokensUsed: int
    estimatedCost: float
    timestamp: int | None = None


class TaskStreamEvent(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    type: StreamEventType
    payload: dict[str, Any]
