from pydantic import BaseModel, Field
from typing import Dict, Any
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    CONFIGURING = "configuring"
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    PAUSED = "paused"
    FAILED = "failed"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            # Normalize to lowercase before lookup
            value = value.lower()
            for member in cls:
                if member.value == value:
                    return member
        return None

class Task(BaseModel):
    name: str
    description: str
    budgetMinutes: int = 180
    status: TaskStatus = TaskStatus.CONFIGURING
    progress: float = 0.0
    elapsed: int = 0
    createdAt: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
    updatedAt: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
    config: Dict[str, Any] = Field(default_factory=dict)
