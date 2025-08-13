from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class Task:
    name: str
    user_prompt: str
    deadline: datetime | None = None          # optional hard cut-off
    budget_minutes: int = 60                  # soft slice the orchestrator can reclaim later
    status: str = "pending"
    scratchpad: list[str] = field(default_factory=list)
    result: str | None = None
