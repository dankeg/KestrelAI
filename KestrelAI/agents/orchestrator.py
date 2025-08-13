from datetime import datetime, timedelta
from .base import LlmWrapper
from dataclass.tasks import Task

ORCH_INSTRUCTIONS = """
You are a project manager. Balance progress across tasks and avoid
getting stuck. If good progress is being made, don't interrupt it. Return JSON with keys:
{{
  "reasoning": perform reasoning about what decision to make
  "decision": "continue" | "switch" | "done",
  "next_task": "name_of_task"
}}
"""

class Orchestrator:
    def __init__(self, tasks: list[Task], llm: LlmWrapper, slice_minutes: int = 15):
        self.tasks = {t.name: t for t in tasks}
        self.llm = llm
        self.slice = timedelta(minutes=slice_minutes)
        self.current = tasks[0].name

    def _review(self, task: Task, latest_notes: str) -> dict:
        msg = [
            {"role": "system", "content": ORCH_INSTRUCTIONS},
            {"role": "user", "content": f"Current time: {datetime.now()}\n"
                                        f"Task statuses: {[ (t.name, t.status) for t in self.tasks.values() ]}\n"
                                        f"Latest update for {task.name}:\n{latest_notes}"},
        ]
        raw = self.llm.chat(msg)
        import json, re
        json_part = re.search(r"{.*}", raw, re.S)
        return json.loads(json_part.group(0)) if json_part else {"decision": "continue"}

    def next_action(self, task: Task, notes: str):
        review = self._review(task, notes)
        decision = review["decision"]

        print(f"The decision is {review}")

        if decision == "switch":
            self.current = review["next_task"]
        elif decision == "done":
            task.status = "done"
            remaining = [t for t in self.tasks.values() if t.status != "done"]
            self.current = remaining[0].name if remaining else None
