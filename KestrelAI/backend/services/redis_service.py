from shared.models import Task
from typing import Optional
from shared.redis_utils import get_task, save_task, send_command

async def get_task_from_redis(task_id: str) -> Optional[Task]:
    try:
        task_data = await get_task(task_id)
        if task_data:
            return Task(**task_data)
        return None
    except Exception as e:
        print(f"Error fetching task: {e}")
        return None

async def save_task_to_redis(task: Task):
    try:
        await save_task(task.id, task.json())
    except Exception as e:
        print(f"Error saving task: {e}")

async def send_command(task_id: str, command_type: str, payload: dict):
    try:
        command = {
            "taskId": task_id,
            "type": command_type,
            "payload": payload
        }
        await send_command(task_id, command_type, payload)
    except Exception as e:
        print(f"Error sending command: {e}")
