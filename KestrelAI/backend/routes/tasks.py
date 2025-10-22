from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Dict, Any
from datetime import datetime
from shared.models import Task, TaskStatus
from shared.redis_utils import get_async_redis_client, RedisConfig

router = APIRouter()


@router.get("/tasks", response_model=List[Task])
async def get_tasks():
    """Get all tasks"""
    # Example logic to fetch tasks from Redis
    return []


@router.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: str):
    """Get a specific task by ID"""
    client = get_async_redis_client()
    task_data = await client.get_task_from_redis(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return Task(**task_data)


@router.post("/tasks", response_model=Task)
async def create_task(task_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Create a new task"""
    task = Task(
        id=task_data.get("id", ""),
        name=task_data.get("name", "New Task"),
        description=task_data.get("description", ""),
        budgetMinutes=task_data.get("budgetMinutes", 180),
        status=TaskStatus.CONFIGURING,
        config=task_data.get("config", {}),
    )
    client = get_async_redis_client()
    await client.save_task_to_redis(task.dict())
    return task


@router.patch("/tasks/{task_id}", response_model=Task)
async def update_task(task_id: str, updates: Dict[str, Any]):
    """Update an existing task"""
    client = get_async_redis_client()
    task_data = await client.get_task_from_redis(task_id)
    task = Task(**task_data) if task_data else None
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    for key, value in updates.items():
        if hasattr(task, key):
            setattr(task, key, value)
    task.updatedAt = int(datetime.now().timestamp() * 1000)
    client = get_async_redis_client()
    await client.save_task_to_redis(task.dict())
    return task


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task"""
    client = get_async_redis_client()
    task_data = await client.get_task_from_redis(task_id)
    task = Task(**task_data) if task_data else None
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    # Logic to delete task from Redis
    return {"message": "Task deleted successfully"}


@router.post("/tasks/{task_id}/start", response_model=Task)
async def start_task(task_id: str):
    """Start a task"""
    client = get_async_redis_client()
    task_data = await client.get_task_from_redis(task_id)
    task = Task(**task_data) if task_data else None
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    task.status = TaskStatus.ACTIVE
    client = get_async_redis_client()
    await client.save_task_to_redis(task.dict())
    await client.send_command(task_id, "start", {})
    return task


@router.post("/tasks/{task_id}/pause", response_model=Task)
async def pause_task(task_id: str):
    """Pause a task"""
    client = get_async_redis_client()
    task_data = await client.get_task_from_redis(task_id)
    task = Task(**task_data) if task_data else None
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    task.status = TaskStatus.PAUSED
    client = get_async_redis_client()
    await client.save_task_to_redis(task.dict())
    await client.send_command(task_id, "pause", {})
    return task


@router.post("/tasks/{task_id}/resume", response_model=Task)
async def resume_task(task_id: str):
    """Resume a paused task"""
    client = get_async_redis_client()
    task_data = await client.get_task_from_redis(task_id)
    task = Task(**task_data) if task_data else None
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    task.status = TaskStatus.ACTIVE
    client = get_async_redis_client()
    await client.save_task_to_redis(task.dict())
    await client.send_command(task_id, "resume", {})
    return task
