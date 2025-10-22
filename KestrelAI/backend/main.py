"""
KestrelAI FastAPI Backend with Redis Queue Integration
Autonomous Research Agent API
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import uuid
import json
import random
from enum import Enum
import asyncio
import logging
import os

# Import shared models and utilities
try:
    from KestrelAI.shared.models import Task, TaskStatus, TaskMetrics, ResearchPlan
    from KestrelAI.shared.redis_utils import (
        RedisConfig, RedisQueues, RedisKeys,
        get_async_redis_client, init_async_redis, close_async_redis
    )
except ImportError:
    # Fallback for different import contexts (Docker, local, etc.)
    from shared.models import Task, TaskStatus, TaskMetrics, ResearchPlan
    from shared.redis_utils import (
        RedisConfig, RedisQueues, RedisKeys,
        get_async_redis_client, init_async_redis, close_async_redis
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="KestrelAI API",
    description="API for autonomous research agent with Redis integration",
    version="2.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis Configuration - now using unified Redis utilities
REDIS_CONFIG = RedisConfig(url=os.getenv("REDIS_URL", "redis://localhost:6379"))


# Enums - TaskStatus is now imported from shared.models


class CommandType(str, Enum):
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    UPDATE_CONFIG = "update_config"
    UPDATE_SETTINGS = "update_settings"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            # Normalize to lowercase before lookup
            value = value.lower()
            for member in cls:
                if member.value == value:
                    return member
        return None


class ActivityType(str, Enum):
    TASK_START = "task_start"
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
            # Normalize to lowercase before lookup
            value = value.lower()
            for member in cls:
                if member.value == value:
                    return member
        return None


class ExportFormat(str, Enum):
    JSON = "json"
    PDF = "pdf"
    MARKDOWN = "markdown"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            # Normalize to lowercase before lookup
            value = value.lower()
            for member in cls:
                if member.value == value:
                    return member
        return None


# Models - Task, TaskStatus, and TaskMetrics are now imported from shared.models


class TaskCommand(BaseModel):
    """Command sent to the agent via Redis"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    taskId: str
    type: CommandType
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now().timestamp() * 1000)
    )


class TaskUpdate(BaseModel):
    """Update received from the agent via Redis"""

    taskId: str
    status: Optional[TaskStatus] = None
    progress: Optional[float] = None
    elapsed: Optional[int] = None
    metrics: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now().timestamp() * 1000)
    )


class ActivityEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    taskId: str
    time: str
    type: ActivityType
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now().timestamp() * 1000)
    )


class SearchEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    taskId: str
    time: str
    query: str
    results: int
    sources: List[str] = Field(default_factory=list)
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now().timestamp() * 1000)
    )


class Report(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    taskId: str
    timestamp: int = Field(
        default_factory=lambda: int(datetime.now().timestamp() * 1000)
    )
    title: str
    content: str
    format: str = "markdown"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SystemMetrics(BaseModel):
    llmCalls: int
    searches: int
    pagesAnalyzed: int
    summaries: int
    checkpoints: int
    tokensUsed: int
    estimatedCost: float


class OllamaMode(str, Enum):
    local = "local"
    docker = "docker"


class Orchestrator(str, Enum):
    hummingbird = "hummingbird"
    kestrel = "kestrel"
    albatross = "albatross"


class Theme(str, Enum):
    amber = "amber"
    blue = "blue"


class AppSettings(BaseModel):
    ollamaMode: OllamaMode = Field(
        default=OllamaMode.local, description="Where to send Ollama calls"
    )
    orchestrator: Orchestrator = Field(
        default=Orchestrator.kestrel, description="Research orchestrator profile"
    )
    theme: Theme = Field(
        default=Theme.amber, description="UI theme color scheme"
    )


# Redis Helper Functions - now using unified Redis utilities
async def get_redis():
    """Get Redis client"""
    return await get_async_redis_client(REDIS_CONFIG).get_redis()


async def init_redis():
    """Initialize Redis connection"""
    try:
        await init_async_redis(REDIS_CONFIG)
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        # Fallback to in-memory storage if Redis is not available
        logger.warning("Running without Redis - using in-memory storage only")


async def close_redis():
    """Close Redis connection"""
    await close_async_redis()


# Task Queue Operations - now using unified Redis utilities
async def send_command(
    task_id: str, command_type: CommandType, payload: Dict[str, Any] = None
):
    """Send command to agent via Redis queue"""
    client = get_async_redis_client(REDIS_CONFIG)
    return await client.send_command(task_id, command_type.value, payload or {})


async def get_task_from_redis(task_id: str) -> Optional[Task]:
    """Get task state from Redis"""
    try:
        client = get_async_redis_client(REDIS_CONFIG)
        task_data = await client.get_task_from_redis(task_id)
        if task_data:
            return Task(**task_data)
        return None
    except Exception as e:
        logger.error(f"Failed to get task from Redis: {e}")
        return None


async def save_task_to_redis(task: Task):
    """Save task state to Redis"""
    try:
        client = get_async_redis_client(REDIS_CONFIG)
        return await client.save_task_to_redis(task.dict())
    except Exception as e:
        logger.error(f"Failed to save task to Redis: {e}")
        return False


async def process_queues():
    """Background task to consume all agent queues and update Redis state keys"""
    while True:
        try:
            r = await get_redis()
            
            # Process task updates
            raw = await r.rpop(RedisQueues.TASK_UPDATES)
            if raw:
                try:
                    update_data = json.loads(raw)
                    task = await get_task_from_redis(update_data.get("taskId"))
                    if task:
                        # Update task fields
                        if "status" in update_data:
                            task.status = TaskStatus(update_data["status"])
                        if "progress" in update_data:
                            task.progress = update_data["progress"]
                        if "elapsed" in update_data:
                            task.elapsed = update_data["elapsed"]
                        if "metrics" in update_data:
                            for k, v in update_data["metrics"].items():
                                if hasattr(task.metrics, k):
                                    setattr(task.metrics, k, v)
                        if "research_plan" in update_data:
                            # Convert dict to ResearchPlan object
                            try:
                                task.research_plan = ResearchPlan(**update_data["research_plan"])
                                logger.info(f"Successfully converted research plan for task {update_data['taskId']}")
                            except Exception as e:
                                logger.error(f"Failed to convert research plan for task {update_data['taskId']}: {e}")
                                logger.error(f"Research plan data: {update_data['research_plan']}")
                                # Store as dict if conversion fails
                                task.research_plan = update_data["research_plan"]
                        task.updatedAt = int(datetime.now().timestamp() * 1000)
                        await save_task_to_redis(task)
                        
                        # Publish update event
                        await r.publish(
                            f"kestrel:task:{update_data['taskId']}:updates",
                            json.dumps({"type": "status", "payload": update_data}),
                        )
                        
                        # Also publish research plan update if it was updated
                        if "research_plan" in update_data:
                            try:
                                await r.publish(
                                    f"kestrel:task:{update_data['taskId']}:updates",
                                    json.dumps({"type": "research_plan", "payload": task.research_plan.dict()}),
                                )
                                logger.info(f"Published research plan update for task {update_data['taskId']}")
                            except Exception as e:
                                logger.error(f"Failed to publish research plan update for task {update_data['taskId']}: {e}")
                        logger.info(f"Processed update for task {update_data['taskId']}")
                except Exception as e:
                    logger.error(f"Error processing task update: {e}")

            # Process activity entries
            raw = await r.rpop(RedisQueues.TASK_ACTIVITIES)
            if raw:
                try:
                    activity_data = json.loads(raw)
                    task_id = activity_data.get("taskId")
                    if task_id:
                        key = RedisKeys.TASK_ACTIVITIES.format(task_id=task_id)
                        await r.lpush(key, raw)
                        # Publish activity event
                        await r.publish(
                            f"kestrel:task:{task_id}:updates",
                            json.dumps({"type": "activity", "payload": activity_data}),
                        )
                        logger.info(f"Processed activity for task {task_id}")
                except Exception as e:
                    logger.error(f"Error processing activity: {e}")

            # Process search entries
            raw = await r.rpop(RedisQueues.TASK_SEARCHES)
            if raw:
                try:
                    search_data = json.loads(raw)
                    task_id = search_data.get("taskId")
                    if task_id:
                        key = RedisKeys.TASK_SEARCHES.format(task_id=task_id)
                        await r.lpush(key, raw)
                        # Publish search event
                        await r.publish(
                            f"kestrel:task:{task_id}:updates",
                            json.dumps({"type": "search", "payload": search_data}),
                        )
                        logger.info(f"Processed search for task {task_id}: {search_data.get('query', 'unknown')}")
                except Exception as e:
                    logger.error(f"Error processing search: {e}")

            # Process report entries
            raw = await r.rpop(RedisQueues.TASK_REPORTS)
            if raw:
                try:
                    report_data = json.loads(raw)
                    task_id = report_data.get("taskId")
                    if task_id:
                        key = RedisKeys.TASK_REPORTS.format(task_id=task_id)
                        await r.lpush(key, raw)
                        # Publish report event
                        await r.publish(
                            f"kestrel:task:{task_id}:updates",
                            json.dumps({"type": "report", "payload": report_data}),
                        )
                        logger.info(f"Processed report for task {task_id}")
                except Exception as e:
                    logger.error(f"Error processing report: {e}")

            # Process metrics updates
            raw = await r.rpop(RedisQueues.TASK_METRICS)
            if raw:
                try:
                    metrics = json.loads(raw)
                    task_id = metrics.get("taskId")
                    if task_id:
                        key = RedisKeys.TASK_METRICS.format(task_id=task_id)
                        await r.set(key, raw)
                        # Publish metrics event
                        await r.publish(
                            f"kestrel:task:{task_id}:updates",
                            json.dumps({"type": "metrics", "payload": metrics}),
                        )
                        logger.info(f"Processed metrics for task {task_id}")
                except Exception as e:
                    logger.error(f"Error processing metrics: {e}")

            # Short sleep to prevent CPU spinning
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error processing queues: {e}")
            await asyncio.sleep(1)


# In-memory fallback storage (for when Redis is not available)
tasks_memory: Dict[str, Task] = {}
activities_memory: Dict[str, List[ActivityEntry]] = {}
searches_memory: Dict[str, List[SearchEntry]] = {}
reports_memory: Dict[str, List[Report]] = {}
settings_memory: AppSettings = AppSettings()

# API Endpoints


@app.get("/")
async def root():
    """Health check endpoint"""
    try:
        await get_redis()
        redis_status = "connected"
    except:
        redis_status = "disconnected"
    return {
        "status": "healthy",
        "service": "KestrelAI API",
        "version": "2.0.0",
        "redis": redis_status,
    }


@app.get("/settings", response_model=AppSettings)
async def get_settings():
    """Get current application settings"""
    try:
        r = await get_redis()
        settings_data = await r.get("kestrel:settings")
        if settings_data:
            return AppSettings(**json.loads(settings_data))
    except:
        pass
    
    # Fallback to in-memory storage
    return settings_memory


@app.post("/settings", response_model=AppSettings)
async def save_settings(settings: AppSettings):
    """Save application settings"""
    global settings_memory
    
    try:
        r = await get_redis()
        await r.set("kestrel:settings", settings.json())
        
        # Also store in memory as fallback
        settings_memory = settings
        
        # Send settings update to all active agents
        await send_command(None, CommandType.UPDATE_SETTINGS, settings.dict())
        
        logger.info(f"Settings updated: {settings.dict()}")
        return settings
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        # Fallback to in-memory storage
        settings_memory = settings
        return settings


@app.get("/api/v1/tasks", response_model=List[Task])
async def get_tasks():
    """Get all tasks"""
    try:
        r = await get_redis()
        task_ids = await r.smembers(RedisKeys.ALL_TASKS)
        tasks = []

        for task_id in task_ids:
            task = await get_task_from_redis(task_id)
            if task:
                tasks.append(task)

        return tasks
    except:
        # Fallback to in-memory storage
        return list(tasks_memory.values())


@app.get("/api/v1/tasks/{task_id}", response_model=Task)
async def get_task(task_id: str):
    """Get a specific task by ID"""
    try:
        task = await get_task_from_redis(task_id)
        if task:
            return task
    except:
        # Fallback to in-memory storage
        if task_id in tasks_memory:
            return tasks_memory[task_id]

    raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")


@app.post("/api/v1/tasks", response_model=Task)
async def create_task(task_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Create a new task"""
    task = Task(
        name=task_data.get("name", "New Research Task"),
        description=task_data.get("description", ""),
        budgetMinutes=task_data.get("budgetMinutes", 180),
        status=TaskStatus.CONFIGURING,
        config=task_data.get("config", {}),
    )

    try:
        await save_task_to_redis(task)

        # Send creation command to agent
        await send_command(
            task.id,
            CommandType.UPDATE_CONFIG,
            {
                "name": task.name,
                "description": task.description,
                "budgetMinutes": task.budgetMinutes,
                "config": task.config,
            },
        )
    except:
        # Fallback to in-memory storage
        tasks_memory[task.id] = task
        activities_memory[task.id] = []
        searches_memory[task.id] = []
        reports_memory[task.id] = []

    return task


@app.patch("/api/v1/tasks/{task_id}", response_model=Task)
async def update_task(task_id: str, updates: Dict[str, Any]):
    """Update an existing task"""
    try:
        task = await get_task_from_redis(task_id)
        if not task:
            task = tasks_memory.get(task_id)
    except:
        task = tasks_memory.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    # Update task fields
    for field, value in updates.items():
        if hasattr(task, field) and field not in ["id", "createdAt"]:
            if field == "metrics" and isinstance(value, dict):
                task.metrics = TaskMetrics(**value)
                logger.error(task.metrics)
                logger.error(f"Pre-existing Values: {task.metrics}")
            else:
                setattr(task, field, value)

    task.updatedAt = int(datetime.now().timestamp() * 1000)

    try:
        await save_task_to_redis(task)

        # Send update command to agent if config changed
        if "config" in updates or "description" in updates:
            await send_command(task_id, CommandType.UPDATE_CONFIG, updates)
    except:
        # Fallback to in-memory storage
        tasks_memory[task_id] = task

    return task


@app.delete("/api/v1/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task"""
    try:
        r = await get_redis()

        # Send stop command to agent
        await send_command(task_id, CommandType.STOP)

        # Remove from Redis
        await r.delete(RedisKeys.TASK_STATE.format(task_id=task_id))
        await r.delete(RedisKeys.TASK_METRICS.format(task_id=task_id))
        await r.delete(RedisKeys.TASK_ACTIVITIES.format(task_id=task_id))
        await r.delete(RedisKeys.TASK_SEARCHES.format(task_id=task_id))
        await r.delete(RedisKeys.TASK_REPORTS.format(task_id=task_id))
        await r.srem(RedisKeys.ALL_TASKS, task_id)
        await r.srem(RedisKeys.ACTIVE_TASKS, task_id)
    except:
        # Fallback to in-memory storage
        if task_id in tasks_memory:
            del tasks_memory[task_id]
        if task_id in activities_memory:
            del activities_memory[task_id]
        if task_id in searches_memory:
            del searches_memory[task_id]
        if task_id in reports_memory:
            del reports_memory[task_id]

    return {"message": "Task deleted successfully"}


@app.post("/api/v1/tasks/{task_id}/start", response_model=Task)
async def start_task(task_id: str):
    """Start a task"""
    try:
        task = await get_task_from_redis(task_id)
        if not task:
            task = tasks_memory.get(task_id)
    except:
        task = tasks_memory.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    if task.status != TaskStatus.CONFIGURING:
        raise HTTPException(
            status_code=400, detail="Task must be in configuring status to start"
        )

    task.status = TaskStatus.ACTIVE
    task.updatedAt = int(datetime.now().timestamp() * 1000)

    try:
        await save_task_to_redis(task)
        await send_command(
            task_id,
            CommandType.START,
            {
                "description": task.description,
                "budgetMinutes": task.budgetMinutes,
                "config": task.config,
            },
        )
    except:
        tasks_memory[task_id] = task

    return task


@app.post("/api/v1/tasks/{task_id}/pause", response_model=Task)
async def pause_task(task_id: str):
    """Pause a task"""
    try:
        task = await get_task_from_redis(task_id)
        if not task:
            task = tasks_memory.get(task_id)
    except:
        task = tasks_memory.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    if task.status != TaskStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Task is not active")

    task.status = TaskStatus.PAUSED
    task.updatedAt = int(datetime.now().timestamp() * 1000)

    try:
        await save_task_to_redis(task)
        await send_command(task_id, CommandType.PAUSE)
    except:
        tasks_memory[task_id] = task

    return task


@app.post("/api/v1/tasks/{task_id}/resume", response_model=Task)
async def resume_task(task_id: str):
    """Resume a paused task"""
    try:
        task = await get_task_from_redis(task_id)
        if not task:
            task = tasks_memory.get(task_id)
    except:
        task = tasks_memory.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    if task.status != TaskStatus.PAUSED:
        raise HTTPException(status_code=400, detail="Task is not paused")

    task.status = TaskStatus.ACTIVE
    task.updatedAt = int(datetime.now().timestamp() * 1000)

    try:
        await save_task_to_redis(task)
        await send_command(task_id, CommandType.RESUME)
    except:
        tasks_memory[task_id] = task

    return task


@app.get("/api/v1/tasks/{task_id}/activity", response_model=List[ActivityEntry])
async def get_task_activity(
    task_id: str,
    limit: int = Query(
        50, ge=1, le=200, description="Maximum number of activities to return"
    ),
):
    """Get activity feed for a task"""
    try:
        r = await get_redis()
        key = RedisKeys.TASK_ACTIVITIES.format(task_id=task_id)

        # Get activities from Redis list
        activities_data = await r.lrange(key, 0, limit - 1)
        activities = [ActivityEntry(**json.loads(data)) for data in activities_data]

        return activities
    except:
        # Fallback to in-memory storage or generate mock data
        if task_id in activities_memory:
            return activities_memory[task_id][:limit]

        # Generate mock activities if none exist
        mock_activities = []
        base_time = datetime.now()
        for i in range(4):
            time_offset = base_time - timedelta(minutes=i * 2)
            mock_activities.append(
                ActivityEntry(
                    taskId=task_id,
                    time=time_offset.strftime("%H:%M:%S"),
                    type=random.choice(list(ActivityType)),
                    message=f"Mock activity {i+1}",
                    timestamp=int(time_offset.timestamp() * 1000),
                )
            )
        return mock_activities[:limit]


@app.get("/api/v1/tasks/{task_id}/searches", response_model=List[SearchEntry])
async def get_task_search_history(
    task_id: str,
    limit: int = Query(
        50, ge=1, le=200, description="Maximum number of searches to return"
    ),
):
    """Get search history for a task"""
    try:
        r = await get_redis()
        key = RedisKeys.TASK_SEARCHES.format(task_id=task_id)

        # Get searches from Redis list
        searches_data = await r.lrange(key, 0, limit - 1)
        searches = [SearchEntry(**json.loads(data)) for data in searches_data]

        return searches
    except:
        # Fallback to in-memory storage or generate mock data
        if task_id in searches_memory:
            return searches_memory[task_id][:limit]

        # Generate mock searches
        mock_searches = []
        base_time = datetime.now()
        queries = [
            "AI research grants 2025",
            "Machine learning fellowships undergraduate",
            "NSF REU programs deadline",
        ]
        for i, query in enumerate(queries):
            time_offset = base_time - timedelta(minutes=i * 5)
            mock_searches.append(
                SearchEntry(
                    taskId=task_id,
                    time=time_offset.strftime("%H:%M:%S"),
                    query=query,
                    results=random.randint(5, 25),
                    sources=["google.com", "bing.com"],
                    timestamp=int(time_offset.timestamp() * 1000),
                )
            )
        return mock_searches[:limit]


@app.get("/api/v1/tasks/{task_id}/reports", response_model=List[Report])
async def get_task_reports(task_id: str):
    """Get all reports for a task"""
    try:
        r = await get_redis()
        key = RedisKeys.TASK_REPORTS.format(task_id=task_id)

        # Get reports from Redis list
        reports_data = await r.lrange(key, 0, -1)
        reports = [Report(**json.loads(data)) for data in reports_data]

        return reports
    except:
        # Fallback to in-memory storage or generate mock data
        if task_id in reports_memory:
            return reports_memory[task_id]

        # Generate mock report
        mock_report = Report(
            taskId=task_id,
            title="Research Summary",
            content="## Mock Report\nThis is a placeholder report.",
            format="markdown",
        )
        return [mock_report]


@app.get("/api/v1/tasks/{task_id}/metrics", response_model=SystemMetrics)
async def get_task_metrics(task_id: str):
    """Get system metrics for a task"""
    try:
        r = await get_redis()
        key = RedisKeys.TASK_METRICS.format(task_id=task_id)

        metrics_data = await r.get(key)
        if metrics_data:
            metrics = json.loads(metrics_data)
            return SystemMetrics(**metrics)
    except:
        pass

    # Return mock metrics
    return SystemMetrics(
        llmCalls=0,
        searches=0,
        pagesAnalyzed=0,
        summaries=0,
        checkpoints=0,
        tokensUsed=0,
        estimatedCost=0.0,
    )


@app.get("/api/v1/tasks/{task_id}/research-plan")
async def get_task_research_plan(task_id: str):
    """Get research plan for a task"""
    try:
        task = await get_task_from_redis(task_id)
        if not task:
            task = tasks_memory.get(task_id)
    except:
        task = tasks_memory.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    # Return research plan if available
    if hasattr(task, 'research_plan') and task.research_plan:
        return task.research_plan
    else:
        return {"message": "Research plan not yet generated"}


@app.get("/api/v1/tasks/{task_id}/export")
async def export_task(
    task_id: str,
    format: ExportFormat = Query(ExportFormat.JSON, description="Export format"),
):
    """Export task data in various formats"""
    try:
        task = await get_task_from_redis(task_id)
        if not task:
            task = tasks_memory.get(task_id)
    except:
        task = tasks_memory.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    # Get related data
    activities = await get_task_activity(task_id, limit=200)
    searches = await get_task_search_history(task_id, limit=200)
    reports = await get_task_reports(task_id)

    if format == ExportFormat.JSON:
        export_data = {
            "task": task.dict(),
            "activities": [a.dict() for a in activities],
            "searches": [s.dict() for s in searches],
            "reports": [r.dict() for r in reports],
            "exported_at": datetime.now().isoformat(),
        }

        content = json.dumps(export_data, indent=2)

        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={task.name.replace(' ', '_')}_{task_id}.json"
            },
        )

    elif format == ExportFormat.MARKDOWN:
        markdown_content = f"""# {task.name}

## Task Details
- **ID**: {task.id}
- **Status**: {task.status}
- **Progress**: {task.progress:.1f}%
- **Time Budget**: {task.budgetMinutes} minutes
- **Time Elapsed**: {task.elapsed} seconds

## Description
{task.description}

## Reports
"""
        for report in reports:
            markdown_content += f"\n### {report.title}\n{report.content}\n"

        return StreamingResponse(
            iter([markdown_content]),
            media_type="text/markdown",
            headers={
                "Content-Disposition": f"attachment; filename={task.name.replace(' ', '_')}_{task_id}.md"
            },
        )

    else:
        raise HTTPException(status_code=501, detail="Format not implemented")


# WebSocket for real-time updates
from fastapi import WebSocket, WebSocketDisconnect


@app.websocket("/ws/tasks/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time task updates"""
    await websocket.accept()

    try:
        while True:
            # Subscribe to task-specific Redis channels for real-time updates
            try:
                r = await get_redis()
                pubsub = r.pubsub()
                await pubsub.subscribe(f"kestrel:task:{task_id}:updates")

                async for message in pubsub.listen():
                    if message["type"] == "message":
                        data = message["data"]
                        if isinstance(data, bytes):
                            data = data.decode("utf-8", errors="ignore")
                        await websocket.send_text(data)
            except:
                # Fallback: send periodic updates
                task = tasks_memory.get(task_id)
                if task:
                    await websocket.send_json({"type": "status", "data": task.dict()})

            await asyncio.sleep(5)

    except WebSocketDisconnect:
        pass


# Background task processor
async def background_processor():
    """Process Redis queues in the background"""
    while True:
        try:
            await get_redis()  # Check if Redis is available
            await process_queues()
        except Exception as e:
            logger.error(f"Background processor error: {e}")
        await asyncio.sleep(1)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize Redis and start background tasks"""
    await init_redis()

    # Start background processor
    asyncio.create_task(background_processor())

    logger.info("KestrelAI API started with Redis integration")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources"""
    await close_redis()
    logger.info("KestrelAI API shutdown complete")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
