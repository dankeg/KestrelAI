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
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="KestrelAI API",
    description="API for autonomous research agent with Redis integration",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis Configuration
REDIS_URL = "redis://redis:6379"
REDIS_POOL: Optional[ConnectionPool] = None
REDIS_CLIENT: Optional[redis.Redis] = None

# Redis Queue Names
class RedisQueues:
    """Redis queue naming convention"""
    TASK_COMMANDS = "kestrel:queue:commands"  # Commands to agent
    TASK_UPDATES = "kestrel:queue:updates"    # Updates from agent
    TASK_ACTIVITIES = "kestrel:queue:activities"  # Activity stream
    TASK_SEARCHES = "kestrel:queue:searches"  # Search queries
    TASK_REPORTS = "kestrel:queue:reports"    # Generated reports
    TASK_METRICS = "kestrel:queue:metrics"    # Metrics updates
    TASK_LOGS = "kestrel:queue:logs"         # Debug logs
    
    @staticmethod
    def task_specific(queue_base: str, task_id: str) -> str:
        """Get task-specific queue name"""
        return f"{queue_base}:{task_id}"

# Redis Keys
class RedisKeys:
    """Redis key patterns"""
    TASK_STATE = "kestrel:task:{task_id}:state"
    TASK_METRICS = "kestrel:task:{task_id}:metrics"
    TASK_ACTIVITIES = "kestrel:task:{task_id}:activities"
    TASK_SEARCHES = "kestrel:task:{task_id}:searches"
    TASK_REPORTS = "kestrel:task:{task_id}:reports"
    ACTIVE_TASKS = "kestrel:tasks:active"
    ALL_TASKS = "kestrel:tasks:all"

# Enums
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

class CommandType(str, Enum):
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    UPDATE_CONFIG = "update_config"

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

# Models
class TaskMetrics(BaseModel):
    searchCount: int = 0
    thinkCount: int = 0
    summaryCount: int = 0
    checkpointCount: int = 0
    webFetchCount: int = 0
    llmTokensUsed: int = 0
    errorCount: int = 0

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str
    budgetMinutes: int = 180
    status: TaskStatus = TaskStatus.CONFIGURING
    progress: float = 0.0
    elapsed: int = 0
    metrics: TaskMetrics = Field(default_factory=TaskMetrics)
    createdAt: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
    updatedAt: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
    config: Dict[str, Any] = Field(default_factory=dict)  # Additional agent config

class TaskCommand(BaseModel):
    """Command sent to the agent via Redis"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    taskId: str
    type: CommandType
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))

class TaskUpdate(BaseModel):
    """Update received from the agent via Redis"""
    taskId: str
    status: Optional[TaskStatus] = None
    progress: Optional[float] = None
    elapsed: Optional[int] = None
    metrics: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))

class ActivityEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    taskId: str
    time: str
    type: ActivityType
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))

class SearchEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    taskId: str
    time: str
    query: str
    results: int
    sources: List[str] = Field(default_factory=list)
    timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))

class Report(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    taskId: str
    timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
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

# Redis Helper Functions
async def get_redis() -> redis.Redis:
    """Get Redis client"""
    global REDIS_CLIENT
    if not REDIS_CLIENT:
        raise HTTPException(status_code=503, detail="Redis not connected")
    return REDIS_CLIENT

async def init_redis():
    """Initialize Redis connection"""
    global REDIS_POOL, REDIS_CLIENT
    try:
        REDIS_POOL = ConnectionPool.from_url(REDIS_URL, decode_responses=True)
        REDIS_CLIENT = redis.Redis(connection_pool=REDIS_POOL)
        await REDIS_CLIENT.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        # Fallback to in-memory storage if Redis is not available
        logger.warning("Running without Redis - using in-memory storage only")

async def close_redis():
    """Close Redis connection"""
    global REDIS_CLIENT, REDIS_POOL
    if REDIS_CLIENT:
        await REDIS_CLIENT.close()
    if REDIS_POOL:
        await REDIS_POOL.disconnect()

# Task Queue Operations
async def send_command(task_id: str, command_type: CommandType, payload: Dict[str, Any] = None):
    """Send command to agent via Redis queue"""
    try:
        r = await get_redis()
        command = TaskCommand(
            taskId=task_id,
            type=command_type,
            payload=payload or {}
        )
        
        # Push to global command queue
        await r.lpush(RedisQueues.TASK_COMMANDS, command.json())
        
        # Also push to task-specific queue for targeted processing
        task_queue = RedisQueues.task_specific(RedisQueues.TASK_COMMANDS, task_id)
        await r.lpush(task_queue, command.json())
        
        logger.info(f"Command sent: {command_type} for task {task_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to send command: {e}")
        return False

async def get_task_from_redis(task_id: str) -> Optional[Task]:
    """Get task state from Redis"""
    try:
        r = await get_redis()
        key = RedisKeys.TASK_STATE.format(task_id=task_id)
        task_data = await r.get(key)
        if task_data:
            return Task(**json.loads(task_data))
        return None
    except Exception as e:
        logger.error(f"Failed to get task from Redis: {e}")
        return None

async def save_task_to_redis(task: Task):
    """Save task state to Redis"""
    try:
        r = await get_redis()
        key = RedisKeys.TASK_STATE.format(task_id=task.id)
        await r.set(key, task.json())
        
        # Add to task lists
        await r.sadd(RedisKeys.ALL_TASKS, task.id)
        if task.status == TaskStatus.ACTIVE:
            await r.sadd(RedisKeys.ACTIVE_TASKS, task.id)
        else:
            await r.srem(RedisKeys.ACTIVE_TASKS, task.id)
            
        logger.info(f"Task {task.id} saved to Redis")
        return True
    except Exception as e:
        logger.error(f"Failed to save task to Redis: {e}")
        return False

async def process_queues():
    """Background task to consume all agent queues and update Redis state keys"""
    while True:
        try:
            r = await get_redis()

            # Process status/progress updates
            data = await r.brpop(RedisQueues.TASK_UPDATES, timeout=1)
            if data:
                _, raw = data
                update = TaskUpdate(**json.loads(raw))
                task = await get_task_from_redis(update.taskId)
                if task:
                    if update.status:
                        task.status = update.status
                    if update.progress is not None:
                        task.progress = update.progress
                    if update.elapsed is not None:
                        task.elapsed = update.elapsed
                    if update.metrics:
                        for k, v in update.metrics.items():
                            if hasattr(task.metrics, k): setattr(task.metrics, k, v)
                    task.updatedAt = int(datetime.now().timestamp() * 1000)
                    await save_task_to_redis(task)
                    # Publish update event
                    await r.publish(f"kestrel:task:{update.taskId}:updates",
                                     json.dumps({"type": "status", "payload": update.dict()}))
                    logger.info(f"Processed update for task {update.taskId}")

            # Process activity entries
            data = await r.brpop(RedisQueues.TASK_ACTIVITIES, timeout=1)
            if data:
                _, raw = data
                logger.error(json.loads(raw))
                entry = ActivityEntry(**json.loads(raw))
                key = RedisKeys.TASK_ACTIVITIES.format(task_id=entry.taskId)
                await r.lpush(key, raw)
                # Publish activity event
                await r.publish(f"kestrel:task:{entry.taskId}:updates",
                                 json.dumps({"type": "activity", "payload": entry.dict()}))

            # Process search entries
            data = await r.brpop(RedisQueues.TASK_SEARCHES, timeout=1)
            if data:
                _, raw = data
                entry = SearchEntry(**json.loads(raw))
                key = RedisKeys.TASK_SEARCHES.format(task_id=entry.taskId)
                await r.lpush(key, raw)
                # Publish search event
                await r.publish(f"kestrel:task:{entry.taskId}:updates",
                                 json.dumps({"type": "search", "payload": entry.dict()}))

            # Process report entries
            data = await r.brpop(RedisQueues.TASK_REPORTS, timeout=1)
            if data:
                _, raw = data
                entry = Report(**json.loads(raw))
                key = RedisKeys.TASK_REPORTS.format(task_id=entry.taskId)
                await r.lpush(key, raw)
                # Publish report event
                await r.publish(f"kestrel:task:{entry.taskId}:updates",
                                 json.dumps({"type": "report", "payload": entry.dict()}))

            # Process metrics updates
            data = await r.brpop(RedisQueues.TASK_METRICS, timeout=1)
            if data:
                _, raw = data
                metrics = json.loads(raw)
                key = RedisKeys.TASK_METRICS.format(task_id=metrics.get("taskId"))
                await r.set(key, raw)
                # Publish metrics event
                await r.publish(f"kestrel:task:{metrics.get('taskId')}:updates",
                                 json.dumps({"type": "metrics", "payload": metrics}))
        except Exception as e:
            logger.error(f"Error processing queues: {e}")
            await asyncio.sleep(1)

# In-memory fallback storage (for when Redis is not available)
tasks_memory: Dict[str, Task] = {}
activities_memory: Dict[str, List[ActivityEntry]] = {}
searches_memory: Dict[str, List[SearchEntry]] = {}
reports_memory: Dict[str, List[Report]] = {}

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    redis_status = "connected" if REDIS_CLIENT else "disconnected"
    return {
        "status": "healthy",
        "service": "KestrelAI API",
        "version": "2.0.0",
        "redis": redis_status
    }

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
        config=task_data.get("config", {})
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
                "config": task.config
            }
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
            setattr(task, field, value)
    
    task.updatedAt = int(datetime.now().timestamp() * 1000)
    
    try:
        await save_task_to_redis(task)
        
        # Send update command to agent if config changed
        if "config" in updates or "description" in updates:
            await send_command(
                task_id,
                CommandType.UPDATE_CONFIG,
                updates
            )
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
        raise HTTPException(status_code=400, detail="Task must be in configuring status to start")
    
    task.status = TaskStatus.ACTIVE
    task.updatedAt = int(datetime.now().timestamp() * 1000)
    
    try:
        await save_task_to_redis(task)
        await send_command(task_id, CommandType.START, {
            "description": task.description,
            "budgetMinutes": task.budgetMinutes,
            "config": task.config
        })
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
    limit: int = Query(50, ge=1, le=200, description="Maximum number of activities to return")
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
            time_offset = base_time - timedelta(minutes=i*2)
            mock_activities.append(ActivityEntry(
                taskId=task_id,
                time=time_offset.strftime("%H:%M:%S"),
                type=random.choice(list(ActivityType)),
                message=f"Mock activity {i+1}",
                timestamp=int(time_offset.timestamp() * 1000)
            ))
        return mock_activities[:limit]

@app.get("/api/v1/tasks/{task_id}/searches", response_model=List[SearchEntry])
async def get_task_search_history(
    task_id: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of searches to return")
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
            "NSF REU programs deadline"
        ]
        for i, query in enumerate(queries):
            time_offset = base_time - timedelta(minutes=i*5)
            mock_searches.append(SearchEntry(
                taskId=task_id,
                time=time_offset.strftime("%H:%M:%S"),
                query=query,
                results=random.randint(5, 25),
                sources=["google.com", "bing.com"],
                timestamp=int(time_offset.timestamp() * 1000)
            ))
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
            format="markdown"
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
        llmCalls=42,
        searches=15,
        pagesAnalyzed=73,
        summaries=8,
        checkpoints=3,
        tokensUsed=15000,
        estimatedCost=0.45
    )

@app.get("/api/v1/tasks/{task_id}/export")
async def export_task(
    task_id: str,
    format: ExportFormat = Query(ExportFormat.JSON, description="Export format")
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
            "exported_at": datetime.now().isoformat()
        }
        
        content = json.dumps(export_data, indent=2)
        
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={task.name.replace(' ', '_')}_{task_id}.json"
            }
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
            }
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
                    if message['type'] == 'message':
                        await websocket.send_text(message['data'])
            except:
                # Fallback: send periodic updates
                task = tasks_memory.get(task_id)
                if task:
                    await websocket.send_json({
                        "type": "status",
                        "data": task.dict()
                    })
            
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        pass

# Background task processor
async def background_processor():
    """Process Redis queues in the background"""
    while True:
        try:
            if REDIS_CLIENT:
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