"""
Unified Redis utilities for KestrelAI
Provides consistent Redis communication between FastAPI backend and model loop
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

from KestrelAI.shared.wire_models import ActivityEntry, Report, SearchEntry, TaskUpdate

try:
    import redis
    import redis.asyncio as async_redis
    from redis.asyncio import ConnectionPool as AsyncConnectionPool
except ImportError:
    raise ImportError("Redis is required. Install with: pip install redis")

logger = logging.getLogger(__name__)


def _json_default(value: Any):
    """JSON serializer fallback for runtime-only structures."""
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_dumps(payload: Any) -> str:
    """Serialize payloads safely for Redis transport/storage."""
    return json.dumps(payload, default=_json_default)


class RedisConfig:
    """Redis configuration"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        url: str | None = None,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.url = url or f"redis://{host}:{port}/{db}"


class RedisQueues:
    """Redis queue naming convention"""

    TASK_COMMANDS = "kestrel:queue:commands"
    TASK_UPDATES = "kestrel:queue:updates"
    TASK_ACTIVITIES = "kestrel:queue:activities"
    TASK_SEARCHES = "kestrel:queue:searches"
    TASK_REPORTS = "kestrel:queue:reports"
    TASK_METRICS = "kestrel:queue:metrics"
    TASK_LOGS = "kestrel:queue:logs"

    @staticmethod
    def task_specific(queue_base: str, task_id: str) -> str:
        """Get task-specific queue name"""
        return f"{queue_base}:{task_id}"


class RedisKeys:
    """Redis key patterns"""

    TASK_STATE = "kestrel:task:{task_id}:state"
    TASK_METRICS = "kestrel:task:{task_id}:metrics"
    TASK_ACTIVITIES = "kestrel:task:{task_id}:activities"
    TASK_SEARCHES = "kestrel:task:{task_id}:searches"
    TASK_REPORTS = "kestrel:task:{task_id}:reports"
    ACTIVE_TASKS = "kestrel:tasks:active"
    ALL_TASKS = "kestrel:tasks:all"


class SyncRedisClient:
    """Synchronous Redis client for model loop"""

    def __init__(self, config: RedisConfig):
        self.config = config
        self.redis = redis.Redis(
            host=config.host, port=config.port, db=config.db, decode_responses=True
        )
        self.task_id: str | None = None
        self.task_config: dict[str, Any] = {}

    def get_next_command(self, timeout: int = 10) -> dict[str, Any] | None:
        """Get next command from queue"""
        if self.task_id:
            queue = RedisQueues.task_specific(RedisQueues.TASK_COMMANDS, self.task_id)
            result = self.redis.brpop(queue, timeout=timeout)
            if result:
                _, data = result
                return json.loads(data)

        result = self.redis.brpop(RedisQueues.TASK_COMMANDS, timeout=timeout)
        if result:
            _, data = result
            return json.loads(data)
        return None

    def send_update(self, task_id: str, **kwargs):
        """Send status update to backend"""
        update = TaskUpdate(taskId=task_id, **kwargs).model_dump(exclude_none=True)
        self.redis.lpush(RedisQueues.TASK_UPDATES, _json_dumps(update))
        self._update_task_state(task_id, update)

    def _update_task_state(self, task_id: str, updates: dict[str, Any]):
        """Update task state in Redis"""
        key = RedisKeys.TASK_STATE.format(task_id=task_id)
        current = self.redis.get(key)
        # Ignore late updates for tasks that no longer exist (e.g. deleted tasks).
        # Writing partial payloads would corrupt the task state shape.
        if not current:
            return

        try:
            task = json.loads(current)
        except (TypeError, json.JSONDecodeError):
            return

        if not isinstance(task, dict) or not task.get("id"):
            return

        safe_updates = {
            key: value
            for key, value in updates.items()
            if key not in {"taskId", "timestamp"}
        }
        task.update(safe_updates)
        task["updatedAt"] = int(time.time() * 1000)
        self.redis.set(key, _json_dumps(task))

    def send_activity(self, task_id: str, activity_type: str, message: str):
        """Send activity log to backend"""
        now = datetime.now()
        activity = ActivityEntry(
            taskId=task_id,
            type=activity_type,
            message=message,
            timestamp=int(time.time() * 1000),
            time=now.strftime("%H:%M:%S"),
        ).model_dump()
        self.redis.lpush(RedisQueues.TASK_ACTIVITIES, _json_dumps(activity))

    def send_search(self, task_id: str, query: str, results: int, sources: list[str]):
        """Send search information to backend"""
        now = datetime.now()
        payload = SearchEntry(
            taskId=task_id,
            query=query,
            results=results,
            sources=sources,
            timestamp=int(time.time() * 1000),
            time=now.strftime("%H:%M:%S"),
        ).model_dump()
        self.redis.lpush(RedisQueues.TASK_SEARCHES, _json_dumps(payload))

    def send_report(
        self,
        task_id: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ):
        """Send report to backend"""
        report = Report(
            taskId=task_id,
            title=title,
            content=content,
            metadata=metadata or {},
            timestamp=int(time.time() * 1000),
            format="markdown",
        ).model_dump()
        self.redis.lpush(RedisQueues.TASK_REPORTS, _json_dumps(report))

    def checkpoint(self, task_id: str, state: dict[str, Any]):
        """Save checkpoint state"""
        self.redis.set(f"kestrel:task:{task_id}:checkpoint", _json_dumps(state))

    def restore_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        """Restore checkpoint state"""
        data = self.redis.get(f"kestrel:task:{task_id}:checkpoint")
        return json.loads(data) if data else None


class AsyncRedisClient:
    """Asynchronous Redis client for FastAPI backend"""

    def __init__(self, config: RedisConfig):
        self.config = config
        self.pool: AsyncConnectionPool | None = None
        self.redis: async_redis.Redis | None = None

    async def connect(self):
        """Initialize async Redis connection"""
        try:
            self.pool = AsyncConnectionPool.from_url(
                self.config.url, decode_responses=True
            )
            self.redis = async_redis.Redis(connection_pool=self.pool)
            await self.redis.ping()
            logger.info("Async Redis connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to async Redis: {e}")
            raise

    async def disconnect(self):
        """Close async Redis connection"""
        if self.redis:
            await self.redis.close()
        if self.pool:
            await self.pool.disconnect()

    async def get_redis(self) -> async_redis.Redis:
        """Get Redis client"""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        return self.redis

    async def send_command(
        self, task_id: str, command_type: str, payload: dict[str, Any] = None
    ):
        """Send command to agent via Redis queue"""
        try:
            r = await self.get_redis()
            command = {
                "id": str(int(time.time() * 1000)),
                "taskId": task_id,
                "type": command_type,
                "payload": payload or {},
                "timestamp": int(time.time() * 1000),
            }

            # Push to global command queue
            await r.lpush(RedisQueues.TASK_COMMANDS, _json_dumps(command))

            # Also push to task-specific queue for targeted processing
            task_queue = RedisQueues.task_specific(RedisQueues.TASK_COMMANDS, task_id)
            await r.lpush(task_queue, _json_dumps(command))

            logger.info(f"Command sent: {command_type} for task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False

    async def get_task_from_redis(self, task_id: str) -> dict[str, Any] | None:
        """Get task state from Redis"""
        try:
            r = await self.get_redis()
            key = RedisKeys.TASK_STATE.format(task_id=task_id)
            task_data = await r.get(key)
            if task_data:
                return json.loads(task_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get task from Redis: {e}")
            return None

    async def save_task_to_redis(self, task_data: dict[str, Any]):
        """Save task state to Redis"""
        try:
            r = await self.get_redis()
            task_id = task_data.get("id")
            if not task_id:
                raise ValueError("Task ID is required")

            key = RedisKeys.TASK_STATE.format(task_id=task_id)
            await r.set(key, _json_dumps(task_data))

            # Add to task lists
            await r.sadd(RedisKeys.ALL_TASKS, task_id)
            if task_data.get("status") == "active":
                await r.sadd(RedisKeys.ACTIVE_TASKS, task_id)
            else:
                await r.srem(RedisKeys.ACTIVE_TASKS, task_id)

            logger.info(f"Task {task_id} saved to Redis")
            return True
        except Exception as e:
            logger.error(f"Failed to save task to Redis: {e}")
            return False


# Global instances
_sync_client: SyncRedisClient | None = None
_async_client: AsyncRedisClient | None = None


def get_sync_redis_client(config: RedisConfig | None = None) -> SyncRedisClient:
    """Get or create sync Redis client"""
    global _sync_client
    if _sync_client is None:
        _sync_client = SyncRedisClient(config or RedisConfig())
    return _sync_client


def get_async_redis_client(config: RedisConfig | None = None) -> AsyncRedisClient:
    """Get or create async Redis client"""
    global _async_client
    if _async_client is None:
        _async_client = AsyncRedisClient(config or RedisConfig())
    return _async_client


async def init_async_redis(config: RedisConfig | None = None):
    """Initialize async Redis connection"""
    client = get_async_redis_client(config)
    await client.connect()


async def close_async_redis():
    """Close async Redis connection"""
    global _async_client
    if _async_client:
        await _async_client.disconnect()
        _async_client = None
