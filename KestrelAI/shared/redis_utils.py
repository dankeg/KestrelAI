from time import time
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from typing import Optional, Dict, Any
import json

# Redis Configuration
REDIS_URL = "redis://redis:6379"
REDIS_POOL: Optional[ConnectionPool] = None
REDIS_CLIENT: Optional[redis.Redis] = None

# Initialize Redis
async def init_redis():
    global REDIS_POOL, REDIS_CLIENT
    if not REDIS_POOL:
        REDIS_POOL = ConnectionPool.from_url(REDIS_URL, decode_responses=True)
    if not REDIS_CLIENT:
        REDIS_CLIENT = redis.Redis(connection_pool=REDIS_POOL)
    await REDIS_CLIENT.ping()

# Close Redis
async def close_redis():
    global REDIS_CLIENT, REDIS_POOL
    if REDIS_CLIENT:
        await REDIS_CLIENT.close()
    if REDIS_POOL:
        await REDIS_POOL.disconnect()

# Get Redis Client
async def get_redis() -> redis.Redis:
    if not REDIS_CLIENT:
        raise Exception("Redis not initialized")
    return REDIS_CLIENT

# Fetch Task from Redis
async def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    try:
        r = await get_redis()
        task_data = await r.get(f"kestrel:task:{task_id}:state")
        if task_data:
            return json.loads(task_data)
        return None
    except Exception as e:
        print(f"Error fetching task: {e}")
        return None

# Save Task to Redis
async def save_task(task_id: str, task_data: Dict[str, Any]):
    try:
        r = await get_redis()
        await r.set(f"kestrel:task:{task_id}:state", json.dumps(task_data))
    except Exception as e:
        print(f"Error saving task: {e}")

# Send Command to Redis Queue
async def send_command(task_id: str, command_type: str, payload: Dict[str, Any]):
    try:
        r = await get_redis()
        command = {
            "taskId": task_id,
            "type": command_type,
            "payload": payload
        }
        await r.lpush("kestrel:queue:commands", json.dumps(command))
    except Exception as e:
        print(f"Error sending command: {e}")


