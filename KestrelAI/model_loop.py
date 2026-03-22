"""
KestrelAI Research Agent with Redis Integration
Extracted from Panel UI to work with Redis queues
"""

import asyncio
import logging
import os
import pathlib
import re
import threading
import time
from datetime import datetime
from typing import Any

from KestrelAI.agents.base import LlmWrapper
from KestrelAI.agents.research_orchestrator import ResearchOrchestrator
from KestrelAI.agents.web_research_agent import ResearchConfig, WebResearchAgent
from KestrelAI.memory.vector_store import MemoryStore
from KestrelAI.shared.llm_capabilities import normalize_runtime_provider
from KestrelAI.shared.models import (
    ResearchPlan,
    Task,
    TaskMetrics,
    TaskStatus,
)
from KestrelAI.shared.models import (
    Subtask as SharedSubtask,
)
from KestrelAI.shared.redis_utils import RedisConfig, get_sync_redis_client
from KestrelAI.shared.runtime_settings import (
    default_local_api_key,
    get_default_llm_provider,
    get_default_model_name,
    get_default_openai_base_url,
    normalize_max_context_tokens,
    normalize_openai_base_url,
    resolve_llm_base_url,
)

try:
    from KestrelAI.graphs.worker_step_runner import LangGraphWorkerStepRunner
except Exception:  # pragma: no cover - dependency-gated path
    LangGraphWorkerStepRunner = None


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Redis configuration
REDIS_CONFIG = RedisConfig(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


# -----------------------------------------------------------------------------
# KestrelAI Research Agent Worker
# -----------------------------------------------------------------------------
class KestrelAgentWorker:
    """Main agent worker that processes research tasks"""

    def __init__(self):
        # Initialize Redis client
        self.redis_client = get_sync_redis_client(REDIS_CONFIG)
        self._active_model_pulls: set[tuple[str, str]] = set()
        self._model_pull_lock = threading.Lock()

        # Initialize AI components
        self.mem = MemoryStore()

        # Load settings from Redis if available, otherwise use defaults
        loaded_settings = self._load_settings_from_redis()
        runtime_config = self._resolve_runtime_config(loaded_settings)
        self._apply_runtime_config(runtime_config)

        # State management
        self.running = False
        self.paused = False
        self.current_task_id = None
        self.current_task_config = {}
        self.tasks: dict[str, Task] = {}
        self.orchestrator = None
        self.orchestrators: dict[str, ResearchOrchestrator] = {}
        self.worker_step_runner = None
        self.task_reports: dict[
            str, list[str]
        ] = {}  # Track all reports per task for accumulation
        self._planning_controls: dict[str, dict[str, Any]] = {}
        self._planning_lock = threading.Lock()

        # Settings management - use loaded settings or defaults
        self.app_settings = {
            "llmProvider": loaded_settings["llmProvider"],
            "ollamaMode": loaded_settings["ollamaMode"],
            "orchestrator": loaded_settings["orchestrator"],
            "modelName": runtime_config["model_name"],
            "executionModelName": runtime_config["execution_model_name"],
            "controlModelName": runtime_config["control_model_name"],
            "openaiBaseUrl": loaded_settings["openaiBaseUrl"],
            "openaiApiKey": loaded_settings["openaiApiKey"],
            "maxContextTokens": runtime_config["max_context_tokens"],
        }

        self.latest_feedback = "No Feedback Yet!"
        self.latest_subtask = ""

        # Metrics tracking
        self.task_metrics: dict[str, dict[str, Any]] = {}
        self.global_metrics = {
            "total_llm_calls": 0,
            "total_searches": 0,
            "total_summaries": 0,
            "total_checkpoints": 0,
            "total_web_fetches": 0,
        }

        # Ensure notes directory exists
        pathlib.Path("notes").mkdir(exist_ok=True)

    def _cancel_planning(self, task_id: str, join_timeout: float = 1.0) -> None:
        """Cancel planning for a task and optionally wait briefly for thread exit."""
        with self._planning_lock:
            control = self._planning_controls.get(task_id)
        if not control:
            return

        loop = control.get("loop")
        planning_task = control.get("task")
        planning_thread = control.get("thread")

        if loop is not None and planning_task is not None and not planning_task.done():
            try:
                loop.call_soon_threadsafe(planning_task.cancel)
                logger.info("Requested planning cancellation for task %s", task_id)
            except Exception as e:
                logger.warning(
                    "Failed to request planning cancellation for task %s: %s",
                    task_id,
                    e,
                )

        if (
            planning_thread is not None
            and planning_thread.is_alive()
            and planning_thread is not threading.current_thread()
        ):
            planning_thread.join(timeout=join_timeout)
            if planning_thread.is_alive():
                logger.warning(
                    "Planning thread still running after cancel request for task %s",
                    task_id,
                )

    @staticmethod
    def _normalize_max_context_tokens(raw_value: Any) -> int:
        """Normalize max context token setting with sane bounds."""
        return normalize_max_context_tokens(raw_value)

    def _resolve_model_name(self, raw_value: Any, endpoint: str, provider: str) -> str:
        """Resolve selected model name from settings/env/available Ollama models."""
        if isinstance(raw_value, str) and raw_value.strip():
            return raw_value.strip()

        env_model = os.getenv("MODEL_NAME", "").strip()
        if env_model:
            return env_model

        if self._should_ensure_ollama_model(endpoint, provider):
            try:
                import ollama

                client = ollama.Client(host=endpoint)
                payload = client.list()
                for model_info in payload.get("models", []):
                    name = str(
                        model_info.get("name") or model_info.get("model") or ""
                    ).strip()
                    if name:
                        logger.info("Resolved model from Ollama tags: %s", name)
                        return name
            except Exception as e:
                logger.warning("Failed to resolve model from Ollama: %s", e)

        # Final fallback to preserve startup continuity when no model can be discovered.
        # Keep this conservative to avoid accidentally selecting oversized local models.
        return get_default_model_name()

    def _resolve_role_model_name(
        self,
        settings: dict[str, Any],
        role_key: str,
        fallback_model_name: str,
        endpoint: str,
        provider: str,
    ) -> str:
        role_value = settings.get(role_key)
        if isinstance(role_value, str) and role_value.strip():
            return self._resolve_model_name(role_value, endpoint, provider)
        return fallback_model_name

    @staticmethod
    def _build_llm_wrapper(
        model_name: str, endpoint: str, provider: str, api_key: str | None
    ) -> LlmWrapper:
        return LlmWrapper(
            model=model_name,
            host=endpoint,
            provider=provider,
            api_key=api_key,
        )

    def _resolve_runtime_config(self, settings: dict[str, Any]) -> dict[str, Any]:
        provider = normalize_runtime_provider(settings.get("llmProvider", "ollama"))
        endpoint = self._resolve_llm_endpoint(settings)
        model_name = self._resolve_model_name(
            settings.get("modelName"),
            endpoint,
            provider,
        )
        execution_model_name = self._resolve_role_model_name(
            settings,
            "executionModelName",
            model_name,
            endpoint,
            provider,
        )
        control_model_name = self._resolve_role_model_name(
            settings,
            "controlModelName",
            model_name,
            endpoint,
            provider,
        )
        return {
            "provider": provider,
            "endpoint": endpoint,
            "model_name": model_name,
            "execution_model_name": execution_model_name,
            "control_model_name": control_model_name,
            "max_context_tokens": self._normalize_max_context_tokens(
                settings.get("maxContextTokens")
            ),
            "api_key": self._resolve_api_key(settings, endpoint, provider),
        }

    def _apply_runtime_config(
        self, runtime_config: dict[str, Any], rebuild_agent: bool = True
    ) -> None:
        endpoint = runtime_config["endpoint"]
        provider = runtime_config["provider"]
        execution_model_name = runtime_config["execution_model_name"]
        control_model_name = runtime_config["control_model_name"]

        if self._should_ensure_ollama_model(endpoint, provider):
            for role_model_name in {execution_model_name, control_model_name}:
                self._ensure_model_available(role_model_name, endpoint)

        self.execution_llm = self._build_llm_wrapper(
            execution_model_name,
            endpoint,
            provider,
            runtime_config["api_key"],
        )
        self.control_llm = self._build_llm_wrapper(
            control_model_name,
            endpoint,
            provider,
            runtime_config["api_key"],
        )
        self.llm = self.execution_llm

        if rebuild_agent:
            self._rebuild_execution_agent(runtime_config["max_context_tokens"])

    def _rebuild_execution_agent(self, max_context_tokens: int) -> None:
        self.agent = WebResearchAgent(
            "main-agent",
            self.execution_llm,
            self.mem,
            config=ResearchConfig(max_context_tokens=max_context_tokens),
        )

    def _load_settings_from_redis(self) -> dict[str, Any]:
        """Load app settings from Redis if available."""
        defaults = {
            "llmProvider": get_default_llm_provider(),
            "ollamaMode": "local",
            "orchestrator": "kestrel",
            "modelName": get_default_model_name(),
            "executionModelName": get_default_model_name(),
            "controlModelName": get_default_model_name(),
            "openaiBaseUrl": get_default_openai_base_url(),
            "openaiApiKey": (
                os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
            ).strip(),
            "maxContextTokens": self._normalize_max_context_tokens(
                os.getenv("MAX_CONTEXT_TOKENS", "32768")
            ),
        }
        try:
            import json

            try:
                import redis
            except ImportError:
                redis = None

            # Use the existing Redis client instead of creating a new connection
            # Wrap in try/except to handle Redis connection errors gracefully
            try:
                settings_data = self.redis_client.redis.get("kestrel:settings")
                if settings_data:
                    settings = json.loads(settings_data)
                    legacy_model_name = settings.get("modelName", defaults["modelName"])
                    merged = defaults | {
                        "llmProvider": settings.get(
                            "llmProvider", defaults["llmProvider"]
                        ),
                        "ollamaMode": settings.get(
                            "ollamaMode", defaults["ollamaMode"]
                        ),
                        "orchestrator": settings.get(
                            "orchestrator", defaults["orchestrator"]
                        ),
                        "modelName": legacy_model_name,
                        "executionModelName": settings.get(
                            "executionModelName", legacy_model_name
                        ),
                        "controlModelName": settings.get(
                            "controlModelName", legacy_model_name
                        ),
                        "openaiBaseUrl": settings.get(
                            "openaiBaseUrl", defaults["openaiBaseUrl"]
                        ),
                        "openaiApiKey": settings.get(
                            "openaiApiKey", defaults["openaiApiKey"]
                        ),
                        "maxContextTokens": self._normalize_max_context_tokens(
                            settings.get(
                                "maxContextTokens", defaults["maxContextTokens"]
                            )
                        ),
                    }
                    logger.info(
                        "Loaded app settings from Redis: %s",
                        self._redact_settings(merged),
                    )
                    return merged
            except Exception as e:
                # Handle Redis connection errors (ConnectionError, TimeoutError, etc.)
                if redis and isinstance(e, (redis.ConnectionError, redis.TimeoutError)):
                    logger.debug(f"Redis not available for settings load: {e}")
                elif isinstance(e, AttributeError):
                    logger.debug(f"Redis client not properly initialized: {e}")
                else:
                    logger.debug(f"Error loading settings from Redis: {e}")
        except Exception as e:
            logger.debug(f"Could not load settings from Redis: {e}")

        return defaults

    def _is_running_in_docker(self) -> bool:
        """Detect if running inside Docker container"""
        # Check for Docker-specific files/environment
        if os.path.exists("/.dockerenv"):
            return True

        if os.path.exists("/proc/self/cgroup"):
            try:
                with open("/proc/self/cgroup") as f:
                    if "docker" in f.read():
                        return True
            except Exception:
                pass

        if os.getenv("container") == "docker":
            return True

        return False

    def _get_ollama_host_for_mode(self, mode: str) -> str:
        """Get Ollama host for a specific mode"""
        host = resolve_llm_base_url(
            mode=mode,
            running_in_docker=self._is_running_in_docker(),
        )
        logger.info("Resolved model endpoint for mode '%s': %s", mode, host)
        return host

    @staticmethod
    def _redact_settings(settings: dict[str, Any]) -> dict[str, Any]:
        redacted = dict(settings)
        if redacted.get("openaiApiKey"):
            redacted["openaiApiKey"] = "***redacted***"
        return redacted

    def _resolve_llm_endpoint(self, settings: dict[str, Any]) -> str:
        provider = str(settings.get("llmProvider") or "ollama").strip().lower()
        if provider == "openai_compatible":
            explicit_base_url = str(settings.get("openaiBaseUrl") or "").strip()
            return normalize_openai_base_url(
                explicit_base_url or get_default_openai_base_url()
            )
        return self._get_ollama_host_for_mode(
            str(settings.get("ollamaMode") or "local")
        )

    @staticmethod
    def _resolve_api_key(
        settings: dict[str, Any], endpoint: str, provider: str
    ) -> str | None:
        if provider != "openai_compatible":
            return None
        explicit_api_key = str(settings.get("openaiApiKey") or "").strip()
        if explicit_api_key:
            return explicit_api_key
        return default_local_api_key(endpoint)

    @staticmethod
    def _should_ensure_ollama_model(
        ollama_host: str, provider: str | None = None
    ) -> bool:
        """
        Return True only when endpoint looks like Ollama.
        Skip model pull checks for non-Ollama OpenAI-compatible providers.
        """
        resolved_provider = (provider or "").strip().lower()
        if resolved_provider == "ollama":
            return True
        if resolved_provider == "openai_compatible":
            return False

        env_provider = os.getenv("LLM_PROVIDER", "openai_compatible").strip().lower()
        if env_provider == "ollama_native":
            return True

        host = (ollama_host or "").lower()
        return "11434" in host or "ollama" in host

    def _ensure_model_available(self, model_name: str, ollama_host: str):
        """Check if model is available, and start a background pull if not."""
        try:
            import ollama

            client = ollama.Client(host=ollama_host)

            # Try to list models to check availability (with timeout handling)
            try:
                # Set a reasonable timeout to avoid blocking initialization
                models = client.list()
                available_models = [
                    str(m.get("name") or m.get("model") or "").strip()
                    for m in models.get("models", [])
                    if str(m.get("name") or m.get("model") or "").strip()
                ]

                # Check if model exists (exact match or name prefix match)
                model_base = model_name.split(":")[0]
                model_found = any(
                    m == model_name or m.startswith(f"{model_base}:")
                    for m in available_models
                )

                if model_found:
                    logger.info(f"Model {model_name} is available")
                else:
                    logger.info(
                        "Model %s not found, starting background pull", model_name
                    )
                    self._start_background_model_pull(model_name, ollama_host)
            except Exception as e:
                # Ollama might not be ready yet - this is okay, will retry on first use
                logger.debug(
                    f"Could not verify model availability (Ollama may not be ready): {e}"
                )
        except ImportError:
            logger.debug("ollama package not available for model checking")
        except Exception as e:
            # Don't fail initialization if model check fails
            logger.debug(f"Error checking model availability: {e}")

    def _start_background_model_pull(self, model_name: str, ollama_host: str) -> None:
        pull_key = (ollama_host, model_name)
        with self._model_pull_lock:
            if pull_key in self._active_model_pulls:
                logger.info("Background pull already active for model %s", model_name)
                return
            self._active_model_pulls.add(pull_key)

        def _pull() -> None:
            try:
                import ollama

                client = ollama.Client(host=ollama_host)
                client.pull(model_name)
                logger.info("Successfully pulled model %s", model_name)
            except Exception as e:
                logger.warning("Could not pull model %s: %s", model_name, e)
                logger.warning(
                    "Continuing - ensure model is available manually if needed"
                )
            finally:
                with self._model_pull_lock:
                    self._active_model_pulls.discard(pull_key)

        threading.Thread(
            target=_pull,
            name=f"ollama-pull-{model_name}",
            daemon=True,
        ).start()

    def run(self):
        """Main agent loop"""
        logger.info("KestrelAI Agent Worker started")

        while True:
            # try:
            # Check for commands
            command = self.redis_client.get_next_command(timeout=10)

            if command:
                self.handle_command(command)

            # Process active task
            if self.running and not self.paused and self.current_task_id:
                self.process_task_step()
            else:
                logger.debug(
                    f"Not processing task: running={self.running}, paused={self.paused}, current_task_id={self.current_task_id}"
                )

            time.sleep(0.5)  # Reasonable frequency for research processing

    def handle_command(self, command: dict[str, Any]):
        """Handle command from backend"""
        cmd_type = command.get("type")
        task_id = command.get("taskId")
        payload = command.get("payload", {})

        logger.info(f"Handling command: {cmd_type} for task {task_id}")

        # Guard against stale lifecycle commands that can arrive late from Redis.
        if cmd_type in {"pause", "resume", "stop"}:
            if task_id and self.current_task_id and task_id != self.current_task_id:
                logger.info(
                    "Ignoring stale %s command for task %s (current task: %s)",
                    cmd_type,
                    task_id,
                    self.current_task_id,
                )
                return

        if cmd_type == "start":
            self.start_task(task_id, payload)
        elif cmd_type == "pause":
            self.pause_task()
        elif cmd_type == "resume":
            self.resume_task()
        elif cmd_type == "stop":
            self.stop_task(completed=False, reason="manual_stop")
        elif cmd_type == "update_config":
            self.update_config(task_id, payload)
        elif cmd_type == "update_settings":
            self.update_settings(payload)

    def start_task(self, task_id: str, config: dict[str, Any]):
        """Start a new research task"""
        previous_task_id = self.current_task_id
        if previous_task_id and previous_task_id != task_id:
            self._cancel_planning(previous_task_id)
        self.current_task_id = task_id
        self.task_id = task_id  # Set for Redis client
        self.current_task_config = config

        # Create Task object aligned to the actual model (budgetMinutes + enum status)
        task = Task(
            name=config.get("name", "Research Task"),
            description=config.get("description", ""),
            budgetMinutes=config.get("budgetMinutes", 180),
            status=TaskStatus.ACTIVE,
        )
        self.tasks[task_id] = task

        # Initialize task metrics before background planning thread starts.
        self.task_metrics[task_id] = {
            "search_count": 0,
            "think_count": 0,
            "summary_count": 0,
            "checkpoint_count": 0,
            "action_count": 0,
            "searches": [],
            "start_time": time.time(),
            "execution_start_time": None,
            "last_research_plan_state": {
                "subtask_index": -1,
                "completed_subtasks": set(),
            },
        }

        # Initialize task reports tracking
        self.task_reports[task_id] = []

        # Initialize orchestrator with task and settings
        logger.info(f"Initializing orchestrator for task {task_id}")
        orchestrator_profile = self.app_settings.get("orchestrator", "kestrel")
        max_context_tokens = self._normalize_max_context_tokens(
            self.app_settings.get("maxContextTokens", 32768)
        )
        task_orchestrator = ResearchOrchestrator(
            [task],
            self.control_llm,
            profile=orchestrator_profile,
            max_context_tokens=max_context_tokens,
        )
        self.orchestrators[task_id] = task_orchestrator
        self.orchestrator = task_orchestrator
        if previous_task_id != task_id:
            # Rebind graph runner to active task context.
            self.worker_step_runner = None
        logger.info(f"Orchestrator initialized for task {task_id}")

        # Initialize planning phase in background thread to avoid blocking main loop
        logger.info(f"Starting planning phase for task {task_id}")

        def run_planning():
            import asyncio

            loop = asyncio.new_event_loop()
            planning_task = None
            control = {
                "thread": threading.current_thread(),
                "loop": loop,
                "task": None,
            }
            try:
                asyncio.set_event_loop(loop)
                planning_timeout_seconds = float(
                    os.getenv(
                        "ORCHESTRATOR_PLANNING_HARD_TIMEOUT_SECONDS",
                        str(
                            float(
                                os.getenv(
                                    "ORCHESTRATOR_PLANNING_TIMEOUT_SECONDS", "120"
                                )
                            )
                            + 30.0
                        ),
                    )
                )
                disable_timeouts = os.getenv(
                    "GLOBAL_DISABLE_TIMEOUTS", "0"
                ).strip().lower() in {"1", "true", "yes", "on"}
                if disable_timeouts or planning_timeout_seconds <= 0:
                    planning_task = loop.create_task(
                        task_orchestrator._planning_phase(task)
                    )
                else:
                    planning_task = loop.create_task(
                        asyncio.wait_for(
                            task_orchestrator._planning_phase(task),
                            timeout=max(5.0, planning_timeout_seconds),
                        )
                    )
                control["task"] = planning_task
                with self._planning_lock:
                    self._planning_controls[task_id] = control

                loop.run_until_complete(planning_task)
                logger.info(f"Planning phase completed for task {task_id}")

                if self.current_task_id != task_id:
                    logger.info(
                        "Skipping research plan publish for inactive task %s",
                        task_id,
                    )
                    return

                # Initialize last research plan state and send initial research plan update
                task_state = task_orchestrator.task_states.get(task.name)
                if task_state and task_state.research_plan:
                    self.task_metrics.setdefault(task_id, {})
                    self.task_metrics[task_id]["last_research_plan_state"] = {
                        "subtask_index": task_state.subtask_index,
                        "completed_subtasks": task_state.completed_subtasks.copy(),
                    }
                # Send research plan update after completion
                self.send_research_plan_update(
                    task_id, task.name, orchestrator=task_orchestrator
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Planning phase hard-timeout reached for task %s; applying fallback plan",
                    task_id,
                )
                try:
                    task_state = task_orchestrator.task_states.get(task.name)
                    if task_state and not getattr(task_state, "research_plan", None):
                        fallback_plan = task_orchestrator._fallback_planning_plan(
                            task,
                            "planning hard-timeout in model loop",
                        )
                        task_state.research_plan = fallback_plan
                        if fallback_plan.subtasks:
                            task_state.create_subtask_agent(
                                0,
                                self.execution_llm,
                                self.mem,
                                task_orchestrator.mcp_manager
                                if getattr(task_orchestrator, "use_mcp", False)
                                else None,
                            )

                    if self.current_task_id == task_id:
                        self.send_research_plan_update(
                            task_id, task.name, orchestrator=task_orchestrator
                        )
                except Exception as recovery_error:
                    logger.error(
                        "Failed to apply fallback planning after hard-timeout for task %s: %s",
                        task_id,
                        recovery_error,
                        exc_info=True,
                    )
            except asyncio.CancelledError:
                logger.info(f"Planning phase cancelled for task {task_id}")
            except Exception as e:
                logger.error(f"Error during planning phase: {e}", exc_info=True)
            finally:
                with self._planning_lock:
                    existing = self._planning_controls.get(task_id)
                    if existing is control:
                        self._planning_controls.pop(task_id, None)

                try:
                    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                    for pending_task in pending:
                        pending_task.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                finally:
                    loop.close()

        planning_thread = threading.Thread(target=run_planning, daemon=True)
        planning_thread.start()
        logger.info(f"Planning phase started in background for task {task_id}")

        # Get initial subtask
        self.latest_subtask = (
            task_orchestrator.get_current_subtask(task.name) or "Initial research"
        )

        self.running = True
        self.paused = False

        # Send initial updates (status as enum value)
        self.redis_client.send_update(
            task_id, status=TaskStatus.ACTIVE.value, progress=0.0
        )
        self.redis_client.send_activity(
            task_id, "task_start", f"🦅 Starting research: {task.name}"
        )

        logger.info(f"Started task {task_id}: {task.name}")

    def pause_task(self):
        """Pause current task"""
        if self.running and not self.paused and self.current_task_id:
            self.paused = True

            # Update in-memory task status
            task = self.tasks.get(self.current_task_id)
            if task:
                task.status = TaskStatus.PAUSED

            # Save checkpoint
            checkpoint_state = {
                "config": self.current_task_config,
                "metrics": self.task_metrics.get(self.current_task_id, {}),
                "global_metrics": self.global_metrics,
            }
            self.redis_client.checkpoint(self.current_task_id, checkpoint_state)

            self.redis_client.send_update(
                self.current_task_id, status=TaskStatus.PAUSED.value
            )
            self.redis_client.send_activity(
                self.current_task_id, "task_pause", "⏸️ Task paused"
            )

            logger.info(f"Paused task {self.current_task_id}")

    def resume_task(self):
        """Resume paused task"""
        if self.running and self.paused and self.current_task_id:
            # Restore from checkpoint
            checkpoint = self.redis_client.restore_checkpoint(self.current_task_id)
            if checkpoint:
                self.task_metrics[self.current_task_id] = checkpoint.get("metrics", {})
                self.global_metrics = checkpoint.get(
                    "global_metrics", self.global_metrics
                )

            self.paused = False

            # Update in-memory task status
            task = self.tasks.get(self.current_task_id)
            if task:
                task.status = TaskStatus.ACTIVE

            self.redis_client.send_update(
                self.current_task_id, status=TaskStatus.ACTIVE.value
            )
            self.redis_client.send_activity(
                self.current_task_id, "task_resume", "▶️ Task resumed"
            )

            logger.info(f"Resumed task {self.current_task_id}")

    def stop_task(self, completed: bool = True, reason: str = ""):
        """Stop current task and persist terminal state."""
        if self.running and self.current_task_id:
            task_id = self.current_task_id
            self._cancel_planning(task_id, join_timeout=2.0)

            # Update in-memory task status
            task = self.tasks.get(task_id)
            if task:
                task.status = TaskStatus.COMPLETE if completed else TaskStatus.FAILED

            # Generate final report
            self.generate_final_report(completed=completed, reason=reason)

            elapsed_seconds = int(
                time.time()
                - self.task_metrics.get(task_id, {}).get("start_time", time.time())
            )
            status_value = (
                TaskStatus.COMPLETE.value if completed else TaskStatus.FAILED.value
            )
            progress_value = 100.0 if completed else 99.0

            # Update status
            self.redis_client.send_update(
                task_id,
                status=status_value,
                progress=progress_value,
                elapsed=elapsed_seconds,
                stopReason=reason,
                completed=completed,
                metrics=self._build_metrics_payload(task_id),
                research_plan=self._build_research_plan_payload(task_id, task),
            )
            if completed:
                self.redis_client.send_activity(
                    task_id, "task_complete", "✅ Task completed"
                )
            else:
                reason_suffix = f" ({reason})" if reason else ""
                self.redis_client.send_activity(
                    task_id,
                    "task_stopped",
                    f"⏹️ Task stopped before completion{reason_suffix}",
                )
                logger.warning(
                    "Stopped task %s before completion (reason=%s, elapsed=%ss)",
                    task_id,
                    reason or "unspecified",
                    elapsed_seconds,
                )

            # Clean up
            self.running = False
            self.paused = False
            self.current_task_id = None
            self.current_task_config = {}
            self.orchestrators.pop(task_id, None)
            self.orchestrator = None
            self.worker_step_runner = None

            logger.info(f"Stopped task {task_id}")

    def update_config(self, task_id: str, config: dict[str, Any]):
        """Update task configuration"""
        if task_id == self.current_task_id:
            self.current_task_config.update(config)

            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.name = config.get("name", task.name)
                task.description = config.get("description", task.description)
                task.budgetMinutes = config.get("budgetMinutes", task.budgetMinutes)

            logger.info(f"Updated config for task {task_id}")

    def update_settings(self, settings: dict[str, Any]):
        """Update application settings"""
        next_settings = dict(settings)
        if "maxContextTokens" in next_settings:
            next_settings["maxContextTokens"] = self._normalize_max_context_tokens(
                next_settings.get("maxContextTokens")
            )
        self.app_settings.update(next_settings)
        runtime_config = self._resolve_runtime_config(self.app_settings)
        self.app_settings["modelName"] = runtime_config["model_name"]
        self.app_settings["executionModelName"] = runtime_config["execution_model_name"]
        self.app_settings["controlModelName"] = runtime_config["control_model_name"]
        self.app_settings["maxContextTokens"] = runtime_config["max_context_tokens"]

        llm_relevant = {
            "llmProvider",
            "ollamaMode",
            "modelName",
            "executionModelName",
            "controlModelName",
            "openaiBaseUrl",
            "openaiApiKey",
        }
        agent_relevant = llm_relevant | {"maxContextTokens"}
        orchestrator_relevant = {
            "orchestrator",
            "llmProvider",
            "ollamaMode",
            "modelName",
            "executionModelName",
            "controlModelName",
            "openaiBaseUrl",
            "openaiApiKey",
            "maxContextTokens",
        }

        if llm_relevant.intersection(next_settings.keys()):
            self._apply_runtime_config(runtime_config, rebuild_agent=False)

        if agent_relevant.intersection(next_settings.keys()):
            self._rebuild_execution_agent(runtime_config["max_context_tokens"])

        if orchestrator_relevant.intersection(next_settings.keys()):
            # Update orchestrator behavior based on setting
            if self.current_task_id and self.current_task_id in self.tasks:
                orchestrator_profile = self.app_settings.get("orchestrator", "kestrel")
                task = self.tasks[self.current_task_id]
                updated_orchestrator = ResearchOrchestrator(
                    [task],
                    self.control_llm,
                    profile=orchestrator_profile,
                    max_context_tokens=runtime_config["max_context_tokens"],
                )
                self.orchestrators[self.current_task_id] = updated_orchestrator
                self.orchestrator = updated_orchestrator
                self.worker_step_runner = None
                # Re-run planning phase with new configuration (async method)
                try:
                    import asyncio

                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    try:
                        loop.run_until_complete(
                            updated_orchestrator._planning_phase(task)
                        )
                        self.latest_subtask = (
                            updated_orchestrator.get_current_subtask(task.name)
                            or "Initial research"
                        )
                        logger.info(
                            f"Orchestrator reinitialized with profile: {orchestrator_profile}"
                        )
                    except Exception as e:
                        logger.error(f"Error reinitializing orchestrator: {e}")
                except Exception as e:
                    logger.error(f"Error reinitializing orchestrator: {e}")
            logger.info(
                "Orchestrator settings updated: profile=%s, control_model=%s, execution_model=%s, endpoint=%s, max_context=%s",
                self.app_settings.get("orchestrator"),
                runtime_config["control_model_name"],
                runtime_config["execution_model_name"],
                runtime_config["endpoint"],
                runtime_config["max_context_tokens"],
            )

        logger.info(
            "Updated app settings: %s", self._redact_settings(self.app_settings)
        )

    def _compute_elapsed_and_progress(
        self, task_id: str, task: Task
    ) -> tuple[float, float]:
        """Compute elapsed seconds and progress percent with safe defaults."""
        metrics = self.task_metrics.get(task_id, {})
        execution_start_time = metrics.get("execution_start_time")
        if isinstance(execution_start_time, (int, float)):
            start_time = execution_start_time
        else:
            start_time = metrics.get("start_time", time.time())
        elapsed = max(0.0, time.time() - start_time)

        budget_minutes = getattr(task, "budgetMinutes", None)
        if not isinstance(budget_minutes, (int, float)) or budget_minutes <= 0:
            budget_minutes = self.current_task_config.get("budgetMinutes", 1)
        budget_seconds = max(float(budget_minutes) * 60.0, 1.0)

        progress = min(100.0, (elapsed / budget_seconds) * 100.0)
        return elapsed, progress

    def _send_progress_heartbeat(self, task_id: str, task: Task) -> None:
        """Emit throttled status/progress updates so long steps do not appear stalled."""
        metrics = self.task_metrics.get(task_id)
        if not metrics:
            return

        now = time.time()
        last_emit = metrics.get("last_progress_emit", 0.0)
        if now - last_emit < 2.0:
            return

        active_orchestrator = self.orchestrators.get(task_id) or self.orchestrator
        task_state = (
            active_orchestrator.task_states.get(task.name)
            if active_orchestrator and hasattr(active_orchestrator, "task_states")
            else None
        )
        if metrics.get("execution_start_time") is None and (
            not task_state or not task_state.research_plan
        ):
            elapsed = max(0.0, now - metrics.get("start_time", now))
            progress = 0.0
        else:
            elapsed, progress = self._compute_elapsed_and_progress(task_id, task)
            if task_state and task_state.research_plan:
                try:
                    progress_info = active_orchestrator.get_task_progress(task.name)
                    plan_progress = float(
                        progress_info.get("progress", progress) or 0.0
                    )
                    progress = max(0.0, min(100.0, plan_progress))
                except Exception:
                    pass
        if hasattr(task, "status") and task.status == TaskStatus.COMPLETE:
            progress = 100.0
        self.redis_client.send_update(
            task_id,
            status=(
                task.status.value
                if hasattr(task, "status")
                else TaskStatus.ACTIVE.value
            ),
            progress=progress,
            elapsed=int(elapsed),
        )
        metrics["last_progress_emit"] = now

    def process_task_step(self):
        """Process one step of the research task"""
        if not self.current_task_id:
            return

        task_id = self.current_task_id
        task = self.tasks.get(task_id)
        if not task:
            return
        active_orchestrator = self.orchestrators.get(task_id) or self.orchestrator
        if not active_orchestrator:
            return
        self.orchestrator = active_orchestrator

        if self.worker_step_runner is None:
            if LangGraphWorkerStepRunner is None:
                logger.error("LangGraph worker step runner unavailable")
                return
            self.worker_step_runner = LangGraphWorkerStepRunner(self)

        try:
            self.worker_step_runner.run(task_id, task)
        except Exception as e:
            logger.error("Error in LangGraph worker step runner: %s", e, exc_info=True)

    def send_research_plan_update(
        self,
        task_id: str,
        task_name: str,
        orchestrator: ResearchOrchestrator | None = None,
    ):
        """Send research plan update to Redis"""
        active_orchestrator = (
            orchestrator or self.orchestrators.get(task_id) or self.orchestrator
        )
        if not active_orchestrator or not hasattr(active_orchestrator, "task_states"):
            logger.warning(
                f"Cannot send research plan update: orchestrator not initialized for task {task_id}"
            )
            return

        task_state = active_orchestrator.task_states.get(task_name)
        if not task_state or not task_state.research_plan:
            logger.warning(
                f"Cannot send research plan update: no research plan for task {task_id}"
            )
            return

        # Convert research plan to dict format using shared Pydantic models so
        # the backend and frontend see a consistent, validated schema.
        try:
            subtasks: list[SharedSubtask] = []
            for i, subtask in enumerate(task_state.research_plan.subtasks):
                status = (
                    "completed"
                    if i in task_state.completed_subtasks
                    else ("in_progress" if i == task_state.subtask_index else "pending")
                )
                subtasks.append(
                    SharedSubtask(
                        order=subtask.order,
                        description=subtask.description,
                        success_criteria=subtask.success_criteria,
                        subtask_type=getattr(subtask, "subtask_type", "general"),
                        status=status,
                    )
                )

            research_plan_model = ResearchPlan(
                restated_task=task_state.research_plan.restated_task,
                subtasks=subtasks,
                current_subtask_index=task_state.subtask_index,
            )

            if hasattr(research_plan_model, "model_dump"):
                research_plan_data = research_plan_model.model_dump()
            else:
                research_plan_data = research_plan_model.dict()

            # Send research plan update
            self.redis_client.send_update(task_id, research_plan=research_plan_data)

            logger.info(
                f"Sent research plan update for task {task_id} with {len(research_plan_data['subtasks'])} subtasks"
            )
            logger.debug(
                f"Research plan details: current_subtask_index={research_plan_data['current_subtask_index']}, "
                f"completed={[i for i, s in enumerate(research_plan_data['subtasks']) if s['status'] == 'completed']}, "
                f"in_progress={[i for i, s in enumerate(research_plan_data['subtasks']) if s['status'] == 'in_progress']}"
            )
        except Exception as e:
            logger.error(
                f"Failed to send research plan update for task {task_id}: {e}",
                exc_info=True,
            )

    def _build_metrics_payload(self, task_id: str) -> dict[str, Any]:
        """Build a JSON-safe metrics snapshot."""
        raw_metrics = self.task_metrics.get(task_id, {})
        metrics_model = TaskMetrics(
            searchCount=raw_metrics.get("search_count", 0),
            thinkCount=raw_metrics.get("think_count", 0),
            summaryCount=raw_metrics.get("summary_count", 0),
            checkpointCount=raw_metrics.get("checkpoint_count", 0),
            webFetchCount=self.global_metrics.get("total_web_fetches", 0),
            llmTokensUsed=self.global_metrics.get("total_llm_calls", 0) * 1000,
            errorCount=0,
        )
        if hasattr(metrics_model, "model_dump"):
            return metrics_model.model_dump()
        return metrics_model.dict()

    def _build_research_plan_payload(
        self, task_id: str, task: Task | None
    ) -> dict[str, Any] | None:
        """Return latest research plan snapshot with explicit subtask status."""
        if task is None:
            return None
        active_orchestrator = self.orchestrators.get(task_id) or self.orchestrator
        if not active_orchestrator or not hasattr(active_orchestrator, "task_states"):
            return None
        task_state = active_orchestrator.task_states.get(task.name)
        if not task_state or not task_state.research_plan:
            return None

        subtasks: list[SharedSubtask] = []
        for i, subtask in enumerate(task_state.research_plan.subtasks):
            status = (
                "completed"
                if i in task_state.completed_subtasks
                else ("in_progress" if i == task_state.subtask_index else "pending")
            )
            subtasks.append(
                SharedSubtask(
                    order=subtask.order,
                    description=subtask.description,
                    success_criteria=subtask.success_criteria,
                    subtask_type=getattr(subtask, "subtask_type", "general"),
                    status=status,
                )
            )
        plan_model = ResearchPlan(
            restated_task=task_state.research_plan.restated_task,
            subtasks=subtasks,
            current_subtask_index=task_state.subtask_index,
        )
        if hasattr(plan_model, "model_dump"):
            return plan_model.model_dump()
        return plan_model.dict()

    def generate_final_report(self, completed: bool = True, reason: str = ""):
        """Generate and send final report"""
        if not self.current_task_id:
            return

        task_id = self.current_task_id
        task = self.tasks.get(task_id)
        if not task:
            return

        raw_metrics = self.task_metrics.get(task_id, {})
        final_metrics = self._build_metrics_payload(task_id)

        # Prefer orchestrator's synthesized report if available.
        safe_name = "".join(
            c if c.isalnum() or c in (" ", "-", "_") else "_" for c in task.name
        ).strip()
        final_notes_file = pathlib.Path(f"notes/{safe_name.upper()}_FINAL_REPORT.txt")
        notes_file = pathlib.Path(f"notes/{safe_name.upper()}.txt")
        if final_notes_file.exists():
            with final_notes_file.open(encoding="utf-8") as fh:
                content = fh.read()
        elif notes_file.exists():
            with notes_file.open(encoding="utf-8") as fh:
                content = fh.read()
        else:
            content = "No research notes available."

        content = re.sub(
            r"(?im)^\s*#\s+Final Research Report.*\n?",
            "",
            content,
        ).strip()
        findings_heading = "" if content.startswith("## ") else "## Findings\n"

        status_label = "Complete" if completed else "Stopped before completion"
        reason_line = (
            f"\n**Stop Reason:** {reason}" if (reason and not completed) else ""
        )
        report_title_prefix = "Final Report" if completed else "Partial Report"

        # Create final report
        report_content = f"""# Research Report: {task.name}

## Task Summary
**Description:** {task.description}
**Duration:** {task.budgetMinutes} minutes budget
**Status:** {status_label}{reason_line}

## Research Metrics
- Total Searches: {raw_metrics.get('search_count', 0)}
- Analysis Steps: {raw_metrics.get('think_count', 0)}
- Summaries Created: {raw_metrics.get('summary_count', 0)}
- Checkpoints: {raw_metrics.get('checkpoint_count', 0)}

{findings_heading}{content}

---
*Report generated at {datetime.now().isoformat()}*"""

        self.redis_client.send_report(
            task_id,
            f"{report_title_prefix} - {task.name}",
            report_content,
            metadata={"final": completed, "reason": reason, "metrics": final_metrics},
        )


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main():
    """Main entry point for the agent worker"""
    worker = KestrelAgentWorker()

    worker.run()


if __name__ == "__main__":
    main()
