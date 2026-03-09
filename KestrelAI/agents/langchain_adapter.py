"""
LangChain adapter layer for provider-agnostic chat + structured outputs.

Default mode targets OpenAI-compatible chat schema so Kestrel can work with
OpenAI, Ollama OpenAI endpoints, vLLM, LM Studio, and similar providers.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from typing import Any, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - dependency-gated path
    AIMessage = HumanMessage = SystemMessage = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - dependency-gated path
    ChatOpenAI = None

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover - dependency-gated path
    ChatOllama = None

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)
_OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai_compatible", "openai-compatible"}
_OLLAMA_NATIVE_PROVIDERS = {"ollama_native", "ollama-native"}


class LangChainChatAdapter:
    """Chat adapter that mirrors the existing LlmWrapper surface."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.6,
        host: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
    ):
        if AIMessage is None or HumanMessage is None or SystemMessage is None:
            raise ImportError(
                "langchain_core messages unavailable. Install langchain-core/langchain."
            )

        self.model = model
        self.temperature = temperature
        resolved_provider = (provider or os.getenv("LLM_PROVIDER", "")).strip().lower()
        self.provider = resolved_provider or "openai_compatible"
        raw_request_timeout_seconds = float(
            os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "300")
        )
        disable_timeouts = os.getenv(
            "GLOBAL_DISABLE_TIMEOUTS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        request_timeout_seconds = (
            None
            if disable_timeouts or raw_request_timeout_seconds <= 0
            else raw_request_timeout_seconds
        )
        request_max_retries = int(os.getenv("LLM_REQUEST_MAX_RETRIES", "1"))

        if self.provider in _OPENAI_COMPATIBLE_PROVIDERS:
            if ChatOpenAI is None:
                raise ImportError(
                    "LangChain OpenAI adapter unavailable. Install langchain-openai."
                )
            resolved_base_url = self._normalize_openai_base_url(
                host
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("OLLAMA_BASE_URL")
                or "http://localhost:11434"
            )
            resolved_api_key = (
                api_key
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("LLM_API_KEY")
                or self._default_local_api_key(resolved_base_url)
            )
            if not resolved_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY (or LLM_API_KEY) is required for non-local OpenAI-compatible endpoints."
                )
            self.host = resolved_base_url
            self.api_key = resolved_api_key
            self.client = ChatOpenAI(
                model=self.model,
                base_url=self.host,
                api_key=self.api_key,
                temperature=self.temperature,
                timeout=request_timeout_seconds,
                max_retries=request_max_retries,
            )
            return

        if self.provider in _OLLAMA_NATIVE_PROVIDERS:
            if ChatOllama is None:
                raise ImportError(
                    "LangChain Ollama adapter unavailable. Install langchain-ollama."
                )
            self.host = host or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.api_key = None
            self.client = ChatOllama(
                model=self.model,
                base_url=self.host,
                temperature=self.temperature,
            )
            return

        raise ValueError(
            f"Unsupported LLM_PROVIDER '{self.provider}'. "
            "Use 'openai_compatible' (default) or 'ollama_native'."
        )

    @classmethod
    def from_llm(cls, llm: Any) -> LangChainChatAdapter:
        """Build adapter configuration from an existing LLM wrapper-like object."""
        raw_model_name = getattr(llm, "model", "gemma3:27b")
        model_name = raw_model_name if isinstance(raw_model_name, str) else "gemma3:27b"

        raw_temperature = getattr(llm, "temperature", 0.6)
        temperature = (
            float(raw_temperature) if isinstance(raw_temperature, (int, float)) else 0.6
        )

        model_host = getattr(llm, "host", None)
        if not isinstance(model_host, str):
            client = getattr(llm, "client", None)
            client_host = getattr(client, "host", None) if client is not None else None
            model_host = client_host if isinstance(client_host, str) else None

        raw_provider = getattr(llm, "provider", None)
        provider = raw_provider if isinstance(raw_provider, str) else None
        raw_api_key = getattr(llm, "api_key", None)
        api_key = raw_api_key if isinstance(raw_api_key, str) else None

        return cls(
            model=model_name,
            temperature=temperature,
            host=model_host,
            provider=provider,
            api_key=api_key,
        )

    def chat(self, messages: list[dict], stream: bool = False):
        lc_messages = self._to_langchain_messages(messages)
        if stream:
            return self.client.stream(lc_messages)
        response = self.client.invoke(lc_messages)
        return response.content if hasattr(response, "content") else str(response)

    def chat_structured(
        self, messages: list[dict], schema: type[TModel], max_retries: int = 2
    ) -> TModel:
        if not hasattr(self.client, "with_structured_output"):
            raise RuntimeError(
                "Structured outputs are not supported by the configured chat client."
            )

        lc_messages = self._to_langchain_messages(messages)
        model_with_schema = self.client.with_structured_output(schema)
        last_error: Exception | None = None

        for _ in range(max_retries + 1):
            try:
                return model_with_schema.invoke(lc_messages)
            except Exception as e:
                last_error = e
                logger.debug("Structured output attempt failed: %s", e)

        raise RuntimeError(
            "Structured output call failed after retries"
        ) from last_error

    async def chat_structured_async(
        self,
        messages: list[dict],
        schema: type[TModel],
        *,
        timeout_seconds: float | None = None,
        retries: int = 2,
        fallback_factory: Callable[[Exception | None], TModel] | None = None,
    ) -> TModel:
        """Async wrapper for structured outputs with retry/timeout/fallback policy."""
        attempts = max(1, retries + 1)
        last_error: Exception | None = None

        for _ in range(attempts):
            try:
                if timeout_seconds is not None and timeout_seconds > 0:
                    return await asyncio.wait_for(
                        asyncio.to_thread(
                            self.chat_structured,
                            messages,
                            schema,
                            0,
                        ),
                        timeout=timeout_seconds,
                    )
                return await asyncio.to_thread(
                    self.chat_structured,
                    messages,
                    schema,
                    0,
                )
            except Exception as e:
                last_error = e
                logger.debug("Async structured output attempt failed: %s", e)

        if fallback_factory is not None:
            return fallback_factory(last_error)
        raise RuntimeError(
            "Structured output call failed after retries"
        ) from last_error

    def _to_langchain_messages(self, messages: list[dict]) -> list[Any]:
        converted: list[Any] = []
        for msg in messages:
            role = (msg.get("role") or "").lower()
            content = msg.get("content", "")
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
        return converted

    @staticmethod
    def _normalize_openai_base_url(raw_base_url: str) -> str:
        base_url = raw_base_url.strip()
        if not base_url:
            return "https://api.openai.com/v1"

        if "://" not in base_url:
            base_url = f"http://{base_url}"

        parsed = urlparse(base_url)
        path = (parsed.path or "").rstrip("/")
        if not path:
            return f"{base_url.rstrip('/')}/v1"
        return base_url.rstrip("/")

    @staticmethod
    def _default_local_api_key(base_url: str) -> str | None:
        host = (urlparse(base_url).hostname or "").lower()
        local_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"}
        if host in local_hosts or "ollama" in host:
            return "not-required"
        return None
