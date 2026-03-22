"""
LangChain adapter layer for provider-agnostic chat + structured outputs.

Default mode targets OpenAI-compatible chat schema so Kestrel can work with
OpenAI, Ollama OpenAI endpoints, vLLM, LM Studio, and similar providers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Callable
from typing import Any, TypeVar

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
_INLINE_REASONING_PATTERN = re.compile(
    r"(?is)<think>\s*(.*?)\s*</think>|<reasoning>\s*(.*?)\s*</reasoning>"
)

from KestrelAI.shared.llm_capabilities import (
    NormalizedChatResponse,
    ObservedLlmCapabilities,
    infer_request_capability_profile,
)
from KestrelAI.shared.runtime_settings import (
    default_local_api_key,
    normalize_openai_base_url,
)


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
        self.request_profile = infer_request_capability_profile(
            model=self.model,
            provider=self.provider,
            host=host or "",
        )
        self._observed_capabilities = ObservedLlmCapabilities(
            reasoning_control=self.request_profile.reasoning_control,
            provider_transport=self.request_profile.provider_transport,
            use_responses_api=self.request_profile.use_responses_api,
        )
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
                use_responses_api=self.request_profile.use_responses_api or None,
                output_version=self.request_profile.output_version,
            )
            self._observed_capabilities.structured_outputs_supported = hasattr(
                self.client,
                "with_structured_output",
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
                reasoning=self.request_profile.reasoning_parameter,
            )
            self._observed_capabilities.structured_outputs_supported = hasattr(
                self.client,
                "with_structured_output",
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
        return self._normalize_response(response).visible_text

    def chat_response(self, messages: list[dict]) -> NormalizedChatResponse:
        lc_messages = self._to_langchain_messages(messages)
        response = self.client.invoke(lc_messages)
        return self._normalize_response(response)

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
        return normalize_openai_base_url(raw_base_url)

    @staticmethod
    def _default_local_api_key(base_url: str) -> str | None:
        return default_local_api_key(base_url)

    def get_capabilities(self) -> dict[str, Any]:
        return self._observed_capabilities.model_dump()

    def _normalize_response(self, response: Any) -> NormalizedChatResponse:
        content = getattr(response, "content", response)
        visible_text, reasoning_text, reasoning_transport = self._extract_text_channels(
            content
        )

        additional_kwargs = getattr(response, "additional_kwargs", None) or {}
        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if not tool_calls and isinstance(additional_kwargs, dict):
            tool_calls = list(additional_kwargs.get("tool_calls") or [])

        separate_reasoning = ""
        if isinstance(additional_kwargs, dict):
            separate_reasoning = self._extract_additional_reasoning(additional_kwargs)
            if separate_reasoning and not reasoning_text:
                reasoning_text = separate_reasoning
                reasoning_transport = (
                    "separate_field"
                    if reasoning_transport == "none"
                    else reasoning_transport
                )

        normalized = NormalizedChatResponse(
            visible_text=(visible_text or "").strip(),
            reasoning_text=(reasoning_text or "").strip(),
            reasoning_present=bool((reasoning_text or "").strip()),
            reasoning_transport=reasoning_transport,
            tool_calls=tool_calls,
            raw_response=response,
        )

        if normalized.reasoning_present:
            self._observed_capabilities.reasoning_transport = (
                normalized.reasoning_transport or "unknown"
            )
        if normalized.tool_calls:
            self._observed_capabilities.tool_calls_supported = True

        return normalized

    def _extract_text_channels(self, content: Any) -> tuple[str, str, str]:
        if isinstance(content, str):
            return self._split_inline_reasoning(content)

        visible_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_transport = "none"

        if isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    (
                        visible_text,
                        reasoning_text,
                        inline_transport,
                    ) = self._split_inline_reasoning(item)
                    if visible_text:
                        visible_parts.append(visible_text)
                    if reasoning_text:
                        reasoning_parts.append(reasoning_text)
                        reasoning_transport = inline_transport
                    continue

                if not isinstance(item, dict):
                    text = str(item).strip()
                    if text:
                        visible_parts.append(text)
                    continue

                item_type = str(item.get("type") or "").strip().lower()
                item_text = self._extract_item_text(item)
                if not item_text:
                    continue

                if item_type in {
                    "reasoning",
                    "thinking",
                    "reasoning_content",
                    "reasoning_summary",
                    "summary_text",
                }:
                    reasoning_parts.append(item_text)
                    reasoning_transport = "content_array"
                else:
                    (
                        visible_text,
                        reasoning_text,
                        inline_transport,
                    ) = self._split_inline_reasoning(item_text)
                    if visible_text:
                        visible_parts.append(visible_text)
                    if reasoning_text:
                        reasoning_parts.append(reasoning_text)
                        reasoning_transport = inline_transport

            return (
                "\n".join(part for part in visible_parts if part).strip(),
                "\n".join(part for part in reasoning_parts if part).strip(),
                reasoning_transport,
            )

        text = str(content).strip()
        return self._split_inline_reasoning(text)

    @staticmethod
    def _extract_item_text(item: dict[str, Any]) -> str:
        direct_fields = ("text", "content", "thinking", "reasoning", "summary")
        for field in direct_fields:
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                joined = " ".join(
                    str(part).strip() for part in value if str(part).strip()
                ).strip()
                if joined:
                    return joined

        nested_text = item.get("text")
        if isinstance(nested_text, dict):
            for field in ("value", "text"):
                value = nested_text.get(field)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return ""

    @classmethod
    def _split_inline_reasoning(cls, text: str) -> tuple[str, str, str]:
        if not isinstance(text, str) or not text.strip():
            return "", "", "none"

        reasoning_chunks: list[str] = []

        def _capture(match: re.Match[str]) -> str:
            chunk = next((group for group in match.groups() if group), "")
            chunk = chunk.strip()
            if chunk:
                reasoning_chunks.append(chunk)
            return " "

        visible_text = _INLINE_REASONING_PATTERN.sub(_capture, text).strip()
        visible_text = re.sub(r"\n{3,}", "\n\n", visible_text)
        reasoning_text = "\n".join(reasoning_chunks).strip()
        if reasoning_text:
            return visible_text, reasoning_text, "inline_text"
        return text.strip(), "", "none"

    @staticmethod
    def _extract_additional_reasoning(additional_kwargs: dict[str, Any]) -> str:
        reasoning_values: list[str] = []
        for field in ("thinking", "reasoning", "reasoning_content"):
            value = additional_kwargs.get(field)
            if isinstance(value, str) and value.strip():
                reasoning_values.append(value.strip())
            elif isinstance(value, list):
                flattened = " ".join(
                    str(part).strip() for part in value if str(part).strip()
                ).strip()
                if flattened:
                    reasoning_values.append(flattened)
            elif isinstance(value, dict):
                for nested_field in ("text", "summary", "content", "value"):
                    nested = value.get(nested_field)
                    if isinstance(nested, str) and nested.strip():
                        reasoning_values.append(nested.strip())
                        break
        return "\n".join(reasoning_values).strip()
