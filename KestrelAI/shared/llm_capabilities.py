from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel


class RequestCapabilityProfile(BaseModel):
    provider_transport: str
    reasoning_control: Literal["none", "openai_responses", "ollama_reasoning"] = "none"
    reasoning_parameter: Any | None = None
    use_responses_api: bool = False
    output_version: Literal["v0", "responses/v1"] = "v0"


class ObservedLlmCapabilities(BaseModel):
    reasoning_transport: str = "none"
    structured_outputs_supported: bool = False
    tool_calls_supported: bool = False
    reasoning_control: str = "none"
    provider_transport: str = "unknown"
    use_responses_api: bool = False


class NormalizedChatResponse(BaseModel):
    visible_text: str
    reasoning_text: str = ""
    reasoning_present: bool = False
    reasoning_transport: str = "none"
    tool_calls: list[Any] = []
    raw_response: Any | None = None


def normalize_runtime_provider(provider: Any) -> str:
    raw_provider = getattr(provider, "value", provider)
    normalized = str(raw_provider or "ollama").strip().lower()
    return "ollama_native" if normalized == "ollama" else "openai_compatible"


def is_official_openai_host(host: str) -> bool:
    hostname = (urlparse(host).hostname or "").lower()
    return hostname == "api.openai.com"


def is_openai_reasoning_model(model: str) -> bool:
    normalized = (model or "").strip().lower()
    if not normalized:
        return False
    return bool(
        re.match(r"^(o\d|o\d+-|gpt-5|gpt-5\.)", normalized)
        or normalized.startswith("gpt-5-")
    )


def infer_ollama_reasoning_parameter(model: str) -> bool | str | None:
    normalized = (model or "").strip().lower()
    if not normalized:
        return None
    if "gpt-oss" in normalized:
        return "low"
    thinking_families = (
        "qwen3",
        "deepseek-r1",
        "deepseek-v3.1",
    )
    if normalized.startswith(thinking_families):
        return True
    return None


def infer_request_capability_profile(
    *, model: str, provider: str, host: str
) -> RequestCapabilityProfile:
    normalized_provider = str(provider or "").strip().lower()
    normalized_host = (host or "").strip()

    if normalized_provider == "ollama_native":
        reasoning_parameter = infer_ollama_reasoning_parameter(model)
        return RequestCapabilityProfile(
            provider_transport="ollama_native",
            reasoning_control=(
                "ollama_reasoning" if reasoning_parameter is not None else "none"
            ),
            reasoning_parameter=reasoning_parameter,
        )

    if normalized_provider in {"openai", "openai_compatible", "openai-compatible"}:
        if is_official_openai_host(normalized_host) and is_openai_reasoning_model(
            model
        ):
            return RequestCapabilityProfile(
                provider_transport="openai_compatible",
                reasoning_control="openai_responses",
                use_responses_api=True,
                output_version="responses/v1",
            )
        return RequestCapabilityProfile(provider_transport="openai_compatible")

    return RequestCapabilityProfile(provider_transport=normalized_provider or "unknown")
