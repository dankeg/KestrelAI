from __future__ import annotations

import os
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from KestrelAI.shared.runtime_settings import (
    get_default_llm_provider,
    get_default_model_name,
    get_default_openai_base_url,
    normalize_max_context_tokens,
    normalize_openai_base_url,
)


class Theme(str, Enum):
    amber = "amber"
    blue = "blue"


class LlmProvider(str, Enum):
    ollama = "ollama"
    openai_compatible = "openai_compatible"


class OllamaMode(str, Enum):
    local = "local"
    docker = "docker"


class Orchestrator(str, Enum):
    hummingbird = "hummingbird"
    kestrel = "kestrel"
    albatross = "albatross"


class SettingsCoreFields(BaseModel):
    llmProvider: LlmProvider = Field(
        default_factory=lambda: LlmProvider(get_default_llm_provider()),
        description="Model provider mode exposed in the UI",
    )
    ollamaMode: OllamaMode = Field(
        default=OllamaMode.local, description="Where to send Ollama calls"
    )
    orchestrator: Orchestrator = Field(
        default=Orchestrator.kestrel, description="Research orchestrator profile"
    )
    theme: Theme = Field(default=Theme.amber, description="UI theme color scheme")
    modelName: str = Field(
        default_factory=get_default_model_name,
        min_length=1,
        description="Selected model identifier for the active LLM provider",
    )
    executionModelName: str | None = Field(
        default=None,
        description="Optional execution model override for worker search/extraction loops",
    )
    controlModelName: str | None = Field(
        default=None,
        description="Optional control model override for orchestration and final report synthesis",
    )
    openaiBaseUrl: str = Field(
        default_factory=get_default_openai_base_url,
        description="Base URL for OpenAI-compatible APIs when that provider is selected",
    )
    openaiApiKey: str = Field(
        default_factory=lambda: (
            os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
        ).strip(),
        description="API key for OpenAI-compatible APIs when required",
    )
    maxContextTokens: int = Field(
        default_factory=lambda: normalize_max_context_tokens(
            os.getenv("MAX_CONTEXT_TOKENS", "32768")
        ),
        ge=2048,
        le=262144,
        description="Maximum context window used by agent/orchestrator token budgeting",
    )

    @field_validator("modelName")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        return value.strip()

    @field_validator("executionModelName", "controlModelName", mode="before")
    @classmethod
    def validate_optional_model_name(cls, value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("openaiBaseUrl")
    @classmethod
    def validate_openai_base_url(cls, value: str) -> str:
        return normalize_openai_base_url(value)

    @field_validator("openaiApiKey")
    @classmethod
    def validate_openai_api_key(cls, value: str) -> str:
        return value.strip()


class AppSettings(SettingsCoreFields):
    pass


class AppSettingsPayload(SettingsCoreFields):
    openaiApiKey: str = Field(
        default="",
        description="Replacement API key for OpenAI-compatible APIs when required",
    )
    clearOpenaiApiKey: bool = Field(
        default=False,
        description="Clear any previously stored OpenAI-compatible API key",
    )

    @field_validator("openaiApiKey")
    @classmethod
    def validate_payload_openai_api_key(cls, value: str) -> str:
        return value.strip()


class AppSettingsResponse(BaseModel):
    llmProvider: LlmProvider
    ollamaMode: OllamaMode
    orchestrator: Orchestrator
    theme: Theme
    modelName: str
    executionModelName: str
    controlModelName: str
    openaiBaseUrl: str
    openaiApiKeySet: bool = False
    maxContextTokens: int


def resolve_role_model_name(
    settings: AppSettings | dict[str, Any], role_key: str
) -> str:
    raw = settings.model_dump() if isinstance(settings, BaseModel) else dict(settings)
    role_value = raw.get(role_key)
    if isinstance(role_value, str) and role_value.strip():
        return role_value.strip()
    return str(raw.get("modelName") or get_default_model_name()).strip()


def redact_settings(settings: AppSettings | dict[str, Any]) -> dict[str, Any]:
    raw = settings.model_dump() if isinstance(settings, BaseModel) else dict(settings)
    api_key = raw.get("openaiApiKey")
    if isinstance(api_key, str) and api_key:
        raw["openaiApiKey"] = "***redacted***"
    return raw


def settings_response(settings: AppSettings | dict[str, Any]) -> AppSettingsResponse:
    raw = settings.model_dump() if isinstance(settings, BaseModel) else dict(settings)
    return AppSettingsResponse(
        llmProvider=LlmProvider(raw.get("llmProvider", get_default_llm_provider())),
        ollamaMode=OllamaMode(raw.get("ollamaMode", OllamaMode.local)),
        orchestrator=Orchestrator(raw.get("orchestrator", Orchestrator.kestrel)),
        theme=Theme(raw.get("theme", Theme.amber)),
        modelName=str(raw.get("modelName") or get_default_model_name()).strip(),
        executionModelName=resolve_role_model_name(raw, "executionModelName"),
        controlModelName=resolve_role_model_name(raw, "controlModelName"),
        openaiBaseUrl=normalize_openai_base_url(
            str(raw.get("openaiBaseUrl") or get_default_openai_base_url())
        ),
        openaiApiKeySet=bool(str(raw.get("openaiApiKey") or "").strip()),
        maxContextTokens=normalize_max_context_tokens(
            raw.get("maxContextTokens", 32768)
        ),
    )


def merge_api_key(
    existing_api_key: str, submitted_api_key: str, clear_api_key: bool
) -> str:
    if clear_api_key:
        return ""
    if submitted_api_key.strip():
        return submitted_api_key.strip()
    return existing_api_key.strip()


def build_stored_settings(
    payload: AppSettingsPayload, existing_settings: AppSettings
) -> AppSettings:
    return AppSettings(
        llmProvider=payload.llmProvider,
        ollamaMode=payload.ollamaMode,
        orchestrator=payload.orchestrator,
        theme=payload.theme,
        modelName=payload.modelName,
        executionModelName=payload.executionModelName or payload.modelName,
        controlModelName=payload.controlModelName or payload.modelName,
        openaiBaseUrl=payload.openaiBaseUrl,
        openaiApiKey=merge_api_key(
            existing_api_key=existing_settings.openaiApiKey,
            submitted_api_key=payload.openaiApiKey,
            clear_api_key=payload.clearOpenaiApiKey,
        ),
        maxContextTokens=payload.maxContextTokens,
    )


def build_agent_settings_payload(settings: AppSettings) -> dict[str, Any]:
    payload = settings.model_dump()
    payload["executionModelName"] = resolve_role_model_name(
        settings, "executionModelName"
    )
    payload["controlModelName"] = resolve_role_model_name(settings, "controlModelName")
    if settings.llmProvider != LlmProvider.openai_compatible:
        payload["openaiApiKey"] = ""
    return payload
