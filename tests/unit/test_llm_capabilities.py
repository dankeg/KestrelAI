from __future__ import annotations

from enum import Enum

from KestrelAI.shared.llm_capabilities import (
    infer_request_capability_profile,
    normalize_runtime_provider,
)


def test_normalize_runtime_provider():
    assert normalize_runtime_provider("ollama") == "ollama_native"
    assert normalize_runtime_provider("openai_compatible") == "openai_compatible"


def test_normalize_runtime_provider_accepts_enum_values():
    class ProviderEnum(str, Enum):
        ollama = "ollama"
        openai_compatible = "openai_compatible"

    assert normalize_runtime_provider(ProviderEnum.ollama) == "ollama_native"
    assert (
        normalize_runtime_provider(ProviderEnum.openai_compatible)
        == "openai_compatible"
    )


def test_infer_request_profile_for_openai_reasoning_model():
    profile = infer_request_capability_profile(
        model="o4-mini",
        provider="openai_compatible",
        host="https://api.openai.com/v1",
    )

    assert profile.reasoning_control == "openai_responses"
    assert profile.use_responses_api is True
    assert profile.output_version == "responses/v1"


def test_infer_request_profile_for_unknown_openai_compatible_provider():
    profile = infer_request_capability_profile(
        model="qwen3-32b",
        provider="openai_compatible",
        host="https://api.example.com/v1",
    )

    assert profile.reasoning_control == "none"
    assert profile.use_responses_api is False


def test_infer_request_profile_for_ollama_thinking_model():
    profile = infer_request_capability_profile(
        model="qwen3:14b",
        provider="ollama_native",
        host="http://localhost:11434",
    )

    assert profile.reasoning_control == "ollama_reasoning"
    assert profile.reasoning_parameter is True
