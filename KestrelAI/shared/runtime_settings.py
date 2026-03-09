from __future__ import annotations

import os

DEFAULT_MODEL_NAME = "gemma3:4b"
DEFAULT_MAX_CONTEXT_TOKENS = 32768
MIN_MAX_CONTEXT_TOKENS = 2048
MAX_MAX_CONTEXT_TOKENS = 262144


def get_default_model_name() -> str:
    """Return the configured model name with a stable repo-level fallback."""
    model_name = (os.getenv("MODEL_NAME") or "").strip()
    return model_name or DEFAULT_MODEL_NAME


def normalize_max_context_tokens(raw_value: object) -> int:
    """Clamp max-context settings to sane bounds."""
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_CONTEXT_TOKENS
    return max(MIN_MAX_CONTEXT_TOKENS, min(value, MAX_MAX_CONTEXT_TOKENS))


def resolve_llm_base_url(
    *, mode: str = "local", running_in_docker: bool = False
) -> str:
    """
    Resolve the main model endpoint.

    `OPENAI_BASE_URL` is authoritative for the OpenAI-compatible mainline.
    `OLLAMA_BASE_URL` remains a compatibility fallback for older local setups.
    """
    explicit_openai = (os.getenv("OPENAI_BASE_URL") or "").strip()
    explicit_ollama = (os.getenv("OLLAMA_BASE_URL") or "").strip()

    if mode == "docker":
        if explicit_openai:
            return explicit_openai.rstrip("/")
        if explicit_ollama:
            return explicit_ollama.rstrip("/")
        return "http://ollama:11434"

    if explicit_openai:
        return explicit_openai.rstrip("/")
    if explicit_ollama and "://ollama:" not in explicit_ollama:
        return explicit_ollama.rstrip("/")

    if running_in_docker:
        return "http://host.docker.internal:11434"
    return "http://localhost:11434"
