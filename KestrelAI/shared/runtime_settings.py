from __future__ import annotations

import os
from urllib.parse import urlparse

DEFAULT_MODEL_NAME = "gemma3:4b"
DEFAULT_MAX_CONTEXT_TOKENS = 32768
MIN_MAX_CONTEXT_TOKENS = 2048
MAX_MAX_CONTEXT_TOKENS = 262144


def get_default_model_name() -> str:
    """Return the configured model name with a stable repo-level fallback."""
    model_name = (os.getenv("MODEL_NAME") or "").strip()
    return model_name or DEFAULT_MODEL_NAME


def normalize_openai_base_url(raw_base_url: str) -> str:
    """Normalize OpenAI-compatible endpoints to a stable /v1-form URL."""
    base_url = (raw_base_url or "").strip()
    if not base_url:
        return "https://api.openai.com/v1"

    if "://" not in base_url:
        base_url = f"http://{base_url}"

    parsed = urlparse(base_url)
    path = (parsed.path or "").rstrip("/")
    if not path:
        return f"{base_url.rstrip('/')}/v1"
    return base_url.rstrip("/")


def default_local_api_key(base_url: str) -> str | None:
    """Return a placeholder API key for local OpenAI-compatible runtimes."""
    host = (urlparse(base_url).hostname or "").lower()
    local_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"}
    if host in local_hosts or "ollama" in host:
        return "not-required"
    return None


def get_default_llm_provider() -> str:
    """
    Resolve the default UI/runtime provider.

    Keep local Ollama as the default when the configured OpenAI-compatible
    endpoint still points at a local Ollama-style runtime.
    """
    explicit_provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit_provider == "ollama_native":
        return "ollama"
    if explicit_provider in {"openai", "openai_compatible", "openai-compatible"}:
        configured_base_url = normalize_openai_base_url(
            os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or ""
        )
        if default_local_api_key(configured_base_url):
            return "ollama"
        return "openai_compatible"

    configured_base_url = normalize_openai_base_url(
        os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or ""
    )
    return (
        "ollama" if default_local_api_key(configured_base_url) else "openai_compatible"
    )


def get_default_openai_base_url() -> str:
    """Return the configured OpenAI-compatible endpoint with normalization."""
    configured = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or "https://api.openai.com/v1"
    )
    return normalize_openai_base_url(configured)


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
