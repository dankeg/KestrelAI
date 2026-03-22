from __future__ import annotations

from fastapi.testclient import TestClient

try:
    from KestrelAI.backend import main
    from KestrelAI.backend.main import AppSettings, LlmProvider, OllamaMode
except ImportError:
    from backend import main
    from backend.main import AppSettings, LlmProvider, OllamaMode


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse, call_log: list[dict]):
        self._response = response
        self._call_log = call_log

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, headers: dict | None = None):
        self._call_log.append({"url": url, "headers": headers or {}})
        return self._response


def test_settings_roundtrip_openai_compatible(monkeypatch):
    async def _raise_get_redis():
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(main, "get_redis", _raise_get_redis)
    main.settings_memory = AppSettings()

    client = TestClient(main.app)
    payload = {
        "llmProvider": "openai_compatible",
        "ollamaMode": "local",
        "orchestrator": "kestrel",
        "theme": "amber",
        "modelName": "gpt-4.1-mini",
        "openaiBaseUrl": "https://api.example.com/v1",
        "openaiApiKey": "sk-test",
        "maxContextTokens": 32768,
    }

    post_response = client.post("/settings", json=payload)
    assert post_response.status_code == 200
    assert post_response.json()["llmProvider"] == "openai_compatible"
    assert post_response.json()["openaiBaseUrl"] == "https://api.example.com/v1"
    assert post_response.json()["openaiApiKeySet"] is True
    assert post_response.json()["executionModelName"] == "gpt-4.1-mini"
    assert post_response.json()["controlModelName"] == "gpt-4.1-mini"
    assert "openaiApiKey" not in post_response.json()

    get_response = client.get("/settings")
    assert get_response.status_code == 200
    assert get_response.json()["llmProvider"] == "openai_compatible"
    assert get_response.json()["modelName"] == "gpt-4.1-mini"
    assert get_response.json()["executionModelName"] == "gpt-4.1-mini"
    assert get_response.json()["controlModelName"] == "gpt-4.1-mini"
    assert get_response.json()["openaiApiKeySet"] is True
    assert "openaiApiKey" not in get_response.json()


def test_settings_roundtrip_role_specific_models(monkeypatch):
    async def _raise_get_redis():
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(main, "get_redis", _raise_get_redis)
    main.settings_memory = AppSettings()

    client = TestClient(main.app)
    payload = {
        "llmProvider": "openai_compatible",
        "ollamaMode": "local",
        "orchestrator": "kestrel",
        "theme": "amber",
        "modelName": "o4-mini",
        "executionModelName": "gpt-4.1-mini",
        "controlModelName": "o4-mini",
        "openaiBaseUrl": "https://api.example.com/v1",
        "openaiApiKey": "sk-test",
        "maxContextTokens": 32768,
    }

    response = client.post("/settings", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["modelName"] == "o4-mini"
    assert body["executionModelName"] == "gpt-4.1-mini"
    assert body["controlModelName"] == "o4-mini"
    assert main.settings_memory.executionModelName == "gpt-4.1-mini"
    assert main.settings_memory.controlModelName == "o4-mini"


def test_settings_preserve_existing_api_key_when_blank(monkeypatch):
    async def _raise_get_redis():
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(main, "get_redis", _raise_get_redis)
    main.settings_memory = AppSettings(
        llmProvider=LlmProvider.openai_compatible,
        modelName="gpt-4.1-mini",
        openaiBaseUrl="https://api.example.com/v1",
        openaiApiKey="sk-existing",
    )

    client = TestClient(main.app)
    payload = {
        "llmProvider": "openai_compatible",
        "ollamaMode": "local",
        "orchestrator": "kestrel",
        "theme": "amber",
        "modelName": "gpt-4.1-mini",
        "openaiBaseUrl": "https://api.example.com/v1",
        "openaiApiKey": "",
        "maxContextTokens": 32768,
    }

    response = client.post("/settings", json=payload)
    assert response.status_code == 200
    assert response.json()["openaiApiKeySet"] is True
    assert main.settings_memory.openaiApiKey == "sk-existing"


def test_settings_can_clear_existing_api_key(monkeypatch):
    async def _raise_get_redis():
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(main, "get_redis", _raise_get_redis)
    main.settings_memory = AppSettings(
        llmProvider=LlmProvider.openai_compatible,
        modelName="gpt-4.1-mini",
        openaiBaseUrl="https://api.example.com/v1",
        openaiApiKey="sk-existing",
    )

    client = TestClient(main.app)
    payload = {
        "llmProvider": "openai_compatible",
        "ollamaMode": "local",
        "orchestrator": "kestrel",
        "theme": "amber",
        "modelName": "gpt-4.1-mini",
        "openaiBaseUrl": "https://api.example.com/v1",
        "openaiApiKey": "",
        "clearOpenaiApiKey": True,
        "maxContextTokens": 32768,
    }

    response = client.post("/settings", json=payload)
    assert response.status_code == 200
    assert response.json()["openaiApiKeySet"] is False
    assert main.settings_memory.openaiApiKey == ""


def test_agent_settings_payload_omits_api_key_for_ollama():
    settings = AppSettings(
        llmProvider=LlmProvider.ollama,
        modelName="gemma3:12b",
        executionModelName="gemma3:4b",
        controlModelName="gemma3:12b",
        openaiApiKey="sk-existing",
    )

    payload = main.build_agent_settings_payload(settings)

    assert payload["openaiApiKey"] == ""
    assert payload["executionModelName"] == "gemma3:4b"
    assert payload["controlModelName"] == "gemma3:12b"


def test_agent_settings_payload_keeps_api_key_for_openai_compatible():
    settings = AppSettings(
        llmProvider=LlmProvider.openai_compatible,
        modelName="o4-mini",
        executionModelName="gpt-4.1-mini",
        controlModelName="o4-mini",
        openaiApiKey="sk-existing",
    )

    payload = main.build_agent_settings_payload(settings)

    assert payload["openaiApiKey"] == "sk-existing"
    assert payload["executionModelName"] == "gpt-4.1-mini"
    assert payload["controlModelName"] == "o4-mini"


def test_openai_compatible_model_discovery(monkeypatch):
    async def _raise_get_redis():
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(main, "get_redis", _raise_get_redis)
    main.settings_memory = AppSettings(
        llmProvider=LlmProvider.openai_compatible,
        modelName="gpt-4.1-mini",
        openaiBaseUrl="https://api.example.com/v1",
        openaiApiKey="sk-test",
    )

    call_log: list[dict] = []

    def _fake_async_client(*args, **kwargs):
        return _FakeAsyncClient(
            _FakeResponse(
                {
                    "data": [
                        {"id": "gpt-4.1-mini"},
                        {"id": "gpt-4.1"},
                    ]
                }
            ),
            call_log,
        )

    monkeypatch.setattr(main.httpx, "AsyncClient", _fake_async_client)

    client = TestClient(main.app)
    response = client.get("/settings/models?provider=openai_compatible")
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "openai_compatible"
    assert payload["baseUrl"] == "https://api.example.com/v1"
    assert payload["models"] == ["gpt-4.1", "gpt-4.1-mini"]
    assert call_log[0]["url"] == "https://api.example.com/v1/models"
    assert call_log[0]["headers"]["Authorization"] == "Bearer sk-test"


def test_ollama_model_discovery(monkeypatch):
    async def _raise_get_redis():
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(main, "get_redis", _raise_get_redis)
    main.settings_memory = AppSettings(
        llmProvider=LlmProvider.ollama,
        ollamaMode=OllamaMode.local,
        modelName="gemma3:12b",
    )

    call_log: list[dict] = []

    def _fake_async_client(*args, **kwargs):
        return _FakeAsyncClient(
            _FakeResponse({"models": [{"name": "gemma3:12b"}, {"name": "qwen3:14b"}]}),
            call_log,
        )

    monkeypatch.setattr(main.httpx, "AsyncClient", _fake_async_client)

    client = TestClient(main.app)
    response = client.get("/settings/models?provider=ollama&mode=local")
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "ollama"
    assert payload["mode"] == "local"
    assert payload["models"] == ["gemma3:12b", "qwen3:14b"]
    assert call_log[0]["url"].endswith("/api/tags")


def test_runtime_capabilities_endpoint(monkeypatch):
    async def _raise_get_redis():
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(main, "get_redis", _raise_get_redis)
    main.settings_memory = AppSettings(
        llmProvider=LlmProvider.openai_compatible,
        modelName="o4-mini",
        executionModelName="gpt-4.1-mini",
        controlModelName="o4-mini",
        openaiBaseUrl="https://api.openai.com/v1",
        openaiApiKey="sk-test",
    )

    client = TestClient(main.app)
    response = client.get("/settings/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "openai_compatible"
    assert payload["runtimeProvider"] == "openai_compatible"
    assert payload["modelName"] == "o4-mini"
    assert payload["requestProfile"]["reasoning_control"] == "openai_responses"
    assert payload["requestProfile"]["use_responses_api"] is True


def test_update_task_rejects_invalid_status(monkeypatch):
    async def _raise_get_redis():
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(main, "get_redis", _raise_get_redis)
    task = main.Task(name="Enum Test", description="desc", budgetMinutes=5)
    main.tasks_memory[task.id] = task

    client = TestClient(main.app)
    response = client.patch(f"/api/v1/tasks/{task.id}", json={"status": "RUNNING_FAST"})
    assert response.status_code == 422
