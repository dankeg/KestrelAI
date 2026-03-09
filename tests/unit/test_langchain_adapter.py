from __future__ import annotations

from types import SimpleNamespace

import pytest

from KestrelAI.agents import langchain_adapter as adapter_module
from KestrelAI.agents.langchain_adapter import LangChainChatAdapter


class _FakeOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, _messages):
        return SimpleNamespace(content="ok")

    def stream(self, _messages):
        yield SimpleNamespace(content="ok")

    def with_structured_output(self, _schema):
        return self


class _FakeOllamaClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, _messages):
        return SimpleNamespace(content="ok")

    def stream(self, _messages):
        yield SimpleNamespace(content="ok")

    def with_structured_output(self, _schema):
        return self


@pytest.mark.unit
def test_openai_compatible_defaults_to_openai_schema(monkeypatch):
    monkeypatch.setattr(adapter_module, "ChatOpenAI", _FakeOpenAIClient)
    monkeypatch.setattr(adapter_module, "ChatOllama", _FakeOllamaClient)
    monkeypatch.setattr(adapter_module, "AIMessage", object)
    monkeypatch.setattr(adapter_module, "HumanMessage", object)
    monkeypatch.setattr(adapter_module, "SystemMessage", object)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    adapter = LangChainChatAdapter(
        model="gemma3:27b",
        host="http://localhost:11434",
        provider="openai_compatible",
    )

    assert adapter.provider == "openai_compatible"
    assert adapter.host == "http://localhost:11434/v1"
    assert adapter.api_key == "not-required"
    assert adapter.client.kwargs["base_url"] == "http://localhost:11434/v1"


@pytest.mark.unit
def test_ollama_native_uses_chatollama(monkeypatch):
    monkeypatch.setattr(adapter_module, "ChatOpenAI", _FakeOpenAIClient)
    monkeypatch.setattr(adapter_module, "ChatOllama", _FakeOllamaClient)
    monkeypatch.setattr(adapter_module, "AIMessage", object)
    monkeypatch.setattr(adapter_module, "HumanMessage", object)
    monkeypatch.setattr(adapter_module, "SystemMessage", object)

    adapter = LangChainChatAdapter(
        model="gemma3:27b",
        host="http://localhost:11434",
        provider="ollama_native",
    )

    assert adapter.provider == "ollama_native"
    assert adapter.host == "http://localhost:11434"
    assert adapter.api_key is None
    assert adapter.client.kwargs["base_url"] == "http://localhost:11434"
