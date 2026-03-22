from __future__ import annotations

from types import SimpleNamespace

import pytest

from KestrelAI.agents import langchain_adapter as adapter_module
from KestrelAI.agents.langchain_adapter import LangChainChatAdapter


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._response = SimpleNamespace(content="ok")

    def invoke(self, _messages):
        return self._response

    def stream(self, _messages):
        yield SimpleNamespace(content="ok")

    def with_structured_output(self, _schema):
        return self


class _FakeOllamaClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._response = SimpleNamespace(content="ok")

    def invoke(self, _messages):
        return self._response

    def stream(self, _messages):
        yield SimpleNamespace(content="ok")

    def with_structured_output(self, _schema):
        return self


@pytest.mark.unit
def test_openai_compatible_defaults_to_openai_schema(monkeypatch):
    monkeypatch.setattr(adapter_module, "ChatOpenAI", _FakeOpenAIClient)
    monkeypatch.setattr(adapter_module, "ChatOllama", _FakeOllamaClient)
    monkeypatch.setattr(adapter_module, "AIMessage", _FakeMessage)
    monkeypatch.setattr(adapter_module, "HumanMessage", _FakeMessage)
    monkeypatch.setattr(adapter_module, "SystemMessage", _FakeMessage)
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
    monkeypatch.setattr(adapter_module, "AIMessage", _FakeMessage)
    monkeypatch.setattr(adapter_module, "HumanMessage", _FakeMessage)
    monkeypatch.setattr(adapter_module, "SystemMessage", _FakeMessage)

    adapter = LangChainChatAdapter(
        model="gemma3:27b",
        host="http://localhost:11434",
        provider="ollama_native",
    )

    assert adapter.provider == "ollama_native"
    assert adapter.host == "http://localhost:11434"
    assert adapter.api_key is None
    assert adapter.client.kwargs["base_url"] == "http://localhost:11434"
    assert adapter.client.kwargs["reasoning"] is None


@pytest.mark.unit
def test_ollama_native_enables_reasoning_for_known_thinking_models(monkeypatch):
    monkeypatch.setattr(adapter_module, "ChatOpenAI", _FakeOpenAIClient)
    monkeypatch.setattr(adapter_module, "ChatOllama", _FakeOllamaClient)
    monkeypatch.setattr(adapter_module, "AIMessage", _FakeMessage)
    monkeypatch.setattr(adapter_module, "HumanMessage", _FakeMessage)
    monkeypatch.setattr(adapter_module, "SystemMessage", _FakeMessage)

    adapter = LangChainChatAdapter(
        model="qwen3:14b",
        host="http://localhost:11434",
        provider="ollama_native",
    )

    assert adapter.client.kwargs["reasoning"] is True
    assert adapter.get_capabilities()["reasoning_control"] == "ollama_reasoning"


@pytest.mark.unit
def test_openai_reasoning_models_use_responses_api(monkeypatch):
    monkeypatch.setattr(adapter_module, "ChatOpenAI", _FakeOpenAIClient)
    monkeypatch.setattr(adapter_module, "ChatOllama", _FakeOllamaClient)
    monkeypatch.setattr(adapter_module, "AIMessage", _FakeMessage)
    monkeypatch.setattr(adapter_module, "HumanMessage", _FakeMessage)
    monkeypatch.setattr(adapter_module, "SystemMessage", _FakeMessage)

    adapter = LangChainChatAdapter(
        model="o4-mini",
        host="https://api.openai.com/v1",
        provider="openai_compatible",
        api_key="sk-test",
    )

    assert adapter.client.kwargs["use_responses_api"] is True
    assert adapter.client.kwargs["output_version"] == "responses/v1"
    assert adapter.get_capabilities()["reasoning_control"] == "openai_responses"


@pytest.mark.unit
def test_chat_response_extracts_inline_reasoning(monkeypatch):
    monkeypatch.setattr(adapter_module, "ChatOpenAI", _FakeOpenAIClient)
    monkeypatch.setattr(adapter_module, "ChatOllama", _FakeOllamaClient)
    monkeypatch.setattr(adapter_module, "AIMessage", _FakeMessage)
    monkeypatch.setattr(adapter_module, "HumanMessage", _FakeMessage)
    monkeypatch.setattr(adapter_module, "SystemMessage", _FakeMessage)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    adapter = LangChainChatAdapter(
        model="gemma3:27b",
        host="http://localhost:11434",
        provider="openai_compatible",
    )
    adapter.client._response = SimpleNamespace(
        content="<think>reason privately</think>Visible answer"
    )

    response = adapter.chat_response([{"role": "user", "content": "Hello"}])

    assert response.visible_text == "Visible answer"
    assert response.reasoning_text == "reason privately"
    assert response.reasoning_present is True
    assert response.reasoning_transport == "inline_text"
    assert adapter.get_capabilities()["reasoning_transport"] == "inline_text"


@pytest.mark.unit
def test_chat_response_extracts_separate_thinking_field(monkeypatch):
    monkeypatch.setattr(adapter_module, "ChatOpenAI", _FakeOpenAIClient)
    monkeypatch.setattr(adapter_module, "ChatOllama", _FakeOllamaClient)
    monkeypatch.setattr(adapter_module, "AIMessage", _FakeMessage)
    monkeypatch.setattr(adapter_module, "HumanMessage", _FakeMessage)
    monkeypatch.setattr(adapter_module, "SystemMessage", _FakeMessage)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    adapter = LangChainChatAdapter(
        model="gemma3:27b",
        host="http://localhost:11434",
        provider="openai_compatible",
    )
    adapter.client._response = SimpleNamespace(
        content="Visible answer",
        additional_kwargs={"thinking": "private reasoning"},
    )

    response = adapter.chat_response([{"role": "user", "content": "Hello"}])

    assert response.visible_text == "Visible answer"
    assert response.reasoning_text == "private reasoning"
    assert response.reasoning_transport == "separate_field"


@pytest.mark.unit
def test_chat_response_extracts_content_array_reasoning_and_tool_calls(monkeypatch):
    monkeypatch.setattr(adapter_module, "ChatOpenAI", _FakeOpenAIClient)
    monkeypatch.setattr(adapter_module, "ChatOllama", _FakeOllamaClient)
    monkeypatch.setattr(adapter_module, "AIMessage", _FakeMessage)
    monkeypatch.setattr(adapter_module, "HumanMessage", _FakeMessage)
    monkeypatch.setattr(adapter_module, "SystemMessage", _FakeMessage)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    adapter = LangChainChatAdapter(
        model="gpt-5-mini",
        host="http://localhost:11434",
        provider="openai_compatible",
    )
    adapter.client._response = SimpleNamespace(
        content=[
            {"type": "reasoning", "summary": "plan first"},
            {"type": "output_text", "text": "Visible answer"},
        ],
        tool_calls=[{"id": "call_1", "name": "search"}],
    )

    response = adapter.chat_response([{"role": "user", "content": "Hello"}])

    assert response.visible_text == "Visible answer"
    assert response.reasoning_text == "plan first"
    assert response.reasoning_transport == "content_array"
    assert response.tool_calls == [{"id": "call_1", "name": "search"}]
    capabilities = adapter.get_capabilities()
    assert capabilities["reasoning_transport"] == "content_array"
    assert capabilities["tool_calls_supported"] is True
