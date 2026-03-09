from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from KestrelAI.agents.tool_executor import LangChainToolExecutor


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_tool_normalizes_hits_and_urls():
    searxng = Mock()
    searxng.search.return_value = [
        {
            "title": "Low Signal",
            "href": "https://reddit.com/r/test",
            "body": "forum discussion",
        },
        {
            "title": "Valid",
            "href": "https://nsf.gov/reu/program",
            "body": "body text",
        },
        {
            "title": "Invalid",
            "href": "not-a-url",
            "body": "ignored",
        },
    ]
    searxng.extract_text.return_value = "extracted content"

    flags = Mock()
    flags.get_or_create_flag.return_value = "[URL_1]"

    executor = LangChainToolExecutor(
        searxng_service=searxng,
        url_flag_manager=flags,
    )

    result = await executor.ainvoke("search_web", {"query": "nsf reu"})

    assert result["query"] == "nsf reu"
    assert len(result["hits"]) == 2
    assert result["hits"][0]["title"] == "Valid"
    assert result["hits"][0]["url"] == "https://nsf.gov/reu/program"
    assert result["hits"][0]["source_tier"] == "authoritative"
    assert result["hits"][0]["official_source"] is True
    assert result["hits"][0]["authority_score"] >= result["hits"][1]["authority_score"]
    assert "[URL_1]" in result["hits"][0]["snippet"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_tool_uses_manager_and_returns_payload():
    mcp_manager = Mock()
    mcp_manager.call_tool = AsyncMock(
        return_value=SimpleNamespace(success=True, data={"k": "v"}, error=None)
    )

    executor = LangChainToolExecutor(
        searxng_service=Mock(),
        url_flag_manager=Mock(),
        mcp_manager=mcp_manager,
        mcp_enabled=lambda: True,
    )

    result = await executor.ainvoke(
        "mcp_call",
        {"tool_name": "query_database", "tool_parameters": {"sql": "select 1"}},
    )

    assert result["success"] is True
    assert result["data"] == {"k": "v"}
    mcp_manager.call_tool.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mcp_tool_respects_enabled_gate():
    mcp_manager = Mock()
    mcp_manager.call_tool = AsyncMock()

    executor = LangChainToolExecutor(
        searxng_service=Mock(),
        url_flag_manager=Mock(),
        mcp_manager=mcp_manager,
        mcp_enabled=lambda: False,
    )

    result = await executor.ainvoke(
        "mcp_call",
        {"tool_name": "query_database", "tool_parameters": {}},
    )

    assert result["success"] is False
    assert "not available" in result["error"].lower()
    mcp_manager.call_tool.assert_not_awaited()
