"""
LangChain tool execution layer for research actions.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from KestrelAI.mcp.langchain_toolkit import LangChainMCPToolkit

from .url_utils import clean_url

try:
    from langchain_core.tools import StructuredTool
except ImportError:  # pragma: no cover - dependency-gated path
    StructuredTool = None

logger = logging.getLogger(__name__)


def _classify_source(url: str) -> dict[str, Any]:
    host = (urlparse(url).netloc or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]

    tier = "general"
    score = 1
    official = False

    high_trust_suffixes = (".gov", ".edu")
    research_hosts = (
        "arxiv.org",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "nih.gov",
        "nsf.gov",
        "grants.gov",
    )
    medium_trust_suffixes = (".org",)

    if host.endswith(high_trust_suffixes) or host in research_hosts:
        tier = "authoritative"
        score = 4
        official = True
    elif any(host.endswith(suffix) for suffix in medium_trust_suffixes):
        tier = "trusted_org"
        score = 3
    elif host.endswith(".com"):
        tier = "commercial"
        score = 2

    if any(
        marker in host
        for marker in (
            "wikipedia.org",
            "reddit.com",
            "quora.com",
            "medium.com",
            "substack.com",
        )
    ):
        tier = "low_signal"
        score = 0
        official = False

    return {
        "domain": host,
        "source_tier": tier,
        "authority_score": score,
        "official_source": official,
    }


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class SearchToolInput(BaseModel):
    query: str = Field(min_length=1)


class MCPToolInput(BaseModel):
    tool_name: str = Field(min_length=1)
    tool_parameters: dict[str, Any] = Field(default_factory=dict)


class LangChainToolExecutor:
    """Unified tool executor backed by LangChain StructuredTool."""

    def __init__(
        self,
        *,
        searxng_service: Any,
        url_flag_manager: Any,
        mcp_manager: Any | None = None,
        mcp_enabled: Callable[[], bool] | None = None,
    ):
        if StructuredTool is None:
            raise ImportError("langchain_core is not installed")

        self.searxng_service = searxng_service
        self.url_flag_manager = url_flag_manager
        self.mcp_manager = mcp_manager
        self.mcp_enabled = mcp_enabled
        self.mcp_toolkit: LangChainMCPToolkit | None = None

        self._tools = {
            "search_web": StructuredTool.from_function(
                func=self._search_web,
                name="search_web",
                description="Search the web and return normalized snippets.",
                args_schema=SearchToolInput,
            ),
            "mcp_call": StructuredTool.from_function(
                coroutine=self._call_mcp_tool,
                name="mcp_call",
                description="Call an MCP tool with structured parameters.",
                args_schema=MCPToolInput,
            ),
        }

        if self.mcp_manager is not None:
            try:
                self.mcp_toolkit = LangChainMCPToolkit(
                    mcp_manager=self.mcp_manager,
                    mcp_enabled=self.mcp_enabled,
                )
                self._tools.update(self.mcp_toolkit.build_tools())
            except Exception as e:
                logger.debug("Failed to initialize MCP toolkit tools: %s", e)
                self.mcp_toolkit = None

    async def ainvoke(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool '{tool_name}'")
        result = await tool.ainvoke(payload)
        return result if isinstance(result, dict) else {"result": result}

    def get_available_tools(self) -> list[str]:
        """List registered tool names."""
        return sorted(self._tools.keys())

    def resolve_mcp_tool_name(self, tool_name: str) -> str | None:
        """Resolve a logical MCP tool name to a registered executor tool key."""
        if self.mcp_toolkit is None:
            return None
        return self.mcp_toolkit.resolve_tool_name(tool_name)

    async def call_mcp_tool(
        self, tool_name: str, tool_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke MCP tool using toolkit-specific tool if available."""
        resolved = self.resolve_mcp_tool_name(tool_name)
        if resolved:
            return await self.ainvoke(resolved, tool_parameters)
        return await self.ainvoke(
            "mcp_call",
            {"tool_name": tool_name, "tool_parameters": tool_parameters},
        )

    def _search_web(self, query: str) -> dict[str, Any]:
        start = time.perf_counter()
        hits = self.searxng_service.search(query)
        processed_hits: list[dict[str, Any]] = []
        disable_timeouts = os.getenv(
            "GLOBAL_DISABLE_TIMEOUTS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        raw_total_timeout_seconds = _env_float(
            "TOOL_SEARCH_TOTAL_TIMEOUT_SECONDS", 25.0
        )
        total_timeout_seconds: float | None
        if disable_timeouts or raw_total_timeout_seconds <= 0:
            total_timeout_seconds = None
        else:
            total_timeout_seconds = max(1.0, raw_total_timeout_seconds)
        max_fetched_hits = max(
            1,
            _env_int("TOOL_SEARCH_MAX_FETCHED_HITS", 3),
        )
        budget_exhausted = False

        for hit in hits[:max_fetched_hits]:
            if (
                total_timeout_seconds is not None
                and (time.perf_counter() - start) >= total_timeout_seconds
            ):
                budget_exhausted = True
                logger.warning(
                    "Tool search budget exhausted for query '%s' after %.2fs",
                    query,
                    time.perf_counter() - start,
                )
                break
            href = str(hit.get("href", ""))
            clean_href = clean_url(href)
            if clean_href is None:
                logger.warning(
                    "Skipping invalid URL from search result: %s", href[:100]
                )
                continue

            body = self.searxng_service.extract_text(clean_href)
            url_flag = self.url_flag_manager.get_or_create_flag(clean_href)
            if url_flag is None:
                continue
            source_meta = _classify_source(clean_href)

            title = str(hit.get("title", ""))
            summary = str(hit.get("body", ""))
            snippet = (
                f"Title: {title}\n"
                f"URL: {url_flag} (see URL reference table)\n"
                f"Summary: {summary[:200]}\n"
                f"Content: {body[:500]}"
            )
            processed_hits.append(
                {
                    "title": title,
                    "url": clean_href,
                    "summary": summary,
                    "content": body,
                    "fetched": bool(body),
                    "snippet": snippet,
                    **source_meta,
                }
            )

        processed_hits.sort(
            key=lambda item: (
                int(item.get("authority_score", 0)),
                int(bool(item.get("fetched"))),
                len(str(item.get("content", ""))),
            ),
            reverse=True,
        )

        return {
            "query": query,
            "search_time": time.perf_counter() - start,
            "hits": processed_hits,
            "budget_exhausted": budget_exhausted,
            "max_fetched_hits": max_fetched_hits,
        }

    async def _call_mcp_tool(
        self, tool_name: str, tool_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        if self.mcp_toolkit is not None:
            return await self.mcp_toolkit.execute_tool(tool_name, tool_parameters)

        if self.mcp_manager is None:
            return {
                "success": False,
                "tool_name": tool_name,
                "data": None,
                "error": "MCP manager unavailable",
            }

        if self.mcp_enabled is not None and not self.mcp_enabled():
            return {
                "success": False,
                "tool_name": tool_name,
                "data": None,
                "error": "MCP not available",
            }

        try:
            result = await self.mcp_manager.call_tool(tool_name, tool_parameters)
            return {
                "success": bool(result.success),
                "tool_name": tool_name,
                "data": result.data,
                "error": result.error,
            }
        except Exception as e:
            return {
                "success": False,
                "tool_name": tool_name,
                "data": None,
                "error": str(e),
            }
