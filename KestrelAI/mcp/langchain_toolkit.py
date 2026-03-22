"""
LangChain toolkit wrapper for MCP-managed tools.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, create_model

try:
    from langchain_core.tools import StructuredTool
except ImportError:  # pragma: no cover - dependency-gated path
    StructuredTool = None

logger = logging.getLogger(__name__)


class _GenericMCPToolInput(BaseModel):
    parameters: dict[str, Any] = Field(default_factory=dict)


def _annotation_for_json_type(type_name: str | None) -> Any:
    if type_name == "string":
        return str
    if type_name == "integer":
        return int
    if type_name == "number":
        return float
    if type_name == "boolean":
        return bool
    if type_name == "array":
        return list[Any]
    if type_name == "object":
        return dict[str, Any]
    return Any


def _sanitize_model_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "MCPTool"


class LangChainMCPToolkit:
    """Expose MCP registry tools as LangChain StructuredTools."""

    def __init__(
        self,
        *,
        mcp_manager: Any,
        mcp_enabled: Callable[[], bool] | None = None,
    ):
        if StructuredTool is None:
            raise ImportError("langchain_core is required for MCP toolkit")
        self.mcp_manager = mcp_manager
        self.mcp_enabled = mcp_enabled
        self._tools_by_executor_name: dict[str, str] = {}

    def build_tools(self) -> dict[str, Any]:
        tools: dict[str, Any] = {}
        for tool_name in self._discover_tool_names():
            tool_info = self._get_tool_info(tool_name)
            executor_name = self._executor_name(tool_name)
            args_schema = self._build_args_schema(tool_name, tool_info)
            description = self._tool_description(tool_name, tool_info)
            tools[executor_name] = StructuredTool.from_function(
                coroutine=self._make_tool_coro(tool_name),
                name=executor_name,
                description=description,
                args_schema=args_schema,
            )
            self._tools_by_executor_name[executor_name] = tool_name
        return tools

    def resolve_tool_name(self, logical_tool_name: str) -> str | None:
        """Map MCP logical tool name to executor tool key."""
        candidate = self._executor_name(logical_tool_name)
        return candidate if candidate in self._tools_by_executor_name else None

    async def execute_tool(
        self, logical_tool_name: str, tool_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute MCP tool through toolkit semantics."""
        return await self._invoke_tool(logical_tool_name, tool_parameters)

    def _discover_tool_names(self) -> list[str]:
        if hasattr(self.mcp_manager, "get_available_tools"):
            try:
                names = list(self.mcp_manager.get_available_tools())
                if names:
                    return sorted(set(str(name) for name in names))
            except Exception as e:
                logger.debug("Failed to get available MCP tools: %s", e)

        # Fallback to tool registry metadata if available.
        registry = getattr(self.mcp_manager, "tool_registry", None)
        if registry is not None and hasattr(registry, "list_tools"):
            try:
                listed = registry.list_tools()
                if listed:
                    return sorted(set(str(t.name) for t in listed))
            except Exception as e:
                logger.debug("Failed to list MCP registry tools: %s", e)
        return []

    def _get_tool_info(self, tool_name: str) -> dict[str, Any] | None:
        if hasattr(self.mcp_manager, "get_tool_info"):
            try:
                return self.mcp_manager.get_tool_info(tool_name)
            except Exception:
                return None
        return None

    def _tool_description(
        self, tool_name: str, tool_info: dict[str, Any] | None
    ) -> str:
        if tool_info and tool_info.get("description"):
            return str(tool_info["description"])
        return f"Execute MCP tool '{tool_name}'"

    def _executor_name(self, tool_name: str) -> str:
        return f"mcp_tool__{_sanitize_model_name(tool_name)}"

    def _build_args_schema(
        self, tool_name: str, tool_info: dict[str, Any] | None
    ) -> type[BaseModel]:
        if not tool_info:
            return _GenericMCPToolInput

        parameters = tool_info.get("parameters")
        if not isinstance(parameters, dict):
            return _GenericMCPToolInput

        properties = parameters.get("properties")
        if not isinstance(properties, dict) or not properties:
            return _GenericMCPToolInput

        required = set(parameters.get("required", []) or [])
        fields: dict[str, tuple[Any, Any]] = {}

        for prop_name, prop_schema in properties.items():
            if not isinstance(prop_schema, dict):
                fields[prop_name] = (Any, Field(default=None))
                continue

            annotation = _annotation_for_json_type(prop_schema.get("type"))
            description = prop_schema.get("description")

            if "default" in prop_schema:
                default = prop_schema["default"]
            elif prop_name in required:
                default = ...
            else:
                default = None

            fields[prop_name] = (
                annotation,
                Field(default=default, description=description),
            )

        if not fields:
            return _GenericMCPToolInput

        model_name = f"MCP_{_sanitize_model_name(tool_name)}_Input"
        return create_model(model_name, **fields)

    def _make_tool_coro(self, tool_name: str):
        async def _run_tool(**kwargs):
            return await self._invoke_tool(tool_name, kwargs)

        return _run_tool

    async def _invoke_tool(
        self, tool_name: str, tool_parameters: dict[str, Any]
    ) -> dict[str, Any]:
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
