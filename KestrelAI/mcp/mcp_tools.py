from __future__ import annotations

"""
MCP Tool Registry and Tool Definitions for KestrelAI
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Categories of MCP tools"""

    DATA_SOURCE = "data_source"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    AUTOMATION = "automation"
    RESEARCH = "research"


@dataclass
class MCPTool:
    """Definition of an MCP tool"""

    name: str
    description: str
    category: ToolCategory
    parameters: dict[str, Any]
    server: str
    enabled: bool = True
    priority: int = 0  # Higher priority tools are preferred

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": self.parameters,
            "server": self.server,
            "enabled": self.enabled,
            "priority": self.priority,
        }


class MCPToolRegistry:
    """Registry for MCP tools available to research agents"""

    def __init__(self):
        self.tools: dict[str, MCPTool] = {}
        self._initialize_default_tools()

    def _initialize_default_tools(self):
        """Initialize with default tool definitions"""
        default_tools = [
            # Data Source Tools
            MCPTool(
                name="search_web",
                description="Enhanced web search with multiple engines and filtering",
                category=ToolCategory.DATA_SOURCE,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "engine": {
                            "type": "string",
                            "enum": ["brave", "google", "bing"],
                            "default": "brave",
                        },
                        "num_results": {
                            "type": "integer",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "time_range": {
                            "type": "string",
                            "enum": ["day", "week", "month", "year", "all"],
                            "default": "all",
                        },
                        "content_type": {
                            "type": "string",
                            "enum": ["web", "news", "images", "videos"],
                            "default": "web",
                        },
                    },
                    "required": ["query"],
                },
                server="brave_search",
                priority=10,
            ),
            MCPTool(
                name="query_database",
                description="Execute SQL queries on structured databases",
                category=ToolCategory.DATA_SOURCE,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute",
                        },
                        "database": {
                            "type": "string",
                            "description": "Database name or path",
                            "default": "default",
                        },
                    },
                    "required": ["query"],
                },
                server="sqlite",
                priority=8,
            ),
            MCPTool(
                name="search_repositories",
                description="Search GitHub repositories for code and documentation",
                category=ToolCategory.DATA_SOURCE,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "language": {
                            "type": "string",
                            "description": "Programming language filter",
                        },
                        "sort": {
                            "type": "string",
                            "enum": ["stars", "forks", "updated"],
                            "default": "stars",
                        },
                        "per_page": {
                            "type": "integer",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100,
                        },
                    },
                    "required": ["query"],
                },
                server="github",
                priority=7,
            ),
            MCPTool(
                name="read_file",
                description="Read contents of files from the filesystem",
                category=ToolCategory.DATA_SOURCE,
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"},
                        "encoding": {"type": "string", "default": "utf-8"},
                    },
                    "required": ["path"],
                },
                server="filesystem",
                priority=6,
            ),
            MCPTool(
                name="search_files",
                description="Search for files by name or content",
                category=ToolCategory.DATA_SOURCE,
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "File name pattern or content to search",
                        },
                        "directory": {
                            "type": "string",
                            "description": "Directory to search in",
                            "default": ".",
                        },
                        "recursive": {"type": "boolean", "default": True},
                        "file_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File extensions to include",
                        },
                    },
                    "required": ["pattern"],
                },
                server="filesystem",
                priority=5,
            ),
            # Analysis Tools
            MCPTool(
                name="analyze_data",
                description="Perform statistical analysis on data",
                category=ToolCategory.ANALYSIS,
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "Data to analyze (JSON, CSV, or text)",
                        },
                        "analysis_type": {
                            "type": "string",
                            "enum": ["summary", "correlation", "trend", "outlier"],
                            "default": "summary",
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific columns to analyze",
                        },
                    },
                    "required": ["data"],
                },
                server="analysis",
                priority=9,
            ),
            MCPTool(
                name="extract_text",
                description="Extract and clean text from web pages or documents",
                category=ToolCategory.ANALYSIS,
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to extract text from",
                        },
                        "selector": {
                            "type": "string",
                            "description": "CSS selector for specific content",
                        },
                        "clean_html": {"type": "boolean", "default": True},
                        "max_length": {
                            "type": "integer",
                            "default": 10000,
                            "description": "Maximum text length",
                        },
                    },
                    "required": ["url"],
                },
                server="puppeteer",
                priority=8,
            ),
            # Research Tools
            MCPTool(
                name="get_repository_info",
                description="Get detailed information about a GitHub repository",
                category=ToolCategory.RESEARCH,
                parameters={
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "description": "Repository owner"},
                        "repo": {"type": "string", "description": "Repository name"},
                        "include_readme": {"type": "boolean", "default": True},
                        "include_issues": {"type": "boolean", "default": False},
                    },
                    "required": ["owner", "repo"],
                },
                server="github",
                priority=7,
            ),
            MCPTool(
                name="navigate_to_page",
                description="Navigate to a web page and extract information",
                category=ToolCategory.RESEARCH,
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to navigate to"},
                        "wait_for": {
                            "type": "string",
                            "description": "CSS selector to wait for",
                        },
                        "screenshot": {"type": "boolean", "default": False},
                        "extract_links": {"type": "boolean", "default": True},
                    },
                    "required": ["url"],
                },
                server="puppeteer",
                priority=6,
            ),
            # Automation Tools
            MCPTool(
                name="write_file",
                description="Write data to files",
                category=ToolCategory.AUTOMATION,
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to write"},
                        "content": {
                            "type": "string",
                            "description": "Content to write",
                        },
                        "encoding": {"type": "string", "default": "utf-8"},
                        "create_dirs": {"type": "boolean", "default": True},
                    },
                    "required": ["path", "content"],
                },
                server="filesystem",
                priority=5,
            ),
            MCPTool(
                name="create_table",
                description="Create database tables for structured data storage",
                category=ToolCategory.AUTOMATION,
                parameters={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table",
                        },
                        "columns": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Column definitions",
                        },
                        "database": {"type": "string", "default": "default"},
                    },
                    "required": ["table_name", "columns"],
                },
                server="sqlite",
                priority=4,
            ),
        ]

        for tool in default_tools:
            self.register_tool(tool)

    def register_tool(self, tool: MCPTool):
        """Register a new MCP tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered MCP tool: {tool.name} from server {tool.server}")

    def get_tool(self, name: str) -> MCPTool | None:
        """Get a tool by name"""
        return self.tools.get(name)

    def get_tools_by_category(self, category: ToolCategory) -> list[MCPTool]:
        """Get all tools in a specific category"""
        return [
            tool
            for tool in self.tools.values()
            if tool.category == category and tool.enabled
        ]

    def get_tools_by_server(self, server: str) -> list[MCPTool]:
        """Get all tools from a specific server"""
        return [
            tool
            for tool in self.tools.values()
            if tool.server == server and tool.enabled
        ]

    def get_enabled_tools(self) -> list[MCPTool]:
        """Get all enabled tools sorted by priority"""
        return sorted(
            [tool for tool in self.tools.values() if tool.enabled],
            key=lambda t: t.priority,
            reverse=True,
        )

    def suggest_tools(self, context: str, task_description: str) -> list[MCPTool]:
        """Suggest relevant tools based on context and task description"""
        # Simple keyword-based tool suggestion
        context_lower = context.lower()
        task_lower = task_description.lower()

        suggested = []

        # Data source suggestions
        if any(
            keyword in context_lower + task_lower
            for keyword in ["search", "find", "lookup", "web"]
        ):
            suggested.extend(self.get_tools_by_category(ToolCategory.DATA_SOURCE))

        if any(
            keyword in context_lower + task_lower
            for keyword in ["database", "sql", "query", "data"]
        ):
            suggested.extend(
                [
                    t
                    for t in self.get_tools_by_category(ToolCategory.DATA_SOURCE)
                    if "database" in t.name or "sql" in t.name
                ]
            )

        if any(
            keyword in context_lower + task_lower
            for keyword in ["github", "repository", "code", "repo"]
        ):
            suggested.extend(self.get_tools_by_server("github"))

        if any(
            keyword in context_lower + task_lower
            for keyword in ["file", "document", "read", "write"]
        ):
            suggested.extend(self.get_tools_by_server("filesystem"))

        # Analysis suggestions
        if any(
            keyword in context_lower + task_lower
            for keyword in ["analyze", "analysis", "statistics", "data"]
        ):
            suggested.extend(self.get_tools_by_category(ToolCategory.ANALYSIS))

        # Research suggestions
        if any(
            keyword in context_lower + task_lower
            for keyword in ["research", "investigate", "explore"]
        ):
            suggested.extend(self.get_tools_by_category(ToolCategory.RESEARCH))

        # Remove duplicates and sort by priority
        seen = set()
        unique_suggested = []
        for tool in suggested:
            if tool.name not in seen:
                seen.add(tool.name)
                unique_suggested.append(tool)

        return sorted(unique_suggested, key=lambda t: t.priority, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert registry to dictionary"""
        return {"tools": {name: tool.to_dict() for name, tool in self.tools.items()}}

    def from_dict(self, data: dict[str, Any]):
        """Load registry from dictionary"""
        self.tools.clear()
        for name, tool_data in data.get("tools", {}).items():
            tool = MCPTool(
                name=tool_data["name"],
                description=tool_data["description"],
                category=ToolCategory(tool_data["category"]),
                parameters=tool_data["parameters"],
                server=tool_data["server"],
                enabled=tool_data.get("enabled", True),
                priority=tool_data.get("priority", 0),
            )
            self.register_tool(tool)


# Global tool registry instance
_tool_registry: MCPToolRegistry | None = None


def get_tool_registry() -> MCPToolRegistry:
    """Get the global tool registry"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = MCPToolRegistry()
    return _tool_registry
