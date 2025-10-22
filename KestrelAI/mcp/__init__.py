"""
MCP (Model Context Protocol) integration for KestrelAI research agents
Production-ready implementation with real MCP protocol only
"""

from .mcp_client import MCPResult
from .mcp_tools import MCPToolRegistry, MCPTool
from .mcp_config import MCPConfig, MCPServerConfig
from .mcp_manager import MCPManager

__all__ = ["MCPManager", "MCPResult", "MCPToolRegistry", "MCPTool", "MCPConfig", "MCPServerConfig"]
