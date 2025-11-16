"""
MCP Configuration for KestrelAI
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server"""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""
    tools: list[str] = field(default_factory=list)  # Available tools from this server


@dataclass
class MCPConfig:
    """Main MCP configuration"""

    servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 3
    enable_logging: bool = True
    log_level: str = "INFO"

    def __post_init__(self):
        """Initialize with default servers if none provided"""
        if not self.servers:
            self.servers = self._get_default_servers()

    def _get_default_servers(self) -> dict[str, MCPServerConfig]:
        """Get default MCP server configurations"""
        return {
            "filesystem": MCPServerConfig(
                name="filesystem",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                description="File system access for reading/writing files",
                tools=["read_file", "write_file", "list_directory", "search_files"],
            ),
            "sqlite": MCPServerConfig(
                name="sqlite",
                command="npx",
                args=[
                    "-y",
                    "@modelcontextprotocol/server-sqlite",
                    "--db-path",
                    "/tmp/kestrel.db",
                ],
                description="SQLite database access for structured data queries",
                tools=["query_database", "create_table", "insert_data", "update_data"],
            ),
            "github": MCPServerConfig(
                name="github",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                description="GitHub API access for repository information",
                tools=[
                    "search_repositories",
                    "get_repository_info",
                    "get_file_contents",
                    "list_issues",
                ],
            ),
            "brave_search": MCPServerConfig(
                name="brave_search",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-brave-search"],
                description="Brave Search API for enhanced web search",
                tools=["search_web", "search_news", "search_images"],
            ),
            "puppeteer": MCPServerConfig(
                name="puppeteer",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-puppeteer"],
                description="Web scraping and browser automation",
                tools=[
                    "navigate_to_page",
                    "take_screenshot",
                    "extract_text",
                    "click_element",
                ],
            ),
        }

    @classmethod
    def from_file(cls, config_path: str) -> MCPConfig:
        """Load configuration from JSON file"""
        try:
            with open(config_path) as f:
                data = json.load(f)

            servers = {}
            for name, server_data in data.get("servers", {}).items():
                servers[name] = MCPServerConfig(**server_data)

            return cls(
                servers=servers,
                timeout=data.get("timeout", 30),
                max_retries=data.get("max_retries", 3),
                enable_logging=data.get("enable_logging", True),
                log_level=data.get("log_level", "INFO"),
            )
        except Exception as e:
            print(f"Failed to load MCP config from {config_path}: {e}")
            return cls()

    def to_file(self, config_path: str):
        """Save configuration to JSON file"""
        data = {
            "servers": {
                name: {
                    "name": server.name,
                    "command": server.command,
                    "args": server.args,
                    "env": server.env,
                    "enabled": server.enabled,
                    "description": server.description,
                    "tools": server.tools,
                }
                for name, server in self.servers.items()
            },
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
        }

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_enabled_servers(self) -> dict[str, MCPServerConfig]:
        """Get only enabled servers"""
        return {name: server for name, server in self.servers.items() if server.enabled}

    def get_available_tools(self) -> dict[str, str]:
        """Get all available tools from enabled servers"""
        tools = {}
        for server in self.get_enabled_servers().values():
            for tool in server.tools:
                tools[tool] = server.name
        return tools


# Global MCP configuration instance
_mcp_config: MCPConfig | None = None


def get_mcp_config() -> MCPConfig:
    """Get the global MCP configuration"""
    global _mcp_config
    if _mcp_config is None:
        config_path = os.getenv("KESTREL_MCP_CONFIG", "mcp_config.json")
        if os.path.exists(config_path):
            _mcp_config = MCPConfig.from_file(config_path)
        else:
            _mcp_config = MCPConfig()
    return _mcp_config


def set_mcp_config(config: MCPConfig):
    """Set the global MCP configuration"""
    global _mcp_config
    _mcp_config = config
