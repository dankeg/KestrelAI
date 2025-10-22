"""
MCP Manager for KestrelAI
Centralized management of MCP servers and tools - production ready
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .mcp_config import get_mcp_config, MCPConfig, MCPServerConfig
from .mcp_tools import get_tool_registry, MCPToolRegistry
from .mcp_client import MCPClient, MCPResult

logger = logging.getLogger(__name__)


@dataclass
class MCPServerStatus:
    """Status information for an MCP server"""
    name: str
    connected: bool
    tools_available: List[str]
    last_error: Optional[str] = None
    connection_time: Optional[float] = None


class MCPManager:
    """Centralized manager for MCP servers and tools - production ready"""
    
    def __init__(self, config: Optional[MCPConfig] = None):
        self.config = config or get_mcp_config()
        self.tool_registry = get_tool_registry()
        self.clients: Dict[str, MCPClient] = {}
        self.server_status: Dict[str, MCPServerStatus] = {}
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize MCP manager and connect to all enabled servers"""
        try:
            logger.info("Initializing MCP Manager...")
            
            # Initialize tool registry
            self.tool_registry = get_tool_registry()
            
            # Create clients for each enabled server
            enabled_servers = self.config.get_enabled_servers()
            
            for server_name, server_config in enabled_servers.items():
                try:
                    # Create client for this server
                    client = MCPClient(server_config)
                    self.clients[server_name] = client
                    
                    # Initialize server status
                    self.server_status[server_name] = MCPServerStatus(
                        name=server_name,
                        connected=False,
                        tools_available=server_config.tools
                    )
                    
                    logger.info(f"Created MCP client for server: {server_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to create client for server {server_name}: {e}")
                    self.server_status[server_name] = MCPServerStatus(
                        name=server_name,
                        connected=False,
                        tools_available=server_config.tools,
                        last_error=str(e)
                    )
            
            # Connect to all servers
            connection_results = await self._connect_all_servers()
            
            self.is_initialized = True
            connected_count = sum(connection_results)
            logger.info(f"MCP Manager initialized successfully. Connected to {connected_count}/{len(connection_results)} servers.")
            
            return connected_count > 0
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP Manager: {e}")
            self.is_initialized = False
            return False
    
    async def _connect_all_servers(self) -> List[bool]:
        """Connect to all MCP servers"""
        connection_results = []
        
        for server_name, client in self.clients.items():
            try:
                logger.info(f"Connecting to MCP server: {server_name}")
                connected = await client.connect()
                
                # Update server status
                self.server_status[server_name].connected = connected
                if connected:
                    self.server_status[server_name].connection_time = asyncio.get_event_loop().time()
                    logger.info(f"Successfully connected to MCP server: {server_name}")
                else:
                    self.server_status[server_name].last_error = "Connection failed"
                    logger.warning(f"Failed to connect to MCP server: {server_name}")
                
                connection_results.append(connected)
                
            except Exception as e:
                logger.error(f"Error connecting to server {server_name}: {e}")
                self.server_status[server_name].last_error = str(e)
                connection_results.append(False)
        
        return connection_results
    
    async def cleanup(self):
        """Cleanup all MCP connections"""
        logger.info("Cleaning up MCP Manager...")
        
        for server_name, client in self.clients.items():
            try:
                await client.disconnect()
                self.server_status[server_name].connected = False
                logger.info(f"Disconnected from MCP server: {server_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from server {server_name}: {e}")
        
        self.clients.clear()
        self.is_initialized = False
        logger.info("MCP Manager cleanup completed")
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any], server_name: Optional[str] = None) -> MCPResult:
        """Call an MCP tool, optionally on a specific server"""
        try:
            # If server is specified, use that client
            if server_name:
                client = self.clients.get(server_name)
                if not client:
                    return MCPResult(
                        success=False,
                        data=None,
                        error=f"Server {server_name} not found",
                        tool_name=tool_name
                    )
                if not client.is_connected:
                    return MCPResult(
                        success=False,
                        data=None,
                        error=f"Server {server_name} is not connected",
                        tool_name=tool_name
                    )
                
                return await client.call_tool(tool_name, parameters)
            
            # Otherwise, try to find the tool on any connected server
            for server_name, client in self.clients.items():
                if client.is_connected:
                    # Check if this server has the tool
                    if tool_name in client.server_config.tools:
                        return await client.call_tool(tool_name, parameters)
            
            return MCPResult(
                success=False,
                data=None,
                error=f"Tool {tool_name} not found on any connected server",
                tool_name=tool_name
            )
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return MCPResult(
                success=False,
                data=None,
                error=str(e),
                tool_name=tool_name
            )
    
    def get_available_tools(self) -> List[str]:
        """Get all available tools from all connected servers"""
        tools = []
        for server_name, status in self.server_status.items():
            if status.connected:
                tools.extend(status.tools_available)
        return tools
    
    def get_server_status(self, server_name: str) -> Optional[MCPServerStatus]:
        """Get status for a specific server"""
        return self.server_status.get(server_name)
    
    def get_all_server_status(self) -> Dict[str, MCPServerStatus]:
        """Get status for all servers"""
        return self.server_status.copy()
    
    def is_server_connected(self, server_name: str) -> bool:
        """Check if a specific server is connected"""
        status = self.server_status.get(server_name)
        return status.connected if status else False
    
    def get_tools_by_server(self, server_name: str) -> List[str]:
        """Get tools available from a specific server"""
        status = self.server_status.get(server_name)
        return status.tools_available if status and status.connected else []
    
    def suggest_tools(self, context: str, task_description: str) -> List[str]:
        """Suggest relevant tools based on context and task description"""
        # Use the tool registry to suggest tools
        suggested_tools = self.tool_registry.suggest_tools(context, task_description)
        return [tool.name for tool in suggested_tools]
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        tool = self.tool_registry.get_tool(tool_name)
        if tool:
            return tool.to_dict()
        return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of MCP system"""
        total_servers = len(self.server_status)
        connected_servers = sum(1 for status in self.server_status.values() if status.connected)
        total_tools = len(self.get_available_tools())
        
        return {
            "initialized": self.is_initialized,
            "total_servers": total_servers,
            "connected_servers": connected_servers,
            "total_tools": total_tools,
            "servers": {
                name: {
                    "connected": status.connected,
                    "tools_count": len(status.tools_available),
                    "tools": status.tools_available,
                    "last_error": status.last_error,
                    "connection_time": status.connection_time
                }
                for name, status in self.server_status.items()
            }
        }
    
    async def restart_server(self, server_name: str) -> bool:
        """Restart a specific MCP server"""
        try:
            logger.info(f"Restarting MCP server: {server_name}")
            
            # Disconnect if connected
            if server_name in self.clients:
                await self.clients[server_name].disconnect()
            
            # Reconnect
            if server_name in self.clients:
                connected = await self.clients[server_name].connect()
                self.server_status[server_name].connected = connected
                
                if connected:
                    logger.info(f"Successfully restarted MCP server: {server_name}")
                else:
                    logger.warning(f"Failed to restart MCP server: {server_name}")
                
                return connected
            
            return False
            
        except Exception as e:
            logger.error(f"Error restarting server {server_name}: {e}")
            self.server_status[server_name].last_error = str(e)
            return False
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()


# Global MCP manager instance
_mcp_manager: Optional[MCPManager] = None


def get_mcp_manager() -> MCPManager:
    """Get the global MCP manager instance"""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPManager()
    return _mcp_manager


def set_mcp_manager(manager: MCPManager):
    """Set the global MCP manager instance"""
    global _mcp_manager
    _mcp_manager = manager


async def initialize_mcp_system() -> bool:
    """Initialize the global MCP system"""
    manager = get_mcp_manager()
    return await manager.initialize()


async def cleanup_mcp_system():
    """Cleanup the global MCP system"""
    manager = get_mcp_manager()
    await manager.cleanup()