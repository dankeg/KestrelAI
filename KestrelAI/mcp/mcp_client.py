"""
Production MCP Client for KestrelAI
Uses only real MCP protocol - no simulations or mocks
"""

import asyncio
import json
import logging
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import os

from .mcp_config import MCPServerConfig
from .mcp_tools import MCPTool

logger = logging.getLogger(__name__)


@dataclass
class MCPResult:
    """Result from an MCP tool call"""
    success: bool
    data: Any
    error: Optional[str] = None
    tool_name: Optional[str] = None
    server: Optional[str] = None
    execution_time: Optional[float] = None


class MCPClient:
    """Production MCP client that uses only real MCP protocol"""
    
    def __init__(self, server_config: MCPServerConfig):
        self.server_config = server_config
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.is_connected = False
        
    async def connect(self) -> bool:
        """Connect to MCP server using real protocol"""
        try:
            # Check if the command exists
            if not self._command_exists(self.server_config.command):
                logger.error(f"Command {self.server_config.command} not found")
                return False
            
            # Start MCP server process
            cmd = [self.server_config.command] + self.server_config.args
            env = {**os.environ, **self.server_config.env}
            
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=0
            )
            
            # Send initialization request
            init_request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "KestrelAI",
                        "version": "1.0.0"
                    }
                }
            }
            
            await self._send_request(init_request)
            
            # Wait for initialization response with timeout
            response = await asyncio.wait_for(self._read_response(), timeout=10.0)
            
            if response and "result" in response:
                self.is_connected = True
                logger.info(f"Successfully connected to MCP server: {self.server_config.name}")
                return True
            else:
                logger.error(f"Failed to initialize MCP server: {self.server_config.name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.server_config.name}: {e}")
            return False
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in the system"""
        try:
            if command == "npx":
                # Check if npx is available
                result = subprocess.run(["which", "npx"], capture_output=True, text=True)
                return result.returncode == 0
            else:
                # For other commands, try to run with --help
                result = subprocess.run([command, "--help"], capture_output=True, text=True)
                return result.returncode == 0
        except Exception:
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.process:
            try:
                # Send shutdown request
                shutdown_request = {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "notifications/shutdown"
                }
                await self._send_request(shutdown_request)
                
                # Terminate process
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5)
                
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
                if self.process:
                    self.process.kill()
            
            self.process = None
        
        self.is_connected = False
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> MCPResult:
        """Call an MCP tool with given parameters"""
        start_time = time.time()
        
        try:
            # Create tool call request
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                }
            }
            
            # Send request and wait for response
            response = await self._send_request_and_wait(request)
            
            execution_time = time.time() - start_time
            
            if response and "result" in response:
                return MCPResult(
                    success=True,
                    data=response["result"],
                    tool_name=tool_name,
                    server=self.server_config.name,
                    execution_time=execution_time
                )
            elif response and "error" in response:
                return MCPResult(
                    success=False,
                    data=None,
                    error=response["error"].get("message", "Unknown error"),
                    tool_name=tool_name,
                    server=self.server_config.name,
                    execution_time=execution_time
                )
            else:
                return MCPResult(
                    success=False,
                    data=None,
                    error="No response from server",
                    tool_name=tool_name,
                    server=self.server_config.name,
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error calling tool {tool_name}: {e}")
            return MCPResult(
                success=False,
                data=None,
                error=str(e),
                tool_name=tool_name,
                server=self.server_config.name,
                execution_time=execution_time
            )
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server"""
        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list"
            }
            
            response = await self._send_request_and_wait(request)
            
            if response and "result" in response:
                return response["result"].get("tools", [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []
    
    def _next_id(self) -> str:
        """Generate next request ID"""
        self.request_id += 1
        return str(self.request_id)
    
    async def _send_request(self, request: Dict[str, Any]):
        """Send JSON-RPC request to server"""
        if not self.process or not self.process.stdin:
            raise Exception("Server not connected")
        
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json)
        self.process.stdin.flush()
    
    async def _send_request_and_wait(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send request and wait for response"""
        request_id = request["id"]
        
        # Create future for this request
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        try:
            # Send request
            await self._send_request(request)
            
            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out")
            return None
        except Exception as e:
            logger.error(f"Error in request {request_id}: {e}")
            return None
        finally:
            # Clean up pending request
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
    
    async def _read_response(self) -> Optional[Dict[str, Any]]:
        """Read response from server"""
        if not self.process or not self.process.stdout:
            return None
        
        try:
            # Read line from stdout
            line = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.process.stdout.readline
                ), timeout=30.0
            )
            
            if line:
                response = json.loads(line.strip())
                
                # Handle response
                if "id" in response:
                    request_id = str(response["id"])
                    if request_id in self.pending_requests:
                        future = self.pending_requests[request_id]
                        if not future.done():
                            future.set_result(response)
                
                return response
            
            return None
            
        except Exception as e:
            logger.error(f"Error reading response: {e}")
            return None

