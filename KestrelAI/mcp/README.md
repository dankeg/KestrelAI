# MCP (Model Context Protocol) Integration for KestrelAI

This module provides comprehensive MCP integration for KestrelAI research agents, enabling them to access powerful external tools and data sources during research tasks.

## Overview

The MCP integration extends KestrelAI's research capabilities by providing access to:

- **Enhanced Data Sources**: Web search, databases, GitHub repositories, file systems
- **Analysis Tools**: Data analysis, text extraction, statistical processing
- **Automation Tools**: File operations, database management, web scraping
- **Specialized Research Tools**: Repository analysis, content extraction, structured data queries

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Research      │◄──►│  MCP Manager     │◄──►│   MCP Servers   │
│   Agents        │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Orchestrator   │    │  Tool Registry   │    │  Tool Clients   │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Production MCP  │    │  Real MCP        │    │  JSON-RPC       │
│ Manager         │    │  Protocol        │    │  Communication  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Production MCP Integration

The MCP integration uses **only real MCP protocol** - no simulations or mocks:

### Real MCP Protocol Implementation
- **JSON-RPC Communication**: Uses actual MCP protocol for server communication
- **Subprocess Management**: Spawns real MCP server processes (npx commands)
- **Tool Discovery**: Dynamically discovers available tools from servers
- **Error Handling**: Proper error handling and timeout management
- **Connection Validation**: Validates server availability before attempting connections

### Production Requirements
- **MCP Servers Required**: System requires actual MCP servers to be installed
- **Command Availability**: Checks for required commands (npx, etc.) before connecting
- **Real Tool Execution**: All tool calls use actual MCP protocol
- **No Fallbacks**: System fails gracefully if MCP servers are unavailable

### Why Production-Only?

1. **Real Capabilities**: Only provides actual MCP functionality
2. **No Confusion**: Users know exactly what capabilities they have
3. **Production Ready**: Designed for real-world usage
4. **Clear Requirements**: Explicit about MCP server requirements
5. **Reliable**: No hidden fallbacks or unexpected behavior

## Components

### 1. MCP Manager (`mcp_manager.py`)
Centralized management of MCP servers, tools, and client connections.

**Key Features:**
- Server connection management
- Tool discovery and registration
- Health monitoring
- Error handling and recovery

### 2. MCP Client (`mcp_client.py`)
Client for interacting with MCP servers and tools.

**Key Features:**
- Tool execution
- Parameter validation
- Result processing
- Error handling

### 3. Tool Registry (`mcp_tools.py`)
Registry and definitions for available MCP tools.

**Key Features:**
- Tool categorization
- Parameter schemas
- Tool suggestions
- Priority management

### 4. Configuration (`mcp_config.py`)
Configuration management for MCP servers and tools.

**Key Features:**
- Server configurations
- Tool definitions
- Environment settings
- JSON-based configuration

## Available Tools

### Data Sources
- **search_web**: Enhanced web search with multiple engines
- **query_database**: SQL queries on structured databases
- **search_repositories**: GitHub repository search
- **read_file**: File system access
- **search_files**: File content search

### Analysis Tools
- **analyze_data**: Statistical data analysis
- **extract_text**: Web page text extraction

### Research Tools
- **get_repository_info**: GitHub repository details
- **navigate_to_page**: Web page navigation and extraction

### Automation Tools
- **write_file**: File writing operations
- **create_table**: Database table creation

## Usage

### Basic Usage

```python
from KestrelAI.mcp.mcp_manager import get_mcp_manager
from KestrelAI.agents.consolidated_orchestrator import ConsolidatedOrchestrator

# Initialize MCP system
mcp_manager = get_mcp_manager()
await mcp_manager.initialize()

# Create MCP-enhanced orchestrator
async with MCPOrchestrator([task], llm, profile="kestrel") as orchestrator:
    # Run research with MCP tools
    result = await orchestrator.next_action(task)
```

### Advanced Usage

```python
from KestrelAI.agents.consolidated_research_agent import ConsolidatedResearchAgent, ResearchConfig

# Create MCP-enhanced research agent
config = ResearchConfig(use_mcp=True, mcp_manager=mcp_manager)
agent = ConsolidatedResearchAgent("mcp-agent", llm, memory, config)
    # Run research step with MCP tools
    result = await agent.run_step(task)
```

### Tool Execution

```python
# Call MCP tools directly
result = await mcp_client.call_tool("search_web", {
    "query": "AI research grants",
    "num_results": 10,
    "engine": "brave"
})
```

## Configuration

### MCP Configuration File (`mcp_config.json`)

```json
{
  "servers": {
    "filesystem": {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "enabled": true,
      "description": "File system access",
      "tools": ["read_file", "write_file", "list_directory"]
    }
  },
  "timeout": 30,
  "max_retries": 3,
  "enable_logging": true,
  "log_level": "INFO"
}
```

### Environment Variables

- `KESTREL_MCP_CONFIG`: Path to MCP configuration file
- `SEARXNG_URL`: SearXNG search engine URL
- `OLLAMA_BASE_URL`: Ollama server URL

## Enhanced Research Agents

### MCP Research Agent
Extends the base research agent with MCP tool capabilities:

- **Enhanced Planning**: MCP tool suggestions during planning
- **Tool Execution**: Direct MCP tool calls during research
- **Data Analysis**: Integration with analysis tools
- **Result Tracking**: Comprehensive tracking of tool usage

### MCP Subtask Agent
Specialized agent for focused subtask research:

- **Tool-Specific Research**: Targeted tool usage for subtasks
- **Progress Tracking**: Enhanced progress monitoring
- **Completion Detection**: Intelligent subtask completion

### MCP Orchestrator
Enhanced orchestrator with MCP awareness:

- **Tool-Aware Planning**: MCP tool integration in planning phase
- **Resource Management**: Efficient tool usage across subtasks
- **Data Organization**: Structured data storage strategies

## Benefits

### Enhanced Research Capabilities
- **Multiple Data Sources**: Access to web, databases, repositories, files
- **Specialized Tools**: Domain-specific research tools
- **Data Analysis**: Built-in analysis and processing capabilities
- **Automation**: Automated data collection and organization

### Improved Research Quality
- **Comprehensive Coverage**: Multiple angles and data sources
- **Structured Data**: Organized data storage and retrieval
- **Cross-Reference**: Data validation across sources
- **Deep Analysis**: Statistical and analytical processing

### Scalability and Flexibility
- **Modular Design**: Easy to add new tools and servers
- **Configuration-Driven**: Flexible server and tool configuration
- **Error Handling**: Robust error handling and recovery
- **Monitoring**: Comprehensive health monitoring

## Examples

See the `examples/mcp_research_example.py` file for a complete working example of MCP-enhanced research.

## Dependencies

- `mcp`: Model Context Protocol client library
- `asyncio`: Asynchronous programming support
- `pydantic`: Data validation and serialization
- `requests`: HTTP client for web requests

## Installation

```bash
# Install with MCP support
poetry install --extras "agent"

# Or install all dependencies
poetry install --extras "all"
```

## Troubleshooting

### Common Issues

1. **MCP Servers Not Connecting**
   - Check server configurations in `mcp_config.json`
   - Verify server commands and arguments
   - Check network connectivity

2. **Tool Execution Failures**
   - Verify tool parameters match schema
   - Check server health status
   - Review error logs

3. **Performance Issues**
   - Adjust timeout settings
   - Limit concurrent tool calls
   - Monitor resource usage

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check MCP system health:

```python
health = await mcp_manager.get_health_status()
print(health)
```

## Future Enhancements

- **Custom Tool Development**: Framework for custom MCP tools
- **Tool Chaining**: Automatic tool chaining and workflows
- **Performance Optimization**: Caching and optimization strategies
- **Advanced Analytics**: Research analytics and insights
- **Integration APIs**: REST APIs for external integrations

## Contributing

When adding new MCP tools or servers:

1. Define tool schema in `mcp_tools.py`
2. Add server configuration in `mcp_config.py`
3. Implement tool execution in `mcp_client.py`
4. Update documentation and examples
5. Add tests for new functionality

## License

This MCP integration is part of the KestrelAI project and follows the same license terms.
