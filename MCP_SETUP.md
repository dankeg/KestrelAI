# MCP Server Setup Guide for KestrelAI

This guide explains how to set up MCP (Model Context Protocol) servers for KestrelAI's production-ready research capabilities.

## Prerequisites

### 1. Install Node.js and npm
```bash
# Install Node.js (version 18 or higher)
# Visit https://nodejs.org/ or use your package manager

# Verify installation
node --version
npm --version
```

### 2. Install npx (comes with npm)
```bash
# npx should be available with npm
npx --version
```

## MCP Server Installation

### 1. Filesystem Server
```bash
# Install filesystem MCP server
npm install -g @modelcontextprotocol/server-filesystem

# Or use npx (recommended)
npx @modelcontextprotocol/server-filesystem --help
```

### 2. SQLite Server
```bash
# Install SQLite MCP server
npm install -g @modelcontextprotocol/server-sqlite

# Or use npx (recommended)
npx @modelcontextprotocol/server-sqlite --help
```

### 3. GitHub Server
```bash
# Install GitHub MCP server
npm install -g @modelcontextprotocol/server-github

# Or use npx (recommended)
npx @modelcontextprotocol/server-github --help
```

### 4. Brave Search Server
```bash
# Install Brave Search MCP server
npm install -g @modelcontextprotocol/server-brave-search

# Or use npx (recommended)
npx @modelcontextprotocol/server-brave-search --help
```

### 5. Puppeteer Server
```bash
# Install Puppeteer MCP server
npm install -g @modelcontextprotocol/server-puppeteer

# Or use npx (recommended)
npx @modelcontextprotocol/server-puppeteer --help
```

## Configuration

### 1. Environment Variables
Set up required environment variables for MCP servers:

```bash
# GitHub server requires GitHub token
export GITHUB_TOKEN="your_github_token_here"

# Brave Search server requires API key
export BRAVE_API_KEY="your_brave_api_key_here"

# Optional: Set custom paths
export MCP_FILESYSTEM_PATH="/path/to/your/files"
export MCP_SQLITE_DB_PATH="/path/to/your/database.db"
```

### 2. MCP Configuration File
The system uses `mcp_config.json` in the project root. Update it with your specific requirements:

```json
{
  "servers": {
    "filesystem": {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/your/path"],
      "env": {},
      "enabled": true,
      "description": "File system access",
      "tools": ["read_file", "write_file", "list_directory", "search_files"]
    },
    "sqlite": {
      "name": "sqlite",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "/path/to/db.db"],
      "env": {},
      "enabled": true,
      "description": "SQLite database access",
      "tools": ["query_database", "create_table", "insert_data", "update_data"]
    }
  },
  "timeout": 30,
  "max_retries": 3,
  "enable_logging": true,
  "log_level": "INFO"
}
```

## Testing MCP Setup

### 1. Test Individual Servers
```bash
# Test filesystem server
npx @modelcontextprotocol/server-filesystem /tmp

# Test SQLite server
npx @modelcontextprotocol/server-sqlite --db-path /tmp/test.db

# Test GitHub server (requires GITHUB_TOKEN)
npx @modelcontextprotocol/server-github
```

### 2. Test KestrelAI MCP Integration
```bash
# Run the MCP research example
cd /path/to/KestrelAI
python examples/mcp_research_example.py
```

## Troubleshooting

### Common Issues

1. **npx not found**
   ```bash
   # Install Node.js and npm
   # npx comes with npm
   ```

2. **MCP server not found**
   ```bash
   # Install the specific MCP server
   npm install -g @modelcontextprotocol/server-<name>
   ```

3. **Permission errors**
   ```bash
   # Use npx instead of global install
   npx @modelcontextprotocol/server-<name>
   ```

4. **API key errors**
   ```bash
   # Set required environment variables
   export GITHUB_TOKEN="your_token"
   export BRAVE_API_KEY="your_key"
   ```

### Debug Mode
Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run your MCP-enabled research
```

## Production Deployment

### 1. Server Requirements
- Node.js 18+ installed
- Required MCP servers installed
- Environment variables configured
- Proper file permissions

### 2. Docker Deployment
```dockerfile
# Add to your Dockerfile
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs
RUN npm install -g @modelcontextprotocol/server-filesystem
RUN npm install -g @modelcontextprotocol/server-sqlite
RUN npm install -g @modelcontextprotocol/server-github
```

### 3. Health Checks
```python
# Check MCP system health
from KestrelAI.mcp import get_mcp_manager

async def check_mcp_health():
    manager = get_mcp_manager()
    await manager.initialize()
    health = manager.get_health_status()
    print(f"Connected servers: {health['connected_servers']}/{health['total_servers']}")
    await manager.cleanup()
```

## Security Considerations

1. **API Keys**: Store API keys securely, never commit to version control
2. **File Permissions**: Limit filesystem access to necessary directories
3. **Network Access**: Restrict network access for MCP servers
4. **Process Isolation**: Run MCP servers in isolated environments

## Support

For MCP server issues:
- Check the official MCP documentation
- Verify server installation and configuration
- Check environment variables and permissions
- Review KestrelAI logs for specific error messages




