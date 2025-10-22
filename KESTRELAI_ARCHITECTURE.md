# KestrelAI Architecture & Documentation

## Overview

KestrelAI is an autonomous research agent system that combines AI-powered research capabilities with a modern web interface and MCP (Model Context Protocol) integration for enhanced research tools.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        KestrelAI System                        │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React + TypeScript)                                 │
│  ├── kestrel-ui/src/KestrelAIApp.tsx                           │
│  ├── Theme System (Amber/Blue)                                 │
│  └── Real-time UI Updates                                      │
├─────────────────────────────────────────────────────────────────┤
│  Backend (FastAPI + Redis)                                     │
│  ├── KestrelAI/backend/main.py                                 │
│  ├── REST API Endpoints                                        │
│  ├── WebSocket Communication                                   │
│  └── Settings Management                                       │
├─────────────────────────────────────────────────────────────────┤
│  Core Engine (Python)                                          │
│  ├── KestrelAI/model_loop.py (KestrelAgentWorker)              │
│  ├── Research Orchestrator                                     │
│  ├── Web Research Agent                                        │
│  └── Memory Store (ChromaDB)                                   │
├─────────────────────────────────────────────────────────────────┤
│  MCP Integration (Optional)                                     │
│  ├── KestrelAI/mcp/mcp_manager.py                              │
│  ├── KestrelAI/mcp/mcp_client.py                               │
│  ├── External Tool Access                                      │
│  └── Enhanced Research Capabilities                             │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Frontend (`kestrel-ui/`)

**Technology Stack:**
- React 18 with TypeScript
- Vite for build tooling
- Tailwind CSS for styling
- Lucide React for icons
- React Markdown for content rendering

**Key Features:**
- **Task Management**: Create, configure, and monitor research tasks
- **Real-time Updates**: Live progress tracking and status updates
- **Theme System**: Switchable color schemes (Amber/Blue)
- **Settings Panel**: Configure research parameters and preferences
- **Responsive Design**: Works on desktop and mobile devices

**Main Components:**
- `KestrelAIApp.tsx`: Main application component
- `SettingsModal`: Theme and configuration management
- `TaskConfiguration`: Task setup and parameter configuration
- `ResearchPlanWidget`: Display research plans and progress
- `Activity`: Real-time activity feed
- `Search`: Search functionality and results

### 2. Backend (`KestrelAI/backend/`)

**Technology Stack:**
- FastAPI for REST API
- Pydantic for data validation
- Redis for task queue and caching
- WebSocket for real-time communication

**Key Features:**
- **REST API**: Full CRUD operations for tasks and settings
- **Real-time Communication**: WebSocket for live updates
- **Settings Management**: Persistent configuration storage
- **Task Queue**: Redis-based task processing
- **CORS Support**: Cross-origin request handling

**API Endpoints:**
- `GET/POST /settings`: Application settings management
- `GET /tasks`: List all tasks
- `POST /tasks`: Create new tasks
- `GET /tasks/{task_id}`: Get specific task details
- `PUT /tasks/{task_id}`: Update task
- `DELETE /tasks/{task_id}`: Delete task
- `POST /tasks/{task_id}/start`: Start task execution
- `POST /tasks/{task_id}/stop`: Stop task execution

### 3. Core Engine (`KestrelAI/model_loop.py`)

**Main Class: `KestrelAgentWorker`**

**Key Responsibilities:**
- **Task Processing**: Execute research tasks from Redis queue
- **Orchestrator Management**: Coordinate research workflow
- **Memory Management**: Store and retrieve research findings
- **Progress Tracking**: Monitor and report task progress
- **Error Handling**: Robust error recovery and logging

**Workflow:**
1. **Task Initialization**: Create research orchestrator for task
2. **Planning Phase**: Generate research plan with subtasks
3. **Execution Phase**: Execute subtasks with research agents
4. **Progress Monitoring**: Track and report progress
5. **Completion**: Store results and update task status

### 4. Research Agents (`KestrelAI/agents/`)

#### Research Orchestrator (`research_orchestrator.py`)
- **Purpose**: High-level coordination of research tasks
- **Features**: Task planning, subtask management, progress tracking
- **Integration**: Works with web research agents and MCP tools

#### Web Research Agent (`web_research_agent.py`)
- **Purpose**: Execute individual research subtasks
- **Features**: Web search, content analysis, result processing
- **Capabilities**: Multi-source research, fact-checking, summarization

#### Base Agent (`base_agent.py`)
- **Purpose**: Abstract base classes for all agents
- **Features**: Common functionality, state management, error handling
- **Classes**: `BaseAgent`, `ResearchAgent`, `OrchestratorAgent`

### 5. Memory System (`KestrelAI/memory/`)

**Technology Stack:**
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Persistent storage for research findings

**Key Features:**
- **Vector Search**: Semantic search across research findings
- **Document Storage**: Store research results and metadata
- **Similarity Matching**: Find related research content
- **Persistence**: Long-term storage of research data

### 6. MCP Integration (`KestrelAI/mcp/`)

**Model Context Protocol (MCP) Integration**

**Purpose**: Extend research capabilities with external tools and data sources

**Components:**
- **MCP Manager**: Centralized management of MCP servers
- **MCP Client**: Communication with MCP servers
- **Tool Registry**: Available tools and their configurations
- **Configuration**: Server and tool settings

**Available Tools:**
- **Data Sources**: Web search, databases, GitHub repositories, file systems
- **Analysis Tools**: Data analysis, text extraction, statistical processing
- **Automation Tools**: File operations, database management, web scraping

**Configuration:**
- JSON-based configuration (`mcp_config.json`)
- Environment variable support
- Server health monitoring
- Error handling and recovery

## Data Models

### Core Models (`KestrelAI/shared/models.py`)

#### Task
```python
class Task(BaseModel):
    id: str
    name: str
    description: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    research_plan: Optional[ResearchPlan]
    metrics: Optional[TaskMetrics]
```

#### ResearchPlan
```python
class ResearchPlan(BaseModel):
    restated_task: str
    subtasks: List[Subtask]
    current_subtask_index: int
    created_at: float
```

#### Subtask
```python
class Subtask(BaseModel):
    order: int
    description: str
    success_criteria: str
    status: str  # "pending", "in_progress", "completed"
    findings: Optional[List[str]]
```

## Configuration

### Application Settings
- **Ollama Mode**: Local or remote LLM configuration
- **Orchestrator**: Research orchestration strategy
- **Theme**: UI color scheme (Amber/Blue)
- **MCP Integration**: Enable/disable MCP tools

### Environment Variables
- `REDIS_HOST`: Redis server host
- `REDIS_PORT`: Redis server port
- `OLLAMA_BASE_URL`: Ollama server URL
- `SEARXNG_URL`: SearXNG search engine URL
- `KESTREL_MCP_CONFIG`: MCP configuration file path

## Development & Deployment

### Local Development
```bash
# Backend
cd /Users/ganeshdanke/Documents/KestrelAI
PYTHONPATH=/Users/ganeshdanke/Documents/KestrelAI python -m KestrelAI.backend.main

# Frontend
cd kestrel-ui
npm run dev

# Model Loop
python KestrelAI/model_loop.py
```

### Docker Deployment
```bash
# Full stack
docker compose up --build

# Individual services
docker compose up backend
docker compose up frontend
docker compose up agent
```

### Testing
```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# All tests
python -m pytest tests/ -v
```

## Key Features

### 1. Autonomous Research
- **Intelligent Planning**: AI-generated research plans with subtasks
- **Multi-source Research**: Web search, databases, repositories
- **Progress Tracking**: Real-time monitoring of research progress
- **Result Synthesis**: Automatic compilation and analysis of findings

### 2. Modern UI
- **Responsive Design**: Works on all device sizes
- **Theme System**: Customizable color schemes
- **Real-time Updates**: Live progress and status updates
- **Intuitive Interface**: Easy-to-use task management

### 3. Extensible Architecture
- **MCP Integration**: Extend capabilities with external tools
- **Plugin System**: Add new research agents and tools
- **API-First**: RESTful API for external integrations
- **Modular Design**: Easy to extend and customize

### 4. Robust Infrastructure
- **Redis Queue**: Reliable task processing
- **Vector Storage**: Semantic search and memory
- **Error Handling**: Comprehensive error recovery
- **Monitoring**: Health checks and logging

## File Structure

```
KestrelAI/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main API server
│   ├── models/             # Data models
│   └── routes/             # API routes
├── agents/                 # Research agents
│   ├── base_agent.py       # Base agent classes
│   ├── research_orchestrator.py  # Main orchestrator
│   ├── web_research_agent.py     # Web research agent
│   ├── base.py             # LLM wrapper
│   └── config.py           # Agent configuration
├── memory/                 # Memory system
│   └── vector_store.py     # ChromaDB integration
├── mcp/                    # MCP integration
│   ├── mcp_manager.py      # MCP server management
│   ├── mcp_client.py       # MCP client
│   ├── mcp_tools.py        # Tool registry
│   └── mcp_config.py       # Configuration
├── shared/                 # Shared utilities
│   ├── models.py           # Data models
│   └── redis_utils.py      # Redis utilities
└── model_loop.py           # Main worker process

kestrel-ui/                 # React frontend
├── src/
│   ├── KestrelAIApp.tsx    # Main app component
│   ├── index.css           # Theme system
│   └── assets/             # Static assets
└── package.json            # Dependencies

tests/                      # Test suite
├── unit/                   # Unit tests
├── integration/            # Integration tests
└── performance/            # Performance tests
```

## Usage Examples

### Basic Research Task
```python
# Create a research task
task = Task(
    id="research_ai_grants",
    name="AI Research Grants 2024",
    description="Find available AI research grants and funding opportunities",
    status=TaskStatus.PENDING
)

# Start the research
worker = KestrelAgentWorker()
await worker.process_task(task)
```

### MCP-Enhanced Research
```python
# Initialize MCP system
mcp_manager = MCPManager()
await mcp_manager.initialize()

# Create MCP-enhanced task
task = Task(
    id="github_analysis",
    name="GitHub Repository Analysis",
    description="Analyze popular AI repositories",
    status=TaskStatus.PENDING,
    use_mcp=True
)
```

### Frontend Integration
```typescript
// Create new task
const task = {
  name: "Research Task",
  description: "Task description",
  status: "pending"
};

const response = await fetch('/api/v1/tasks', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(task)
});
```

## Troubleshooting

### Common Issues

1. **Redis Connection Errors**
   - Ensure Redis server is running
   - Check Redis host/port configuration
   - Verify network connectivity

2. **MCP Server Issues**
   - Verify Node.js and npx installation
   - Check MCP server configurations
   - Review server health status

3. **Import Errors**
   - Check Python path configuration
   - Verify module installations
   - Review import statements

4. **Frontend Build Issues**
   - Clear node_modules and reinstall
   - Check Node.js version compatibility
   - Verify build configuration

### Debug Mode
```bash
# Enable debug logging
export KESTREL_LOG_LEVEL=DEBUG
python -m KestrelAI.backend.main
```

## Contributing

### Development Setup
1. Clone the repository
2. Install dependencies: `poetry install`
3. Set up environment variables
4. Run tests: `pytest tests/`
5. Start development servers

### Code Standards
- Follow PEP 8 for Python code
- Use TypeScript for frontend code
- Write comprehensive tests
- Document new features
- Follow existing patterns

## License

This project follows the license terms specified in the LICENSE file.

---

*This documentation reflects the current state of KestrelAI after cleanup and consolidation. For the most up-to-date information, refer to the source code and test files.*
