# KestrelAI User Guide - MCP-Enhanced Research

This guide explains how to interact with KestrelAI's MCP-enhanced research system as a user.

## 🚀 Getting Started

### 1. Prerequisites
Before using KestrelAI with MCP capabilities, ensure you have:

- **Node.js 18+** installed
- **MCP servers** installed (see [MCP_SETUP.md](MCP_SETUP.md))
- **API keys** for external services (GitHub, Brave Search, etc.)
- **KestrelAI** running (Docker or local installation)

### 2. Launch KestrelAI
```bash
# Using Docker (recommended)
docker compose up --build

# Or local installation
cd KestrelAI
poetry install --extras "all"
python -m KestrelAI.backend.main
```

Navigate to `http://localhost:5173` in your browser.

## 🎯 User Interaction Flow

### **Step 1: Create a Research Task**

#### **Option A: Use Templates**
1. Click **"New Task"** in the sidebar
2. Select from pre-built templates:
   - **🎓 ML Fellowships**: Find AI/ML funding opportunities
   - **📚 AI Conferences**: Find conferences accepting submissions
   - **🏆 ML Competitions**: Find student competitions

#### **Option B: Custom Task**
1. Click **"New Task"** in the sidebar
2. Fill in the task configuration:
   - **Task Name**: Descriptive name for your research
   - **Research Objective**: Detailed description of what you want to research
   - **Time Budget**: How long to spend on research (30-360 minutes)

### **Step 2: Configure Research Parameters**

#### **Orchestrator Selection**
Choose your research style:
- **🦅 Kestrel** (Default): Balanced exploration with medium-depth insights
- **⚡ Hummingbird**: Fast, focused research with minimal exploration
- **🌊 Albatross**: Long-horizon research with deep dives and cross-topic synthesis

#### **MCP Tool Integration**
The system automatically:
- **Detects available MCP servers** on startup
- **Suggests relevant tools** based on your research objective
- **Plans tool usage** during the planning phase
- **Executes tools** during research execution

### **Step 3: Start Research**

1. Click **"Start Research"** button
2. The system will:
   - **Initialize MCP servers** (if available)
   - **Generate research plan** with MCP tool recommendations
   - **Begin autonomous research** using available tools

### **Step 4: Monitor Progress**

#### **Real-Time Dashboard**
- **Progress Bar**: Overall task completion percentage
- **Live Activity Feed**: Real-time research actions
- **Search Intelligence**: Web searches and data collection
- **System Metrics**: LLM calls, searches, analysis performed

#### **MCP Tool Usage**
- **Tools Used**: Which MCP tools are being utilized
- **Data Sources**: What data sources are being accessed
- **Analysis Results**: Results from data analysis tools

### **Step 5: Review Results**

#### **Research Reports**
- **Checkpoint Reports**: Periodic progress summaries
- **Final Report**: Comprehensive research findings
- **Export Options**: JSON, Markdown, PDF formats

#### **MCP-Enhanced Results**
Reports include:
- **Multi-source data**: Web, databases, repositories, files
- **Structured analysis**: Statistical analysis and insights
- **Cross-referenced findings**: Data validation across sources
- **Tool-specific insights**: Results from specialized tools

## 🔧 MCP Tool Integration

### **Automatic Tool Selection**
The system automatically selects and uses MCP tools based on your research objective:

#### **Data Sources**
- **Web Search**: Enhanced search with multiple engines
- **Database Queries**: Structured data from SQLite databases
- **GitHub Repositories**: Code and documentation analysis
- **File Systems**: Local file access and search

#### **Analysis Tools**
- **Data Analysis**: Statistical analysis of collected data
- **Text Extraction**: Web page content extraction
- **Content Analysis**: Document and code analysis

#### **Research Tools**
- **Repository Analysis**: GitHub repository deep dives
- **Web Scraping**: Automated web page navigation
- **Data Storage**: Organized data storage and retrieval

### **Tool Usage Examples**

#### **Example 1: AI Research Grants**
```
User Input: "Find AI/ML research grants for undergraduate students"

System Uses:
- search_web: "AI research grants undergraduate 2025"
- search_repositories: "AI funding opportunities"
- query_database: Store and cross-reference findings
- analyze_data: Statistical analysis of grant data
```

#### **Example 2: Conference Research**
```
User Input: "Find AI conferences accepting submissions"

System Uses:
- search_web: "AI conferences 2025 call for papers"
- get_repository_info: Conference organization repositories
- extract_text: Extract submission deadlines and requirements
- write_file: Store organized conference data
```

## 📊 User Interface Features

### **Task Management**
- **Create Tasks**: Template-based or custom task creation
- **Monitor Progress**: Real-time progress tracking
- **Pause/Resume**: Control research execution
- **Export Results**: Multiple export formats

### **Settings Panel**
- **Ollama Runtime**: Local vs Docker configuration
- **Orchestrator Profile**: Research style selection
- **MCP Status**: Server connection status and available tools

### **Dashboard Views**
- **Task Overview**: High-level progress and metrics
- **Activity Feed**: Real-time research actions
- **Search History**: Web searches and data collection
- **Reports**: Research findings and analysis

## 🎮 Interactive Examples

### **Example 1: Research AI Fellowships**

1. **Create Task**:
   - Name: "AI Fellowships Research"
   - Description: "Find currently open AI/ML fellowships for undergraduate students in the US"
   - Budget: 120 minutes

2. **System Behavior**:
   - **Planning**: Generates subtasks with MCP tool recommendations
   - **Execution**: Uses web search, GitHub analysis, database storage
   - **Analysis**: Statistical analysis of fellowship data
   - **Reporting**: Comprehensive report with all findings

3. **User Experience**:
   - Watch real-time progress in dashboard
   - See MCP tools being used in activity feed
   - Review structured reports with multi-source data
   - Export results in preferred format

### **Example 2: Conference Research**

1. **Create Task**:
   - Name: "AI Conference Submissions"
   - Description: "Find AI conferences with open submission deadlines"
   - Budget: 90 minutes

2. **System Behavior**:
   - **Web Search**: Find conference websites and CFPs
   - **Repository Analysis**: Check conference organization repos
   - **Text Extraction**: Extract deadlines and requirements
   - **Data Organization**: Store structured conference data

3. **User Experience**:
   - Monitor search progress in real-time
   - See extracted deadlines and requirements
   - Get organized conference data
   - Export to calendar or spreadsheet format

## 🔍 Monitoring and Control

### **Real-Time Monitoring**
- **Progress Tracking**: Visual progress bars and percentages
- **Activity Feed**: Live updates of research actions
- **Tool Usage**: Which MCP tools are being used
- **Data Collection**: Sources being accessed and analyzed

### **User Controls**
- **Pause Research**: Temporarily stop research execution
- **Resume Research**: Continue paused research
- **Export Data**: Download results in various formats
- **Task Management**: Create, edit, delete research tasks

### **MCP Status Monitoring**
- **Server Status**: Which MCP servers are connected
- **Tool Availability**: Which tools are available for use
- **Error Handling**: Clear error messages if servers unavailable
- **Health Checks**: System health and connectivity status

## 📈 Advanced Usage

### **Custom MCP Tools**
Users can add custom MCP tools by:
1. Installing additional MCP servers
2. Updating `mcp_config.json` configuration
3. Restarting KestrelAI to load new tools

### **Research Orchestration**
- **Multi-task Research**: Run multiple research tasks simultaneously
- **Task Dependencies**: Chain research tasks with dependencies
- **Data Sharing**: Share findings between related tasks
- **Batch Processing**: Process multiple research objectives

### **Integration with External Systems**
- **API Integration**: Connect to external APIs via MCP tools
- **Database Integration**: Store results in external databases
- **File System Integration**: Access local and remote file systems
- **Repository Integration**: Analyze code and documentation repositories

## 🚨 Error Handling

### **MCP Server Issues**
- **Server Not Found**: Clear error message with installation instructions
- **Connection Failed**: Detailed error information and troubleshooting
- **Tool Unavailable**: Graceful degradation with alternative approaches
- **API Key Issues**: Clear guidance on required API keys

### **Research Issues**
- **No Results Found**: System suggests alternative search strategies
- **Tool Execution Failed**: Automatic retry with different parameters
- **Data Processing Errors**: Error logging and recovery mechanisms
- **Timeout Issues**: Configurable timeouts and retry logic

## 🎯 Best Practices

### **Task Design**
- **Be Specific**: Clear, detailed research objectives
- **Set Realistic Budgets**: Appropriate time allocation for research scope
- **Use Templates**: Leverage pre-built templates for common research types
- **Iterative Approach**: Break complex research into smaller tasks

### **MCP Configuration**
- **Install Required Servers**: Ensure all needed MCP servers are installed
- **Configure API Keys**: Set up required API keys for external services
- **Test Connections**: Verify MCP server connectivity before starting research
- **Monitor Resources**: Check system resources and performance

### **Research Monitoring**
- **Regular Check-ins**: Monitor progress and adjust as needed
- **Review Checkpoints**: Review periodic progress reports
- **Export Results**: Save important findings regularly
- **Iterate and Refine**: Use results to refine and expand research

This user interaction model provides a seamless experience where users can leverage powerful MCP tools through a simple, intuitive interface while maintaining full visibility into the research process and results.




