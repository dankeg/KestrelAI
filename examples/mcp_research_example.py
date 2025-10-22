#!/usr/bin/env python3
"""
Example script demonstrating MCP-enhanced research capabilities in KestrelAI
"""

import asyncio
import sys
import os

# Add the parent directory to the path so we can import KestrelAI modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KestrelAI.agents.base import LlmWrapper
from KestrelAI.agents.research_orchestrator import ResearchOrchestrator
from KestrelAI.agents.web_research_agent import WebResearchAgent, ResearchConfig
from KestrelAI.memory.vector_store import MemoryStore
from KestrelAI.shared.models import Task, TaskStatus
from KestrelAI.mcp.mcp_manager import MCPManager
from KestrelAI.mcp.mcp_config import get_mcp_config


async def main():
    """Main example function"""
    print("🦅 KestrelAI MCP Research Example")
    print("=" * 50)
    
    # Initialize components
    print("Initializing components...")
    
    # Initialize LLM wrapper
    llm = LlmWrapper(temperature=0.7)
    
    # Initialize memory store
    memory = MemoryStore()
    
    # Initialize MCP manager
    mcp_config = get_mcp_config()
    mcp_manager = MCPManager(mcp_config)
    
    # Create a sample research task
    task = Task(
        name="AI Research Grants Analysis",
        description="Find and analyze currently open AI/ML research grants and funding opportunities for undergraduate students in the US. Include eligibility requirements, deadlines, application processes, and funding amounts.",
        budgetMinutes=60,
        status=TaskStatus.CONFIGURING
    )
    
    print(f"Created task: {task.name}")
    print(f"Description: {task.description}")
    print()
    
    # Initialize MCP system
    print("Initializing MCP system...")
    async with mcp_manager:
        mcp_initialized = mcp_manager.is_initialized
        
        if mcp_initialized:
            print("✅ MCP system initialized successfully")
            
            # Show available tools
            health_status = mcp_manager.get_health_status()
            print(f"Connected servers: {health_status['connected_servers']}/{health_status['total_servers']}")
            
            # Show server status
            for server_name, status in health_status['servers'].items():
                status_icon = "✅" if status['connected'] else "❌"
                print(f"  {status_icon} {server_name}: {len(status['tools'])} tools")
            
            print()
            
            # Initialize MCP orchestrator
            print("Initializing MCP orchestrator...")
            orchestrator = ResearchOrchestrator([task], llm, profile="kestrel", mcp_manager=mcp_manager, use_mcp=True)
            await orchestrator.initialize_mcp()
            print("✅ MCP orchestrator initialized")
            
            try:
                # Run the planning phase
                print("Running planning phase...")
                await orchestrator._planning_phase(task)
                
                # Show the generated plan
                task_state = orchestrator.task_states[task.name]
                if task_state.research_plan:
                    print("✅ Research plan generated:")
                    print(f"  Restated task: {task_state.research_plan.restated_task}")
                    print(f"  Subtasks: {len(task_state.research_plan.subtasks)}")
                    
                    for i, subtask in enumerate(task_state.research_plan.subtasks):
                        print(f"    {i+1}. {subtask.description}")
                
                print()
                
                # Run a few research steps
                print("Running research steps...")
                for step in range(3):
                    print(f"Step {step + 1}:")
                    result = await orchestrator.next_action(task)
                    print(f"  Result: {result}")
                    print()
                    
                    # Show progress
                    progress = orchestrator.get_task_progress(task.name)
                    print(f"  Progress: {progress['progress']:.1f}%")
                    print()
                
                print("✅ Research completed")
                
            finally:
                # Cleanup
                await orchestrator.cleanup_mcp()
            
        else:
            print("❌ MCP system failed to initialize - no MCP servers available")
            print("This example requires MCP servers to be installed and configured.")
            print("Please install MCP servers or check your configuration.")
    
    print("✅ Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())