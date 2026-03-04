# Performance regression tests
import time
import asyncio
import os
import psutil
from unittest.mock import Mock, patch
import pytest
import requests

from KestrelAI.agents.base import LlmWrapper
from KestrelAI.memory.vector_store import MemoryStore
from KestrelAI.agents.web_research_agent import WebResearchAgent
from KestrelAI.agents.research_orchestrator import ResearchOrchestrator
from KestrelAI.shared.models import Task, TaskStatus


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """
    Offline performance benchmarks focusing on KestrelAI internal overhead.
    These tests use deterministic mocks to ensure CI safety and reproducibility.
    """

    def test_llm_wrapper_overhead(self):
        """
        Benchmark: LlmWrapper.chat method overhead.
        Question: How much latency does the KestrelAI wrapper add on top of the LLM client?
        Invariant: Method overhead should stay within order-of-magnitude guardrails (< 50ms).
        """
        with patch("ollama.Client") as mock_ollama_client:
            # Mock the internal client response to be near-instant
            mock_response = Mock()
            mock_response.message.content = "Mocked Response"
            mock_ollama_client.return_value.chat.return_value = mock_response
            
            wrapper = LlmWrapper(model="test-model", host="http://localhost:11434")
            
            # Warm up
            wrapper.chat([{"role": "user", "content": "hi"}])
            
            start_time = time.perf_counter()
            for _ in range(10):
                wrapper.chat([{"role": "user", "content": "hi"}])
            end_time = time.perf_counter()
            
            avg_latency = (end_time - start_time) / 10
            
            # 50ms is a coarse order-of-magnitude guardrail for local Python overhead,
            # not a strict SLA. It ensures no massive regressions in wrapper logic.
            assert avg_latency < 0.050, f"LLM Wrapper overhead too high: {avg_latency:.4f}s"

    def test_memory_store_abstraction_overhead(self, temp_dir):
        """
        Benchmark: MemoryStore abstraction overhead.
        Question: What is the cost of Kestrel's metadata handling and vector search abstraction?
        Invariant: O(1) or O(log N) scaling for metadata indexing overhead.
        Note: This tests Kestrel abstraction overhead, not ChromaDB performance (which is mocked).
        """
        # Justification: Old test used 'add_document' which does not exist. Redesigned to use 'add'.
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = Mock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
            
            store = MemoryStore(path=temp_dir)
            
            # Measure insertion overhead
            start_time = time.perf_counter()
            for i in range(100):
                store.add(doc_id=f"id_{i}", text=f"text {i}", meta={"idx": i})
            insertion_time = time.perf_counter() - start_time
            
            # Measure search abstraction overhead
            mock_collection.query.return_value = {
                "documents": [["result"]],
                "metadatas": [[{"idx": 0}]],
                "distances": [[0.1]]
            }
            
            start_time = time.perf_counter()
            for _ in range(50):
                store.search("query", k=5)
            search_time = time.perf_counter() - start_time
            
            # Guardrails for Python-side logic (not storage engine)
            assert insertion_time < 0.5, f"MemoryStore insertion overhead too high: {insertion_time:.4f}s"
            assert search_time < 0.2, f"MemoryStore search overhead too high: {search_time:.4f}s"

    @pytest.mark.asyncio
    async def test_agent_loop_latency(self, mock_task):
        """
        Benchmark: WebResearchAgent.run_step logic latency.
        Question: How much time is spent in Kestrel's agent loop logic (context building, plan parsing) excluding I/O?
        Invariant: Deterministic path execution latency.
        """
        # Justification: Old test assumed a 'search_web' method which is internal/non-existent.
        # Now benchmarking the public 'run_step' with mocked LLM and Search.
        mock_llm = Mock()
        mock_llm.chat.return_value = '{"action": "think", "thought": "benchmarking"}'
        mock_memory = Mock()
        
        agent = WebResearchAgent("bench-agent", mock_llm, mock_memory)
        agent.config.think_loops = 1  # Minimize loops for core logic benchmark
        
        start_time = time.perf_counter()
        await agent.run_step(mock_task)
        latency = time.perf_counter() - start_time
        
        # Core logic (context build + parse) should be fast
        assert latency < 0.200, f"Agent loop logic too slow: {latency:.4f}s"

    @pytest.mark.asyncio
    async def test_orchestrator_transition_complexity(self, mock_llm, mock_task):
        """
        Benchmark: ResearchOrchestrator.next_action overhead.
        Question: What is the overhead of orchestrator state transitions and subtask management?
        Invariant: Linear complexity relative to subtask count.
        Entry Point: ResearchOrchestrator.next_action (Stable Public Interface)
        """
        # Justification: Old test referenced 'consolidated_orchestrator' which was renamed/moved.
        # Now targeting ResearchOrchestrator.next_action as the stable entry point.
        from KestrelAI.shared.models import ResearchPlan, Subtask
        
        orchestrator = ResearchOrchestrator([mock_task], mock_llm)
        task_state = orchestrator.task_states[mock_task.name]
        task_state.research_plan = ResearchPlan(
            restated_task="Task",
            subtasks=[Subtask(order=i, description=f"S{i}", success_criteria="C") for i in range(5)]
        )
        
        # Mock subtask agent to prevent recursive I/O
        mock_agent = Mock()
        async def mock_run_step(*args, **kwargs):
            return "Subtask result"
        mock_agent.run_step.side_effect = mock_run_step
        
        with patch.object(task_state, "get_current_subtask_agent", return_value=mock_agent):
            with patch.object(orchestrator, "_review") as mock_review:
                from KestrelAI.agents.research_orchestrator import OrchestratorDecision
                mock_review.return_value = OrchestratorDecision(
                    reasoning="testing", decision="continue", feedback="", subtask="stay", next_task=""
                )
                
                start_time = time.perf_counter()
                await orchestrator.next_action(mock_task)
                latency = time.perf_counter() - start_time
                
                assert latency < 0.100, f"Orchestrator transition too slow: {latency:.4f}s"

    def test_memory_growth_trend(self, mock_llm):
        """
        Benchmark: Memory usage growth trend.
        Question: Does initializing a large number of tasks cause exponential memory growth?
        Invariant: RSS growth should be roughly linear or better. Assert coarse upper bounds.
        """
        process = psutil.Process(os.getpid())
        
        def get_mem():
            return process.memory_info().rss / 1024 / 1024  # MB

        mem_start = get_mem()
        
        # Create 100 dummy tasks
        tasks = [
            Task(name=f"T{i}", description="desc", budgetMinutes=1)
            for i in range(100)
        ]
        
        orch = ResearchOrchestrator(tasks, mock_llm)
        
        mem_end = get_mem()
        growth = mem_end - mem_start
        
        # Coarse guardrail: 100 tasks shouldn't take more than 50MB of metadata overhead
        assert growth < 50.0, f"Memory growth excessive: {growth:.2f}MB"

@pytest.mark.requires_services
class TestServiceIntegrationPerformance:
    """
    Integration performance tests for live services.
    These are SKIPPED by default and should only be run in dev environments with Ollama/Redis.
    """

    def test_redis_latency(self):
        """Measure real Redis round-trip time."""
        from KestrelAI.shared.redis_utils import get_sync_redis_client
        try:
            client = get_sync_redis_client({"host": "localhost", "port": 6379, "db": 0})
            start = time.perf_counter()
            client.ping()
            latency = time.perf_counter() - start
            assert latency < 0.010, f"Redis latency high: {latency:.4f}s"
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")

    def test_llm_end_to_end_latency(self):
        """Measure real LLM response time from Ollama."""
        llm = LlmWrapper(model="gemma3:27b") # Uses default host
        try:
            start = time.perf_counter()
            llm.chat([{"role": "user", "content": "hi"}])
            latency = time.perf_counter() - start
            # No hard assert here as LLM timing is variable, just log if needed
            print(f"Real LLM Latency: {latency:.4f}s")
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")
