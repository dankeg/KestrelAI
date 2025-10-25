# Performance regression tests
import pytest
import time
import requests
from unittest.mock import patch, Mock


@pytest.mark.performance
@pytest.mark.requires_services


class TestPerformanceRegression:
    """Test performance regression prevention."""
    
    @pytest.fixture
    def performance_benchmarks(self):
        """Performance benchmarks to prevent regression."""
        return {
            "llm_response_time": 10.0,  # seconds - should be much faster with local Ollama
            "planning_phase_time": 30.0,  # seconds
            "task_creation_time": 1.0,  # seconds
            "redis_operation_time": 0.1,  # seconds
            "memory_store_operation_time": 0.5,  # seconds
            "web_search_time": 5.0,  # seconds
            "orchestrator_initialization_time": 2.0,  # seconds
        }
    
    def test_llm_performance_regression(self, performance_benchmarks):
        """Test LLM performance to prevent regression."""
        from KestrelAI.agents.base import LlmWrapper
        
        llm = LlmWrapper(model="gemma3:27b", host="http://localhost:11434")
        
        start_time = time.time()
        try:
            response = llm.chat([{"role": "user", "content": "Hello"}])
            response_time = time.time() - start_time
            
            # With local Ollama, this should be much faster than Docker Ollama
            assert response_time < performance_benchmarks["llm_response_time"]
            assert response is not None
            
        except Exception as e:
            # If connection fails, that's a different issue
            pytest.skip(f"LLM not accessible: {e}")
    
    def test_redis_performance_regression(self, performance_benchmarks):
        """Test Redis performance to prevent regression."""
        from KestrelAI.shared.redis_utils import get_sync_redis_client
        
        try:
            client = get_sync_redis_client({
                "host": "localhost",
                "port": 6379,
                "db": 0
            })
            
            # Test ping performance
            start_time = time.time()
            result = client.ping()
            ping_time = time.time() - start_time
            
            assert result is True
            assert ping_time < performance_benchmarks["redis_operation_time"]
            
            # Test set/get performance
            start_time = time.time()
            client.set("perf_test_key", "perf_test_value")
            set_time = time.time() - start_time
            
            start_time = time.time()
            value = client.get("perf_test_key")
            get_time = time.time() - start_time
            
            assert value == b"perf_test_value"
            assert set_time < performance_benchmarks["redis_operation_time"]
            assert get_time < performance_benchmarks["redis_operation_time"]
            
            # Clean up
            client.delete("perf_test_key")
            
        except Exception as e:
            pytest.skip(f"Redis not accessible: {e}")
    
    def test_memory_store_performance_regression(self, performance_benchmarks, temp_dir):
        """Test memory store performance to prevent regression."""
        from KestrelAI.memory.vector_store import MemoryStore
        
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = Mock()
            mock_collection.add.return_value = None
            mock_collection.query.return_value = {
                "documents": [["test document"]],
                "metadatas": [[{"metadata": "test"}]],
                "distances": [[0.1]]
            }
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            memory_store = MemoryStore(path=temp_dir)
            
            # Test add document performance
            start_time = time.time()
            memory_store.add_document("test_id", "test content", {"metadata": "test"})
            add_time = time.time() - start_time
            
            assert add_time < performance_benchmarks["memory_store_operation_time"]
            
            # Test search performance
            start_time = time.time()
            results = memory_store.search("test query", n_results=5)
            search_time = time.time() - start_time
            
            assert len(results) > 0
            assert search_time < performance_benchmarks["memory_store_operation_time"]
    
    def test_web_search_performance_regression(self, performance_benchmarks):
        """Test web search performance to prevent regression."""
        from KestrelAI.agents.web_research_agent import WebResearchAgent
        
        mock_llm = Mock()
        mock_memory = Mock()
        agent = WebResearchAgent("test-agent", mock_llm, mock_memory)
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {
                        "title": "Test Result",
                        "url": "https://example.com",
                        "content": "Test content"
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            start_time = time.time()
            results = agent.search_web("test query")
            search_time = time.time() - start_time
            
            assert len(results) == 1
            assert search_time < performance_benchmarks["web_search_time"]
    
    def test_orchestrator_performance_regression(self, performance_benchmarks, mock_llm, mock_task):
        """Test orchestrator performance to prevent regression."""
        from KestrelAI.agents.consolidated_orchestrator import ResearchOrchestrator
        
        # Test initialization performance
        start_time = time.time()
        orchestrator = ResearchOrchestrator([mock_task], mock_llm, profile="kestrel")
        init_time = time.time() - start_time
        
        assert init_time < performance_benchmarks["orchestrator_initialization_time"]
        assert orchestrator is not None
    
    def test_planning_phase_performance_regression(self, performance_benchmarks, mock_llm, mock_task):
        """Test planning phase performance to prevent regression."""
        from KestrelAI.agents.consolidated_orchestrator import ResearchOrchestrator
        import asyncio
        
        orchestrator = ResearchOrchestrator([mock_task], mock_llm, profile="kestrel")
        
        with patch.object(orchestrator.llm, 'chat') as mock_chat:
            mock_chat.return_value = '''
            {
                "restated_task": "Test restated task",
                "subtasks": [
                    {
                        "order": 1,
                        "description": "Test subtask 1",
                        "success_criteria": "Test criteria 1",
                        "status": "pending",
                        "findings": []
                    }
                ],
                "current_subtask_index": 0
            }
            '''
            
            async def test_planning():
                start_time = time.time()
                await orchestrator._planning_phase(mock_task)
                planning_time = time.time() - start_time
                
                assert planning_time < performance_benchmarks["planning_phase_time"]
                
                # Verify planning completed
                task_state = orchestrator.task_states[mock_task.name]
                assert task_state.research_plan is not None
            
            # Run async test
            asyncio.run(test_planning())
    
    def test_end_to_end_performance_regression(self, performance_benchmarks):
        """Test end-to-end performance to prevent regression."""
        api_base_url = "http://localhost:8000/api/v1"
        
        # Skip if services not available
        try:
            requests.get(f"{api_base_url}/tasks", timeout=5)
        except requests.exceptions.RequestException:
            pytest.skip("Backend API not accessible")
        
        # Test complete workflow performance
        start_time = time.time()
        
        # Create task
        task_data = {
            "name": "Performance Regression Test",
            "description": "Test end-to-end performance",
            "budgetMinutes": 1
        }
        
        response = requests.post(
            f"{api_base_url}/tasks",
            json=task_data,
            timeout=10
        )
        
        creation_time = time.time() - start_time
        assert creation_time < performance_benchmarks["task_creation_time"]
        assert response.status_code == 201
        
        task_id = response.json()["id"]
        
        # Start task and measure planning phase time
        start_time = time.time()
        response = requests.post(
            f"{api_base_url}/tasks/{task_id}/start",
            timeout=10
        )
        assert response.status_code == 200
        
        # Wait for planning phase to complete
        max_wait_time = 60  # 1 minute max wait
        planning_start = time.time()
        
        while time.time() - planning_start < max_wait_time:
            response = requests.get(f"{api_base_url}/tasks/{task_id}", timeout=5)
            assert response.status_code == 200
            
            task_status = response.json()
            if task_status.get("research_plan") is not None:
                break
            
            time.sleep(2)
        
        planning_time = time.time() - start_time
        assert planning_time < performance_benchmarks["planning_phase_time"]
        
        # Clean up
        requests.delete(f"{api_base_url}/tasks/{task_id}", timeout=5)
    
    def test_memory_usage_regression(self):
        """Test memory usage to prevent regression."""
        import psutil
        import os
        
        # Get current process memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        from KestrelAI.agents.consolidated_orchestrator import ResearchOrchestrator
        from KestrelAI.shared.models import Task, TaskStatus
        
        mock_llm = Mock()
        tasks = []
        
        # Create multiple tasks to test memory usage
        for i in range(10):
            task = Task(
                name=f"Memory Test Task {i}",
                description=f"Memory test task {i}",
                budgetMinutes=5,
                status=TaskStatus.ACTIVE
            )
            tasks.append(task)
        
        orchestrator = ResearchOrchestrator(tasks, mock_llm, profile="kestrel")
        
        # Check memory usage after operations
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f}MB"
    
    def test_concurrent_performance_regression(self, performance_benchmarks):
        """Test concurrent operations performance to prevent regression."""
        import concurrent.futures
        import threading
        
        api_base_url = "http://localhost:8000/api/v1"
        
        # Skip if services not available
        try:
            requests.get(f"{api_base_url}/tasks", timeout=5)
        except requests.exceptions.RequestException:
            pytest.skip("Backend API not accessible")
        
        def create_task(task_num):
            """Create a task."""
            task_data = {
                "name": f"Concurrent Perf Test {task_num}",
                "description": f"Concurrent performance test {task_num}",
                "budgetMinutes": 1
            }
            
            start_time = time.time()
            response = requests.post(
                f"{api_base_url}/tasks",
                json=task_data,
                timeout=10
            )
            creation_time = time.time() - start_time
            
            if response.status_code == 201:
                task_id = response.json()["id"]
                return task_id, creation_time
            return None, creation_time
        
        # Test concurrent task creation
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_task, i) for i in range(5)]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        # Filter successful results
        successful_results = [r for r in results if r[0] is not None]
        task_ids = [r[0] for r in successful_results]
        creation_times = [r[1] for r in successful_results]
        
        # Verify performance requirements
        assert len(successful_results) > 0
        assert total_time < performance_benchmarks["task_creation_time"] * 2  # Allow some overhead for concurrency
        
        for creation_time in creation_times:
            assert creation_time < performance_benchmarks["task_creation_time"]
        
        # Clean up
        for task_id in task_ids:
            try:
                requests.delete(f"{api_base_url}/tasks/{task_id}", timeout=5)
            except:
                pass  # Ignore cleanup errors
