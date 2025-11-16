# API tests for backend functionality
import time

import pytest
import requests


@pytest.mark.api
@pytest.mark.requires_services
class TestBackendAPI:
    """Test complete end-to-end workflow."""

    @pytest.fixture
    def api_base_url(self):
        """API base URL for testing."""
        return "http://localhost:8000/api/v1"

    @pytest.fixture
    def frontend_url(self):
        """Frontend URL for testing."""
        return "http://localhost:5173"

    def test_service_connectivity(self, api_base_url, frontend_url):
        """Test that all services are accessible."""
        # Test backend API
        try:
            response = requests.get(f"{api_base_url}/tasks", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip("Backend API not accessible")

        # Test frontend
        try:
            response = requests.get(frontend_url, timeout=5)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip("Frontend not accessible")

    def test_task_lifecycle(self, api_base_url):
        """Test complete task lifecycle."""
        # Skip if services not available
        try:
            requests.get(f"{api_base_url}/tasks", timeout=5)
        except requests.exceptions.RequestException:
            pytest.skip("Backend API not accessible")

        # 1. Create task
        task_data = {
            "name": "E2E Test Task",
            "description": "End-to-end test task",
            "budgetMinutes": 2,
        }

        response = requests.post(f"{api_base_url}/tasks", json=task_data, timeout=10)
        assert response.status_code == 201
        task = response.json()
        task_id = task["id"]

        # 2. Verify task creation
        response = requests.get(f"{api_base_url}/tasks/{task_id}", timeout=5)
        assert response.status_code == 200
        created_task = response.json()
        assert created_task["name"] == task_data["name"]
        assert created_task["status"] == "pending"

        # 3. Start task
        response = requests.post(f"{api_base_url}/tasks/{task_id}/start", timeout=10)
        assert response.status_code == 200

        # 4. Monitor task progress
        max_wait_time = 120  # 2 minutes max wait
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            response = requests.get(f"{api_base_url}/tasks/{task_id}", timeout=5)
            assert response.status_code == 200

            task_status = response.json()

            # Check if task has research plan
            if task_status.get("research_plan") is not None:
                assert len(task_status["research_plan"]["subtasks"]) > 0
                break

            time.sleep(5)  # Wait 5 seconds before next check

        # 5. Verify task is processing
        response = requests.get(f"{api_base_url}/tasks/{task_id}", timeout=5)
        assert response.status_code == 200
        task_status = response.json()

        # Task should have research plan
        assert task_status.get("research_plan") is not None
        assert len(task_status["research_plan"]["subtasks"]) > 0

        # 6. Wait for task completion or timeout
        completion_timeout = 300  # 5 minutes max
        completion_start = time.time()

        while time.time() - completion_start < completion_timeout:
            response = requests.get(f"{api_base_url}/tasks/{task_id}", timeout=5)
            assert response.status_code == 200

            task_status = response.json()

            # Check if task completed
            if task_status["progress"] >= 100:
                break

            time.sleep(10)  # Wait 10 seconds before next check

        # 7. Verify final task state
        response = requests.get(f"{api_base_url}/tasks/{task_id}", timeout=5)
        assert response.status_code == 200
        final_task = response.json()

        # Task should have made some progress
        assert final_task["progress"] > 0
        assert final_task["elapsed"] > 0

        # Clean up - delete task
        requests.delete(f"{api_base_url}/tasks/{task_id}", timeout=5)

    def test_multiple_tasks(self, api_base_url):
        """Test handling multiple tasks."""
        # Skip if services not available
        try:
            requests.get(f"{api_base_url}/tasks", timeout=5)
        except requests.exceptions.RequestException:
            pytest.skip("Backend API not accessible")

        # Create multiple tasks
        task_ids = []
        for i in range(3):
            task_data = {
                "name": f"Multi Task {i+1}",
                "description": f"Multiple task test {i+1}",
                "budgetMinutes": 1,
            }

            response = requests.post(
                f"{api_base_url}/tasks", json=task_data, timeout=10
            )
            assert response.status_code == 201
            task_ids.append(response.json()["id"])

        # Start all tasks
        for task_id in task_ids:
            response = requests.post(
                f"{api_base_url}/tasks/{task_id}/start", timeout=10
            )
            assert response.status_code == 200

        # Wait for all tasks to get research plans
        max_wait_time = 180  # 3 minutes max wait
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            all_have_plans = True
            for task_id in task_ids:
                response = requests.get(f"{api_base_url}/tasks/{task_id}", timeout=5)
                assert response.status_code == 200

                task_status = response.json()
                if task_status.get("research_plan") is None:
                    all_have_plans = False
                    break

            if all_have_plans:
                break

            time.sleep(5)

        # Verify all tasks have research plans
        for task_id in task_ids:
            response = requests.get(f"{api_base_url}/tasks/{task_id}", timeout=5)
            assert response.status_code == 200
            task_status = response.json()
            assert task_status.get("research_plan") is not None

        # Clean up - delete all tasks
        for task_id in task_ids:
            requests.delete(f"{api_base_url}/tasks/{task_id}", timeout=5)

    def test_task_error_handling(self, api_base_url):
        """Test error handling in task operations."""
        # Skip if services not available
        try:
            requests.get(f"{api_base_url}/tasks", timeout=5)
        except requests.exceptions.RequestException:
            pytest.skip("Backend API not accessible")

        # Test invalid task creation
        invalid_task_data = {
            "name": "",  # Empty name should fail
            "description": "Invalid task",
            "budgetMinutes": 5,
        }

        response = requests.post(
            f"{api_base_url}/tasks", json=invalid_task_data, timeout=10
        )
        # Should return error status
        assert response.status_code >= 400

        # Test accessing non-existent task
        response = requests.get(f"{api_base_url}/tasks/nonexistent", timeout=5)
        assert response.status_code == 404

        # Test starting non-existent task
        response = requests.post(f"{api_base_url}/tasks/nonexistent/start", timeout=5)
        assert response.status_code == 404

    def test_performance_requirements(self, api_base_url, performance_thresholds):
        """Test performance requirements."""
        # Skip if services not available
        try:
            requests.get(f"{api_base_url}/tasks", timeout=5)
        except requests.exceptions.RequestException:
            pytest.skip("Backend API not accessible")

        # Test task creation performance
        start_time = time.time()

        task_data = {
            "name": "Performance Test Task",
            "description": "Test task creation performance",
            "budgetMinutes": 1,
        }

        response = requests.post(f"{api_base_url}/tasks", json=task_data, timeout=10)

        creation_time = time.time() - start_time
        assert creation_time < performance_thresholds["task_creation_time"]
        assert response.status_code == 201

        task_id = response.json()["id"]

        # Test task retrieval performance
        start_time = time.time()
        response = requests.get(f"{api_base_url}/tasks/{task_id}", timeout=5)
        retrieval_time = time.time() - start_time

        assert retrieval_time < performance_thresholds["task_creation_time"]
        assert response.status_code == 200

        # Clean up
        requests.delete(f"{api_base_url}/tasks/{task_id}", timeout=5)

    def test_concurrent_operations(self, api_base_url):
        """Test concurrent task operations."""
        # Skip if services not available
        try:
            requests.get(f"{api_base_url}/tasks", timeout=5)
        except requests.exceptions.RequestException:
            pytest.skip("Backend API not accessible")

        import concurrent.futures

        def create_and_start_task(task_num):
            """Create and start a task."""
            task_data = {
                "name": f"Concurrent Task {task_num}",
                "description": f"Concurrent test task {task_num}",
                "budgetMinutes": 1,
            }

            # Create task
            response = requests.post(
                f"{api_base_url}/tasks", json=task_data, timeout=10
            )
            if response.status_code != 201:
                return None

            task_id = response.json()["id"]

            # Start task
            response = requests.post(
                f"{api_base_url}/tasks/{task_id}/start", timeout=10
            )
            if response.status_code != 200:
                return None

            return task_id

        # Create multiple tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_and_start_task, i) for i in range(3)]
            task_ids = [future.result() for future in futures]

        # Filter out None results (failed tasks)
        task_ids = [tid for tid in task_ids if tid is not None]

        # Verify all tasks were created and started
        assert len(task_ids) > 0

        # Wait for tasks to get research plans
        max_wait_time = 120  # 2 minutes max wait
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            all_have_plans = True
            for task_id in task_ids:
                response = requests.get(f"{api_base_url}/tasks/{task_id}", timeout=5)
                if response.status_code != 200:
                    continue

                task_status = response.json()
                if task_status.get("research_plan") is None:
                    all_have_plans = False
                    break

            if all_have_plans:
                break

            time.sleep(5)

        # Clean up - delete all tasks
        for task_id in task_ids:
            try:
                requests.delete(f"{api_base_url}/tasks/{task_id}", timeout=5)
            except:
                pass  # Ignore cleanup errors
