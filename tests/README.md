# KestrelAI Test Suite

This directory contains the comprehensive test suite for KestrelAI, organized by test type and functionality.

## Test Organization

### Directory Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_core.py         # Core functionality tests
│   └── test_agents.py       # Agent functionality tests
├── integration/             # Integration tests
│   ├── test_model_loop.py   # Model loop integration tests
│   └── test_memory_store.py # Memory store integration tests
├── api/                     # Backend API tests
│   └── test_backend_api.py  # API endpoint tests
├── frontend/                # Frontend tests
│   └── test_theme_switching.py # Theme switching tests
├── e2e/                     # End-to-end tests
│   └── test_workflow.py     # Complete workflow tests
├── performance/             # Performance tests
│   └── test_regression.py   # Performance regression tests
├── utils/                   # Test utilities
│   ├── check_services.py    # Service availability checker
│   ├── test_fixtures.py     # Common test fixtures
│   └── test_config.py       # Test configuration
├── conftest.py              # Test configuration and fixtures
├── run_tests.py             # Test runner script
└── README.md               # This file
```

### Test Categories

- **Unit Tests** (`tests/unit/`): Fast, isolated tests with mocked dependencies
- **Integration Tests** (`tests/integration/`): Tests that verify component interactions
- **API Tests** (`tests/api/`): Backend API endpoint tests
- **Frontend Tests** (`tests/frontend/`): Frontend UI and theme tests
- **End-to-End Tests** (`tests/e2e/`): Complete workflow tests
- **Performance Tests** (`tests/performance/`): Performance and regression tests

## Running Tests

### Using the Test Runner (Recommended)

```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py unit integration
python tests/run_tests.py api frontend
python tests/run_tests.py e2e performance

# Run with coverage
python tests/run_tests.py --coverage

# Run in parallel
python tests/run_tests.py --parallel

# Skip service check
python tests/run_tests.py --skip-service-check
```

### Basic pytest Commands

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m api
pytest -m frontend
pytest -m e2e
pytest -m performance

# Run specific test files
pytest tests/unit/test_core.py
pytest tests/api/test_backend_api.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=KestrelAI --cov-report=html
```

### Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.api`: Backend API tests
- `@pytest.mark.frontend`: Frontend UI tests
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.slow`: Slow running tests
- `@pytest.mark.requires_services`: Tests requiring external services

### Service Requirements

Some tests require external services to be running:

- **Backend API** (`http://localhost:8000`) - Required for API tests
- **Frontend** (`http://localhost:5173`) - Required for frontend tests
- **Ollama** (`http://localhost:11434`) - Optional for LLM tests
- **SearXNG** (`http://localhost:8080`) - Optional for search tests
- **Redis** (`localhost:6379`) - Optional for caching tests

Check service availability:

```bash
python tests/utils/check_services.py
```

## Test Configuration

### Environment Variables

Tests use the following environment variables:

- `PYTHONPATH`: Set to project root
- `REDIS_HOST`: Redis server host
- `REDIS_PORT`: Redis server port
- `OLLAMA_BASE_URL`: Ollama server URL
- `SEARXNG_URL`: SearXNG server URL

### Test Configuration

Test configuration is managed in `tests/utils/test_config.py`:

- Targets and URLs for services
- Performance thresholds
- Test data constants
- Service requirements by category

### Fixtures

Common test fixtures are available in `tests/utils/test_fixtures.py`:

- `temp_dir`: Temporary directory for test files
- `sample_task`: Sample task for testing
- `sample_research_plan`: Sample research plan
- `mock_llm`: Mock LLM wrapper
- `mock_memory_store`: Mock memory store
- `mock_redis`: Mock Redis client
- `mock_ollama_client`: Mock Ollama client
- `mock_chromadb`: Mock ChromaDB client
- `mock_sentence_transformer`: Mock SentenceTransformer
- `mock_requests`: Mock HTTP requests

## Writing Tests

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch
from tests.utils.test_fixtures import TestDataFactory

@pytest.mark.unit
class TestMyComponent:
    """Test my component functionality."""
    
    def test_basic_functionality(self, mock_llm):
        """Test basic functionality."""
        # Test implementation
        assert True
    
    @pytest.mark.requires_services
    def test_with_external_service(self):
        """Test with external service."""
        # Test implementation
        assert True
```

### Using Test Utilities

```python
from tests.utils.test_fixtures import TestData

def test_with_test_data():
    """Test using test data factory."""
    task = TestDataFactory.create_task(
        name="Custom Task",
        description="Custom description"
    )
    assert task.name == "Custom Task"

def test_with_config():
    """Test using test configuration."""
    config = get_test_config()
    assert config["api_base_url"] == "http://localhost:8000/api/v1"
```

### Best Practices

1. **Use descriptive test names**: Test names should clearly describe what is being tested
2. **Mock external dependencies**: Use mocks for external services in unit tests
3. **Test edge cases**: Include tests for error conditions and edge cases
4. **Use fixtures**: Leverage pytest fixtures for common setup
5. **Mark tests appropriately**: Use the correct pytest markers
6. **Keep tests fast**: Unit tests should run quickly
7. **Test one thing**: Each test should verify one specific behavior
8. **Use test utilities**: Leverage the test utilities for common patterns

## Continuous Integration

Tests are automatically run in CI/CD pipelines with the following configuration:

- Python 3.9+
- pytest with coverage reporting
- Service availability checking
- Performance regression detection

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH is set correctly
2. **Service Connection Errors**: Check that required services are running
3. **Timeout Errors**: Increase timeout values for slow tests
4. **Mock Issues**: Verify mock configurations and return values

### Debug Mode

Run tests in debug mode for detailed output:

```bash
pytest -v -s --tb=long
```

### Service Debugging

Check service availability:

```bash
python tests/utils/check_services.py
```

## Performance Testing

Performance tests ensure that the system meets performance requirements:

- **LLM Response Time**: < 10 seconds
- **Planning Phase Time**: < 30 seconds
- **Task Creation Time**: < 1 second
- **Redis Operation Time**: < 0.1 seconds
- **API Response Time**: < 2 seconds
- **Frontend Load Time**: < 3 seconds

Run performance tests:

```bash
pytest -m performance
```

## Coverage

Test coverage is tracked and reported:

```bash
pytest --cov=KestrelAI --cov-report=html
```

Coverage reports are generated in `htmlcov/` directory.

## Test Utilities

### Service Checker

The service checker (`tests/utils/check_services.py`) provides:

- Service availability checking
- Configuration management
- Test-specific service validation

### Test Data Factory

The test data factory (`tests/utils/test_fixtures.py`) provides:

- Sample data creation
- Mock object setup
- Common test patterns

### Test Configuration

Test configuration (`tests/utils/test_config.py`) provides:

- Service URLs and settings
- Performance thresholds
- Test categorization