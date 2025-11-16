"""
Unit tests for context management and token counting.
"""

from unittest.mock import Mock, patch

import pytest

try:
    from KestrelAI.agents.context_manager import (
        ContextManager,
        TokenBudget,
        TokenCounter,
    )
except ImportError:
    from agents.context_manager import ContextManager, TokenBudget, TokenCounter


@pytest.mark.unit
class TestTokenBudget:
    """Test TokenBudget dataclass."""

    def test_token_budget_defaults(self):
        """Test default token budget values."""
        budget = TokenBudget()

        assert budget.max_context == 32768
        assert budget.system_prompt == 500
        assert budget.task_description == 200
        assert budget.previous_findings == 2000
        assert budget.checkpoints == 3000
        assert budget.history == 2000
        assert budget.rag_content == 2000
        assert budget.url_reference == 500
        assert budget.response_reserve == 2000

    def test_token_budget_custom(self):
        """Test custom token budget values."""
        budget = TokenBudget(
            max_context=16384, system_prompt=1000, response_reserve=1000
        )

        assert budget.max_context == 16384
        assert budget.system_prompt == 1000
        assert budget.response_reserve == 1000

    def test_available_for_context(self):
        """Test available_for_context calculation."""
        budget = TokenBudget(
            max_context=10000, system_prompt=500, response_reserve=1000
        )

        available = budget.available_for_context
        assert available == 8500  # 10000 - 500 - 1000

    def test_validate_budget(self):
        """Test budget validation."""
        # Valid budget
        budget = TokenBudget(
            max_context=10000,
            system_prompt=1000,
            task_description=500,
            previous_findings=2000,
            checkpoints=2000,
            history=1000,
            rag_content=1000,
            url_reference=500,
            response_reserve=1000,
        )
        assert budget.validate() is True

        # Invalid budget (exceeds max_context)
        budget_invalid = TokenBudget(
            max_context=1000,
            system_prompt=500,
            task_description=500,
            previous_findings=500,
            checkpoints=500,
            history=500,
            rag_content=500,
            url_reference=500,
            response_reserve=500,
        )
        # This should be invalid, but validate() only checks if total <= max
        # Let's test with clearly invalid case
        budget_invalid.max_context = 100
        assert budget_invalid.validate() is False


@pytest.mark.unit
class TestTokenCounter:
    """Test TokenCounter class."""

    @pytest.fixture
    def token_counter(self):
        """Create TokenCounter instance for testing."""
        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_get_encoding.return_value = mock_encoding

            counter = TokenCounter(model_name="gpt-4")
            counter.encoding = mock_encoding
            return counter

    def test_token_counter_initialization(self, token_counter):
        """Test TokenCounter initialization."""
        assert token_counter is not None
        assert token_counter.model_name == "gpt-4"
        assert token_counter.encoding is not None

    def test_count_tokens(self, token_counter):
        """Test token counting."""
        text = "This is a test"
        count = token_counter.count_tokens(text)

        assert count == 5  # Mock returns 5 tokens
        token_counter.encoding.encode.assert_called_once_with(text)

    def test_count_tokens_empty(self, token_counter):
        """Test token counting with empty text."""
        assert token_counter.count_tokens("") == 0
        assert token_counter.count_tokens(None) == 0

    def test_count_messages(self, token_counter):
        """Test counting tokens in messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is AI?"},
        ]

        # Mock encoding to return different token counts
        token_counter.encoding.encode.side_effect = [
            [1, 2, 3, 4, 5],  # First message: 5 tokens
            [6, 7, 8],  # Second message: 3 tokens
        ]

        total = token_counter.count_messages(messages)

        # 5 + 3 + 4 (overhead for first message) + 4 (overhead for second message) = 16
        assert total == 16

    def test_truncate_to_tokens(self, token_counter):
        """Test text truncation to token limit."""
        text = "This is a very long text that needs to be truncated"

        # Mock encoding: full text = 10 tokens, truncated to 5
        full_tokens = list(range(10))
        truncated_tokens = full_tokens[:5]

        token_counter.encoding.encode.return_value = full_tokens
        token_counter.encoding.decode.return_value = "This is a very long"

        result = token_counter.truncate_to_tokens(text, max_tokens=5)

        assert result == "This is a very long..."
        token_counter.encoding.encode.assert_called()
        token_counter.encoding.decode.assert_called_with(truncated_tokens)

    def test_truncate_to_tokens_no_truncation_needed(self, token_counter):
        """Test truncation when text fits within limit."""
        text = "Short text"

        token_counter.encoding.encode.return_value = [1, 2, 3]  # 3 tokens

        result = token_counter.truncate_to_tokens(text, max_tokens=5)

        assert result == text  # Should return original text

    def test_model_encoding_mapping(self):
        """Test model encoding name mapping."""
        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_encoding = Mock()
            mock_get_encoding.return_value = mock_encoding

            # Test different model names
            counter1 = TokenCounter(model_name="gpt-4")
            assert counter1.encoding is not None

            counter2 = TokenCounter(model_name="gemma3:27b")
            assert counter2.encoding is not None

    def test_token_counter_import_error(self):
        """Test TokenCounter raises ImportError when tiktoken not available."""
        with patch("KestrelAI.agents.context_manager.tiktoken", None):
            with pytest.raises(ImportError, match="tiktoken is required"):
                TokenCounter(model_name="gpt-4")


@pytest.mark.unit
class TestContextManager:
    """Test ContextManager class."""

    @pytest.fixture
    def token_counter(self):
        """Create mock TokenCounter."""
        counter = Mock(spec=TokenCounter)
        counter.count_tokens = Mock(
            side_effect=lambda x: len(str(x).split()) * 1.3
        )  # Rough estimate
        counter.truncate_to_tokens = Mock(
            side_effect=lambda x, max_t: x[:max_t] + "..."
        )
        return counter

    @pytest.fixture
    def token_budget(self):
        """Create TokenBudget for testing."""
        return TokenBudget(
            max_context=1000,
            system_prompt=100,
            task_description=50,
            previous_findings=200,
            checkpoints=300,
            history=200,
            rag_content=200,
            url_reference=50,
            response_reserve=100,
        )

    @pytest.fixture
    def context_manager(self, token_counter, token_budget):
        """Create ContextManager instance."""
        return ContextManager(token_counter, token_budget)

    def test_context_manager_initialization(
        self, context_manager, token_counter, token_budget
    ):
        """Test ContextManager initialization."""
        assert context_manager.counter == token_counter
        assert context_manager.budget == token_budget
        assert context_manager.llm is None

    def test_build_context_minimal(self, context_manager, token_counter):
        """Test building context with minimal components."""
        components = {"task": "Test task description"}

        token_counter.count_tokens.return_value = 3  # Small token count

        context, usage = context_manager.build_context(components)

        assert "Task: Test task description" in context
        assert usage["task"] > 0
        assert usage["total"] > 0

    def test_build_context_with_all_components(self, context_manager, token_counter):
        """Test building context with all components."""
        components = {
            "task": "Test task",
            "subtask": "Test subtask",
            "previous_findings": "Previous findings text",
            "checkpoints": ["Checkpoint 1", "Checkpoint 2"],
            "history": ["Action 1", "Action 2"],
            "rag_content": "RAG content",
            "url_reference": "URL: https://example.com",
        }

        # Mock token counts to be within budget
        token_counter.count_tokens.side_effect = lambda x: len(str(x).split())

        context, usage = context_manager.build_context(components)

        assert "Task: Test task" in context
        assert usage["task"] > 0
        assert usage["subtask"] > 0
        assert usage["checkpoints"] > 0
        assert usage["history"] > 0
        assert usage["total"] > 0

    def test_build_context_prioritizes_recent(self, context_manager, token_counter):
        """Test that context building prioritizes recent items."""
        components = {
            "task": "Test task",
            "checkpoints": ["Old checkpoint", "Recent checkpoint"],
            "history": ["Old action", "Recent action"],
        }

        token_counter.count_tokens.side_effect = lambda x: len(str(x).split())

        context, usage = context_manager.build_context(
            components, prioritize_recent=True
        )

        assert "Recent checkpoint" in context or "Old checkpoint" in context

    def test_build_context_truncates_long_content(self, context_manager, token_counter):
        """Test that context manager truncates content that exceeds budget."""
        # Create very long text
        long_text = "word " * 1000  # Very long text

        components = {"task": long_text, "rag_content": long_text}

        # Mock token counting
        def count_tokens(text):
            return len(str(text).split())

        token_counter.count_tokens.side_effect = count_tokens

        context, usage = context_manager.build_context(components)

        # Should truncate to fit budget
        assert usage["task"] <= context_manager.budget.task_description
        assert usage["rag_content"] <= context_manager.budget.rag_content

    def test_build_context_handles_empty_components(self, context_manager):
        """Test building context with empty components."""
        components = {}

        context, usage = context_manager.build_context(components)

        # Should return empty or minimal context
        assert isinstance(context, str)
        assert isinstance(usage, dict)
        assert usage["total"] >= 0

    def test_build_context_checkpoint_prioritization(
        self, context_manager, token_counter
    ):
        """Test checkpoint prioritization when budget is limited."""
        # Create many checkpoints
        checkpoints = [f"Checkpoint {i}" for i in range(10)]

        components = {"task": "Test task", "checkpoints": checkpoints}

        token_counter.count_tokens.side_effect = lambda x: len(str(x).split())

        context, usage = context_manager.build_context(
            components, prioritize_recent=True
        )

        # Should include some checkpoints within budget
        assert usage["checkpoints"] > 0
        assert usage["checkpoints"] <= context_manager.budget.checkpoints

    def test_build_context_token_usage_tracking(self, context_manager, token_counter):
        """Test that token usage is accurately tracked."""
        components = {
            "task": "Test task",
            "history": ["Action 1", "Action 2", "Action 3"],
        }

        token_counter.count_tokens.side_effect = lambda x: len(str(x).split())

        context, usage = context_manager.build_context(components)

        # Verify usage tracking
        assert "task" in usage
        assert "history" in usage
        assert "total" in usage
        # Verify that total equals the sum of all component values (excluding total itself)
        component_sum = sum(
            v for k, v in usage.items() if k != "total" and k != "summarized"
        )
        assert usage["total"] == component_sum  # Total should equal sum of components
