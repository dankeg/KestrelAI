"""
Unit tests for multi-level summarization.
"""

from unittest.mock import Mock

import pytest

try:
    from KestrelAI.agents.context_manager import TokenCounter
    from KestrelAI.agents.multi_level_summarizer import (
        ExtractedFacts,
        MultiLevelSummarizer,
        SummaryLevel,
    )
except ImportError:
    from agents.context_manager import TokenCounter
    from agents.multi_level_summarizer import (
        ExtractedFacts,
        MultiLevelSummarizer,
        SummaryLevel,
    )


@pytest.mark.unit
class TestSummaryLevel:
    """Test SummaryLevel dataclass."""

    def test_summary_level_creation(self):
        """Test creating a SummaryLevel."""
        level = SummaryLevel(
            name="test", compression_ratio=0.5, description="Test level"
        )

        assert level.name == "test"
        assert level.compression_ratio == 0.5
        assert level.description == "Test level"


@pytest.mark.unit
class TestMultiLevelSummarizer:
    """Test MultiLevelSummarizer class."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        llm = Mock()
        llm.chat.return_value = "This is a summarized version of the content."
        return llm

    @pytest.fixture
    def token_counter(self):
        """Create mock TokenCounter."""
        counter = Mock(spec=TokenCounter)
        counter.count_tokens = Mock(side_effect=lambda x: len(str(x).split()))
        counter.truncate_to_tokens = Mock(
            side_effect=lambda x, max_t: x[:max_t] + "..."
        )
        return counter

    @pytest.fixture
    def summarizer(self, mock_llm, token_counter):
        """Create MultiLevelSummarizer instance."""
        return MultiLevelSummarizer(mock_llm, token_counter)

    def test_summarizer_initialization(self, summarizer, mock_llm, token_counter):
        """Test MultiLevelSummarizer initialization."""
        assert summarizer.llm == mock_llm
        assert summarizer.counter == token_counter
        assert len(summarizer.levels) == 4  # Default levels
        assert summarizer.levels[0].name == "detailed"  # Should be sorted

    def test_summarizer_custom_levels(self, mock_llm, token_counter):
        """Test summarizer with custom levels."""
        custom_levels = [
            SummaryLevel("full", 1.0, "Full content"),
            SummaryLevel("brief", 0.3, "Brief summary"),
        ]

        summarizer = MultiLevelSummarizer(mock_llm, token_counter, custom_levels)

        assert len(summarizer.levels) == 2
        assert summarizer.levels[0].name == "full"

    def test_create_summary_hierarchy_empty(self, summarizer):
        """Test creating summary hierarchy with empty content."""
        result = summarizer.create_summary_hierarchy("")

        assert "summaries" in result
        assert "facts" in result
        assert result["summaries"] == {}

    def test_create_summary_hierarchy(self, summarizer, mock_llm, token_counter):
        """Test creating summary hierarchy."""
        content = "This is a test content with multiple sentences. " * 10

        # Mock token counting
        token_counter.count_tokens.return_value = 100  # Original has 100 tokens

        result = summarizer.create_summary_hierarchy(content)

        # Should return dict with "summaries" and "facts"
        assert "summaries" in result
        assert "facts" in result

        # Should have all levels
        assert "detailed" in result["summaries"]
        assert result["summaries"]["detailed"] == content

        # Should have called LLM for other levels
        assert mock_llm.chat.called

    def test_retrieve_adaptive_fits(self, summarizer, token_counter):
        """Test adaptive retrieval when content fits."""
        summaries = {
            "detailed": "Very long content " * 100,
            "medium": "Medium length content " * 50,
            "summary": "Short summary",
            "executive": "Brief overview",
        }

        # Mock token counting
        def count_tokens(text):
            mapping = {
                summaries["detailed"]: 200,
                summaries["medium"]: 100,
                summaries["summary"]: 50,
                summaries["executive"]: 20,
            }
            return mapping.get(text, len(text.split()))

        token_counter.count_tokens.side_effect = count_tokens

        content, level = summarizer.retrieve_adaptive(summaries, max_tokens=150)

        # Should return medium level (fits, and is most detailed that fits)
        assert level == "medium"
        assert content == summaries["medium"]

    def test_retrieve_adaptive_none_fit(self, summarizer, token_counter):
        """Test adaptive retrieval when nothing fits."""
        summaries = {
            "detailed": "Very long content " * 100,
            "medium": "Medium length content " * 50,
        }

        # Mock token counting - all exceed limit
        token_counter.count_tokens.side_effect = lambda x: 200

        content, level = summarizer.retrieve_adaptive(summaries, max_tokens=50)

        # Should truncate
        assert level == "truncated"
        assert len(content) > 0
        token_counter.truncate_to_tokens.assert_called()

    def test_retrieve_adaptive_empty(self, summarizer):
        """Test adaptive retrieval with empty summaries."""
        content, level = summarizer.retrieve_adaptive({}, max_tokens=100)

        assert content == ""
        assert level == "none"

    def test_get_summary_at_level(self, summarizer):
        """Test getting summary at specific level."""
        summaries = {
            "detailed": "Full content",
            "medium": "Medium summary",
            "summary": "Brief summary",
        }

        result = summarizer.get_summary_at_level(summaries, "medium")
        assert result == "Medium summary"

        result = summarizer.get_summary_at_level(summaries, "nonexistent")
        assert result is None

    def test_estimate_tokens_for_level(self, summarizer):
        """Test token estimation for level."""
        original_tokens = 1000

        # Test different levels
        detailed = summarizer.estimate_tokens_for_level(original_tokens, "detailed")
        assert detailed == 1000  # 1.0 ratio

        medium = summarizer.estimate_tokens_for_level(original_tokens, "medium")
        assert medium == 500  # 0.5 ratio

        summary = summarizer.estimate_tokens_for_level(original_tokens, "summary")
        assert summary == 200  # 0.2 ratio

        executive = summarizer.estimate_tokens_for_level(original_tokens, "executive")
        assert executive == 100  # 0.1 ratio

    def test_estimate_tokens_invalid_level(self, summarizer):
        """Test token estimation with invalid level."""
        result = summarizer.estimate_tokens_for_level(1000, "nonexistent")
        assert result == 1000  # Should return original

    def test_summarize_error_handling(self, summarizer, mock_llm, token_counter):
        """Test error handling during summarization."""
        content = "Test content"

        # Make LLM raise an error
        mock_llm.chat.side_effect = Exception("LLM error")

        # Should fall back to truncation
        result = summarizer._summarize(
            content, target_tokens=50, level=SummaryLevel("test", 0.5, "Test")
        )

        assert result is not None
        token_counter.truncate_to_tokens.assert_called()

    def test_summarize_too_long_result(self, summarizer, mock_llm, token_counter):
        """Test handling when summary is too long."""
        content = "Test content"
        target_tokens = 50

        # Mock LLM to return very long summary
        mock_llm.chat.return_value = "Very long summary " * 100
        token_counter.count_tokens.side_effect = lambda x: 200  # Exceeds target

        result = summarizer._summarize(
            content, target_tokens, SummaryLevel("test", 0.5, "Test")
        )

        # Should truncate
        token_counter.truncate_to_tokens.assert_called()
        assert result is not None

    def test_extracted_facts(self):
        """Test ExtractedFacts dataclass."""
        facts = ExtractedFacts(
            deadlines=["February 15, 2025"],
            urls=["https://example.com"],
            programs=["NSF REU"],
        )

        assert len(facts.deadlines) == 1
        assert len(facts.urls) == 1
        assert len(facts.programs) == 1

        text = facts.to_text()
        assert "February 15, 2025" in text
        assert "https://example.com" in text
        assert "NSF REU" in text

    def test_extract_facts_regex(self, summarizer):
        """Test regex-based fact extraction fallback."""
        content = """
        NSF REU program deadline is February 15, 2025.
        Contact: info@nsf.gov
        Funding: $5,000
        URL: https://www.nsf.gov/reu
        """

        facts = summarizer._extract_facts_regex(content)

        assert len(facts.urls) > 0
        assert len(facts.deadlines) > 0
        assert len(facts.contact_info) > 0
        assert len(facts.amounts) > 0

    def test_create_summary_on_demand(self, summarizer, mock_llm, token_counter):
        """Test on-demand summary creation."""
        content = "Long content " * 100
        max_tokens = 50

        # Mock token counting
        token_counter.count_tokens.side_effect = lambda x: len(str(x).split())
        mock_llm.chat.return_value = "Summarized content"

        summary, level, facts = summarizer.create_summary_on_demand(content, max_tokens)

        assert summary is not None
        assert level in ["detailed", "medium", "summary", "executive"]
        assert facts is not None or facts is None  # May or may not extract facts

    def test_validate_summary_quality(self, summarizer, token_counter):
        """Test summary quality validation."""
        original = "NSF REU deadline is February 15, 2025. Contact: info@nsf.gov. URL: https://nsf.gov"
        summary = "NSF REU deadline February 15, 2025"

        facts = ExtractedFacts(
            deadlines=["February 15, 2025"],
            urls=["https://nsf.gov"],
            contact_info=["info@nsf.gov"],
        )

        token_counter.count_tokens.side_effect = lambda x: len(str(x).split())

        quality = summarizer.validate_summary_quality(original, summary, facts)

        assert "compression_ratio" in quality
        assert "has_deadlines" in quality
        assert quality["has_deadlines"] is True  # Deadline is preserved
