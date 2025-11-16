"""
Standalone unit tests for hybrid retrieval system.
Tests that don't require ChromaDB/NumPy dependencies.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock ChromaDB and sentence-transformers before any imports
sys.modules["chromadb"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

# Now import our modules
try:
    from KestrelAI.memory.hybrid_retriever import HybridRetriever
except ImportError:
    from memory.hybrid_retriever import HybridRetriever


@pytest.mark.unit
class TestHybridRetrieverStandalone:
    """Standalone tests for HybridRetriever that don't require real dependencies."""

    def test_tokenize(self):
        """Test tokenization function."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store, enable_bm25=False)

        text = "NSF REU program deadline February 15th"
        tokens = retriever._tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "nsf" in tokens
        assert "reu" in tokens
        assert "deadline" in tokens
        assert "february" in tokens

    def test_tokenize_empty(self):
        """Test tokenization with empty text."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store, enable_bm25=False)

        tokens = retriever._tokenize("")
        assert tokens == []

    def test_tokenize_special_chars(self):
        """Test tokenization with special characters."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store, enable_bm25=False)

        text = "NSF-REU program (deadline: 02/15/2024)"
        tokens = retriever._tokenize(text)

        assert "nsf" in tokens
        assert "reu" in tokens
        assert "deadline" in tokens

    def test_normalize_scores(self):
        """Test score normalization."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store, enable_bm25=False)

        results = [
            {"score": 0.5, "content": "doc1"},
            {"score": 1.0, "content": "doc2"},
            {"score": 0.0, "content": "doc3"},
        ]

        normalized = retriever._normalize_scores(results, "test")

        assert len(normalized) == 3
        assert all("normalized_score" in r for r in normalized)
        assert all(0.0 <= r["normalized_score"] <= 1.0 for r in normalized)
        # Highest score should be normalized to 1.0
        assert max(r["normalized_score"] for r in normalized) == 1.0
        # Lowest score should be normalized to 0.0
        assert min(r["normalized_score"] for r in normalized) == 0.0

    def test_normalize_scores_single_result(self):
        """Test score normalization with single result."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store, enable_bm25=False)

        results = [{"score": 0.5, "content": "doc1"}]

        normalized = retriever._normalize_scores(results, "test")

        assert len(normalized) == 1
        assert normalized[0]["normalized_score"] == 1.0

    def test_normalize_scores_empty(self):
        """Test score normalization with empty results."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store, enable_bm25=False)

        normalized = retriever._normalize_scores([], "test")

        assert normalized == []

    def test_normalize_scores_same_scores(self):
        """Test normalization when all scores are the same."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store, enable_bm25=False)

        results = [
            {"score": 0.5, "content": "doc1"},
            {"score": 0.5, "content": "doc2"},
            {"score": 0.5, "content": "doc3"},
        ]

        normalized = retriever._normalize_scores(results, "test")

        assert len(normalized) == 3
        # All should be normalized to 1.0 when scores are equal
        assert all(r["normalized_score"] == 1.0 for r in normalized)

    def test_fuse_results(self):
        """Test result fusion."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store, enable_bm25=False)

        vector_results = [
            {
                "content": "NSF REU program information",
                "metadata": {"task": "test"},
                "doc_id": "doc1",
                "normalized_score": 0.9,
                "score": 0.9,
            },
            {
                "content": "NSF GRFP fellowship details",
                "metadata": {"task": "test"},
                "doc_id": "doc2",
                "normalized_score": 0.7,
                "score": 0.7,
            },
        ]

        bm25_results = [
            {
                "content": "NSF REU program information",
                "metadata": {"task": "test"},
                "doc_id": "doc1",
                "normalized_score": 0.8,
                "score": 0.8,
            },
            {
                "content": "Research opportunities overview",
                "metadata": {"task": "test"},
                "doc_id": "doc3",
                "normalized_score": 0.6,
                "score": 0.6,
            },
        ]

        fused = retriever._fuse_results(vector_results, bm25_results)

        assert isinstance(fused, list)
        assert len(fused) == 3  # doc1, doc2, doc3

        # Check structure
        for result in fused:
            assert "content" in result
            assert "metadata" in result
            assert "doc_id" in result
            assert "fused_score" in result
            assert "vector_score" in result
            assert "bm25_score" in result

        # doc1 should have highest score (found by both methods)
        doc1 = next(r for r in fused if r["doc_id"] == "doc1")
        assert doc1["fused_score"] > 0
        assert doc1["vector_score"] == 0.9
        assert doc1["bm25_score"] == 0.8

    def test_fuse_results_vector_only(self):
        """Test fusion with only vector results."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store, enable_bm25=False)

        vector_results = [
            {
                "content": "Test document",
                "metadata": {"task": "test"},
                "doc_id": "doc1",
                "normalized_score": 0.9,
                "score": 0.9,
            }
        ]

        fused = retriever._fuse_results(vector_results, [])

        assert len(fused) == 1
        assert fused[0]["doc_id"] == "doc1"
        assert fused[0]["fused_score"] > 0
        assert fused[0]["vector_score"] == 0.9
        assert fused[0]["bm25_score"] == 0.0

    def test_fuse_results_bm25_only(self):
        """Test fusion with only BM25 results."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store, enable_bm25=False)

        bm25_results = [
            {
                "content": "Test document",
                "metadata": {"task": "test"},
                "doc_id": "doc1",
                "normalized_score": 0.8,
                "score": 0.8,
            }
        ]

        fused = retriever._fuse_results([], bm25_results)

        assert len(fused) == 1
        assert fused[0]["doc_id"] == "doc1"
        assert fused[0]["fused_score"] > 0
        assert fused[0]["vector_score"] == 0.0
        assert fused[0]["bm25_score"] == 0.8

    def test_fuse_results_empty(self):
        """Test fusion with empty results."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store, enable_bm25=False)

        fused = retriever._fuse_results([], [])

        assert fused == []

    def test_invalidate_bm25_index(self):
        """Test BM25 index invalidation."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store, enable_bm25=False)

        # Set some values
        retriever.bm25_index = Mock()
        retriever.bm25_documents = ["doc1", "doc2"]
        retriever.bm25_doc_ids = ["id1", "id2"]
        retriever.bm25_metadatas = [{"meta": 1}, {"meta": 2}]
        retriever._bm25_initialized = True

        # Invalidate
        retriever.invalidate_bm25_index()

        assert retriever.bm25_index is None
        assert retriever.bm25_documents == []
        assert retriever.bm25_doc_ids == []
        assert retriever.bm25_metadatas == []
        assert retriever._bm25_initialized is False

    def test_initialization_with_bm25(self):
        """Test initialization with BM25 enabled."""
        mock_store = Mock()
        # Mock BM25Okapi to be available
        with patch("KestrelAI.memory.hybrid_retriever.BM25Okapi", Mock()):
            retriever = HybridRetriever(mock_store, enable_bm25=True)

            assert retriever.memory_store == mock_store
            # enable_bm25 will be True if BM25Okapi is available, False otherwise
            # This test verifies the initialization doesn't crash
            assert retriever.vector_weight == 0.6
            assert retriever.bm25_weight == 0.4

    def test_initialization_without_bm25(self):
        """Test initialization with BM25 disabled."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store, enable_bm25=False)

        assert retriever.enable_bm25 is False

    def test_initialization_defaults(self):
        """Test default initialization values."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store)

        assert retriever.memory_store == mock_store
        assert retriever.vector_weight == 0.6
        assert retriever.bm25_weight == 0.4
        assert retriever.bm25_index is None
        assert retriever._bm25_initialized is False
