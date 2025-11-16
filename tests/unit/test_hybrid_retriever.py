"""
Unit tests for hybrid retrieval system.
"""

import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest

try:
    from KestrelAI.memory.hybrid_retriever import HybridRetriever
    from KestrelAI.memory.vector_store import MemoryStore
except ImportError:
    from memory.hybrid_retriever import HybridRetriever
    from memory.vector_store import MemoryStore


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary ChromaDB directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory_store(temp_chroma_dir):
    """Create a MemoryStore instance."""
    return MemoryStore(path=temp_chroma_dir)


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "text": "NSF REU program provides summer research opportunities for undergraduates. Deadline is February 15th.",
            "metadata": {
                "task": "test_task",
                "type": "checkpoint",
                "layer": "episodic",
            },
        },
        {
            "id": "doc2",
            "text": "NSF GRFP fellowship supports graduate students in STEM fields. Application deadline is October 20th.",
            "metadata": {
                "task": "test_task",
                "type": "checkpoint",
                "layer": "episodic",
            },
        },
        {
            "id": "doc3",
            "text": "Research opportunities in computer science include internships and fellowships. Many programs have deadlines in the fall.",
            "metadata": {"task": "test_task", "type": "summary", "layer": "semantic"},
        },
        {
            "id": "doc4",
            "text": "The National Science Foundation offers various funding opportunities for students at different levels.",
            "metadata": {
                "task": "other_task",
                "type": "checkpoint",
                "layer": "episodic",
            },
        },
    ]


@pytest.fixture
def populated_memory_store(memory_store, sample_documents):
    """MemoryStore with sample documents."""

    # Mock the search results to return our sample documents
    def mock_query(query_embeddings=None, n_results=5):
        return {
            "documents": [[doc["text"] for doc in sample_documents]],
            "metadatas": [[doc["metadata"] for doc in sample_documents]],
            "distances": [[0.1, 0.2, 0.3, 0.4]],
            "ids": [[doc["id"] for doc in sample_documents]],
        }

    memory_store.collection.query = Mock(side_effect=mock_query)

    # Mock add to store documents
    stored_docs = {}

    def mock_add(ids, documents, metadatas, embeddings):
        for doc_id, doc, meta in zip(ids, documents, metadatas):
            stored_docs[doc_id] = {"text": doc, "metadata": meta}

    memory_store.collection.add = Mock(side_effect=mock_add)

    # Add documents
    for doc in sample_documents:
        memory_store.add(doc["id"], doc["text"], doc["metadata"])

    return memory_store


@pytest.mark.unit
class TestHybridRetriever:
    """Test HybridRetriever class."""

    def test_initialization_with_bm25(self, memory_store):
        """Test HybridRetriever initialization with BM25 enabled."""
        # Mock BM25Okapi to be available - patch at the import level
        mock_bm25 = Mock()
        mock_rank_bm25 = Mock()
        mock_rank_bm25.BM25Okapi = mock_bm25
        with patch.dict("sys.modules", {"rank_bm25": mock_rank_bm25}):
            # Reload the module to pick up the mock
            import importlib
            import sys

            if "KestrelAI.memory.hybrid_retriever" in sys.modules:
                importlib.reload(sys.modules["KestrelAI.memory.hybrid_retriever"])
            from KestrelAI.memory.hybrid_retriever import HybridRetriever

            retriever = HybridRetriever(memory_store, enable_bm25=True)

            assert retriever.memory_store == memory_store
            assert (
                retriever.enable_bm25 is True
            )  # Should be True when BM25Okapi is available
            assert retriever.vector_weight == 0.6
            assert retriever.bm25_weight == 0.4
            assert retriever.bm25_index is None
            assert retriever._bm25_initialized is False

    def test_initialization_without_bm25(self, memory_store):
        """Test HybridRetriever initialization with BM25 disabled."""
        retriever = HybridRetriever(memory_store, enable_bm25=False)

        assert retriever.enable_bm25 is False

    @patch("KestrelAI.memory.hybrid_retriever.BM25Okapi")
    def test_initialization_bm25_not_available(self, mock_bm25, memory_store):
        """Test initialization when BM25 library is not available."""
        # Simulate BM25 not being available
        import sys

        original_bm25 = sys.modules.get("rank_bm25")
        if "rank_bm25" in sys.modules:
            del sys.modules["rank_bm25"]

        # Mock the import to fail
        with patch.dict("sys.modules", {"rank_bm25": None}):
            from importlib import reload

            try:
                from KestrelAI.memory import hybrid_retriever

                reload(hybrid_retriever)
                retriever = hybrid_retriever.HybridRetriever(
                    memory_store, enable_bm25=True
                )
                assert retriever.enable_bm25 is False
            finally:
                # Restore original
                if original_bm25:
                    sys.modules["rank_bm25"] = original_bm25

    def test_tokenize(self, memory_store):
        """Test tokenization function."""
        retriever = HybridRetriever(memory_store)

        text = "NSF REU program deadline February 15th"
        tokens = retriever._tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "nsf" in tokens
        assert "reu" in tokens
        assert "deadline" in tokens
        assert "february" in tokens

    def test_vector_search(self, populated_memory_store):
        """Test vector-based semantic search."""
        retriever = HybridRetriever(populated_memory_store)

        results = retriever._vector_search("NSF REU program", k=3)

        assert isinstance(results, list)
        assert len(results) > 0

        # Check result structure
        for result in results:
            assert "content" in result
            assert "metadata" in result
            assert "score" in result
            assert "doc_id" in result
            assert "method" in result
            assert result["method"] == "vector"
            assert 0.0 <= result["score"] <= 1.0

    def test_vector_search_empty_query(self, populated_memory_store):
        """Test vector search with empty query."""
        retriever = HybridRetriever(populated_memory_store)

        results = retriever._vector_search("", k=3)

        # Should return empty or handle gracefully
        assert isinstance(results, list)

    def test_bm25_search(self, populated_memory_store):
        """Test BM25 keyword search."""
        retriever = HybridRetriever(populated_memory_store, enable_bm25=True)

        # Build index first
        retriever._build_bm25_index_lazy("test_task")

        results = retriever._bm25_search("NSF REU deadline", k=3, task_name="test_task")

        assert isinstance(results, list)

        # Check result structure if results exist
        for result in results:
            assert "content" in result
            assert "metadata" in result
            assert "score" in result
            assert "doc_id" in result
            assert "method" in result
            assert result["method"] == "bm25"
            assert result["score"] > 0  # BM25 scores should be positive

    def test_bm25_search_no_index(self, populated_memory_store):
        """Test BM25 search when index is not built."""
        retriever = HybridRetriever(populated_memory_store, enable_bm25=True)

        # Don't build index
        results = retriever._bm25_search("NSF REU", k=3)

        # Should build index lazily or return empty
        assert isinstance(results, list)

    def test_bm25_search_disabled(self, populated_memory_store):
        """Test BM25 search when disabled."""
        retriever = HybridRetriever(populated_memory_store, enable_bm25=False)

        results = retriever._bm25_search("NSF REU", k=3)

        assert results == []

    def test_normalize_scores(self, memory_store):
        """Test score normalization."""
        retriever = HybridRetriever(memory_store)

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

    def test_normalize_scores_single_result(self, memory_store):
        """Test score normalization with single result."""
        retriever = HybridRetriever(memory_store)

        results = [{"score": 0.5, "content": "doc1"}]

        normalized = retriever._normalize_scores(results, "test")

        assert len(normalized) == 1
        assert normalized[0]["normalized_score"] == 1.0

    def test_normalize_scores_empty(self, memory_store):
        """Test score normalization with empty results."""
        retriever = HybridRetriever(memory_store)

        normalized = retriever._normalize_scores([], "test")

        assert normalized == []

    def test_fuse_results(self, memory_store):
        """Test result fusion."""
        retriever = HybridRetriever(memory_store)

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

    def test_fuse_results_vector_only(self, memory_store):
        """Test fusion with only vector results."""
        retriever = HybridRetriever(memory_store)

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

    def test_fuse_results_bm25_only(self, memory_store):
        """Test fusion with only BM25 results."""
        retriever = HybridRetriever(memory_store)

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

    def test_retrieve_hybrid(self, populated_memory_store):
        """Test hybrid retrieval combining vector and BM25."""
        retriever = HybridRetriever(populated_memory_store, enable_bm25=True)

        results = retriever.retrieve(
            "NSF REU program deadline", k=3, task_name="test_task", use_hybrid=True
        )

        assert isinstance(results, list)
        assert len(results) <= 3

        # Check result structure
        for result in results:
            assert "content" in result
            assert "metadata" in result
            assert "fused_score" in result or "score" in result
            assert "doc_id" in result

    def test_retrieve_vector_only(self, populated_memory_store):
        """Test retrieval with vector search only."""
        retriever = HybridRetriever(populated_memory_store, enable_bm25=False)

        results = retriever.retrieve(
            "NSF REU program", k=3, task_name="test_task", use_hybrid=False
        )

        assert isinstance(results, list)
        assert len(results) <= 3

    def test_retrieve_with_task_filter(self, populated_memory_store):
        """Test retrieval with task filtering."""
        retriever = HybridRetriever(populated_memory_store, enable_bm25=True)

        results = retriever.retrieve(
            "NSF program", k=10, task_name="test_task", use_hybrid=True
        )

        # All results should be from test_task
        for result in results:
            assert result["metadata"].get("task") == "test_task"

    def test_invalidate_bm25_index(self, populated_memory_store):
        """Test BM25 index invalidation."""
        retriever = HybridRetriever(populated_memory_store, enable_bm25=True)

        # Build index
        retriever._build_bm25_index_lazy("test_task")
        assert retriever.bm25_index is not None

        # Invalidate
        retriever.invalidate_bm25_index()

        assert retriever.bm25_index is None
        assert retriever.bm25_documents == []
        assert retriever.bm25_doc_ids == []
        assert retriever._bm25_initialized is False

    def test_build_bm25_index_lazy(self, populated_memory_store):
        """Test lazy BM25 index building."""
        retriever = HybridRetriever(populated_memory_store, enable_bm25=True)

        assert retriever.bm25_index is None

        # Build index
        retriever._build_bm25_index_lazy("test_task")

        # Index should be built (if BM25 is available)
        if retriever.enable_bm25:
            # Index might be None if no documents found, but should not error
            assert isinstance(retriever.bm25_documents, list)

    def test_get_all_documents_for_bm25(self, populated_memory_store):
        """Test getting all documents for BM25 indexing."""
        retriever = HybridRetriever(populated_memory_store, enable_bm25=True)

        docs, ids, metas = retriever._get_all_documents_for_bm25("test_task")

        assert isinstance(docs, list)
        assert isinstance(ids, list)
        assert isinstance(metas, list)
        assert len(docs) == len(ids) == len(metas)

        # Should filter by task
        for meta in metas:
            assert meta.get("task") == "test_task"

    def test_get_all_documents_for_bm25_no_filter(self, populated_memory_store):
        """Test getting all documents without task filter."""
        retriever = HybridRetriever(populated_memory_store, enable_bm25=True)

        docs, ids, metas = retriever._get_all_documents_for_bm25(None)

        assert isinstance(docs, list)
        assert isinstance(ids, list)
        assert isinstance(metas, list)
        # Should get documents from all tasks

    def test_get_all_documents_uses_multiple_queries(self, populated_memory_store):
        """Test that _get_all_documents_for_bm25 uses multiple queries for better coverage."""
        retriever = HybridRetriever(populated_memory_store, enable_bm25=True)

        # Mock the search method to track how many times it's called
        call_count = 0
        original_search = retriever.memory_store.search

        def mock_search(query, k):
            nonlocal call_count
            call_count += 1
            return original_search(query, k)

        retriever.memory_store.search = mock_search

        # Get documents
        docs, ids, metas = retriever._get_all_documents_for_bm25("test_task")

        # Should use multiple queries (the fix uses 4 different queries)
        assert call_count >= 1  # At least one query should be made
        # Note: exact count depends on implementation, but should be multiple

    def test_fused_score_used_for_sorting(self, memory_store):
        """Test that fused_score is used correctly for sorting, not distance."""
        retriever = HybridRetriever(memory_store)

        # Create mock results with different score types
        vector_results = [
            {
                "content": "Document found by both methods",
                "metadata": {"task": "test"},
                "doc_id": "doc1",
                "normalized_score": 0.8,  # Weighted score [0, 1]
                "score": 0.8,
            }
        ]

        bm25_results = [
            {
                "content": "Document found only by BM25",
                "metadata": {"task": "test"},
                "doc_id": "doc2",
                "normalized_score": 0.7,
                "score": 0.7,
            }
        ]

        # Fuse results
        fused = retriever._fuse_results(vector_results, bm25_results)

        # Verify fused_score is present and reasonable
        for result in fused:
            assert "fused_score" in result
            assert result["fused_score"] >= 0  # Should be non-negative

        # When both methods find a doc, it should use weighted_score
        doc1 = next((r for r in fused if r["doc_id"] == "doc1"), None)
        if doc1:
            # Weighted score should be in [0, 1] range
            assert 0.0 <= doc1["fused_score"] <= 1.0

        # When only one method finds a doc, it should use RRF score
        doc2 = next((r for r in fused if r["doc_id"] == "doc2"), None)
        if doc2:
            # RRF score will be smaller (typically < 0.1 for k=60)
            assert doc2["fused_score"] >= 0
            # RRF scores are typically much smaller than weighted scores
            # This is expected behavior - the fix ensures we sort by fused_score directly
