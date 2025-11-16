# Integration tests for MemoryStore functionality
import tempfile
import time
from unittest.mock import Mock, patch

import pytest

try:
    from KestrelAI.memory.vector_store import MemoryStore
except ImportError:
    from memory.vector_store import MemoryStore


@pytest.mark.integration
class TestMemoryStoreIntegration:
    """Test MemoryStore integration functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_memory_store_initialization(self, temp_dir):
        """Test MemoryStore initialization with mocked model for speed."""
        # Mock the SentenceTransformer to avoid model download
        with patch("sentence_transformers.SentenceTransformer") as mock_transformer:
            mock_model = Mock()
            mock_model.encode.return_value = [
                [0.1, 0.2, 0.3, 0.4, 0.5]
            ]  # Mock embedding
            mock_transformer.return_value = mock_model

            memory_store = MemoryStore(path=temp_dir)

            assert memory_store.client is not None
            assert memory_store.collection is not None
            assert memory_store.model is not None
            assert memory_store.collection.name == "research_mem"

    def test_memory_store_add_document(self, temp_dir):
        """Test adding documents to MemoryStore."""
        with patch("sentence_transformers.SentenceTransformer") as mock_transformer:
            mock_model = Mock()
            mock_model.encode.return_value = [
                [0.1, 0.2, 0.3, 0.4, 0.5]
            ]  # Mock embedding
            mock_transformer.return_value = mock_model

            memory_store = MemoryStore(path=temp_dir)

            # Add a test document
            doc_id = "test_doc_1"
            text = "This is a test document for memory store testing."
            metadata = {"source": "test", "type": "example"}

            memory_store.add(doc_id, text, metadata)

            # Verify the document was added by searching for it
            results = memory_store.search(text, k=1)

            assert results is not None
            assert len(results["ids"][0]) >= 1  # Should find at least one document
            assert doc_id in results["ids"][0]  # Should find our test document

    def test_memory_store_search(self, temp_dir):
        """Test searching in MemoryStore."""
        with patch("sentence_transformers.SentenceTransformer") as mock_transformer:
            mock_model = Mock()
            mock_model.encode.return_value = [
                [0.1, 0.2, 0.3, 0.4, 0.5]
            ]  # Mock embedding
            mock_transformer.return_value = mock_model

            memory_store = MemoryStore(path=temp_dir)

            # Add multiple test documents
            documents = [
                (
                    "doc1",
                    "Machine learning is a subset of artificial intelligence.",
                    {"topic": "ML"},
                ),
                (
                    "doc2",
                    "Natural language processing helps computers understand text.",
                    {"topic": "NLP"},
                ),
                (
                    "doc3",
                    "Computer vision enables machines to interpret visual information.",
                    {"topic": "CV"},
                ),
            ]

            for doc_id, text, metadata in documents:
                memory_store.add(doc_id, text, metadata)

            # Search for machine learning related content
            results = memory_store.search(
                "artificial intelligence machine learning", k=2
            )

            assert results is not None
            assert len(results["ids"][0]) >= 1  # Should find at least one document
            assert len(results["distances"][0]) >= 1  # Should have similarity scores

    def test_memory_store_delete_all(self, temp_dir):
        """Test deleting all documents from MemoryStore."""
        with patch("sentence_transformers.SentenceTransformer") as mock_transformer:
            mock_model = Mock()
            mock_model.encode.return_value = [
                [0.1, 0.2, 0.3, 0.4, 0.5]
            ]  # Mock embedding
            mock_transformer.return_value = mock_model

            memory_store = MemoryStore(path=temp_dir)

            # Add a test document
            memory_store.add("test_doc", "Test content", {"test": True})

            # Verify document exists
            results = memory_store.search("test content", k=1)
            assert len(results["ids"][0]) >= 1

            # Delete all documents
            memory_store.delete_all()

            # Verify document is deleted
            results = memory_store.search("test content", k=1)
            assert len(results["ids"][0]) == 0  # Should find no documents

    def test_memory_store_persistence(self, temp_dir):
        """Test that MemoryStore persists data across instances."""
        # Skip this test if using in-memory mode (which is forced in test environments)
        # In-memory mode doesn't persist across instances by design
        import os

        if os.getenv("TESTING") or os.getenv("PYTEST_CURRENT_TEST"):
            pytest.skip(
                "Persistence test skipped in test mode (uses in-memory storage)"
            )

        with patch("sentence_transformers.SentenceTransformer") as mock_transformer:
            mock_model = Mock()
            mock_model.encode.return_value = [
                [0.1, 0.2, 0.3, 0.4, 0.5]
            ]  # Mock embedding
            mock_transformer.return_value = mock_model

            # Create first instance and add document
            memory_store1 = MemoryStore(path=temp_dir)
            memory_store1.add(
                "persistent_doc", "This should persist", {"persistent": True}
            )

            # Create second instance (should load existing data)
            memory_store2 = MemoryStore(path=temp_dir)

            # Search for the document in the second instance
            results = memory_store2.search("This should persist", k=1)

            assert results is not None
            assert len(results["ids"][0]) >= 1
            assert "persistent_doc" in results["ids"][0]

    def test_memory_store_performance(self, temp_dir):
        """Test MemoryStore performance with multiple documents."""
        with patch("sentence_transformers.SentenceTransformer") as mock_transformer:
            mock_model = Mock()
            mock_model.encode.return_value = [
                [0.1, 0.2, 0.3, 0.4, 0.5]
            ]  # Mock embedding
            mock_transformer.return_value = mock_model

            memory_store = MemoryStore(path=temp_dir)

            # Add multiple documents
            start_time = time.time()
            for i in range(10):
                doc_id = f"perf_doc_{i}"
                text = f"Performance test document number {i} with some content."
                metadata = {"index": i, "test": "performance"}
                memory_store.add(doc_id, text, metadata)

            add_time = time.time() - start_time

            # Search performance
            start_time = time.time()
            results = memory_store.search("performance test", k=5)
            search_time = time.time() - start_time

            # Performance assertions (should be reasonable for 10 documents)
            assert (
                add_time < 10.0
            )  # Adding 10 documents should take less than 10 seconds
            assert search_time < 5.0  # Search should take less than 5 seconds
            assert len(results["ids"][0]) >= 1  # Should find at least one document
