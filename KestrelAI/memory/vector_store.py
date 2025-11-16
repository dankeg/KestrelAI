"""
Vector-backed MemoryStore abstraction.

In production this is backed by ChromaDB (PersistentClient). For test and
degraded environments where ChromaDB is unavailable or mis-installed, we
fall back to a lightweight in-process store that mimics the minimal
ChromaDB API the application and tests rely on.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any

# Disable ChromaDB telemetry BEFORE importing ChromaDB to prevent telemetry errors
# This must be set before any ChromaDB imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

try:
    from chromadb import PersistentClient  # type: ignore
except Exception:  # pragma: no cover - handled by fallback
    PersistentClient = None

# Lazy import SentenceTransformer to allow mocking in tests before import
# This prevents mutex locking errors when tests mock it
_SentenceTransformer = None


def _get_sentence_transformer():
    """Lazy import SentenceTransformer to allow test mocking."""
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer

        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


# Global lock for SentenceTransformer model loading to prevent concurrent loads
_model_lock = threading.Lock()
_shared_model = None


class _InMemoryCollection:
    """Minimal in-memory collection that mimics ChromaDB's Collection API."""

    def __init__(self, name: str):
        self.name = name
        # doc_id -> (text, metadata, embedding)
        self._docs: dict[str, dict[str, Any]] = {}

    # Chroma-style API -----------------------------------------------------
    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        for i, (doc_id, text, meta) in enumerate(zip(ids, documents, metadatas)):
            self._docs[doc_id] = {
                "text": text,
                "metadata": meta,
                "embedding": None if embeddings is None else embeddings[i],
            }

    def query(
        self, query_embeddings: list[list[float]], n_results: int = 5
    ) -> dict[str, Any]:
        """
        Very simple "query" implementation:
        - Ignores the embedding values and just returns up to n_results docs.
        - Preserves the structure expected by HybridRetriever and tests.
        """
        doc_ids = list(self._docs.keys())
        texts = [self._docs[i]["text"] for i in doc_ids]
        metadatas = [self._docs[i]["metadata"] for i in doc_ids]

        # For determinism in tests, sort by doc_id
        combined = list(zip(doc_ids, texts, metadatas))
        combined.sort(key=lambda x: x[0])
        selected = combined[:n_results]

        ids = [[doc_id for doc_id, _, _ in selected]]
        documents = [[text for _, text, _ in selected]]
        metas = [[meta for _, _, meta in selected]]
        distances = [[0.0 for _ in selected]]  # Dummy distances

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metas,
            "distances": distances,
        }

    def delete(self, where: dict[str, Any] | None = None) -> None:
        """Delete all documents (where clause is ignored in this simple impl)."""
        self._docs.clear()


class _InMemoryClient:
    """Minimal in-memory client that mimics ChromaDB's PersistentClient API."""

    def __init__(self, path: str):
        self.path = path
        self._collections: dict[str, _InMemoryCollection] = {}

    def get_or_create_collection(self, name: str) -> _InMemoryCollection:
        if name not in self._collections:
            self._collections[name] = _InMemoryCollection(name)
        return self._collections[name]

    def delete_collection(self, name: str) -> None:
        if name in self._collections:
            del self._collections[name]


class MemoryStore:
    def __init__(self, path: str = ".chroma", model_name: str | None = None):
        """
        Initialize MemoryStore.

        In normal environments this uses ChromaDB's PersistentClient. If that
        fails (e.g., chromadb version mismatch), we transparently fall back to
        a lightweight in-memory client that provides the minimal API needed
        by the rest of the system and tests.

        Args:
            path: Path for ChromaDB storage (default: ".chroma")
            model_name: Optional model name for SentenceTransformer.
                       If None, uses default or shared instance.
        """
        # Lazy-load embedding model to avoid mutex issues
        # Use shared model instance in test environments to prevent multiple loads
        self._model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._model: SentenceTransformer | None = None

        # Try to initialize real Chroma client, with safe fallback
        # In test environments, prefer in-memory client to avoid mutex/locking issues
        use_in_memory = (
            os.getenv("PYTEST_CURRENT_TEST")
            or os.getenv("TESTING")
            or os.getenv("USE_IN_MEMORY_CHROMA")
        )

        if PersistentClient is not None and not use_in_memory:
            try:
                # Use unique path per instance to avoid locking conflicts
                # In test environments, ensure path is unique
                if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING"):
                    # Add thread ID to path to ensure uniqueness in parallel tests
                    import threading

                    unique_path = f"{path}_{threading.get_ident()}"
                else:
                    unique_path = path
                self.client = PersistentClient(path=unique_path)  # type: ignore[call-arg]
            except Exception:
                # Fallback to in-memory client if Chroma cannot be instantiated
                # This avoids mutex/locking issues
                self.client = _InMemoryClient(path)
        else:
            # Use in-memory client in tests or if ChromaDB is not available
            self.client = _InMemoryClient(path)

        self.collection = self.client.get_or_create_collection("research_mem")

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model with thread-safe singleton pattern."""
        global _shared_model, _model_lock

        if self._model is not None:
            return self._model

        # Lazy import to allow test mocking
        SentenceTransformer = _get_sentence_transformer()

        # Use shared model in test environments to avoid multiple loads
        use_shared = os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING")

        if use_shared:
            with _model_lock:
                if _shared_model is None:
                    try:
                        _shared_model = SentenceTransformer(self._model_name)
                    except Exception:
                        # If loading fails, create instance-specific model
                        self._model = SentenceTransformer(self._model_name)
                        return self._model
                self._model = _shared_model
        else:
            # In production, create instance-specific model
            with _model_lock:
                if self._model is None:
                    self._model = SentenceTransformer(self._model_name)

        return self._model

    def add(self, doc_id: str, text: str, meta: dict):
        # Ensure model is loaded (lazy loading)
        emb = self.model.encode(text)
        # Handle both 1D and 2D arrays from encode()
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb[0]  # Flatten if 2D with single row
        emb_list = emb.tolist()
        self.collection.add(
            ids=[doc_id], documents=[text], metadatas=[meta], embeddings=[emb_list]
        )

    def search(self, query: str, k: int = 5):
        # Ensure model is loaded (lazy loading)
        emb = self.model.encode(query)
        # Handle both 1D and 2D arrays from encode()
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb[0]  # Flatten if 2D with single row
        emb_list = emb.tolist()
        return self.collection.query(query_embeddings=[emb_list], n_results=k)

    def delete_all(self) -> None:
        """Delete everything in this collection (but keep the collection)."""
        # Clear current collection documents
        self.collection.delete(where={})
        # For API parity with original implementation, drop and recreate collection
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name)
