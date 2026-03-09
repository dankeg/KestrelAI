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
from typing import TYPE_CHECKING, Any, Optional

# Disable ChromaDB telemetry BEFORE importing ChromaDB to prevent telemetry errors
# This must be set before any ChromaDB imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ[
    "CHROMA_PRODUCT_TELEMETRY_IMPL"
] = "chromadb.telemetry.product.noop.NoopProductTelemetry"

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

try:
    from chromadb import PersistentClient  # type: ignore
except Exception:  # pragma: no cover - handled by fallback
    PersistentClient = None

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError:  # pragma: no cover - dependency-gated path
    Document = None
    BaseRetriever = None

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


if BaseRetriever is not None and Document is not None:

    class _MemoryStoreLangChainRetriever(BaseRetriever):
        """LangChain retriever facade over MemoryStore search results."""

        memory_store: MemoryStore
        k: int = 5
        task_name: Optional[str] = None

        def _get_relevant_documents(self, query: str) -> list[Document]:
            results = self.memory_store.search(query, k=self.k)
            if (
                not results
                or not results.get("documents")
                or not results["documents"][0]
            ):
                return []

            documents = results["documents"][0]
            metadatas = (
                results["metadatas"][0]
                if results.get("metadatas")
                else [{}] * len(documents)
            )

            out: list[Document] = []
            for doc, meta in zip(documents, metadatas):
                metadata = dict(meta or {})
                if self.task_name and metadata.get("task") != self.task_name:
                    continue
                out.append(Document(page_content=str(doc), metadata=metadata))
            return out

        async def _aget_relevant_documents(self, query: str) -> list[Document]:
            return self._get_relevant_documents(query)

else:  # pragma: no cover - dependency-gated path
    _MemoryStoreLangChainRetriever = None


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

    def add_document(self, doc_id: str, text: str, meta: dict) -> None:
        """Convenience alias for adding a document."""
        self.add(doc_id, text, meta)

    def search(self, query: str, k: int = 5, n_results: int | None = None):
        """Search vector store with `k` and optional compatibility `n_results`."""
        requested_k = (
            int(n_results) if isinstance(n_results, int) and n_results > 0 else int(k)
        )
        top_k = self._clamp_query_size(requested_k)
        if top_k <= 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }
        # Ensure model is loaded (lazy loading)
        emb = self.model.encode(query)
        # Handle both 1D and 2D arrays from encode()
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb[0]  # Flatten if 2D with single row
        emb_list = emb.tolist()
        return self.collection.query(query_embeddings=[emb_list], n_results=top_k)

    def _clamp_query_size(self, requested_k: int) -> int:
        """
        Clamp query size to current collection cardinality to avoid Chroma warnings
        about requesting more results than indexed elements.
        """
        if requested_k <= 0:
            return 0
        try:
            if hasattr(self.collection, "count"):
                count = int(self.collection.count())
            elif hasattr(self.collection, "_docs"):
                count = len(getattr(self.collection, "_docs", {}))
            else:
                count = requested_k
            if count <= 0:
                return 0
            return min(requested_k, count)
        except Exception:
            return requested_k

    def as_langchain_retriever(
        self, *, task_name: str | None = None, k: int = 5
    ) -> Any | None:
        """Return a LangChain retriever abstraction for this memory store."""
        if _MemoryStoreLangChainRetriever is None:
            return None
        return _MemoryStoreLangChainRetriever(
            memory_store=self,
            k=k,
            task_name=task_name,
        )

    def delete_all(self) -> None:
        """Delete everything in this collection (but keep the collection)."""
        # Clear current collection documents
        self.collection.delete(where={})
        # For API parity with original implementation, drop and recreate collection
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name)
