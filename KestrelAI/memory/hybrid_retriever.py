from __future__ import annotations

"""
Hybrid Retrieval System
Combines vector-based semantic search with BM25 keyword search for improved retrieval quality.
"""

import logging
import os
from typing import Any, Optional

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    logging.warning("rank-bm25 not installed. BM25 keyword search will be disabled.")

from .vector_store import MemoryStore

try:
    from langchain.retrievers import EnsembleRetriever
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError:  # pragma: no cover - dependency-gated path
    EnsembleRetriever = None
    BaseRetriever = None
    Document = None

logger = logging.getLogger(__name__)


if BaseRetriever is not None and Document is not None:

    class _VectorLangChainRetriever(BaseRetriever):
        """LangChain wrapper around HybridRetriever vector retrieval."""

        hybrid: HybridRetriever
        k: int = 10
        task_name: Optional[str] = None

        def _get_relevant_documents(self, query: str) -> list[Document]:
            results = self.hybrid._vector_search(query, k=self.k)
            if self.task_name:
                results = [
                    r
                    for r in results
                    if r.get("metadata", {}).get("task") == self.task_name
                ]

            docs: list[Document] = []
            for result in results:
                metadata = dict(result.get("metadata", {}))
                metadata["doc_id"] = result.get("doc_id")
                metadata["vector_score"] = result.get("score", 0.0)
                metadata["method"] = "vector"
                docs.append(
                    Document(
                        page_content=result.get("content", ""),
                        metadata=metadata,
                    )
                )
            return docs

        async def _aget_relevant_documents(self, query: str) -> list[Document]:
            return self._get_relevant_documents(query)

    class _BM25LangChainRetriever(BaseRetriever):
        """LangChain wrapper around HybridRetriever BM25 retrieval."""

        hybrid: HybridRetriever
        k: int = 10
        task_name: Optional[str] = None

        def _get_relevant_documents(self, query: str) -> list[Document]:
            results = self.hybrid._bm25_search(
                query,
                k=self.k,
                task_name=self.task_name,
            )
            docs: list[Document] = []
            for result in results:
                metadata = dict(result.get("metadata", {}))
                metadata["doc_id"] = result.get("doc_id")
                metadata["bm25_score"] = result.get("score", 0.0)
                metadata["method"] = "bm25"
                docs.append(
                    Document(
                        page_content=result.get("content", ""),
                        metadata=metadata,
                    )
                )
            return docs

        async def _aget_relevant_documents(self, query: str) -> list[Document]:
            return self._get_relevant_documents(query)

else:  # pragma: no cover - dependency-gated path
    _VectorLangChainRetriever = None
    _BM25LangChainRetriever = None


class HybridRetriever:
    """
    Hybrid retrieval system combining:
    1. Vector search (semantic similarity)
    2. BM25 keyword search (exact matches)
    3. Fusion and reranking
    """

    def __init__(self, memory_store: MemoryStore, enable_bm25: bool = True):
        """
        Initialize hybrid retriever.

        Args:
            memory_store: The vector store for semantic search
            enable_bm25: Whether to enable BM25 keyword search
        """
        self.memory_store = memory_store
        self.enable_bm25 = enable_bm25 and BM25Okapi is not None

        # BM25 index (built lazily)
        self.bm25_index: BM25Okapi | None = None
        self.bm25_doc_ids: list[str] = []
        self.bm25_documents: list[str] = []
        self.bm25_metadatas: list[dict[str, Any]] = []
        self._bm25_initialized = False

        # Fusion weights
        self.vector_weight = 0.6  # Weight for vector search results
        self.bm25_weight = 0.4  # Weight for BM25 results
        self.langchain_retrieval_enabled = bool(
            EnsembleRetriever is not None
            and _VectorLangChainRetriever is not None
            and _BM25LangChainRetriever is not None
        )

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25"""
        # Basic tokenization: lowercase, split on whitespace and punctuation
        import re

        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def _build_bm25_index(self):
        """Build BM25 index from all documents in the collection"""
        if not self.enable_bm25 or self._bm25_initialized:
            return

        try:
            # Get all documents from ChromaDB
            # Note: ChromaDB doesn't have a direct "get all" method, so we'll build incrementally
            # For now, we'll rebuild on each search if needed
            # In production, you might want to maintain this index separately

            # Try to get all documents (this is a limitation - ChromaDB doesn't expose this easily)
            # We'll build the index lazily during searches
            self._bm25_initialized = True
            logger.debug("BM25 index will be built lazily during searches")
        except Exception as e:
            logger.warning(f"Failed to initialize BM25 index: {e}")
            self.enable_bm25 = False

    def _get_all_documents_for_bm25(
        self, task_name: str | None = None
    ) -> tuple[list[str], list[str], list[dict]]:
        """
        Get documents for BM25 indexing.
        Prefer direct collection.get() to avoid expensive broad vector queries.
        """
        try:
            max_docs = max(50, int(os.getenv("HYBRID_BM25_MAX_DOCS", "500")))
            collection = getattr(self.memory_store, "collection", None)

            documents: list[str] = []
            doc_ids: list[str] = []
            metadatas: list[dict[str, Any]] = []

            if collection is not None and hasattr(collection, "get"):
                payload = collection.get(include=["documents", "metadatas"])
                doc_ids = list(payload.get("ids", []) or [])
                documents = list(payload.get("documents", []) or [])
                metadatas = list(payload.get("metadatas", []) or [])
                if documents and len(metadatas) < len(documents):
                    metadatas.extend({} for _ in range(len(documents) - len(metadatas)))
            if not documents:
                # Fallback for simplified collection implementations.
                all_docs_map: dict[str, tuple[str, dict[str, Any]]] = {}
                queries = [
                    "research information data",
                    "document text content",
                    "checkpoint summary report",
                    "findings results analysis",
                ]
                for query in queries:
                    if len(all_docs_map) >= max_docs:
                        break
                    try:
                        results = self.memory_store.search(query, k=max_docs)
                    except Exception as e:
                        logger.debug("BM25 fallback query failed (%s): %s", query, e)
                        continue
                    if not results or not results.get("documents"):
                        continue
                    query_ids = results.get("ids", [])[0] if results.get("ids") else []
                    query_docs = results["documents"][0]
                    query_meta = (
                        results["metadatas"][0]
                        if results.get("metadatas")
                        else [{}] * len(query_docs)
                    )
                    for doc_id, doc, meta in zip(query_ids, query_docs, query_meta):
                        normalized_id = (
                            str(doc_id) if doc_id else f"doc_{len(all_docs_map)}"
                        )
                        all_docs_map.setdefault(
                            normalized_id,
                            (str(doc), dict(meta or {})),
                        )
                        if len(all_docs_map) >= max_docs:
                            break
                doc_ids = list(all_docs_map.keys())
                documents = [item[0] for item in all_docs_map.values()]
                metadatas = [item[1] for item in all_docs_map.values()]

            if not documents:
                return [], [], []

            if len(documents) > max_docs:
                doc_ids = doc_ids[:max_docs]
                documents = documents[:max_docs]
                metadatas = metadatas[:max_docs]

            if len(doc_ids) < len(documents):
                doc_ids.extend(f"doc_{i}" for i in range(len(doc_ids), len(documents)))

            if task_name:
                filtered_documents: list[str] = []
                filtered_ids: list[str] = []
                filtered_metas: list[dict[str, Any]] = []
                for doc_id, doc, meta in zip(doc_ids, documents, metadatas):
                    if meta.get("task") == task_name:
                        filtered_documents.append(doc)
                        filtered_ids.append(doc_id)
                        filtered_metas.append(meta)
                return filtered_documents, filtered_ids, filtered_metas

            return documents, doc_ids, metadatas
        except Exception as e:
            logger.warning(f"Error getting documents for BM25: {e}")
            return [], [], []

    def _build_bm25_index_lazy(self, task_name: str | None = None):
        """Build BM25 index lazily from current documents"""
        if not self.enable_bm25:
            return

        try:
            documents, doc_ids, metadatas = self._get_all_documents_for_bm25(task_name)

            if not documents:
                self.bm25_index = None
                return

            # Tokenize documents
            tokenized_docs = [self._tokenize(doc) for doc in documents]

            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.bm25_documents = documents
            self.bm25_doc_ids = doc_ids
            self.bm25_metadatas = metadatas

            logger.debug(f"Built BM25 index with {len(documents)} documents")
        except Exception as e:
            logger.warning(f"Error building BM25 index: {e}")
            self.enable_bm25 = False

    def _bm25_search(
        self, query: str, k: int = 10, task_name: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Perform BM25 keyword search.

        Returns:
            List of result dictionaries with 'content', 'metadata', 'score', 'doc_id'
        """
        if not self.enable_bm25:
            return []

        # Build index if needed
        if self.bm25_index is None:
            self._build_bm25_index_lazy(task_name)

        if self.bm25_index is None or not self.bm25_documents:
            return []

        try:
            # Tokenize query
            query_tokens = self._tokenize(query)

            if not query_tokens:
                return []

            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)

            # Get top k results
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:k]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include results with positive scores
                    results.append(
                        {
                            "content": self.bm25_documents[idx],
                            "metadata": self.bm25_metadatas[idx],
                            "score": float(scores[idx]),
                            "doc_id": (
                                self.bm25_doc_ids[idx]
                                if idx < len(self.bm25_doc_ids)
                                else f"doc_{idx}"
                            ),
                            "method": "bm25",
                        }
                    )

            return results
        except Exception as e:
            logger.warning(f"Error in BM25 search: {e}")
            return []

    def _vector_search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """
        Perform vector-based semantic search.

        Returns:
            List of result dictionaries with 'content', 'metadata', 'score', 'doc_id'
        """
        try:
            results = self.memory_store.search(query, k=k)

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
            distances = (
                results["distances"][0]
                if results.get("distances")
                else [0.0] * len(documents)
            )
            doc_ids = (
                results["ids"][0]
                if results.get("ids")
                else [f"doc_{i}" for i in range(len(documents))]
            )

            # Convert distance to similarity score (1 - normalized distance)
            # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
            # Convert to similarity: similarity = 1 - (distance / 2)
            results_list = []
            for doc, meta, dist, doc_id in zip(
                documents, metadatas, distances, doc_ids
            ):
                similarity = max(0.0, 1.0 - (dist / 2.0))  # Normalize to [0, 1]
                results_list.append(
                    {
                        "content": doc,
                        "metadata": meta,
                        "score": similarity,
                        "doc_id": doc_id,
                        "method": "vector",
                    }
                )

            return results_list
        except Exception as e:
            logger.warning(f"Error in vector search: {e}")
            return []

    def _normalize_scores(
        self, results: list[dict[str, Any]], method: str
    ) -> list[dict[str, Any]]:
        """Normalize scores to [0, 1] range for fusion"""
        if not results:
            return results

        scores = [r["score"] for r in results]
        if not scores:
            return results

        min_score = min(scores)
        max_score = max(scores)

        # Normalize
        if max_score > min_score:
            normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        else:
            normalized = [1.0] * len(scores)

        # Update scores
        for i, result in enumerate(results):
            result["normalized_score"] = normalized[i]

        return results

    def _fuse_results(
        self, vector_results: list[dict], bm25_results: list[dict]
    ) -> list[dict]:
        """
        Fuse results from vector and BM25 searches using reciprocal rank fusion (RRF).

        RRF formula: score = sum(1 / (k + rank)) for each method
        where k is a constant (typically 60) and rank is the position in results
        """
        # Create a map of doc_id -> combined result
        fused_map: dict[str, dict[str, Any]] = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result.get("doc_id", f"vector_{rank}")
            if doc_id not in fused_map:
                fused_map[doc_id] = {
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "doc_id": doc_id,
                    "vector_score": result.get("normalized_score", result["score"]),
                    "bm25_score": 0.0,
                    "vector_rank": rank,
                    "bm25_rank": None,
                }
            else:
                fused_map[doc_id]["vector_score"] = result.get(
                    "normalized_score", result["score"]
                )
                fused_map[doc_id]["vector_rank"] = rank

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result.get("doc_id", f"bm25_{rank}")
            if doc_id not in fused_map:
                fused_map[doc_id] = {
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "doc_id": doc_id,
                    "vector_score": 0.0,
                    "bm25_score": result.get("normalized_score", result["score"]),
                    "vector_rank": None,
                    "bm25_rank": rank,
                }
            else:
                fused_map[doc_id]["bm25_score"] = result.get(
                    "normalized_score", result["score"]
                )
                fused_map[doc_id]["bm25_rank"] = rank

        # Calculate RRF scores
        k = 60  # RRF constant
        fused_results = []
        for doc_id, result in fused_map.items():
            rrf_score = 0.0

            # Add vector contribution
            if result["vector_rank"] is not None:
                rrf_score += 1.0 / (k + result["vector_rank"])

            # Add BM25 contribution
            if result["bm25_rank"] is not None:
                rrf_score += 1.0 / (k + result["bm25_rank"])

            # Weighted combination (alternative to RRF)
            weighted_score = (
                self.vector_weight * result["vector_score"]
                + self.bm25_weight * result["bm25_score"]
            )

            # Use weighted score if both methods found it, otherwise use RRF
            if result["vector_rank"] is not None and result["bm25_rank"] is not None:
                final_score = weighted_score
            else:
                final_score = rrf_score

            result["fused_score"] = final_score
            fused_results.append(result)

        # Sort by fused score
        fused_results.sort(key=lambda x: x["fused_score"], reverse=True)

        return fused_results

    def retrieve(
        self,
        query: str,
        k: int = 10,
        task_name: str | None = None,
        use_hybrid: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid retrieval combining vector and BM25 search.

        Args:
            query: Search query
            k: Number of results to return
            task_name: Optional task name to filter results
            use_hybrid: Whether to use hybrid retrieval (True) or just vector search (False)

        Returns:
            List of result dictionaries with 'content', 'metadata', 'score', 'doc_id', 'fused_score'
        """
        if not use_hybrid or not self.enable_bm25:
            # Fall back to vector search only
            results = self._vector_search(query, k=k)
            # Filter by task if specified
            if task_name:
                results = [r for r in results if r["metadata"].get("task") == task_name]
            return results

        if self.langchain_retrieval_enabled:
            langchain_results = self._retrieve_with_langchain_ensemble(
                query=query,
                k=k,
                task_name=task_name,
            )
            if langchain_results:
                return langchain_results

        # Perform both searches
        # Get more results from each method to have better fusion
        vector_results = self._vector_search(query, k=k * 2)
        bm25_results = self._bm25_search(query, k=k * 2, task_name=task_name)

        # Filter vector results by task if specified
        if task_name:
            vector_results = [
                r for r in vector_results if r["metadata"].get("task") == task_name
            ]

        # Normalize scores
        vector_results = self._normalize_scores(vector_results, "vector")
        bm25_results = self._normalize_scores(bm25_results, "bm25")

        # Fuse results
        fused_results = self._fuse_results(vector_results, bm25_results)

        # Return top k
        return fused_results[:k]

    def _retrieve_with_langchain_ensemble(
        self, *, query: str, k: int, task_name: str | None
    ) -> list[dict[str, Any]]:
        """Use LangChain EnsembleRetriever while preserving the project return schema."""
        if not self.langchain_retrieval_enabled:
            return []

        try:
            expanded_k = max(k * 2, 10)
            vector_retriever = _VectorLangChainRetriever(
                hybrid=self,
                k=expanded_k,
                task_name=task_name,
            )
            bm25_retriever = _BM25LangChainRetriever(
                hybrid=self,
                k=expanded_k,
                task_name=task_name,
            )
            ensemble = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[self.vector_weight, self.bm25_weight],
                c=60,
            )
            docs = ensemble.invoke(query)
            if not docs:
                return []

            results: list[dict[str, Any]] = []
            for rank, doc in enumerate(docs[:k], start=1):
                metadata = dict(getattr(doc, "metadata", {}) or {})
                if task_name and metadata.get("task") != task_name:
                    continue

                doc_id = metadata.get("doc_id", f"doc_{rank}")
                vector_score = float(metadata.get("vector_score", 0.0))
                bm25_score = float(metadata.get("bm25_score", 0.0))
                fallback_rrf = 1.0 / (60 + rank)
                weighted_score = (self.vector_weight * vector_score) + (
                    self.bm25_weight * bm25_score
                )
                fused_score = max(fallback_rrf, weighted_score)

                results.append(
                    {
                        "content": getattr(doc, "page_content", ""),
                        "metadata": metadata,
                        "doc_id": doc_id,
                        "vector_score": vector_score,
                        "bm25_score": bm25_score,
                        "fused_score": fused_score,
                        "method": "langchain_ensemble",
                    }
                )
            return results
        except Exception as e:
            logger.warning(
                "LangChain ensemble retrieval failed, using built-in fusion: %s",
                e,
            )
            return []

    def invalidate_bm25_index(self):
        """Invalidate BM25 index (call when documents are added/removed)"""
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_doc_ids = []
        self.bm25_metadatas = []
        self._bm25_initialized = False
