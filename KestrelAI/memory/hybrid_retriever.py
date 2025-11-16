from __future__ import annotations

"""
Hybrid Retrieval System
Combines vector-based semantic search with BM25 keyword search for improved retrieval quality.
"""

import logging
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    logging.warning("rank-bm25 not installed. BM25 keyword search will be disabled.")

from .vector_store import MemoryStore

logger = logging.getLogger(__name__)


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
        Get all documents for BM25 indexing.
        Since ChromaDB doesn't easily expose "get all", we'll use a workaround:
        search with multiple broad queries to get as many documents as possible.

        FIXED: Use multiple queries and higher k value to retrieve more documents.
        This is still not perfect but better than a single query with k=1000.
        """
        try:
            # Use multiple generic queries to try to retrieve all documents
            # This is a limitation - ideally we'd have direct access to all documents
            # Try different query terms to maximize coverage
            all_doc_ids = set()
            all_documents = []
            all_metadatas = []
            doc_id_to_index = {}  # Map doc_id to index in lists

            queries = [
                "research information data",
                "document text content",
                "checkpoint summary report",
                "findings results analysis",
            ]

            # Use larger k value to get more documents
            k = 2000  # Increased from 1000

            for query in queries:
                try:
                    results = self.memory_store.search(query, k=k)

                    if not results or not results.get("documents"):
                        continue

                    doc_ids = results.get("ids", [])[0] if results.get("ids") else []
                    documents = results["documents"][0]
                    metadatas = (
                        results["metadatas"][0]
                        if results.get("metadatas")
                        else [{}] * len(documents)
                    )

                    # Add documents, avoiding duplicates
                    for doc_id, doc, meta in zip(doc_ids, documents, metadatas):
                        if doc_id not in all_doc_ids:
                            all_doc_ids.add(doc_id)
                            all_documents.append(doc)
                            all_metadatas.append(meta)
                            doc_id_to_index[doc_id] = len(all_documents) - 1
                except Exception as e:
                    logger.debug(
                        f"Error in BM25 document retrieval query '{query}': {e}"
                    )
                    continue

            if not all_documents:
                return [], [], []

            # Filter by task if specified
            if task_name:
                filtered = []
                filtered_ids = []
                filtered_metas = []
                for doc_id, idx in doc_id_to_index.items():
                    meta = all_metadatas[idx]
                    if meta.get("task") == task_name:
                        filtered.append(all_documents[idx])
                        filtered_ids.append(doc_id)
                        filtered_metas.append(meta)
                return filtered, filtered_ids, filtered_metas

            return all_documents, list(all_doc_ids), all_metadatas
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

    def invalidate_bm25_index(self):
        """Invalidate BM25 index (call when documents are added/removed)"""
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_doc_ids = []
        self.bm25_metadatas = []
        self._bm25_initialized = False
