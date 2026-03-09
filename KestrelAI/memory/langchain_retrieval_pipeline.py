"""
LangChain-based retrieval and compression pipeline.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

try:
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda
except ImportError:  # pragma: no cover - dependency-gated path
    Document = None
    RunnableLambda = None

if TYPE_CHECKING:
    from KestrelAI.agents.context_manager import TokenCounter
    from KestrelAI.agents.multi_level_summarizer import MultiLevelSummarizer
    from KestrelAI.memory.hybrid_retriever import HybridRetriever
    from KestrelAI.memory.vector_store import MemoryStore

logger = logging.getLogger(__name__)


class LangChainRetrievalPipeline:
    """
    Retrieve + select + compress pipeline that keeps existing task/layer semantics,
    while routing post-retrieval steps through LangChain runnables.
    """

    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        hybrid_retriever: HybridRetriever | None = None,
        token_counter: TokenCounter | None = None,
        summarizer: MultiLevelSummarizer | None = None,
        context_management_enabled: bool = False,
        debug: bool = False,
    ):
        if Document is None or RunnableLambda is None:
            raise ImportError("langchain_core is not installed")

        self.memory_store = memory_store
        self.hybrid_retriever = hybrid_retriever
        self.token_counter = token_counter
        self.summarizer = summarizer
        self.context_management_enabled = context_management_enabled
        self.debug = debug

        self._pipeline = RunnableLambda(self._select_content) | RunnableLambda(
            self._compress_content
        )

    def retrieve(
        self,
        *,
        task_name: str,
        query: str,
        max_tokens: int | None = None,
        fallback_entries: Iterable[str] | None = None,
    ) -> str:
        documents = self._collect_documents(task_name=task_name, query=query)
        if not documents:
            return self._fallback_content(fallback_entries)

        output = self._pipeline.invoke(
            {"documents": documents, "max_tokens": max_tokens}
        )
        content = str(output) if output is not None else ""
        return content or self._fallback_content(fallback_entries)

    def retrieve_documents(
        self,
        *,
        task_name: str,
        query: str,
        k: int = 20,
    ) -> list[Document]:
        """Return raw LangChain documents collected through the retrieval abstraction."""
        documents = self._collect_documents(task_name=task_name, query=query)
        if k <= 0:
            return documents
        return documents[:k]

    def _collect_documents(self, *, task_name: str, query: str) -> list[Document]:
        task_docs: list[Document] = []

        if self.hybrid_retriever is not None:
            hybrid_results = self.hybrid_retriever.retrieve(
                query,
                k=20,
                task_name=task_name,
                use_hybrid=True,
            )
            for result in hybrid_results:
                meta = dict(result.get("metadata", {}))
                fused_score = float(result.get("fused_score", result.get("score", 0.0)))
                doc_id = result.get("doc_id")
                if doc_id:
                    meta["doc_id"] = doc_id
                meta["fused_score"] = fused_score
                meta["layer"] = meta.get("layer", "episodic")
                meta["checkpoint_index"] = meta.get("checkpoint_index", -1)
                meta["distance"] = self._distance_from_fused_score(fused_score)
                task_docs.append(
                    Document(
                        page_content=result.get("content", ""),
                        metadata=meta,
                    )
                )
        else:
            retriever_factory = getattr(
                self.memory_store, "as_langchain_retriever", None
            )
            retriever = (
                retriever_factory(task_name=task_name, k=20)
                if callable(retriever_factory)
                else None
            )
            if retriever is not None and hasattr(retriever, "invoke"):
                lc_docs = retriever.invoke(query)
                if not isinstance(lc_docs, list):
                    lc_docs = []
                for doc in lc_docs:
                    metadata = dict(doc.metadata or {})
                    metadata["layer"] = metadata.get("layer", "episodic")
                    metadata["checkpoint_index"] = metadata.get("checkpoint_index", -1)
                    metadata["distance"] = float(metadata.get("distance", 0.0))
                    metadata["fused_score"] = float(metadata.get("fused_score", 0.0))
                    task_docs.append(
                        Document(
                            page_content=doc.page_content,
                            metadata=metadata,
                        )
                    )
            if not task_docs:
                results = self.memory_store.search(query, k=20)
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
                ids = results["ids"][0] if results.get("ids") else [""] * len(documents)

                for doc, meta, dist, doc_id in zip(
                    documents, metadatas, distances, ids
                ):
                    if meta.get("task") != task_name:
                        continue
                    metadata = dict(meta)
                    if doc_id:
                        metadata["doc_id"] = doc_id
                    metadata["layer"] = metadata.get("layer", "episodic")
                    metadata["checkpoint_index"] = metadata.get("checkpoint_index", -1)
                    metadata["distance"] = dist
                    metadata["fused_score"] = metadata.get("fused_score", 0.0)
                    task_docs.append(
                        Document(
                            page_content=doc,
                            metadata=metadata,
                        )
                    )

        task_docs.sort(
            key=lambda doc: (
                float(doc.metadata.get("fused_score", 0.0)),
                int(doc.metadata.get("checkpoint_index", -1)),
                -float(doc.metadata.get("distance", 1.0)),
            ),
            reverse=True,
        )
        return task_docs

    def _select_content(self, payload: dict[str, Any]) -> dict[str, Any]:
        documents: list[Document] = payload.get("documents", [])
        max_tokens = payload.get("max_tokens")

        if not documents:
            return {"content": "", "max_tokens": max_tokens}

        if (
            isinstance(max_tokens, int)
            and max_tokens > 0
            and self.context_management_enabled
            and self.token_counter is not None
        ):
            selected_docs = self._select_documents_by_budget(documents, max_tokens)
        else:
            episodic_docs = [
                doc for doc in documents if doc.metadata.get("layer") == "episodic"
            ]
            selected_docs = episodic_docs[:5] if episodic_docs else documents[:5]

        content = "\n\n---\n\n".join(doc.page_content for doc in selected_docs)
        return {"content": content, "max_tokens": max_tokens}

    def _compress_content(self, payload: dict[str, Any]) -> str:
        content = str(payload.get("content", ""))
        max_tokens = payload.get("max_tokens")

        if not content:
            return ""

        if (
            not isinstance(max_tokens, int)
            or max_tokens <= 0
            or not self.context_management_enabled
            or self.summarizer is None
            or self.token_counter is None
        ):
            return content

        content_tokens = self.token_counter.count_tokens(content)
        if content_tokens <= max_tokens:
            return content

        summary, level, _facts = self.summarizer.create_summary_on_demand(
            content,
            max_tokens=max_tokens,
            preserve_facts=True,
        )
        if self.debug:
            logger.debug(
                "Summarized retrieved content: %s -> %s tokens (level: %s)",
                content_tokens,
                self.token_counter.count_tokens(summary),
                level,
            )
        return summary

    def _select_documents_by_budget(
        self, documents: list[Document], max_tokens: int
    ) -> list[Document]:
        selected: list[Document] = []
        tokens_used = 0

        grouped: dict[int, list[Document]] = defaultdict(list)
        for doc in documents:
            idx = int(doc.metadata.get("checkpoint_index", -1))
            grouped[idx].append(doc)

        for checkpoint_idx in sorted(grouped.keys(), reverse=True):
            group = grouped[checkpoint_idx]

            layered_candidates = [
                next(
                    (doc for doc in group if doc.metadata.get("layer") == "episodic"),
                    None,
                ),
                next(
                    (doc for doc in group if doc.metadata.get("layer") == "semantic"),
                    None,
                ),
                next(
                    (doc for doc in group if doc.metadata.get("layer") == "summary"),
                    None,
                ),
            ]

            picked = False
            for candidate in layered_candidates:
                if candidate is None:
                    continue
                doc_tokens = self.token_counter.count_tokens(candidate.page_content)
                if tokens_used + doc_tokens <= max_tokens:
                    selected.append(candidate)
                    tokens_used += doc_tokens
                    picked = True
                    break

            if not picked:
                break

        return selected or documents[:5]

    @staticmethod
    def _distance_from_fused_score(fused_score: float) -> float:
        if fused_score > 1.0:
            normalized = 1.0
        elif fused_score > 0.1:
            normalized = fused_score
        else:
            normalized = min(0.5, fused_score * 5.0)
        return 1.0 - normalized

    @staticmethod
    def _fallback_content(fallback_entries: Iterable[str] | None) -> str:
        if not fallback_entries:
            return "(No previous findings)"

        entries = list(fallback_entries)
        if not entries:
            return "(No previous findings)"

        recent_entries = entries[-5:] if len(entries) > 5 else entries
        return "\n\n".join(str(entry) for entry in recent_entries)
