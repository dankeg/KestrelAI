from __future__ import annotations

from unittest.mock import Mock

import pytest

from KestrelAI.memory.langchain_retrieval_pipeline import LangChainRetrievalPipeline


@pytest.mark.unit
def test_retrieval_pipeline_filters_task_and_compresses_when_over_budget():
    memory_store = Mock()
    memory_store.search.return_value = {
        "documents": [
            [
                "task a detailed content",
                "task b should be filtered out",
            ]
        ],
        "metadatas": [
            [
                {"task": "task-a", "layer": "episodic", "checkpoint_index": 2},
                {"task": "task-b", "layer": "episodic", "checkpoint_index": 1},
            ]
        ],
        "distances": [[0.2, 0.3]],
    }

    token_counter = Mock()
    token_counter.count_tokens.side_effect = lambda text: len(str(text).split())

    summarizer = Mock()
    summarizer.create_summary_on_demand.return_value = (
        "compressed content",
        "summary",
        None,
    )

    pipeline = LangChainRetrievalPipeline(
        memory_store=memory_store,
        hybrid_retriever=None,
        token_counter=token_counter,
        summarizer=summarizer,
        context_management_enabled=True,
    )

    result = pipeline.retrieve(
        task_name="task-a",
        query="task",
        max_tokens=2,
        fallback_entries=["fallback1", "fallback2"],
    )

    assert result == "compressed content"
    summarizer.create_summary_on_demand.assert_called_once()


@pytest.mark.unit
def test_retrieval_pipeline_fallbacks_to_recent_entries_when_no_docs():
    memory_store = Mock()
    memory_store.search.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    pipeline = LangChainRetrievalPipeline(
        memory_store=memory_store,
        hybrid_retriever=None,
        token_counter=None,
        summarizer=None,
        context_management_enabled=False,
    )

    result = pipeline.retrieve(
        task_name="task-a",
        query="anything",
        max_tokens=None,
        fallback_entries=["a", "b", "c", "d", "e", "f"],
    )

    assert result == "b\n\nc\n\nd\n\ne\n\nf"
