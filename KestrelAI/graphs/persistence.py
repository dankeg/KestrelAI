"""
Shared LangGraph runtime persistence primitives.
"""

from __future__ import annotations

import threading

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
except ImportError:  # pragma: no cover - dependency-gated path
    InMemorySaver = None
    InMemoryStore = None

_lock = threading.Lock()
_runtime: tuple[object | None, object | None] | None = None


def get_langgraph_runtime() -> tuple[object | None, object | None]:
    """
    Return singleton (checkpointer, store) runtime components.
    """
    global _runtime
    if _runtime is not None:
        return _runtime

    with _lock:
        if _runtime is None:
            if InMemorySaver is None or InMemoryStore is None:
                _runtime = (None, None)
            else:
                _runtime = (InMemorySaver(), InMemoryStore())
    return _runtime
