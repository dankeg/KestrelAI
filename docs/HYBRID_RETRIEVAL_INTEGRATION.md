# Hybrid Retrieval Integration Summary

## Overview

The hybrid retrieval system has been fully integrated into KestrelAI's research agent architecture. It combines vector-based semantic search with BM25 keyword search to improve retrieval quality and accuracy.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              WebResearchAgent                            │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  HybridRetriever                                 │  │
│  │  - Vector Search (ChromaDB)                      │  │
│  │  - BM25 Keyword Search                           │  │
│  │  - Result Fusion (RRF + Weighted)                │  │
│  └──────────────────────────────────────────────────┘  │
│                        │                                 │
│                        ▼                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │  MemoryStore (ChromaDB)                          │  │
│  │  - Episodic Layer (detailed)                     │  │
│  │  - Semantic Layer (summarized)                   │  │
│  │  - Summary Layer (compressed)                    │  │
│  └──────────────────────────────────────────────────┘  │
│                        │                                 │
│                        ▼                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ContextManager                                  │  │
│  │  - Token Budget Management                       │  │
│  │  - Full Context Summarization                    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. Initialization (`WebResearchAgent.__init__`)

**Location:** `KestrelAI/agents/web_research_agent.py:270-278`

```python
# Initialize hybrid retriever
try:
    self.hybrid_retriever = HybridRetriever(memory, enable_bm25=True)
    self.hybrid_retrieval_enabled = True
    logger.info("Hybrid retrieval enabled (vector + BM25)")
except Exception as e:
    logger.warning(f"Failed to initialize hybrid retriever: {e}. Using vector search only.")
    self.hybrid_retriever = None
    self.hybrid_retrieval_enabled = False
```

**Features:**
- Graceful fallback if BM25 library is not available
- Automatic detection of BM25 availability
- Logging for debugging

### 2. Document Addition (`BaseAgent._add_to_rag`)

**Location:** `KestrelAI/agents/base_agent.py:156-162`

```python
# Invalidate BM25 index if hybrid retriever exists
try:
    if hasattr(self, 'hybrid_retriever') and self.hybrid_retriever:
        self.hybrid_retriever.invalidate_bm25_index()
except AttributeError:
    pass  # Not all agents have hybrid_retriever
```

**Features:**
- Automatic index invalidation when documents are added
- Ensures BM25 index stays up-to-date
- Safe for agents without hybrid retriever

### 3. Retrieval (`WebResearchAgent._retrieve_from_rag`)

**Location:** `KestrelAI/agents/web_research_agent.py:1230-1331`

**Flow:**
1. Check if hybrid retrieval is enabled
2. If enabled, use `HybridRetriever.retrieve()` with:
   - Query (from task description or current focus)
   - Task name filtering
   - k=20 results for better filtering
3. Convert hybrid results to internal format
4. Fall back to vector-only search if hybrid unavailable
5. Filter by task name
6. Sort by fused_score, checkpoint_index, distance
7. Select appropriate summary level based on token budget
8. Apply final summarization if needed

**Key Features:**
- Seamless integration with hierarchical memory layers
- Respects token budgets
- Works with summary level selection
- Graceful fallback to vector-only search

### 4. Context Building Integration

**Location:** `KestrelAI/agents/web_research_agent.py:638`

```python
rag_content = self._retrieve_from_rag(task, query=rag_query, max_tokens=self.token_budget.rag_content)
```

**Integration:**
- Uses hybrid retrieval for RAG content
- Respects token budget from `TokenBudget`
- Uses current focus or task description as query
- Results are integrated into context via `ContextManager`

### 5. Final Report Generation

**Location:** `KestrelAI/agents/web_research_agent.py:1085`

```python
rag_content = self._retrieve_from_rag(task, query=task.description, max_tokens=self.token_budget.rag_content if self.token_budget else None)
```

**Integration:**
- Uses hybrid retrieval for final report generation
- Retrieves most relevant information using task description
- Respects token budgets

## Data Flow

### Document Storage Flow

```
1. Agent creates checkpoint/report
   ↓
2. _add_to_rag() called
   ↓
3. Document stored in ChromaDB (vector store)
   ↓
4. BM25 index invalidated (will rebuild on next search)
   ↓
5. Summary hierarchies stored at multiple layers
```

### Retrieval Flow

```
1. _retrieve_from_rag() called with query
   ↓
2. HybridRetriever.retrieve() called
   ↓
3. Vector search (semantic similarity)
   ↓
4. BM25 search (keyword matching) [if enabled]
   ↓
5. Results fused using RRF + weighted combination
   ↓
6. Results filtered by task name
   ↓
7. Sorted by fused_score, checkpoint_index, distance
   ↓
8. Summary level selected based on token budget
   ↓
9. Final summarization if needed
   ↓
10. Returned to context builder
```

## Key Features

### 1. Hybrid Search
- **Vector Search:** Semantic similarity using embeddings
- **BM25 Search:** Keyword-based exact matching
- **Fusion:** Combines both using RRF and weighted scores

### 2. Hierarchical Memory Integration
- Works seamlessly with episodic/semantic/summary layers
- Selects appropriate summary level based on token budget
- Preserves detailed information when needed

### 3. Token Budget Awareness
- Respects token budgets from `TokenBudget`
- Uses `_select_documents_by_budget()` for layer selection
- Applies final summarization if content exceeds budget

### 4. Error Handling
- Graceful fallback to vector-only search
- Handles missing BM25 library gracefully
- Logs warnings for debugging
- Falls back to scratchpad if retrieval fails

### 5. Index Management
- Lazy BM25 index building (built on first search)
- Automatic invalidation when documents added
- Efficient memory usage

## Configuration

### Dependencies

**Required:**
- `chromadb` - Vector store
- `sentence-transformers` - Embeddings

**Optional:**
- `rank-bm25` - BM25 keyword search (enables hybrid mode)

### Initialization Options

```python
# Enable hybrid retrieval (default)
retriever = HybridRetriever(memory_store, enable_bm25=True)

# Disable BM25 (vector-only)
retriever = HybridRetriever(memory_store, enable_bm25=False)
```

### Fusion Weights

Default weights (configurable in `HybridRetriever.__init__`):
- `vector_weight = 0.6` (60% weight for semantic similarity)
- `bm25_weight = 0.4` (40% weight for keyword matching)

## Performance Considerations

### BM25 Index Building
- Built lazily on first search
- Rebuilt when invalidated (after document addition)
- Uses ChromaDB search to get documents (workaround for "get all")

### Memory Usage
- BM25 index stored in memory
- Index invalidated when documents added (prevents stale data)
- Efficient tokenization and scoring

### Retrieval Performance
- Vector search: O(n) where n = number of documents
- BM25 search: O(n) where n = number of documents
- Fusion: O(k) where k = number of results
- Overall: Efficient for typical document counts

## Testing

### Unit Tests
- `tests/unit/test_hybrid_retriever_standalone.py` - Core functionality
- 15 tests covering tokenization, normalization, fusion, initialization

### Integration Tests
- `tests/integration/test_hybrid_retrieval_integration.py` - Full integration
- 10 tests covering agent integration, error handling, task filtering

### Test Coverage
- ✅ Tokenization
- ✅ Score normalization
- ✅ Result fusion
- ✅ Index management
- ✅ Agent integration
- ✅ Error handling
- ✅ Task filtering
- ✅ Token budget respect

## Usage Examples

### Basic Usage

```python
from KestrelAI.memory.hybrid_retriever import HybridRetriever
from KestrelAI.memory.vector_store import MemoryStore

# Initialize
memory_store = MemoryStore()
retriever = HybridRetriever(memory_store, enable_bm25=True)

# Retrieve
results = retriever.retrieve(
    query="NSF REU program deadline",
    k=10,
    task_name="research_task",
    use_hybrid=True
)
```

### Integration with Agent

```python
# Automatically initialized in WebResearchAgent
agent = WebResearchAgent("agent-id", llm, memory_store)

# Retrieval happens automatically in _retrieve_from_rag()
# Called during context building and report generation
```

## Future Enhancements

1. **Persistent BM25 Index:** Store index on disk to avoid rebuilding
2. **Incremental Updates:** Update index incrementally instead of rebuilding
3. **Query Expansion:** Expand queries for better retrieval
4. **Reranking:** Add cross-encoder reranking for better relevance
5. **Performance Metrics:** Track retrieval performance and quality

## Troubleshooting

### BM25 Not Available
- **Symptom:** `WARNING: rank-bm25 not installed`
- **Solution:** Install with `pip install rank-bm25` or `poetry add rank-bm25`
- **Impact:** Falls back to vector-only search (still works)

### Index Not Updating
- **Symptom:** Old documents appearing in results
- **Solution:** Index invalidates automatically on document addition
- **Note:** Index rebuilds on next search (lazy building)

### Poor Retrieval Quality
- **Check:** Query relevance, document quality
- **Try:** Adjust fusion weights, increase k parameter
- **Debug:** Check logs for retrieval scores

## Summary

The hybrid retrieval system is fully integrated and working. It:
- ✅ Combines vector and BM25 search for better results
- ✅ Integrates seamlessly with hierarchical memory
- ✅ Respects token budgets and summary levels
- ✅ Handles errors gracefully
- ✅ Maintains index consistency
- ✅ Is fully tested and documented

The system is production-ready and provides improved retrieval quality over vector-only search.



