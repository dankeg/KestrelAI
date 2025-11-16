# Advanced Long-Term Memory Systems for Research Information

## Executive Summary

This document explores advanced techniques for long-term storage and retrieval of research information that go beyond basic RAG. The focus is on maintaining coherency, preserving detailed information, and enabling efficient access to accumulated knowledge over extended research sessions.

## Current System Analysis

### Current RAG Implementation
- **Vector Store**: ChromaDB with persistent storage
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Storage**: Simple document embeddings with metadata (task, type, timestamp, length)
- **Retrieval**: Basic semantic search with fixed k=5 results
- **Limitations**:
  - No hierarchical organization
  - No summarization layers
  - No relationship tracking between documents
  - No distinction between episodic (specific events) and semantic (general knowledge) memory
  - Fixed retrieval size regardless of context needs

## Advanced Memory Architectures

### 1. Hierarchical Memory Systems

#### Concept
Organize information in multiple layers of abstraction, from detailed facts to high-level summaries.

#### Implementation Strategy

**Three-Tier Architecture:**

1. **Episodic Layer (Detailed)**
   - Raw research findings, specific search results, detailed checkpoints
   - High granularity, preserves all details
   - Indexed by: timestamp, source, task, subtask

2. **Semantic Layer (Summarized)**
   - Summaries of related findings, synthesized knowledge
   - Medium granularity, key facts preserved
   - Indexed by: topic, entity, concept

3. **Episodic Summary Layer (Compressed)**
   - High-level summaries of research sessions
   - Low granularity, essential insights only
   - Indexed by: task, time period, major themes

**Benefits:**
- Fast access to summaries for context building
- Detailed information available when needed
- Progressive summarization prevents information loss
- Efficient token usage in context windows

**Implementation:**
```python
class HierarchicalMemoryStore:
    def __init__(self):
        # Three separate collections
        self.episodic_store = MemoryStore(path=".chroma/episodic")
        self.semantic_store = MemoryStore(path=".chroma/semantic")
        self.summary_store = MemoryStore(path=".chroma/summary")
        
        # Track relationships
        self.relationships = {}  # summary_id -> [episodic_ids]
    
    def add_episodic(self, content: str, metadata: dict):
        """Store detailed information"""
        doc_id = self.episodic_store.add(content, metadata)
        return doc_id
    
    def create_semantic_summary(self, episodic_ids: List[str], query: str):
        """Summarize multiple episodic entries into semantic knowledge"""
        episodes = [self.episodic_store.get(id) for id in episodic_ids]
        summary = self._summarize_episodes(episodes, query)
        summary_id = self.semantic_store.add(summary, {
            "type": "semantic",
            "source_episodes": episodic_ids,
            "query": query
        })
        return summary_id
    
    def create_episodic_summary(self, task_id: str, time_period: str):
        """Create high-level summary of research session"""
        episodes = self.episodic_store.search_by_task(task_id, time_period)
        summary = self._create_session_summary(episodes)
        summary_id = self.summary_store.add(summary, {
            "type": "episodic_summary",
            "task": task_id,
            "period": time_period
        })
        return summary_id
```

### 2. Graph-Based Memory (Knowledge Graphs)

#### Concept
Store information as a knowledge graph with entities, relationships, and properties. Enables structured querying and relationship traversal.

#### Key Techniques

**Graph RAG (FG-RAG - Fine-Grained Graph RAG):**
- Build entity knowledge graph from documents
- Pre-generate community summaries for related entities
- Use graph traversal for multi-hop reasoning
- Context-aware entity expansion during retrieval

**Benefits:**
- Captures relationships between concepts
- Enables multi-hop reasoning
- Better handling of complex queries
- Maintains coherence through graph structure

**Implementation Options:**
- **Neo4j**: Full-featured graph database, excellent for complex relationships
- **NetworkX + Vector Store**: Lightweight, Python-native, good for smaller graphs
- **LangGraph**: Specialized for LLM agent workflows with graph-based state management

**Example Structure:**
```
Entity: "NSF REU Program"
  - Properties: deadline, eligibility, funding_amount
  - Relationships:
    - RELATED_TO -> "Undergraduate Research"
    - SIMILAR_TO -> "NSF GRFP"
    - PART_OF -> "NSF Programs"
  - Documents: [doc1, doc2, doc3]
  - Summary: "NSF REU provides summer research opportunities..."
```

**Implementation:**
```python
class GraphMemoryStore:
    def __init__(self):
        self.vector_store = MemoryStore()
        # Use NetworkX for graph structure
        import networkx as nx
        self.graph = nx.DiGraph()
        self.entity_store = {}  # entity_id -> entity_data
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities and relationships from text"""
        # Use LLM or NER model to extract entities
        entities = self._extract_entities_llm(text)
        return entities
    
    def add_document(self, doc_id: str, text: str, metadata: dict):
        """Add document and build graph"""
        # Store in vector store
        self.vector_store.add(doc_id, text, metadata)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Add to graph
        for entity in entities:
            entity_id = entity['id']
            if entity_id not in self.graph:
                self.graph.add_node(entity_id, **entity)
            
            # Add relationships
            for rel in entity.get('relationships', []):
                target_id = rel['target']
                if target_id not in self.graph:
                    self.graph.add_node(target_id)
                self.graph.add_edge(entity_id, target_id, 
                                  relationship=rel['type'])
            
            # Link document to entity
            self.graph.add_edge(entity_id, doc_id, 
                              relationship='MENTIONED_IN')
    
    def query_graph(self, query: str, max_hops: int = 2):
        """Query graph with multi-hop reasoning"""
        # Find relevant entities
        relevant_entities = self.vector_store.search(query, k=10)
        
        # Expand graph from relevant entities
        expanded = set()
        for entity_id in relevant_entities:
            # Get neighbors within max_hops
            neighbors = nx.single_source_shortest_path_length(
                self.graph, entity_id, cutoff=max_hops
            )
            expanded.update(neighbors.keys())
        
        # Retrieve documents connected to expanded entities
        documents = []
        for node in expanded:
            if node.startswith('doc_'):
                documents.append(self.vector_store.get(node))
        
        return documents
```

### 3. Multi-Level Summarization

#### Concept
Create summaries at multiple granularities, enabling efficient navigation from high-level overviews to detailed information.

#### Techniques

**HIRO (Hierarchical Information Retrieval Optimization):**
- Organizes documents at various levels of summarization
- Uses Depth-First Search with recursive similarity scoring
- Branch pruning to minimize context without losing information
- 10.85% improvement in performance on complex datasets

**RECON (Reasoning with Condensation):**
- Summarization module within reasoning loop
- Reduces context length by 35%
- Two-stage training: relevance pretraining + multi-aspect distillation
- Improves multi-hop question answering

**Implementation:**
```python
class MultiLevelSummarizer:
    def __init__(self, llm):
        self.llm = llm
        self.summary_levels = {
            'detailed': 1.0,      # Full content
            'medium': 0.5,        # 50% compression
            'summary': 0.2,       # 20% compression
            'executive': 0.1      # 10% compression
        }
    
    def create_summary_hierarchy(self, content: str) -> Dict[str, str]:
        """Create summaries at multiple levels"""
        summaries = {}
        
        # Start with full content
        summaries['detailed'] = content
        
        # Create progressively more compressed summaries
        current = content
        for level in ['medium', 'summary', 'executive']:
            ratio = self.summary_levels[level]
            target_tokens = int(len(current.split()) * ratio)
            current = self._summarize(current, target_tokens)
            summaries[level] = current
        
        return summaries
    
    def retrieve_adaptive(self, query: str, max_tokens: int):
        """Retrieve content at appropriate level based on token budget"""
        # Try to get detailed first
        detailed = self.vector_store.search(query, k=1)
        tokens = self.count_tokens(detailed)
        
        if tokens <= max_tokens:
            return detailed
        
        # Fall back to summaries
        for level in ['medium', 'summary', 'executive']:
            summary = self.summary_store.search(query, level, k=1)
            tokens = self.count_tokens(summary)
            if tokens <= max_tokens:
                return summary
        
        # If still too large, truncate
        return self._truncate(detailed, max_tokens)
```

### 4. Episodic vs Semantic Memory Distinction

#### Concept
Separate specific events (episodic) from general knowledge (semantic), similar to human memory systems.

#### Implementation

**Episodic Memory:**
- Specific research actions: "Searched for X on date Y, found Z"
- Checkpoints with timestamps
- Search results and their sources
- Task-specific findings

**Semantic Memory:**
- General knowledge: "NSF REU programs typically have deadlines in February"
- Synthesized insights across multiple tasks
- Entity properties and relationships
- Reusable knowledge

**Benefits:**
- Better organization of information
- Efficient retrieval (semantic for general, episodic for specific)
- Prevents information overload
- Maintains both detailed and abstract knowledge

**Implementation:**
```python
class DualMemoryStore:
    def __init__(self):
        self.episodic = MemoryStore(path=".chroma/episodic")
        self.semantic = MemoryStore(path=".chroma/semantic")
        self.semanticizer = SemanticExtractor()
    
    def add_episodic(self, content: str, metadata: dict):
        """Store specific event/action"""
        doc_id = self.episodic.add(content, {
            **metadata,
            "memory_type": "episodic",
            "timestamp": datetime.now()
        })
        return doc_id
    
    def extract_semantic(self, episodic_content: str) -> str:
        """Extract general knowledge from episodic content"""
        # Use LLM to extract semantic knowledge
        prompt = f"""Extract general, reusable knowledge from this specific research finding.
        Focus on facts, relationships, and insights that could be useful in other contexts.
        
        Content: {episodic_content}
        
        Extract semantic knowledge:"""
        
        semantic = self.llm.chat([{"role": "user", "content": prompt}])
        return semantic
    
    def add_semantic(self, semantic_content: str, source_episodic_ids: List[str]):
        """Store general knowledge"""
        doc_id = self.semantic.add(semantic_content, {
            "memory_type": "semantic",
            "sources": source_episodic_ids,
            "timestamp": datetime.now()
        })
        return doc_id
    
    def query(self, query: str, memory_type: str = "both"):
        """Query appropriate memory type"""
        if memory_type == "episodic":
            return self.episodic.search(query)
        elif memory_type == "semantic":
            return self.semantic.search(query)
        else:
            # Query both and merge
            episodic_results = self.episodic.search(query, k=3)
            semantic_results = self.semantic.search(query, k=3)
            return self._merge_results(episodic_results, semantic_results)
```

### 5. Progressive Summarization with Compression

#### Concept
Continuously compress older information while preserving key details, preventing context overflow.

#### Strategy

**Time-Based Compression:**
- Recent information: Full detail
- Medium age: Summarized
- Old information: Highly compressed or archived

**Importance-Based Compression:**
- High importance: Preserve detail longer
- Medium importance: Summarize earlier
- Low importance: Compress or remove

**Implementation:**
```python
class ProgressiveCompressor:
    def __init__(self):
        self.compression_schedule = {
            'recent': (0, 7),      # Days 0-7: No compression
            'medium': (7, 30),     # Days 7-30: Medium compression
            'old': (30, 90),       # Days 30-90: High compression
            'archived': (90, None) # Days 90+: Archive or remove
        }
    
    def compress_old_content(self, content: str, age_days: int) -> str:
        """Compress content based on age"""
        if age_days < 7:
            return content  # No compression
        elif age_days < 30:
            return self._medium_compress(content)
        elif age_days < 90:
            return self._high_compress(content)
        else:
            return self._archive(content)
    
    def _medium_compress(self, content: str) -> str:
        """Compress to ~50% of original"""
        # Use LLM to create summary preserving key facts
        prompt = f"""Summarize this research finding, preserving:
        - Key facts and numbers
        - Important entities and relationships
        - Actionable information
        
        Content: {content}
        
        Summary:"""
        return self.llm.chat([{"role": "user", "content": prompt}])
    
    def _high_compress(self, content: str) -> str:
        """Compress to ~20% of original"""
        # Extract only essential facts
        prompt = f"""Extract only the most essential facts from this content.
        Focus on: entities, key numbers, critical relationships.
        
        Content: {content}
        
        Essential facts:"""
        return self.llm.chat([{"role": "user", "content": prompt}])
```

### 6. Hybrid Retrieval Strategies

#### Concept
Combine multiple retrieval methods for better coverage and relevance.

#### Techniques

**Fusion Retrieval:**
- Combine keyword-based (BM25) and vector-based (semantic) search
- Rerank results using cross-encoders
- Query expansion for broader coverage

**Sentence Window Retrieval:**
- Retrieve sentences with surrounding context
- Preserves local coherence
- Better for detailed information access

**Auto-Merging Retrieval:**
- Automatically merge related chunks
- Reduces redundancy
- Creates more complete context

**Implementation:**
```python
class HybridRetriever:
    def __init__(self):
        self.vector_store = MemoryStore()
        self.keyword_index = BM25Index()  # Sparse retrieval
        self.reranker = CrossEncoderReranker()
    
    def retrieve(self, query: str, k: int = 5):
        # 1. Vector search (semantic)
        vector_results = self.vector_store.search(query, k=k*2)
        
        # 2. Keyword search (exact matches)
        keyword_results = self.keyword_index.search(query, k=k*2)
        
        # 3. Merge and deduplicate
        merged = self._merge_results(vector_results, keyword_results)
        
        # 4. Rerank with cross-encoder
        reranked = self.reranker.rerank(query, merged)
        
        # 5. Return top k
        return reranked[:k]
    
    def retrieve_with_context(self, query: str, window_size: int = 2):
        """Retrieve with sentence window"""
        results = self.retrieve(query)
        
        expanded = []
        for result in results:
            # Get surrounding sentences
            context = self._get_sentence_window(result, window_size)
            expanded.append({
                'content': result['content'],
                'context': context,
                'score': result['score']
            })
        
        return expanded
```

## Recommended Architecture

### Integrated System Design

```
┌─────────────────────────────────────────────────────────┐
│                  Research Agent                          │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐      ┌─────────▼──────────┐
│ Context Manager│      │  Memory Manager    │
│ (Token Budget) │      │  (Long-term)       │
└───────┬────────┘      └─────────┬──────────┘
        │                         │
        │              ┌──────────┴──────────┐
        │              │                     │
┌───────▼──────────────▼──────────┐  ┌──────▼──────────────┐
│   Hierarchical Memory Store     │  │  Graph Memory Store │
│                                 │  │                     │
│  ┌──────────────┐              │  │  Entities           │
│  │ Episodic     │              │  │  Relationships      │
│  │ (Detailed)   │              │  │  Properties         │
│  └──────────────┘              │  └─────────────────────┘
│                                 │
│  ┌──────────────┐              │
│  │ Semantic     │              │
│  │ (Summarized) │              │
│  └──────────────┘              │
│                                 │
│  ┌──────────────┐              │
│  │ Summary      │              │
│  │ (Compressed) │              │
│  └──────────────┘              │
└─────────────────────────────────┘
```

### Key Components

1. **Context Manager**: Handles token budgets and immediate context
2. **Memory Manager**: Orchestrates long-term storage and retrieval
3. **Hierarchical Store**: Multi-level organization (episodic/semantic/summary)
4. **Graph Store**: Relationship tracking and multi-hop reasoning
5. **Progressive Compressor**: Age-based compression
6. **Hybrid Retriever**: Multiple retrieval strategies

## Implementation Priority

### Phase 1: Foundation (High Priority)
1. ✅ Implement token counting
2. ✅ Basic context management with token budgets
3. ✅ Hierarchical summarization (3 levels)
4. ✅ Episodic vs Semantic memory separation

### Phase 2: Enhancement (Medium Priority)
1. Graph-based memory with entity extraction
2. Progressive compression system
3. Hybrid retrieval (vector + keyword + reranking)
4. Multi-level summarization with adaptive retrieval

### Phase 3: Advanced (Lower Priority)
1. Full knowledge graph implementation (Neo4j)
2. Advanced graph traversal algorithms
3. Automated relationship discovery
4. Cross-task knowledge synthesis

## Benefits

1. **Coherency**: Hierarchical organization maintains context flow
2. **Detail Preservation**: Multiple levels ensure nothing is lost
3. **Efficiency**: Summaries enable fast context building
4. **Scalability**: Progressive compression handles long sessions
5. **Intelligence**: Graph structure enables complex reasoning
6. **Flexibility**: Multiple retrieval strategies for different needs

## References

- **RECON**: Reasoning with Condensation for Efficient RAG (arXiv:2510.10448)
- **FG-RAG**: Fine-Grained Graph RAG (arXiv:2504.07103)
- **HIRO**: Hierarchical Information Retrieval Optimization (arXiv:2406.09979)
- **Refiner**: Extract-and-Restructure Paradigm (arXiv:2406.11357)
- **RankCoT**: Refining Knowledge through Ranking Chain-of-Thoughts (arXiv:2502.17888)



