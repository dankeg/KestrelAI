# Context Management Analysis & Recommendations

## Current Issues

### 1. **No Token Counting**
- Currently only measures character length: `len(context)` (line 298 in `web_research_agent.py`)
- Character count ≠ token count (varies by model, typically 1 token ≈ 4 characters for English)
- No awareness of actual token budget limits
- Cannot accurately determine if context fits within model's context window

### 2. **Arbitrary Pruning**
- `history: deque(maxlen=20)` - fixed count, not token-based
- `scratchpad[-5:]` - last 5 entries, regardless of size
- `state.snips = deque(maxlen=10)` - fixed count
- No consideration of:
  - Token budget remaining
  - Importance/relevance of content
  - Recency vs. relevance trade-offs

### 3. **Incoherent Context Building**
- Simply concatenates all context parts without prioritization
- No intelligent selection of what to include/exclude
- May include redundant information
- No summarization of older content to preserve key information

### 4. **No Token Budget Management**
- No allocation of tokens to different context components:
  - System prompt
  - Task description
  - Previous findings
  - Checkpoints
  - History
  - RAG content
  - URL reference table
- No reserve for response generation

## Recommended Solutions

### 1. **Implement Token Counting**

**Use tiktoken for accurate token counting:**
- tiktoken is the official tokenizer used by OpenAI and many other models
- Supports various models including those compatible with Ollama
- Fast and accurate token counting

**Implementation:**
```python
import tiktoken

class TokenCounter:
    def __init__(self, model_name: str = "gpt-4"):
        # Map Ollama models to tiktoken encodings
        self.encoding = tiktoken.encoding_for_model(model_name)
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def count_messages(self, messages: List[Dict]) -> int:
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.get("content", ""))
            # Add overhead for message formatting (typically 4 tokens per message)
            total += 4
        return total
```

### 2. **Token Budget Allocation**

**Define clear token budgets:**
```python
@dataclass
class TokenBudget:
    max_context: int = 32768  # Model's context window (adjust per model)
    system_prompt: int = 500
    task_description: int = 200
    previous_findings: int = 2000
    checkpoints: int = 3000
    history: int = 2000
    rag_content: int = 2000
    url_reference: int = 500
    response_reserve: int = 2000  # Reserve for LLM response
    
    @property
    def available_for_context(self) -> int:
        return self.max_context - self.system_prompt - self.response_reserve
```

### 3. **Intelligent Context Pruning**

**Priority-based pruning:**
1. **Always include (highest priority):**
   - System prompt
   - Current task description
   - Current subtask description (if applicable)

2. **Include with priority:**
   - Most recent checkpoint (highest priority)
   - Recent history (recency-weighted)
   - Relevant RAG content (relevance-weighted)
   - Previous findings (summarized if too long)

3. **Prune intelligently:**
   - Older checkpoints → summarize or exclude
   - Old history → exclude or summarize
   - Less relevant RAG content → exclude
   - URL reference table → truncate if needed

**Implementation Strategy:**
```python
class ContextManager:
    def __init__(self, token_budget: TokenBudget, token_counter: TokenCounter):
        self.budget = token_budget
        self.counter = token_counter
    
    def build_context(self, components: Dict[str, str]) -> str:
        """Build context within token budget"""
        context_parts = []
        tokens_used = 0
        
        # 1. Always include system prompt and task
        essential = [
            ("task", components.get("task", "")),
            ("subtask", components.get("subtask", "")),
        ]
        
        for name, content in essential:
            tokens = self.counter.count_tokens(content)
            if tokens_used + tokens <= self.budget.available_for_context:
                context_parts.append(content)
                tokens_used += tokens
        
        # 2. Add checkpoints (most recent first)
        checkpoints = components.get("checkpoints", [])
        for checkpoint in reversed(checkpoints):  # Most recent first
            tokens = self.counter.count_tokens(checkpoint)
            if tokens_used + tokens <= self.budget.checkpoints:
                context_parts.append(checkpoint)
                tokens_used += tokens
            else:
                # Summarize if too long
                summary = self._summarize(checkpoint, self.budget.checkpoints - tokens_used)
                if summary:
                    context_parts.append(summary)
                    tokens_used += self.counter.count_tokens(summary)
                break
        
        # 3. Add history (recency-weighted)
        history = components.get("history", [])
        for entry in reversed(history):  # Most recent first
            tokens = self.counter.count_tokens(entry)
            if tokens_used + tokens <= self.budget.history:
                context_parts.append(entry)
                tokens_used += tokens
            else:
                break
        
        # 4. Add RAG content (relevance-weighted)
        rag_content = components.get("rag_content", "")
        if rag_content:
            tokens = self.counter.count_tokens(rag_content)
            if tokens_used + tokens <= self.budget.rag_content:
                context_parts.append(rag_content)
            else:
                # Truncate or summarize
                truncated = self._truncate_to_tokens(rag_content, self.budget.rag_content)
                context_parts.append(truncated)
        
        return "\n\n".join(context_parts)
```

### 4. **Sliding Window with Summarization**

**For long-running tasks:**
- When history exceeds token budget, summarize older entries
- Keep recent entries verbatim
- Create compressed summaries of older content
- Maintain a "summary of summaries" for very long contexts

**Implementation:**
```python
def summarize_history(self, history: List[str], max_tokens: int) -> str:
    """Summarize history to fit within token budget"""
    if not history:
        return ""
    
    # Keep most recent entries verbatim
    recent = []
    recent_tokens = 0
    for entry in reversed(history):
        tokens = self.counter.count_tokens(entry)
        if recent_tokens + tokens <= max_tokens * 0.6:  # 60% for recent
            recent.insert(0, entry)
            recent_tokens += tokens
        else:
            break
    
    # Summarize older entries
    older = history[:-len(recent)] if recent else history
    if older:
        summary = self._create_summary(older, max_tokens - recent_tokens)
        return "\n".join(recent) + "\n\n[Earlier context summarized]\n" + summary
    
    return "\n".join(recent)
```

### 5. **RAG Content Selection**

**Improve RAG retrieval:**
- Use semantic similarity scores to rank content
- Select top-k most relevant chunks within token budget
- Consider recency in addition to relevance
- Avoid redundant information

**Implementation:**
```python
def retrieve_rag_content(self, task: Task, query: str, max_tokens: int) -> str:
    """Retrieve relevant RAG content within token budget"""
    # Get more results than needed
    results = self.memory.search(query, k=20)
    
    selected = []
    tokens_used = 0
    
    # Sort by relevance (distance) and recency
    sorted_results = sorted(
        results['documents'][0],
        key=lambda x: (x['distance'], -x['metadata']['timestamp'])
    )
    
    for doc in sorted_results:
        content = doc['text']
        tokens = self.counter.count_tokens(content)
        
        if tokens_used + tokens <= max_tokens:
            selected.append(content)
            tokens_used += tokens
        else:
            # Try to fit partial content
            remaining = max_tokens - tokens_used
            if remaining > 100:  # Minimum viable chunk
                truncated = self._truncate_to_tokens(content, remaining)
                selected.append(truncated)
            break
    
    return "\n\n".join(selected)
```

### 6. **Dynamic Token Budget Adjustment**

**Adapt based on model and usage:**
- Detect model's actual context window
- Adjust budgets based on what's available
- Reserve tokens for response generation
- Monitor and log token usage

## Implementation Plan

### Phase 1: Token Counting Infrastructure
1. Add tiktoken dependency
2. Create TokenCounter class
3. Integrate token counting into context building
4. Add logging for token usage

### Phase 2: Token Budget Management
1. Define TokenBudget dataclass
2. Create ContextManager class
3. Implement priority-based context building
4. Add token budget validation

### Phase 3: Intelligent Pruning
1. Implement recency-weighted history selection
2. Add checkpoint summarization
3. Improve RAG content selection
4. Add content deduplication

### Phase 4: Advanced Features
1. Implement sliding window summarization
2. Add context compression for long tasks
3. Create context quality metrics
4. Add monitoring and alerting

## Benefits

1. **Accurate Context Management**: Know exactly how many tokens are being used
2. **Prevent Context Overflow**: Never exceed model's context window
3. **Better Coherence**: Prioritize important/recent information
4. **Efficient Resource Usage**: Maximize use of available context
5. **Scalability**: Handle long-running tasks without degradation
6. **Debuggability**: Clear visibility into what's included/excluded

## References

- tiktoken: https://github.com/openai/tiktoken
- Context Window Management: Various papers on LLM context management
- RAG Best Practices: Semantic search and retrieval optimization



