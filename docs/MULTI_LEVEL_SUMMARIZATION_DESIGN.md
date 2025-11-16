# Multi-Level Summarization Design Explanation

## Overview

The multi-level summarization system creates **hierarchical summaries** at different compression levels, allowing the system to retrieve the appropriate level of detail based on available token budget.

## How It Works

### 1. **Progressive Summarization (Sequential Compression)**

The system creates summaries **sequentially**, where each level builds on the previous:

```
Original Content (1000 tokens)
    ↓
Medium Summary (500 tokens) ← Summarizes original
    ↓
Summary (200 tokens) ← Summarizes medium summary
    ↓
Executive (100 tokens) ← Summarizes summary
```

**Key Design Decision:** Each level summarizes the **previous level**, not the original.

**Why?**
- More consistent compression ratios
- Each level is a "refinement" of the previous
- Avoids information loss from repeatedly summarizing the same long text

**Potential Issue:** Error accumulation - each summarization can lose information, and that loss compounds.

### 2. **Compression Levels**

```python
DEFAULT_LEVELS = [
    SummaryLevel("detailed", 1.0, "Full content"),      # 100% - no compression
    SummaryLevel("medium", 0.5, "50% compression"),     # 50% - half size
    SummaryLevel("summary", 0.2, "20% compression"),    # 20% - one-fifth size
    SummaryLevel("executive", 0.1, "10% compression")   # 10% - one-tenth size
]
```

### 3. **Adaptive Retrieval**

When you need content, the system picks the **most detailed level that fits**:

```python
# You have 300 tokens available
summaries = {
    "detailed": 1000 tokens,  # Too big ❌
    "medium": 500 tokens,     # Too big ❌
    "summary": 200 tokens,    # Fits! ✅
    "executive": 100 tokens   # Fits, but less detailed
}

# Returns: summary level (200 tokens)
```

## Design Flow

### Creating Summaries

```python
summarizer = MultiLevelSummarizer(llm, token_counter)

# Step 1: Start with original
current = "Long research content..."  # 1000 tokens

# Step 2: Create medium (50% of original)
medium = llm.summarize(current, target=500 tokens)
# Result: 500 tokens

# Step 3: Create summary (20% of medium, not original!)
summary = llm.summarize(medium, target=100 tokens)
# Result: 100 tokens

# Step 4: Create executive (10% of summary)
executive = llm.summarize(summary, target=10 tokens)
# Result: 10 tokens
```

### Retrieving Content

```python
# You need content for context, have 150 tokens available
content, level = summarizer.retrieve_adaptive(summaries, max_tokens=150)

# System checks:
# - detailed (1000 tokens) > 150? Yes, skip
# - medium (500 tokens) > 150? Yes, skip
# - summary (100 tokens) <= 150? Yes! Return this
# Result: ("summary content", "summary")
```

## Design Decisions & Rationale

### ✅ **Progressive Summarization (Sequential)**

**Decision:** Each level summarizes the previous level, not the original.

**Pros:**
- More consistent compression
- Each level is a refinement
- Avoids repeatedly processing very long text

**Cons:**
- **Error accumulation** - information loss compounds
- If medium summary misses something, it's gone forever
- More LLM calls (costs more)

**Alternative Considered:** Each level summarizes original independently
- Pros: No error accumulation
- Cons: Less consistent compression, more expensive (summarize long text multiple times)

### ✅ **Fixed Compression Ratios**

**Decision:** Use fixed ratios (1.0, 0.5, 0.2, 0.1)

**Pros:**
- Predictable token usage
- Easy to reason about
- Works well for most cases

**Cons:**
- May not match actual content structure
- Some content compresses better/worse than others
- Rigid - doesn't adapt to content type

**Alternative Considered:** Dynamic compression based on content analysis
- More complex, but could be more accurate

### ✅ **LLM-Based Summarization**

**Decision:** Use LLM to create summaries (not extractive methods)

**Pros:**
- Coherent summaries (not just chunks)
- Preserves meaning and relationships
- Can follow instructions about what to preserve

**Cons:**
- **Expensive** - multiple LLM calls per content
- **Slow** - sequential processing
- **Unpredictable** - LLM may not hit target token count exactly
- **Quality varies** - depends on LLM quality

**Alternative Considered:** Extractive summarization (sentence selection)
- Faster and cheaper
- But doesn't create coherent summaries (just chunks)

## Potential Concerns

### 1. **Error Accumulation** ⚠️

**Problem:** Each summarization can lose information, and that loss compounds.

**Example:**
```
Original: "NSF REU deadline is February 15, 2025, requires 3.5 GPA"
    ↓ (medium summary might miss date)
Medium: "NSF REU requires 3.5 GPA"
    ↓ (summary might miss GPA requirement)
Summary: "NSF REU program exists"
    ↓ (executive loses everything)
Executive: "Research opportunities"
```

**Mitigation:**
- Use prompts that emphasize preserving key facts
- Could add fact extraction step before summarization
- Could store key facts separately

### 2. **Cost & Latency** ⚠️

**Problem:** Creating all levels requires multiple LLM calls.

**Example:**
- Original: 1000 tokens
- Medium: 1 LLM call (~2-5 seconds, costs tokens)
- Summary: 1 LLM call (~2-5 seconds, costs tokens)
- Executive: 1 LLM call (~2-5 seconds, costs tokens)
- **Total: 3 LLM calls, 6-15 seconds, significant token cost**

**Mitigation:**
- Only create summaries when needed (lazy evaluation)
- Cache summaries
- Could use cheaper/faster model for summarization

### 3. **Token Count Accuracy** ⚠️

**Problem:** LLM may not hit target token count exactly.

**Example:**
- Target: 200 tokens
- LLM produces: 250 tokens (25% over)
- System truncates, potentially losing important ending

**Current Mitigation:**
- Check if summary exceeds target by 50%, truncate if so
- But truncation can cut off mid-sentence

**Better Approach:**
- Iterative refinement (ask LLM to shorten if too long)
- Or accept variance and adjust budgets

### 4. **Quality Consistency** ⚠️

**Problem:** LLM summarization quality varies.

**Factors:**
- Model quality
- Prompt quality
- Content complexity
- Randomness in generation

**Mitigation:**
- Well-crafted prompts (current implementation)
- Could add quality checks
- Could use multiple attempts and pick best

### 5. **When to Create Summaries?** ⚠️

**Current Design:** Create all levels upfront when `create_summary_hierarchy()` is called.

**Concerns:**
- Expensive if summaries aren't used
- Slow if done synchronously
- Wastes resources if only one level is needed

**Better Approach:**
- **Lazy evaluation** - create summaries on-demand
- **Caching** - store summaries once created
- **Background processing** - create summaries asynchronously

## Usage Pattern

### Current Usage (As Designed)

```python
# 1. Create all summaries upfront
summaries = summarizer.create_summary_hierarchy(long_content)
# Takes: 3 LLM calls, 6-15 seconds

# 2. Later, retrieve appropriate level
content, level = summarizer.retrieve_adaptive(summaries, max_tokens=200)
# Takes: milliseconds (just dictionary lookup)
```

### Better Usage (Lazy Evaluation)

```python
# 1. Store original, don't summarize yet
original = long_content

# 2. When needed, create summary on-demand
if need_summary:
    summary = summarizer._summarize(original, target_tokens=200, level=medium_level)
    # Only 1 LLM call, only when needed
```

## Integration Points

### Where Summarization Happens

1. **Checkpoints** - When creating checkpoints, could create summaries
2. **RAG Content** - When retrieving from RAG, could summarize if too long
3. **History** - When history gets long, could summarize older entries
4. **Previous Reports** - When passing to next subtask, could summarize

### Current Integration (Not Yet Done)

The summarizer is created but **not yet integrated** into the agent workflow. It needs to be:

1. Called when content is stored (checkpoints, summaries)
2. Used when building context (retrieve appropriate level)
3. Cached to avoid re-summarization

## Recommendations

### Short-Term Improvements

1. **Add lazy evaluation** - Only create summaries when needed
2. **Add caching** - Store summaries once created
3. **Improve prompts** - Better instructions for preserving key facts
4. **Add fact extraction** - Extract key facts before summarization

### Long-Term Improvements

1. **Extractive + Abstractive hybrid** - Use extractive for speed, abstractive for coherence
2. **Quality metrics** - Measure summary quality, re-summarize if poor
3. **Adaptive compression** - Adjust ratios based on content type
4. **Parallel summarization** - Create multiple levels in parallel (if using multiple models)

## Summary

The multi-level summarization design is **sound but has trade-offs**:

**Strengths:**
- ✅ Enables adaptive retrieval based on token budget
- ✅ Creates coherent summaries (not just chunks)
- ✅ Progressive compression is efficient

**Weaknesses:**
- ⚠️ Error accumulation from sequential summarization
- ⚠️ Expensive (multiple LLM calls)
- ⚠️ Slow (sequential processing)
- ⚠️ Quality varies

**Key Question:** Is the cost/quality trade-off worth it for your use case?

For research information where **accuracy is critical**, you might want to:
- Use summarization more sparingly
- Add fact extraction to preserve key details
- Use extractive methods for some levels
- Only summarize when absolutely necessary



