# Enhanced Multi-Level Summarization Design

## Overview

The enhanced multi-level summarization system prioritizes **quality and research accuracy** through fact extraction, preservation, and research-specific prompts.

## Key Enhancements

### 1. **Fact Extraction & Preservation**

**ExtractedFacts Dataclass:**
- Extracts critical research information before summarization
- Preserves: deadlines, dates, requirements, contact info, URLs, programs, amounts, eligibility
- Facts are appended to summaries to ensure they're never lost

**How It Works:**
```python
# Before summarization
facts = extract_facts(content)
# facts.deadlines = ["February 15, 2025"]
# facts.urls = ["https://nsf.gov/reu"]

# During summarization
summary = summarize(content, target_tokens, facts)
# Summary includes facts section: "--- Key Facts ---\nDeadlines: February 15, 2025..."
```

**Benefits:**
- ✅ Critical information never lost
- ✅ Facts preserved even in highly compressed summaries
- ✅ Research accuracy maintained

### 2. **Research-Specific Prompts**

**Enhanced System Prompts:**
- Emphasize preserving ALL actionable information
- Prioritize accuracy over brevity
- Explicit instructions to preserve exact dates, numbers, requirements
- Research-focused language (programs, grants, deadlines, eligibility)

**Example Prompt:**
```
CRITICAL PRESERVATION REQUIREMENTS:
- ALL deadlines and dates (exact dates, not approximations)
- ALL requirements and eligibility criteria (exact wording)
- ALL contact information
- ALL URLs and application links
- ALL funding amounts and costs
- Program names, grant names (exact names)
```

### 3. **Lazy Evaluation (On-Demand Summarization)**

**New Method: `create_summary_on_demand()`**
- Only creates summary when needed
- More efficient than creating all levels upfront
- Chooses appropriate level based on token budget
- Still extracts and preserves facts

**Usage:**
```python
# Instead of creating all levels upfront
summary, level, facts = summarizer.create_summary_on_demand(
    content, 
    max_tokens=500
)
# Only 1 LLM call, only when needed
```

### 4. **Quality Validation**

**New Method: `validate_summary_quality()`**
- Checks if critical facts are preserved
- Reports missing information
- Provides quality metrics
- Helps identify when summaries lose important information

**Usage:**
```python
quality = summarizer.validate_summary_quality(original, summary, facts)
# Returns: {
#   "has_deadlines": True,
#   "has_urls": True,
#   "missing_facts": ["Contact: info@nsf.gov"],
#   "compression_ratio": 0.3
# }
```

## Integration Points

### With WebResearchAgent

**Checkpoint Creation:**
```python
# In _create_checkpoint()
checkpoint = self._chat([...])

# Create summary hierarchy for storage
hierarchy = self.summarizer.create_summary_hierarchy(checkpoint)
# Store both summaries and facts
self._store_checkpoint_with_summaries(checkpoint, hierarchy)
```

**Context Building:**
```python
# In _build_context()
# Retrieve appropriate summary level based on token budget
summary, level, facts = self.summarizer.create_summary_on_demand(
    checkpoint,
    max_tokens=self.token_budget.checkpoints
)
```

**RAG Storage:**
```python
# When storing to RAG, create summaries
hierarchy = self.summarizer.create_summary_hierarchy(content)
# Store summaries at different levels
for level_name, summary in hierarchy["summaries"].items():
    self.memory.add(f"{doc_id}_{level_name}", summary, metadata)
```

### With URL Flag System

**Integration:**
- Facts extraction preserves URLs
- URL flags are maintained in summaries
- Facts section includes URL flags, not full URLs
- URL reference table included when facts are appended

## Design Decisions

### ✅ **Fact Extraction Before Summarization**

**Why:** Ensures facts are captured before any information loss occurs.

**Trade-off:** Adds one LLM call, but guarantees fact preservation.

### ✅ **Facts Appended to Summaries**

**Why:** Even if summary loses information, facts section preserves it.

**Trade-off:** Uses more tokens, but ensures accuracy.

### ✅ **Research-Specific Prompts**

**Why:** Generic prompts don't emphasize research-critical information.

**Trade-off:** More verbose prompts, but better results for research.

### ✅ **Lazy Evaluation Option**

**Why:** Don't create summaries if not needed.

**Trade-off:** Slightly more complex API, but more efficient.

## Quality Assurance

### Fact Preservation Guarantees

1. **Extraction:** Facts extracted before summarization
2. **Inclusion:** Facts explicitly included in summarization prompts
3. **Appendage:** Facts appended to summaries as separate section
4. **Validation:** Quality checks verify facts are preserved

### Error Handling

- **LLM extraction fails:** Falls back to regex-based extraction
- **Summary too long:** Truncates but preserves facts section
- **Summary loses facts:** Facts section ensures they're still available

## Usage Examples

### Example 1: Checkpoint with Summarization

```python
# Create checkpoint
checkpoint = "NSF REU deadline February 15, 2025. Contact info@nsf.gov..."

# Create summaries for storage
hierarchy = summarizer.create_summary_hierarchy(checkpoint)
# Returns: {
#   "summaries": {
#     "detailed": "...",
#     "medium": "...",
#     "summary": "..."
#   },
#   "facts": ExtractedFacts(deadlines=["February 15, 2025"], ...)
# }

# Store in RAG with summaries
for level, summary in hierarchy["summaries"].items():
    memory.add(f"checkpoint_{level}", summary, metadata)
```

### Example 2: Context Building with Token Budget

```python
# Need checkpoint in context, have 200 tokens available
summary, level, facts = summarizer.create_summary_on_demand(
    checkpoint,
    max_tokens=200
)

# summary = "NSF REU program...\n\n--- Key Facts ---\nDeadlines: February 15, 2025..."
# level = "summary"
# facts = ExtractedFacts(...)
```

### Example 3: Quality Validation

```python
# After creating summary, validate quality
quality = summarizer.validate_summary_quality(original, summary, facts)

if quality["missing_facts"]:
    logger.warning(f"Summary lost facts: {quality['missing_facts']}")
    # Could re-summarize or use more detailed level
```

## Benefits for Research Quality

1. **No Information Loss:** Facts are preserved even in compressed summaries
2. **Accuracy:** Research-specific prompts emphasize exact details
3. **Validation:** Quality checks ensure important information isn't lost
4. **Flexibility:** Lazy evaluation allows efficient usage
5. **Integration:** Works seamlessly with existing URL flag system

## Next Steps for Integration

1. **Integrate into checkpoint creation** - Create summaries when checkpoints are created
2. **Integrate into context building** - Use on-demand summarization based on token budget
3. **Integrate into RAG storage** - Store summaries at multiple levels
4. **Add quality monitoring** - Log quality metrics to track information preservation



