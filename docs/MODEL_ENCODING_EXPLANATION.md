# Model Encoding Explanation

## What is a Tokenizer Encoding?

When we count "tokens" for an LLM, we're not counting words or characters - we're counting the atomic units that the model actually processes. Different models use different **tokenization schemes** (called "encodings") to break text into tokens.

### Why Different Encodings Matter

**Example:**
- The text "Hello, world!" might be:
  - **3 tokens** with one encoding: `["Hello", ",", " world!"]`
  - **4 tokens** with another: `["Hello", ",", " world", "!"]`
  - **2 tokens** with another: `["Hello,", " world!"]`

If we use the wrong encoding, our token count will be **wrong**, and we might:
- Exceed the model's context window
- Underutilize available context
- Make incorrect pruning decisions

## What are `cl100k_base` and `o200k_base`?

These are **encoding names** used by tiktoken (OpenAI's tokenizer library):

- **`cl100k_base`**: Used by GPT-3.5, GPT-4, and many other models
  - "cl" = "ChatGPT" / "cl100k" = 100k vocabulary size
  - Most common encoding
  
- **`o200k_base`**: Used by GPT-4o and newer models
  - "o200k" = 200k vocabulary size
  - Larger vocabulary = fewer tokens for same text

- **`p50k_base`**: Older encoding used by GPT-3

## How Our Code Handles This

```python
MODEL_ENCODINGS = {
    "gpt-4": "cl100k_base",        # GPT-4 uses cl100k_base
    "gpt-4o": "o200k_base",        # GPT-4o uses o200k_base
    "gemma3": "cl100k_base",       # Gemma models use cl100k_base
    "default": "cl100k_base",      # Fallback
}
```

### The Mapping Process

1. **You provide a model name**: `"gemma3:27b"`

2. **We look it up**:
   ```python
   model_name = "gemma3:27b"
   model_lower = "gemma3:27b".lower()  # "gemma3:27b"
   
   # Check exact match: "gemma3:27b" not in dict
   # Check partial match: "gemma3" in "gemma3:27b" ✓
   # Return: "cl100k_base"
   ```

3. **We get the encoding**:
   ```python
   encoding = tiktoken.get_encoding("cl100k_base")
   ```

4. **We use it to count tokens**:
   ```python
   tokens = encoding.encode("Your text here")
   token_count = len(tokens)
   ```

## Why This Matters for Your System

Since you're using **Ollama with Gemma models**, the code will:

1. Detect "gemma3" in the model name
2. Use `cl100k_base` encoding (which is correct for Gemma)
3. Count tokens accurately

## What If the Model Isn't in the List?

The code has a **fallback mechanism**:

```python
# If model not found, use default
return self.MODEL_ENCODINGS["default"]  # "cl100k_base"
```

Most models use `cl100k_base`, so this is usually safe. However, if you're using a model with a different encoding, you can:

1. **Add it to the mapping**:
   ```python
   MODEL_ENCODINGS = {
       # ... existing mappings
       "your-model": "correct_encoding_name",
   }
   ```

2. **Or specify the encoding directly** (if we add that feature)

## Real Example

```python
from KestrelAI.agents.context_manager import TokenCounter

# Your Ollama model
counter = TokenCounter(model_name="gemma3:27b")

# This will:
# 1. Detect "gemma3" in the name
# 2. Use "cl100k_base" encoding
# 3. Count tokens accurately

text = "Research AI opportunities for undergraduates"
tokens = counter.count_tokens(text)
print(f"Text has {tokens} tokens")
```

## Common Encodings Reference

| Encoding | Used By | Vocabulary Size |
|----------|---------|----------------|
| `cl100k_base` | GPT-3.5, GPT-4, Gemma, LLaMA | 100k |
| `o200k_base` | GPT-4o | 200k |
| `p50k_base` | GPT-3 (older) | 50k |
| `r50k_base` | GPT-3 (older) | 50k |

## Summary

- **Encoding** = How the model breaks text into tokens
- **Different models** = Different encodings
- **Wrong encoding** = Wrong token count = Context overflow/underflow
- **Our code** = Automatically maps model names to correct encodings
- **Fallback** = Uses `cl100k_base` if model unknown (safe for most cases)

The encoding mapping is just a **convenience feature** - it automatically picks the right tokenizer for your model so you don't have to know the encoding name yourself.



