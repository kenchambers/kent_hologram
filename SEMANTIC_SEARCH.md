# Semantic Code Search - Quick Reference

This repository has **semantic search** powered by EmbeddixDB. Use it to navigate the codebase faster!

## Quick Start

### For Claude Code Users

Claude is pre-configured to use semantic search. Just ask questions naturally:

```
> "Where is the ARC solver implemented?"
> "How does neural consolidation work?"
> "Show me the chat interface code"
```

Claude will automatically use semantic search first before falling back to traditional file search.

### For Developers

Use the MCP tool directly:

```python
# In Claude Code
mcp__embeddixdb-kent-hologram__search_code(
    query="neural consolidation background worker",
    limit=5
)
```

## Common Queries

### Architecture Questions
- "holographic fact storage and retrieval"
- "fractal space vector generation"
- "HDC binding and bundling operations"

### Feature Implementation
- "ARC solver iterative reasoning"
- "neural consolidation long-term memory"
- "chat interface conversational learning"
- "confidence scoring and refusal policy"

### Components
- "resonator factorization ALS solver"
- "object detector for ARC grids"
- "ChromaDB persistence adapter"
- "metacognitive loop self-monitoring"

## Advanced Usage

### Filter by File Type
```python
search_code(
    query="consolidation",
    filters={"file_type": "code"}  # Exclude docs
)
```

### Filter by Code Structure
```python
search_code(
    query="memory storage",
    filters={"block_type": "class"}  # Only classes
)
```

### Adjust Result Count
```python
search_code(
    query="ARC transformation",
    limit=10  # Get more results
)
```

## Available Filters

- `file_type`: "code" | "documentation"
- `extension`: ".py" | ".md" | etc.
- `block_type`: "function" | "class" | "module_docstring" | "text_chunk"

## Collection Stats

Check what's indexed:

```python
mcp__embeddixdb-kent-hologram__get_collection_stats()
```

Returns:
- Total documents indexed
- Code vs documentation ratio
- File type distribution
- Index health status

## Why Use Semantic Search?

### Traditional Search (Grep/Glob)
```
❌ Requires exact keywords
❌ Returns too many irrelevant matches
❌ Doesn't understand context
❌ You have to guess file names
```

### Semantic Search (EmbeddixDB)
```
✅ Understands natural language
✅ Returns most relevant code first
✅ Provides context snippets
✅ Ranked by relevance score
```

## Examples

### Example 1: Find Implementation
**Question**: "How does the system prevent hallucinations?"

**Semantic Search Query**:
```python
search_code(
    query="hallucination prevention refusal bounded vocabulary",
    limit=5
)
```

**Results** (ranked by relevance):
1. `hologram/safety/refusal.py` - RefusalPolicy class
2. `hologram/core/codebook.py` - Vocabulary constraints
3. `hologram/retrieval/confidence.py` - Confidence thresholds
4. `README.md` - Bounded hallucination explanation

### Example 2: Understand Component
**Question**: "What does the resonator do?"

**Semantic Search Query**:
```python
search_code(
    query="resonator factorization subject verb object decomposition",
    limit=5,
    filters={"file_type": "code"}
)
```

**Results**:
1. `hologram/core/resonator.py` - Main Resonator class
2. `hologram/arc/transform_resonator.py` - ARC-specific resonator
3. `hologram/core/operations.py` - Bind/unbind operations
4. `tests/test_resonator.py` - Usage examples

### Example 3: Find Related Files
**Question**: "What files are involved in neural consolidation?"

**Semantic Search Query**:
```python
search_code(
    query="neural consolidation background thread long-term memory",
    limit=10
)
```

**Results** (all relevant files):
1. `hologram/consolidation/neural_memory.py`
2. `hologram/consolidation/manager.py`
3. `hologram/consolidation/types.py`
4. `scripts/arc_trainer.py` (uses consolidation)
5. `docs/CONSCIOUS_HOLOGRAM_ARCH.md` (architecture)

## Tips for Best Results

1. **Be Specific**: "ARC solver beam search" > "search"
2. **Include Context**: "chat interface conversational mode" > "chat"
3. **Use Domain Terms**: "HDC binding bundling" > "combining vectors"
4. **Combine Concepts**: "neural memory consolidation background worker" finds exact implementation

## Integration with Claude Code

Claude Code is configured to:
1. **Always try semantic search first**
2. Read `.claude/INSTRUCTIONS.md` for guidance
3. Fall back to Glob/Grep only when needed
4. Use semantic search results to inform which files to read

See `.claude/INSTRUCTIONS.md` for complete workflow details.

## Collection Information

**Collection**: `kent_hologram_code`

**Indexed Content**:
- All Python source files (`src/hologram/**/*.py`)
- All documentation (`*.md` files)
- Examples and scripts
- Test files

**Not Indexed**:
- Virtual environment files (`.venv/`)
- Build artifacts
- Git metadata
- Binary files

## Troubleshooting

### "No results found"
- Try broader terms: "consolidation" instead of "ConsolidationManager"
- Check spelling and terminology
- Use `get_collection_stats()` to verify index is healthy

### "Too many irrelevant results"
- Add more specific context to query
- Use filters to narrow scope
- Reduce `limit` to see only top matches

### "Results don't match what I'm looking for"
- Try different keywords or phrasing
- Include surrounding concepts
- Use multiple smaller searches instead of one broad search

## Learn More

- **EmbeddixDB Documentation**: `./embeddixdb/README.md`
- **Claude Instructions**: `.claude/INSTRUCTIONS.md`
- **Project Overview**: `README.md`
