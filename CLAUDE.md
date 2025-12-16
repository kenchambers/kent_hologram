# Kent Hologram - AI Coding Instructions

## CRITICAL: Semantic Search First

**YOU MUST use semantic search before Glob/Grep for code exploration.**

```
mcp__embeddixdb-kent-hologram__search_code(query="your question", limit=5)
```

### Why This Matters
- Semantic search understands code meaning, not just keywords
- Returns ranked, relevant results with context
- Glob/Grep returns noisy, unranked matches

### Search Examples
| Instead of Grep for... | Use semantic search query |
|------------------------|---------------------------|
| "consolidation" | "neural memory consolidation process" |
| "solver" | "ARC puzzle solver implementation" |
| "chat" | "conversational interface chat mode" |

### When Glob/Grep is OK
- Exact string replacement (after semantic search finds files)
- Verifying all occurrences of a known symbol
- Semantic search returned 0 results

## Project Structure
- `hologram/core/` - HDC primitives
- `hologram/arc/` - ARC solver
- `hologram/chat/` - Chat interface
- `hologram/consolidation/` - Memory consolidation
