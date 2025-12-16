# Developer Guide - Kent Hologram

Welcome to Kent Hologram! This guide will help you navigate and contribute to the codebase efficiently.

## 🔍 Primary Navigation: Semantic Search

**Before you start exploring**, know that this repository has **AI-powered semantic search**. Use it first!

### Quick Start with Semantic Search

Instead of guessing file locations or grepping blindly:

```python
# In Claude Code or any MCP-enabled tool
mcp__embeddixdb-kent-hologram__search_code(
    query="how does neural consolidation work",
    limit=5
)
```

You'll get ranked results with:
- File paths
- Relevance scores (0-1)
- Code context snippets
- Direct links to implementations

### Examples

**Question**: "Where is the ARC solver?"

```python
search_code(query="ARC solver implementation", limit=3)
```

**Results**:
1. `src/hologram/arc/solver.py` - Main solver class (0.87 relevance)
2. `src/hologram/arc/iterative_solver.py` - Multi-step solver (0.81 relevance)
3. `scripts/arc_trainer.py` - Training orchestrator (0.73 relevance)

---

**Question**: "How does the system prevent hallucinations?"

```python
search_code(query="hallucination prevention bounded vocabulary", limit=5)
```

**Results**:
1. `src/hologram/safety/refusal.py` - RefusalPolicy class
2. `src/hologram/core/codebook.py` - Vocabulary constraints
3. `src/hologram/retrieval/confidence.py` - Confidence thresholds
4. `README.md` - Architecture explanation

## 🏗️ Architecture Overview

### Core Modules

Use semantic search to explore each:

#### 1. HDC Primitives (`hologram/core/`)

```python
search_code("HDC vector operations binding bundling")
```

- **VectorSpace**: Random vector generation with deterministic seeds
- **Codebook**: Vocabulary management (concepts → vectors)
- **Operations**: Bind, bundle, unbind operations
- **Resonator**: Factorization via Alternating Least Squares

#### 2. Memory Systems (`hologram/memory/`, `hologram/consolidation/`)

```python
search_code("holographic fact storage consolidation")
```

- **FactStore**: S-P-O (subject-predicate-object) fact storage
- **MemoryTrace**: Holographic bundling of facts
- **NeuralMemory**: Long-term consolidation
- **ConsolidationManager**: Background worker for sleep-inspired learning

#### 3. ARC Solver (`hologram/arc/`)

```python
search_code("ARC solver transformation detection")
```

- **HolographicARCSolver**: Main solver orchestration
- **IterativeSolver**: Multi-step state traversal
- **ObjectDetector**: Grid pattern detection
- **TransformationResonator**: Resonance-based factorization

#### 4. Conversational Learning (`hologram/conversation/`, `hologram/chat/`)

```python
search_code("conversational intent classification learning")
```

- **IntentClassifier**: HDC-based intent detection
- **ConversationMemory**: Multi-level context tracking
- **ChatInterface**: Interactive REPL
- **VentriloquistGenerator**: SLM-based natural responses

#### 5. Safety & Retrieval (`hologram/safety/`, `hologram/retrieval/`)

```python
search_code("confidence scoring refusal citation")
```

- **RefusalPolicy**: Confidence-based refusal
- **CitationEnforcer**: Fact attribution
- **ConfidenceScorer**: Similarity interpretation

## 📚 Common Development Tasks

### Task 1: Add a New Transformation Type

**Step 1**: Search for existing transformations
```python
search_code("ARC transformation types actions targets modifiers")
```

**Step 2**: Read relevant files identified by search

**Step 3**: Follow patterns found in results

### Task 2: Understand How Memory Persists

**Step 1**: Search for persistence
```python
search_code("ChromaDB persistence save load facts")
```

**Step 2**: Examine top results

**Step 3**: Look at examples in results

### Task 3: Debug Consolidation Issues

**Step 1**: Search consolidation logic
```python
search_code("neural consolidation background thread queue")
```

**Step 2**: Find consolidation manager and worker

**Step 3**: Check tests and usage examples

### Task 4: Add New Intent Type

**Step 1**: Search intent classification
```python
search_code("intent classification HDC examples teaching")
```

**Step 2**: Study IntentClassifier implementation

**Step 3**: Follow example patterns

## 🧪 Running Tests

### Unit Tests

```bash
# Run all tests
uv run pytest

# Run specific module
uv run pytest tests/arc/

# With coverage
uv run pytest --cov=hologram
```

### Integration Tests

```bash
# Run ARC benchmark
uv run python scripts/arc_trainer.py --max-rounds 1

# Run chat interface
uv run python -m hologram.chat --no-persist
```

## 🎯 Best Practices

### 1. Search Before You Code

```python
# ❌ Don't: Blindly grep or guess file locations
grep -r "consolidation" src/

# ✅ Do: Use semantic search with context
search_code("consolidation manager background worker neural")
```

### 2. Use Filters for Precision

```python
# Only search functions
search_code(
    "resonator factorization",
    filters={"block_type": "function"}
)

# Only search code (no docs)
search_code(
    "ARC solver",
    filters={"file_type": "code"}
)
```

### 3. Start Broad, Then Narrow

```python
# Step 1: Broad search
search_code("neural memory", limit=10)

# Step 2: Narrow based on results
search_code("neural memory consolidation background thread", limit=5)
```

### 4. Read Documentation Results Too

```python
# Include documentation in search
search_code("neural consolidation architecture", limit=10)
# Results include both code AND architecture docs
```

## 📖 Key Documentation Files

Use semantic search to find documentation:

```python
# Architecture documentation
search_code(
    "architecture diagrams holographic consciousness",
    filters={"file_type": "documentation"}
)

# Training guides
search_code(
    "training guide overnight crew ARC",
    filters={"extension": ".md"}
)
```

**Important Docs**:
- `README.md` - Main overview and quick start
- `SEMANTIC_SEARCH.md` - Semantic search guide (this tool!)
- `TRAINING_GUIDE.md` - How to train the system
- `docs/CONSCIOUS_HOLOGRAM_ARCH.md` - Complete architecture
- `docs/HDC_FOR_BEGINNERS.md` - HDC concepts explained
- `CONTEXT_FLOW_DIAGRAMS.md` - Data flow visualizations

## 🔧 Development Workflow

### Standard Flow

1. **Search for similar code**
   ```python
   search_code("similar to what I want to add")
   ```

2. **Read relevant implementations**
   - Use file paths from search results
   - Check function signatures and patterns

3. **Write your code**
   - Follow established patterns
   - Use same abstractions

4. **Add tests**
   ```python
   search_code("test examples for similar feature")
   ```

5. **Run validation**
   ```bash
   uv run pytest tests/
   ```

### Example: Adding a New ARC Transformation

```python
# 1. Search existing transformations
search_code("ARC transformation executor apply", limit=5)
# → Find TransformationExecutor in arc/executor.py

# 2. Search action types
search_code("ACTIONS TARGETS MODIFIERS vocabulary", limit=3)
# → Find vocabulary definitions in arc/types.py

# 3. Search verification
search_code("verify transformation training pairs", limit=3)
# → Find SearchVerifier in arc/search_verifier.py

# 4. Read identified files and implement
# 5. Add tests following existing test patterns
```

## 🚀 Performance Tips

### Semantic Search is Fast
- Typical query: **< 100ms**
- Much faster than reading multiple files
- Pre-ranked by relevance

### Use Collection Stats

```python
mcp__embeddixdb-kent-hologram__get_collection_stats()
```

Shows:
- Total documents indexed (~500+)
- Index health
- Available filters

## 🐛 Troubleshooting

### "Can't find what I'm looking for"

1. Try different terminology
   ```python
   # Instead of: "save memory"
   search_code("persistence ChromaDB save load facts")
   ```

2. Broaden your search
   ```python
   # Increase limit
   search_code("consolidation", limit=15)
   ```

3. Remove filters
   ```python
   # Don't restrict file type initially
   search_code("neural memory")  # All results
   ```

### "Too many results"

1. Add more context
   ```python
   # Too broad
   search_code("memory")

   # Better
   search_code("neural memory consolidation background worker")
   ```

2. Use filters
   ```python
   search_code("memory", filters={"block_type": "class"})
   ```

3. Reduce limit
   ```python
   search_code("memory", limit=3)  # Top 3 only
   ```

## 📞 Getting Help

### For Claude Code Users

Claude is configured to use semantic search automatically. Just ask:

```
> "How does the ARC solver work?"
> "Where is neural consolidation implemented?"
> "Show me examples of HDC binding"
```

### Manual Search

Run slash command:
```
/search-help
```

Or read full guide:
```
SEMANTIC_SEARCH.md
```

## 🎓 Learning Path

### Week 1: Understand HDC Basics
```python
# Day 1-2: Vector operations
search_code("HDC vector binding bundling operations")

# Day 3-4: Codebook and vocabulary
search_code("codebook vocabulary concept encoding")

# Day 5-7: Fact storage
search_code("fact store subject predicate object holographic")
```

### Week 2: Memory Systems
```python
# Day 1-3: Working memory
search_code("memory trace bundling interference patterns")

# Day 4-7: Neural consolidation
search_code("neural consolidation long-term background thread")
```

### Week 3: ARC Solver
```python
# Day 1-3: Basic solver
search_code("ARC solver object detection transformation")

# Day 4-5: Iterative reasoning
search_code("iterative solver state traversal multi-step")

# Day 6-7: Training
search_code("ARC trainer curriculum difficulty")
```

### Week 4: Conversational Learning
```python
# Day 1-3: Intent classification
search_code("intent classifier HDC learning examples")

# Day 4-5: Chat interface
search_code("chat interface conversational REPL")

# Day 6-7: Integration
search_code("chatbot persistent learning ventriloquist")
```

## 🏆 Contribution Guidelines

1. **Search first, code second**
   - Use semantic search to find similar implementations
   - Follow existing patterns

2. **Document with context**
   - Good docstrings help semantic search
   - Use domain terminology

3. **Test thoroughly**
   - Find test examples via semantic search
   - Match existing test patterns

4. **Update docs**
   - Add relevant documentation
   - Use keywords that help search

## 🔗 Quick Links

- **Semantic Search Guide**: [SEMANTIC_SEARCH.md](./SEMANTIC_SEARCH.md)
- **Claude Instructions**: [.claude/INSTRUCTIONS.md](./.claude/INSTRUCTIONS.md)
- **Main README**: [README.md](./README.md)
- **Architecture**: [docs/CONSCIOUS_HOLOGRAM_ARCH.md](./docs/CONSCIOUS_HOLOGRAM_ARCH.md)
- **HDC Concepts**: [docs/HDC_FOR_BEGINNERS.md](./docs/HDC_FOR_BEGINNERS.md)

---

**Remember**: Semantic search is your superpower. Use it first, use it often! 🚀
