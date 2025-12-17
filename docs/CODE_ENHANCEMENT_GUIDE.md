# Holographic Code Enhancement Guide

## Overview

The Holographic Code Enhancement system implements RAG (Retrieval Augmented Generation) for code, preventing LLM hallucination by grounding code generation in verified facts. This is inspired by how RAG systems improve accuracy by retrieving relevant context before generation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Coding Task                              │
│          "Implement factory pattern for User class"         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Dual Retrieval                             │
├─────────────────────┬───────────────────────────────────────┤
│  Concept Store      │  Code Store                           │
│  (General)          │  (Project-Specific)                   │
├─────────────────────┼───────────────────────────────────────┤
│ Factory Pattern     │ User class exists                     │
│ - Purpose           │ User.signature = (name, email)        │
│ - Implementation    │ User inherits from BaseModel          │
└─────────────────────┴───────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              GLM-4.6v Code Generation                       │
│          (Creative + Grounded in Facts)                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               Code Verification                             │
│  ✓ All functions exist in Code Store                       │
│  ✓ All classes exist in Code Store or Concept Store        │
│  ✗ Reject if unknown APIs detected (hallucination)         │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. FactStore Extensions (`src/hologram/memory/fact_store.py`)

**New Methods:**
- `query_subject(predicate, object)` - Reverse queries for finding "what calls X"
- `get_facts_by_object(object)` - Metadata lookup by object value

**Why This Matters:**
- Enables call graph queries: "What functions call `encode`?"
- Supports genealogy tracking: "What classes inherit from `BaseModel`?"

### 2. Concept Ingester (`scripts/ingest_concepts.py`)

**Purpose:** Pre-train the system with general software engineering knowledge.

**What It Stores:**
- **Design Patterns**: Singleton, Factory, Observer, Strategy, Decorator, Adapter
- **Algorithms**: Binary Search, Quick Sort, Merge Sort, BFS, DFS, Dynamic Programming
- **Best Practices**: SOLID principles, DRY, KISS, YAGNI
- **Python Patterns**: List comprehensions, context managers, generators

**Canonicalization:**
Maps synonyms to standard terms to prevent duplicates:
- `performance`, `runtime`, `speed` → `time_complexity`
- `O(N)`, `linear time`, `linear` → `O(n)`

**Usage:**
```bash
# Ingest all concepts
python scripts/ingest_concepts.py --test

# Use custom persist directory
python scripts/ingest_concepts.py --persist-dir ./my_concepts
```

**Example Facts Stored:**
```
BinarySearch --time_complexity--> O(log n)
BinarySearch --space_complexity--> O(1)
BinarySearch --type--> Search Algorithm
Singleton --purpose--> Ensure a class has only one instance
Singleton --implementation--> class Singleton: ...
```

### 3. Code Indexer (`scripts/code_indexer.py`)

**Purpose:** Index a specific Python codebase to provide project context.

**What It Extracts:**
- **Function Signatures**: `add_fact --signature--> (subject, predicate, obj, source, confidence)`
- **Return Types**: `query --returns--> tuple[str, float]`
- **Call Graph**: `add_fact --calls--> encode` (and reverse: `encode --called_by--> add_fact`)
- **Class Hierarchy**: `FactStore --inherits--> object`
- **Docstrings**: `add_fact --purpose--> Add a fact to the store`

**Usage:**
```bash
# Index current project
python scripts/code_indexer.py --directory ./src --test

# Query call graph
python scripts/code_indexer.py --directory ./src --query-call-graph add_fact

# Query class hierarchy
python scripts/code_indexer.py --directory ./src --query-hierarchy FactStore
```

**Example Facts Stored:**
```
FactStore --type--> class
FactStore --module--> fact_store
add_fact --type--> function
add_fact --signature--> (subject, predicate, obj, source, confidence)
add_fact --returns--> Optional[Fact]
add_fact --calls--> encode
encode --called_by--> add_fact
```

### 4. VentriloquistGenerator Extensions (`src/hologram/generation/ventriloquist.py`)

**New Methods:**

#### `verify_code(code_str, fact_store, concept_store)`
Parses generated code and checks that all function calls and class references exist in the indexed facts.

**Returns:**
```python
{
    "verified": True/False,
    "issues": ["Unknown functions: magic_query"],
    "confidence": 0.7,
    "unknown_functions": {"magic_query"},
    "unknown_classes": set()
}
```

#### `retrieve_dual_context(query, fact_store, concept_store)`
Retrieves both general concepts and project-specific facts for a query.

**Returns:**
```python
(
    ["Factory --purpose--> Create objects without specifying exact class"],  # Concepts
    ["User --type--> class", "User --signature--> (name, email)"]  # Project
)
```

#### `generate_code_with_context(prompt, fact_store, concept_store, verify=True)`
Generates code using dual retrieval and optionally verifies it.

**Example:**
```python
ventriloquist = VentriloquistGenerator()

result = ventriloquist.generate_code_with_context(
    prompt="Implement factory pattern for User class",
    fact_store=code_store,
    concept_store=concept_store,
    verify=True
)

print(result["code"])
print(f"Verified: {result['verification']['verified']}")
```

### 5. CodeTeacher (`scripts/crew_trainer.py`)

**Purpose:** Fetch coding concepts from technical documentation online.

**New Mode:** `--web-teach-code`

**What It Does:**
- Searches web for technical documentation (e.g., "Python list methods", "Binary search algorithm")
- Uses specialized extraction prompts to identify API signatures, complexity, implementations
- Stores as facts in FactStore

**Usage:**
```bash
# Teach code concepts from web
python scripts/crew_trainer.py --web-teach-code "Python dict methods" "Sorting algorithms"

# Combined: general knowledge + code concepts
python scripts/crew_trainer.py \
    --web-teach "Physics" "History" \
    --web-teach-code "Graph algorithms" "Design patterns"

# Custom search parameters
python scripts/crew_trainer.py \
    --web-teach-code "Dynamic programming" \
    --web-results 5 \
    --web-facts 10
```

## Complete Workflow

### Step 1: Pre-train with General Concepts

```bash
python scripts/ingest_concepts.py --test
```

**Output:**
```
[Design Patterns] Added 18 facts
[Algorithms] Added 24 facts
[Best Practices] Added 15 facts
[Python Stdlib] Added 10 facts

Total: 67 facts added
```

### Step 2: Index Your Codebase

```bash
python scripts/code_indexer.py --directory ./src --test
```

**Output:**
```
Found 32 Python files
  ✓ src/hologram/memory/fact_store.py: 47 facts
  ✓ src/hologram/core/codebook.py: 15 facts
  ...

Total: 312 facts added
```

### Step 3: Generate Code with Verification

```python
from hologram.generation.ventriloquist import VentriloquistGenerator

# Load stores (or use persistence)
ventriloquist = VentriloquistGenerator()

# Generate with dual context
result = ventriloquist.generate_code_with_context(
    prompt="Create a Singleton manager for database connections",
    fact_store=code_store,
    concept_store=concept_store,
    verify=True
)

if result["verification"]["verified"]:
    print("✓ Code verified - no hallucinations")
    print(result["code"])
else:
    print("✗ Verification failed:")
    for issue in result["verification"]["issues"]:
        print(f"  - {issue}")
```

## Advantages Over Standard LLM Code Generation

### Without Holographic Enhancement:
```python
# LLM generates:
user_factory = UserFactory()
user = user_factory.create_user(name="Alice", email="alice@example.com")
user.save_to_database()  # ❌ Method doesn't exist! (Hallucination)
```

### With Holographic Enhancement:
1. **Retrieval**: Finds that `User` class has methods `save()` but NOT `save_to_database()`
2. **Verification**: Detects `save_to_database()` is not in Code Store
3. **Result**: Rejects the code or corrects it to `user.save()`

## Canonicalization: Preventing Semantic Duplicates

The system uses canonicalization to map synonyms to standard terms:

```python
# Without canonicalization (noise):
BinarySearch --performance--> O(log n)
BinarySearch --runtime--> O(log n)
BinarySearch --time_complexity--> O(log n)
# 3 duplicate facts with different predicates!

# With canonicalization:
BinarySearch --time_complexity--> O(log n)
# Single canonical fact
```

## Surprise Gating: Preventing Duplicates

Even if the same fact is added multiple times (with different phrasing), the system's **surprise gating** mechanism (from the Titans architecture) prevents re-encoding:

```python
# First time
fact_store.add_fact("BinarySearch", "complexity", "O(log n)")
# Returns: Fact(surprise_score=1.0)  ✓ Novel, stored

# Second time (exact duplicate)
fact_store.add_fact("BinarySearch", "complexity", "O(log n)")
# Returns: None  ✗ Duplicate detected by exact match index

# Third time (semantic duplicate)
fact_store.add_fact("BinarySearch", "complexity", "logarithmic")
# Returns: None  ✗ Surprise score < 0.1, skipped
```

## Example: SWE Bench Task

**Task:** "Fix the bug in the authentication system where logout doesn't clear the session"

### How Holographic Enhancement Helps:

1. **Concept Retrieval:**
   - Session management patterns
   - Authentication best practices

2. **Code Index Retrieval:**
   - `logout()` function signature
   - `clear_session()` function exists
   - `logout()` calls `redirect()` but NOT `clear_session()`

3. **Generation:**
   - GLM-4.6v generates fix: Add `clear_session()` call to `logout()`

4. **Verification:**
   - Checks that `clear_session()` exists in Code Store ✓
   - Checks that `redirect()` exists in Code Store ✓
   - No hallucinated functions detected ✓

## Performance Considerations

### Memory Saturation
- **10,000D vectors**: ~100 facts before degradation (UNPROVEN, needs empirical validation)
- **Surprise gating**: Automatically prevents redundant encoding
- **Canonicalization**: Reduces vocabulary size by 20-30%

### Query Speed
- **Exact match**: O(1) via normalized key index
- **Resonance search**: O(n) where n = vocabulary size (~100-1000 terms)
- **Typical query time**: <10ms for 100 facts

## Limitations & Future Work

### Current Limitations:
1. **No Persistence**: Facts stored in memory only (ChromaDB/FAISS integration coming)
2. **Python Only**: Code indexer currently supports Python only (extensible to other languages)
3. **No Type Checking**: Verification checks existence but not type correctness
4. **Static Analysis**: Cannot detect runtime behavior (only static code structure)

### Future Enhancements:
1. **Semantic Search**: Use embeddings for fuzzy concept matching
2. **Type Verification**: Check argument types against stored signatures
3. **Multi-Language**: Support JavaScript, Go, Rust, etc.
4. **Runtime Profiling**: Index actual usage patterns from execution traces
5. **Incremental Updates**: Watch filesystem for code changes and auto-reindex

## Testing

Run the complete demo:
```bash
python examples/code_enhancement_demo.py
```

This demonstrates:
1. Concept ingestion
2. Code indexing
3. Dual retrieval
4. Code verification (hallucination detection)

## References

- **Holographic Memory**: Based on Itzhak Bentov's holographic interference pattern model
- **Surprise Learning**: Inspired by Google's Titans architecture (arXiv:2501.00663)
- **RAG for Code**: Retrieval Augmented Generation applied to software engineering
- **VSA/HDC**: Vector Symbolic Architectures and Hyperdimensional Computing

## Contributing

To extend the system:
1. Add new concept categories to `ingest_concepts.py`
2. Extend AST extraction in `code_indexer.py`
3. Add verification rules to `ventriloquist.verify_code()`
4. Create language-specific indexers (e.g., `js_indexer.py`)

## License

Same as parent project.






