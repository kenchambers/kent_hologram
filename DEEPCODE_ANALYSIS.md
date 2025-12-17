# DeepCode Concepts for Hologram: Architectural Analysis

## Executive Summary

**Verdict: SELECTIVE ENHANCEMENT with MINIMAL ADDITIONS**

DeepCode's graph-based concepts would enhance Hologram for specific coding tasks WITHOUT refactoring. However, **HDC resonance already captures most semantic relationships**. The gap is not in understanding *what* relates to what, but in **efficient traversal of multi-hop dependencies** and **structural validation across file boundaries**.

---

## Part 1: How Hologram Currently Works (Detailed Trace)

### Layer 1: Storage (HDC Resonance)

**File:** `src/hologram/memory/fact_store.py`

```python
# Facts encoded as: bind(bind(subject, predicate), object)
key = Operations.bind(s_vec, p_vec)  # bind(subject, predicate)
store(key, o_vec)                     # store object

# Example: "encode() calls helper()"
fs.add_fact("encode", "calls", "helper")
# Internally: bind(encode_vec, calls_vec) → helper_vec
```

**Retrieval via Resonance:**

```python
# Query: "What does encode call?"
s_vec = codebook.encode("encode")
p_vec = codebook.encode("calls")
key = bind(s_vec, p_vec)
similarities = memory.resonance(key, candidates)  # Line 329
best_idx = argmax(similarities)
answer = candidates[best_idx]  # "helper"
```

**Critical property:** Resonance finds semantic similarity. It answers:
- **1-hop queries:** "What is the predicate of subject?" (✓ Works)
- **Reverse queries:** "What calls helper()?" (✓ Works via query_subject)
- **Fuzzy matching:** "encode-like functions?" (✓ Works, but lossy)

### Layer 2: Code Vectorization (Structure Loss)

**File:** `src/hologram/swe/encoder.py`

```python
def encode_patch(self, patch: CodePatch) -> torch.Tensor:
    # Structure: bind(operation, role_op) + bind(file, role_file) + bind(location, role_loc)
    observation = bundle(
        bind(op_vec, self._role_operation),     # "modify_line"
        bind(file_vec, self._role_file),        # "encoder.py"
        bind(loc_vec, self._role_location),     # "line 95"
    )
    return observation
```

**What this captures:**
- OPERATION: What changed (add_line, modify_function, etc.)
- FILE: Which file changed
- LOCATION: Where in the file

**What this LOSES:**
- **Call graph structure:** "encode() → helper()" is just a fact, not a searchable dependency graph
- **File import relationships:** "encoder.py imports types.py" stored as fact, not queryable as graph
- **Multi-file impact chains:** "If encode() signature changes, what breaks in callers?" requires traversal

### Layer 3: Code Generation (Pattern Matching)

**File:** `src/hologram/swe/generator.py`, lines 95-210

```python
def generate(self, task: SWETask, max_patches: int = 5) -> PatchResult:
    # Step 1: Encode issue
    issue_vec = self._encoder.encode_issue(task.issue_text)

    # Step 2: Query memory for learned patterns
    memory_label, memory_confidence = self._memory.query(issue_vec)

    # Step 3: Retrieve or fall back to template
    if memory_confidence >= confidence_threshold:
        memory_pattern = self.get_learned_pattern(memory_label)
        operation, target_file, location, content = memory_pattern
    else:
        operation = "modify_line"  # Template fallback
        target_file = list(task.code_before.keys())[0]
        location = "1"
        content = "# Generated patch"

    # Step 4: Create single patch
    patch = CodePatch(file=target_file, operation=operation, ...)
    patches.append(patch)

    # Step 5: Verify patch (simple cosine similarity)
    patch_vec = self._encoder.encode_patch(patch)
    verification_score = Similarity.cosine(issue_vec, patch_vec)
```

**Critical limitation:** Single-patch generation with no impact analysis.

---

## Part 2: Where HDC Resonance Fails

### Gap 1: Transitive Dependencies (Multi-Hop Queries)

**Problem:** Answering "What transitively depends on encode()?"

```
Graph needed:
├─ encode() [source function]
├─ calls helper()
├─ helper() calls util.py:transform()
├─ transform() is imported by processor.py
└─ processor.py is used by main.py

Question: "If encode() breaks, what in main.py fails?"
Answer requires: encode() → helper() → transform() → processor.py → main.py
```

**What Hologram stores (as S-P-O triples):**

```python
fs.add_fact("encode", "calls", "helper")          # S-P-O triple 1
fs.add_fact("helper", "calls", "transform")        # S-P-O triple 2
fs.add_fact("processor", "imports", "transform")   # S-P-O triple 3
fs.add_fact("main", "uses", "processor")           # S-P-O triple 4
```

**What HDC resonance can answer:**

```python
# ✓ Direct: "What does encode call?"
answer, conf = fs.query("encode", "calls")
# Returns: "helper" with high confidence

# ✗ Transitive: "What transitively depends on encode?"
# No built-in method. Would need manual chain:
# 1. Query "encode calls X" → X = "helper"
# 2. Query "helper calls Y" → Y = "transform"
# 3. Query "Z imports transform" → Z = "processor"
# 4. Query "W uses processor" → W = "main"
# Each hop adds noise via resonance overlap
```

**Resonance decay with transitive hops:**

```python
# Each query filters candidates via resonance
# Confidence = cosine_similarity(query_key, best_candidate)

# Hop 1: encode → helper (conf = 0.95)
# Hop 2: helper → transform (conf = 0.92)
# Hop 3: processor → transform (conf = 0.88)  ← Reverse query adds noise
# Hop 4: main → processor (conf = 0.84)        ← Accumulated error

# Final confidence ≈ 0.95 * 0.92 * 0.88 * 0.84 ≈ 0.69
# At 4+ hops, confidence becomes unreliable
```

**Why graph traversal helps:**
```
Graph: Exact edges, no resonance decay
Traversal: encode() → [callers] = explicit set of functions that call encode()
          No confidence loss, O(n) instead of O(n * resonance)
```

---

### Gap 2: Multi-File Change Coordination

**Problem:** Detecting when a change in FILE_A requires cascading updates in FILE_B

**Concrete example:**

```python
# File A: encoder.py (CHANGE)
class CodeEncoder:
    def encode_issue(self, issue_text: str) -> torch.Tensor:  # ← Signature changed
        # Before: encode_issue(issue_text: str) → torch.Tensor
        # After:  encode_issue(issue_text: str, context: dict) → torch.Tensor
        pass

# File B: generator.py (DEPENDENT)
class CodeGenerator:
    def generate(self, task: SWETask) -> PatchResult:
        issue_vec = self._encoder.encode_issue(task.issue_text)
        # ↑ Breaks! Missing 'context' parameter
```

**What Hologram does:**

```python
# Facts stored:
fs.add_fact("generator.py", "imports", "encoder.py")
fs.add_fact("CodeGenerator", "calls", "encode_issue")

# Query: "What imports encoder.py?"
answer, conf = fs.query("generator.py", "imports")
# Returns: "encoder.py" (not helpful—gives source, not files that import it)

# Query: "What calls encode_issue?"
answer, conf = fs.query_subject("calls", "encode_issue")
# ✓ Returns: "CodeGenerator" (identifies the function)
# ✗ But doesn't know CodeGenerator is in generator.py (loses file context)
```

**Actual facts stored (S-P-O):**

```
("CodeGenerator", "calls", "encode_issue")  ← Loses: which file?
("generator.py", "imports", "encoder.py")   ← Loses: what does it import from it?
```

**Multi-file impact analysis requires:**

```python
# Graph edges (explicit):
encode_issue() in encoder.py
    ↓ called by
CodeGenerator.generate() in generator.py
    ↓ change requires
encode_issue call site in generator.py:99

# Backward compatibility check:
old_signature = (issue_text: str) → torch.Tensor
new_signature = (issue_text: str, context: dict) → torch.Tensor
incompatible = True
files_to_update = [generator.py, ...]
```

**Graph vs. Resonance:**

| Task | HDC Resonance | Explicit Graph |
|------|---------------|-----------------|
| Find who calls X | ✓ via query_subject | ✓ Direct lookup O(1) |
| Find N-hop callers | ✗ (resonance decay) | ✓ Fast traversal O(n) |
| Signature compatibility | ✗ (not stored) | ✓ With metadata |
| Multi-file impact | ✗ (no context) | ✓ Direct edges |

---

### Gap 3: Structural Validation Across Files

**Problem:** CodeGenerator can't validate that proposed patches work multi-file

**Current process (generator.py, lines 144-176):**

```python
if used_learned_pattern and memory_pattern is not None:
    operation, target_file, location, content = memory_pattern
else:
    # Template fallback—SINGLE FILE ONLY
    target_file = list(task.code_before.keys())[0]
    location = "1"
    content = "# Generated patch"

patch = CodePatch(file=target_file, operation=operation, location=location, content=content)
patches.append(patch)

# Verification: Simple cosine similarity
patch_vec = self._encoder.encode_patch(patch)
verification_score = Similarity.cosine(issue_vec, patch_vec)  # ← No multi-file validation
```

**What it's missing:**

```python
# For a real SWE task: "Function X signature changed"
# Correct solution requires:
# 1. Change X's signature in file_a.py
# 2. Update call sites in file_b.py, file_c.py, file_d.py
# 3. Update imports if X moved
# 4. Update mocks in test_file.py

# Current approach: Generates 1 patch, verifies via resonance
# Missing: Cross-file validation

# Graph-based check:
def validate_patch_set(patches: List[CodePatch], call_graph: CallGraph):
    for patch in patches:
        for dependent_file in call_graph.get_files_importing(patch.file):
            validate_import_consistency(patch, dependent_file)
        for call_site in call_graph.get_call_sites(patch.function):
            validate_signature_compatibility(patch, call_site)
```

---

## Part 3: Where DeepCode's Graph Would Help (Minimal Additions)

### Addition 1: Transitive Dependency Resolver

**File location:** New `src/hologram/swe/dependency_graph.py`

**What it does:** Layer on TOP of FactStore, not replacing it.

```python
class CodeDependencyGraph:
    """
    Thin layer for traversing fact triples as DAG.

    Uses FactStore for edge storage, adds graph traversal methods.
    NO SCHEMA CHANGES to FactStore.
    """

    def __init__(self, fact_store: FactStore):
        self._fs = fact_store  # Use existing facts

    def get_transitive_dependents(self, function: str, max_depth: int = 5):
        """Find all functions that transitively call/depend on function.

        Args:
            function: Function name (e.g., "encode")
            max_depth: Maximum hops to traverse

        Returns:
            Set[str]: All transitively dependent functions

        Example:
            encode → [helper, util.transform]
            helper → [processor, main]
            Result: {helper, util.transform, processor, main}
        """
        visited = set()
        queue = [(function, 0)]

        while queue:
            current, depth = queue.pop(0)
            if depth > max_depth or current in visited:
                continue

            visited.add(current)

            # Query FactStore: "Who calls current?"
            # Use the S-P-O facts we already store
            dependents = self._fs.get_facts_by_object(current)  # Line 434 in fact_store.py
            for fact in dependents:
                if fact.predicate in ("calls", "uses", "depends_on"):
                    queue.append((fact.subject, depth + 1))

        return visited

    def get_files_affected(self, function: str):
        """Find all files that would be affected if function changes.

        Returns:
            Dict[str, List[str]]: {file: [affected_functions]}
        """
        affected_functions = self.get_transitive_dependents(function)
        file_map = defaultdict(list)

        for func in affected_functions:
            # Query: "What file contains func?"
            # Would require adding this fact type to FactStore:
            # fs.add_fact(func, "defined_in", file)
            file, conf = self._fs.query(func, "defined_in")
            file_map[file].append(func)

        return dict(file_map)
```

**Usage in CodeGenerator:**

```python
# In generator.py, add after line 115:

graph = CodeDependencyGraph(self._fact_store)
affected_files = graph.get_files_affected(changed_function)

# Now generate patches for all affected files
for file, functions in affected_files.items():
    for func in functions:
        # Generate patch to update call site
        patch = self._generate_update_patch(func, changed_function)
        patches.append(patch)
```

**Effort:** ~50 lines, uses existing FactStore API, no refactoring.

---

### Addition 2: Structural Metadata for Impact Analysis

**File location:** Enhance `src/hologram/swe/encoder.py` with metadata tracking

**Current limitation:** CodePatch stores only (operation, file, location), loses context.

```python
# Current CodePatch (from types.py):
@dataclass
class CodePatch:
    file: str           # Which file
    operation: str      # What operation
    location: str       # Where in file
    content: str        # What content

# MISSING:
# - function_name: Which function in the file?
# - signature_change: Did signature change?
# - affected_calls: Which call sites break?
```

**Enhancement:**

```python
# In encoder.py, add metadata extraction:

class CodeEncoder:
    def extract_function_signature(self, code: str, location: str) -> Optional[str]:
        """Extract function signature from location marker.

        Example:
            location = "def encode_issue(self, issue_text: str)"
            returns: "(self, issue_text: str) -> torch.Tensor"
        """
        # Parse location to extract signature
        if location.startswith("def "):
            # Extract from code_before context
            match = re.search(r'def \w+\((.*?)\)', location)
            if match:
                return f"({match.group(1)})"
        return None

    def encode_patch_with_metadata(self, patch: CodePatch, code_context: str) -> Tuple[torch.Tensor, dict]:
        """Encode patch + extract structural metadata.

        Returns:
            (patch_vector, metadata)
        """
        patch_vec = self.encode_patch(patch)

        metadata = {
            "file": patch.file,
            "location": patch.location,
            "operation": patch.operation,
            "function_name": self._extract_function_name(patch.location, code_context),
            "signature": self.extract_function_signature(code_context, patch.location),
            "is_signature_change": "def " in patch.location and patch.operation in ["modify_function", "add_function"],
        }

        return patch_vec, metadata
```

**Storage in NeuralMemory:**

```python
# In generator.py, enhance learn_from_task():

fact = ConsolidationFact(
    key_vector=issue_vec,
    value_index=0,
    value_label=label,
    metadata={  # NEW: Store structural info
        "files_changed": changed_files,
        "operations": [patch.operation for patch in patches],
        "affected_functions": self._extract_affected_functions(task.code_after),
    }
)
self._memory.consolidate([fact], epochs=20, batch_size=8)
```

**Effort:** ~30 lines, enhances existing classes, no refactoring.

---

### Addition 3: Multi-File Patch Validation

**File location:** New `src/hologram/swe/validator.py`

**What it does:** Check that a patch set is valid across file boundaries

```python
class MultiFilePatchValidator:
    """
    Validate patch sets for consistency across files.

    Uses CodeDependencyGraph + CodeEncoder metadata.
    """

    def __init__(self, fact_store: FactStore, encoder: CodeEncoder):
        self._fs = fact_store
        self._encoder = encoder
        self._graph = CodeDependencyGraph(fact_store)

    def validate_patch_set(self, patches: List[CodePatch], code_before: Dict[str, str]) -> List[str]:
        """Check patch set for multi-file consistency.

        Returns:
            List[str]: Validation errors (empty if valid)
        """
        errors = []

        # Group patches by file
        patches_by_file = defaultdict(list)
        for patch in patches:
            patches_by_file[patch.file].append(patch)

        # Check 1: If function signature changed, find all call sites
        for file, file_patches in patches_by_file.items():
            for patch in file_patches:
                if patch.operation in ["modify_function", "add_function"]:
                    # Find all files that call this function
                    func_name = self._extract_function_name(patch.location, code_before[file])
                    callers = self._graph.get_transitive_dependents(func_name)

                    # Check if patches exist for all call sites
                    for caller_file in callers:
                        if caller_file not in patches_by_file:
                            errors.append(
                                f"Function {func_name} changed in {file}, "
                                f"but no patch for call site in {caller_file}"
                            )

        return errors
```

**Usage in CodeGenerator:**

```python
# In generator.py, after generating patches:

validator = MultiFilePatchValidator(self._fact_store, self._encoder)
validation_errors = validator.validate_patch_set(patches, task.code_before)

return PatchResult(
    patches=patches,
    confidence=memory_confidence if used_learned_pattern else verification_score,
    verification_passed=len(validation_errors) == 0,  # Enhanced validation
    validation_errors=validation_errors,  # NEW: Report errors
    factorization={...}
)
```

**Effort:** ~40 lines, new class, no refactoring of existing code.

---

## Part 4: Why NOT to Use Full DeepCode Architecture

### Problem 1: Redundancy

Hologram already has:
- ✓ Vector-based similarity (resonance)
- ✓ Vocabulary indexing (Codebook)
- ✓ Structural decomposition (CodeResonator factorization)

DeepCode adds:
- Explicit graph edges (redundant with FactStore S-P-O)
- Graph traversal (can be thin layer on FactStore)
- Neural code understanding (Hologram has this via NeuralMemory)

**Cost of full adoption:** Duplication, conflicting update logic.

### Problem 2: Over-Engineering

Hologram's design principle: "No refactoring, minimal additions."

Full DeepCode blueprint:
- Requires defining explicit node/edge schemas
- Needs graph consistency checking
- Adds 500+ LOC of graph infrastructure

Minimal approach:
- Reuse FactStore for edges
- Add thin traversal layer (~100 LOC)
- Integrate with existing CodeGenerator

---

## Part 5: Specific Recommendations

### Recommendation 1: Add Transitive Dependency Resolver (HIGH IMPACT)

**File:** `src/hologram/swe/dependency_graph.py` (NEW)

**Why:** Directly addresses Gap 1 (multi-hop queries)

**What to add:**

```python
class CodeDependencyGraph:
    def get_transitive_dependents(self, function: str, max_depth: int = 5) -> Set[str]:
        """Already shown above"""
        pass

    def get_files_affected(self, function: str) -> Dict[str, List[str]]:
        """Already shown above"""
        pass

    def get_import_chain(self, from_file: str, to_file: str) -> List[str]:
        """Find chain of imports from one file to another"""
        pass
```

**Integration point:**

```python
# In generator.py, line 122-135:
if self._memory is not None:
    memory_label, memory_confidence = self._memory.query(issue_vec)

    # NEW: Check if this is a function change
    if memory_confidence >= confidence_threshold:
        # Identify changed function
        changed_func = self._extract_function_name(memory_label)

        # Get all files that would be affected
        graph = CodeDependencyGraph(self._fact_store)
        affected_files = graph.get_files_affected(changed_func)

        # Generate patches for all affected locations
        for file, functions in affected_files.items():
            for func in functions:
                patch = self._generate_update_patch(func, changed_func, file)
                patches.append(patch)
```

**Effort:** ~80 lines, no refactoring, ~4 hours

---

### Recommendation 2: Store Function-File Relationships (MEDIUM IMPACT)

**File:** Enhance `src/hologram/swe/encoder.py`

**What to add:**

When storing facts about code changes, also store where functions are defined:

```python
# In learn_from_task():

# Extract functions changed
for file in task.code_after:
    functions = self._extract_functions_defined(task.code_after[file])
    for func in functions:
        # Store: "function X defined in file Y"
        self._fact_store.add_fact(func, "defined_in", file)

        # Store: "file Y contains function X"
        self._fact_store.add_fact(file, "defines", func)
```

**Why:** Closes the gap in Gap 2 (multi-file coordination). Currently FactStore knows:

```python
("CodeGenerator", "calls", "encode_issue")  # ✓ Who calls whom
("generator.py", "imports", "encoder.py")   # ✓ Which files import which
```

But MISSING:

```python
("encode_issue", "defined_in", "encoder.py")     # ✗ Missing
("CodeGenerator.generate", "calls", "encode_issue")  # ✗ Missing scoped call
```

**Effort:** ~20 lines, ~2 hours

---

### Recommendation 3: Add Patch Validation (MEDIUM IMPACT)

**File:** `src/hologram/swe/validator.py` (NEW)

**What to add:**

```python
class MultiFilePatchValidator:
    def validate_patch_set(self, patches: List[CodePatch], code_before: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Check if patches form a valid solution"""
        # Check: No hanging call sites
        # Check: Imports are updated
        # Check: No signature mismatches
```

**Integration point:**

```python
# In generator.py, line 186-189:
if used_learned_pattern:
    verification_passed = memory_confidence >= confidence_threshold
else:
    verification_passed = verification_score > 0.1

# NEW: Multi-file validation
validator = MultiFilePatchValidator(self._fact_store, self._encoder)
is_valid, errors = validator.validate_patch_set(patches, task.code_before)
if not is_valid:
    verification_passed = False  # Fail if invalid
```

**Effort:** ~60 lines, ~3 hours

---

## Part 6: Summary Table

| Concept | Hologram Has | DeepCode Adds | Hologram Needs | Effort |
|---------|-------------|--------------|----------------|---------|
| Vector encoding | ✓ (resonance) | ✓ (explicit) | Use resonance | — |
| S-P-O facts | ✓ (FactStore) | ✓ (graph nodes) | Query facts | — |
| Transitive queries | ✗ (resonance decay) | ✓ (BFS) | Add traversal layer | ~80 LOC |
| Function-file mapping | Partial | ✓ (explicit) | Enhance facts | ~20 LOC |
| Multi-file validation | ✗ | ✓ (blueprint) | Add validator | ~60 LOC |
| Total minimal additions | | | | ~160 LOC |

---

## Part 7: Code Examples - How to Integrate

### Example 1: Query Multi-Hop Dependency

**Before (Hologram only):**

```python
# In generator.py
# Question: "What functions transitively depend on encode()?"

# Can't answer directly without manual recursion
def find_all_dependents(func_name, seen=None):
    if seen is None:
        seen = set()
    if func_name in seen:
        return seen

    seen.add(func_name)

    # Query: Who calls func_name?
    dependents_facts = self._fact_store.get_facts_by_object(func_name)
    for fact in dependents_facts:
        if fact.predicate == "calls":
            find_all_dependents(fact.subject, seen)

    return seen

all_dependents = find_all_dependents("encode")
```

**After (With minimal additions):**

```python
# In generator.py with CodeDependencyGraph

from hologram.swe.dependency_graph import CodeDependencyGraph

graph = CodeDependencyGraph(self._fact_store)
all_dependents = graph.get_transitive_dependents("encode")

# Much cleaner, reusable, optimized
```

---

### Example 2: Validate Multi-File Patch

**Before (Hologram only):**

```python
# Single patch generation
patch = CodePatch(file=target_file, operation=operation, ...)
patches = [patch]

# Verification: Only cosine similarity, no cross-file check
patch_vec = self._encoder.encode_patch(patch)
verification_score = Similarity.cosine(issue_vec, patch_vec)
verification_passed = verification_score > 0.1
```

**After (With validator):**

```python
# Generate multiple patches
patches = [
    CodePatch(file="encoder.py", operation="modify_function", ...),
    CodePatch(file="generator.py", operation="modify_line", ...),  # Update call site
    CodePatch(file="test_encoder.py", operation="modify_function", ...),  # Update mock
]

# Validate across files
validator = MultiFilePatchValidator(self._fact_store, self._encoder)
is_valid, errors = validator.validate_patch_set(patches, task.code_before)

if not is_valid:
    print(f"Patches incomplete: {errors}")
    # Can suggest missing patches
    for error in errors:
        # "Function encode() signature changed in encoder.py, but call in generator.py:99 not updated"
        pass

verification_passed = is_valid and memory_confidence >= confidence_threshold
```

---

## Part 8: Final Decision Matrix

### Should you add Graph Traversal?

| Scenario | HDC Resonance Sufficient? | Needs Graph? | Recommendation |
|----------|--------------------------|--------------|-----------------|
| Find who calls function X | Yes (query_subject) | Overkill | Use FactStore |
| Find all transitive callers | No (resonance decay) | Yes | **ADD traversal** |
| Detect breaking changes | No (loses context) | Yes | **ADD metadata** |
| Validate multi-file patches | No (single file) | Yes | **ADD validator** |
| Learn new patterns | Yes (memory sufficient) | No | Keep as-is |
| Verify single-file patches | Yes (cosine similarity) | No | Keep as-is |

---

## Conclusion

**Do NOT adopt full DeepCode architecture.** Instead:

1. **Keep Hologram's HDC resonance** - it's semantically superior for learning
2. **Add 3 minimal layers (~160 LOC total):**
   - CodeDependencyGraph: Transitive traversal on FactStore
   - Enhanced metadata: Function-file relationships
   - MultiFilePatchValidator: Cross-file validation

3. **Integration points:** Only modify CodeGenerator.generate() and CodeGenerator.learn_from_task()

4. **Benefit:** Multi-file impact analysis without refactoring existing HDC architecture

This approach:
- ✓ Solves the 3 key gaps (transitive deps, multi-file context, validation)
- ✓ Reuses existing FactStore infrastructure
- ✓ Maintains HDC semantic learning advantage
- ✓ Requires no refactoring of core architecture
- ✓ ~160 LOC of clean, new code
