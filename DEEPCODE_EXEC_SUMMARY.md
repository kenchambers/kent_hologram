# DeepCode Analysis: Executive Summary

## Question
Should Hologram adopt DeepCode's graph-based concepts for better multi-file impact analysis?

## Answer
**NO** to full adoption. **YES** to 3 minimal additions (~160 LOC, no refactoring).

---

## The Core Issue

Hologram's CodeGenerator (src/hologram/swe/generator.py) does this:

```python
# Current: Single-file patch generation
patch = CodePatch(file=target_file, operation=operation, ...)  # Line 171
patches = [patch]  # Only one file!

# Problem: If encode_issue() signature changes:
#   ✗ Generates patch for encoder.py (the source)
#   ✗ MISSES patch for generator.py (the caller)
#   ✗ MISSES patch for test_encoder.py (the test)
```

**Why?** No way to find "all functions that call encode_issue()" → "all files containing those functions"

---

## Where HDC Resonance WORKS

```python
# From fact_store.py
fs.add_fact("CodeGenerator.generate", "calls", "encode_issue")
fs.query_subject("calls", "encode_issue")
→ ("CodeGenerator.generate", 0.92 confidence)  ✓ Excellent
```

Hologram's HDC resonance is **semantically superior** for pattern learning and fuzzy matching.

---

## Where It FAILS

### Gap 1: Transitive Dependencies

```
Question: "What transitively depends on encode_issue()?"
         encode() → helper() → util.transform() → processor.py → main.py

resonance approach:
  - Hop 1: query_subject("calls", "encode_issue") → helper (0.92)
  - Hop 2: query_subject("calls", "helper") → transform (0.88)
  - Hop 3: query_subject("calls", "transform") → ?? (0.84)
  - Hop 4: ...

  Confidence decay: 0.92 × 0.88 × 0.84 × 0.78 = 0.60 (unreliable)

graph approach:
  - BFS: {encode, helper, transform, processor, main}
  - Confidence: 1.0 (exact edges)
  - Cost: O(n) instead of O(n²) resonance
```

### Gap 2: Multi-File Context Lost

```python
Stored facts:
  ("CodeGenerator.generate", "calls", "encode_issue")  ← What calls what
  ("generator.py", "imports", "encoder.py")             ← Which files import which

Missing:
  ("CodeGenerator.generate", "defined_in", "generator.py")  ← Link function to file
  ("encode_issue", "signature", "(self, text: str, context: dict)")  ← Signature details

Result: Can't answer "Will this change break generator.py:99?"
```

### Gap 3: Single-Patch Generation

```python
# From generator.py:95-210
patches = []
target_file = list(task.code_before.keys())[0]  # Line 139 - FIRST FILE ONLY
location = "1"
patch = CodePatch(...)  # Line 171
patches.append(patch)   # Line 177 - Only one patch!

# Should be:
affected_files = graph.get_files_affected(changed_function)
for file, functions in affected_files.items():
    for func in functions:
        patches.append(CodePatch(file=file, ...))
```

---

## The 3 Minimal Additions

### Addition 1: Transitive Dependency Resolver

**File:** `src/hologram/swe/dependency_graph.py` (NEW, ~80 lines)

```python
class CodeDependencyGraph:
    def __init__(self, fact_store):
        self._fs = fact_store  # Reuse existing FactStore

    def get_transitive_dependents(self, function: str, max_depth: int = 5) -> Set[str]:
        """BFS on fact triples to find all callers"""
        visited = set()
        queue = [(function, 0)]
        while queue:
            current, depth = queue.pop(0)
            if depth > max_depth or current in visited:
                continue
            visited.add(current)
            # Query: "Who calls current?"
            dependents = self._fs.get_facts_by_object(current)
            for fact in dependents:
                if fact.predicate in ("calls", "uses"):
                    queue.append((fact.subject, depth + 1))
        return visited

    def get_files_affected(self, function: str) -> Dict[str, List[str]]:
        """Map transitive dependents to files"""
        affected_functions = self.get_transitive_dependents(function)
        file_map = defaultdict(list)
        for func in affected_functions:
            file, conf = self._fs.query(func, "defined_in")
            file_map[file].append(func)
        return dict(file_map)
```

**Why:** Solves multi-hop queries without resonance decay. Reuses FactStore, no refactoring.

---

### Addition 2: Enhanced Metadata

**File:** `src/hologram/swe/encoder.py` (enhance)

```python
# When learning from task (learn_from_task):
for file in task.code_after:
    functions = self._extract_functions_defined(task.code_after[file])
    for func in functions:
        # Store missing facts:
        self._fact_store.add_fact(func, "defined_in", file)
        self._fact_store.add_fact(file, "defines", func)

        # Extract and store signature:
        sig = self.extract_function_signature(task.code_after[file], func)
        if sig:
            self._fact_store.add_fact(func, "signature", sig)
```

**Why:** Closes context gap. "What file defines this function?" is now answerable.

---

### Addition 3: Multi-File Validator

**File:** `src/hologram/swe/validator.py` (NEW, ~60 lines)

```python
class MultiFilePatchValidator:
    def validate_patch_set(self, patches: List[CodePatch], code_before: Dict[str, str]) -> List[str]:
        """Check patches for breaking changes"""
        errors = []
        graph = CodeDependencyGraph(self._fact_store)

        for patch in patches:
            if patch.operation in ["modify_function"]:
                # Find all call sites
                func = self._extract_function_name(patch)
                callers = graph.get_transitive_dependents(func)

                # Check if patches exist for all callers
                patched_files = {p.file for p in patches}
                for caller_file in callers:
                    if caller_file not in patched_files:
                        errors.append(
                            f"Function {func} modified, but {caller_file} not patched"
                        )
        return errors
```

**Why:** Prevents incomplete patch sets. Catches the encoder.py/generator.py issue.

---

## Integration Points

### In `src/hologram/swe/generator.py`

**Change 1: Import**
```python
from hologram.swe.dependency_graph import CodeDependencyGraph
from hologram.swe.validator import MultiFilePatchValidator
```

**Change 2: Lines 122-135 (Multi-file generation)**
```python
if self._memory is not None:
    memory_label, memory_confidence = self._memory.query(issue_vec)

    # NEW: Get all affected files
    if memory_confidence >= confidence_threshold:
        graph = CodeDependencyGraph(self._fact_store)
        changed_func = self._extract_function_name(memory_label)
        affected_files = graph.get_files_affected(changed_func)

        for file, functions in affected_files.items():
            for func in functions:
                # Generate patch for each call site
                patch = self._generate_update_patch(func, changed_func, file)
                patches.append(patch)
```

**Change 3: Lines 186-189 (Enhanced verification)**
```python
if used_learned_pattern:
    verification_passed = memory_confidence >= confidence_threshold
else:
    verification_passed = verification_score > 0.1

# NEW: Cross-file validation
validator = MultiFilePatchValidator(self._fact_store, self._encoder)
validation_errors = validator.validate_patch_set(patches, task.code_before)
if validation_errors:
    verification_passed = False
```

---

## Comparison: Before vs After

### Before (Current Hologram)

```
Input:  SWE Task: "encode_issue() signature changed"
        code_before: {encoder.py, generator.py, test_encoder.py}

Process:
  1. Encode issue → vector
  2. Query memory
  3. Generate patch for encoder.py only
  4. Verify via cosine similarity

Output: PatchResult(patches=[encoder.py], verified=True)
  ✗ INCOMPLETE - Missing generator.py and test_encoder.py patches
```

### After (With 3 additions)

```
Input:  SWE Task: "encode_issue() signature changed"
        code_before: {encoder.py, generator.py, test_encoder.py}

Process:
  1. Encode issue → vector
  2. Query memory, identify changed_func = "encode_issue"
  3. Build dependency graph: encode_issue → [CodeGenerator.generate, tests.test_encoder]
  4. Map to files: {generator.py: [CodeGenerator.generate], test_encoder.py: [test_encoder]}
  5. Generate patches for all affected files
  6. Validate: Check that all callers are patched
  7. Verify via cosine similarity + cross-file validation

Output: PatchResult(
    patches=[
        CodePatch(file=encoder.py, ...),
        CodePatch(file=generator.py, ...),  # ← NEW: Call site update
        CodePatch(file=test_encoder.py, ...)  # ← NEW: Test update
    ],
    verified=True,
    validation_errors=[]  # ← NEW: Cross-file validation
)
  ✓ COMPLETE - All affected files covered
```

---

## Cost Analysis

| Item | Current | Addition | Effort |
|------|---------|----------|--------|
| Lines of code | ~2500 (hologram/) | +160 new | ~6 hours |
| Files modified | ~5 | generator.py only | 1 file |
| Files created | ~40 | dependency_graph.py, validator.py | 2 files |
| Breaking changes | — | 0 | None |
| Refactoring needed | — | No | None |
| Tests needed | — | ~20 new tests | 2 hours |

---

## Decision Matrix

### When to Use Hologram (Resonance)
- ✓ Learning patterns from code
- ✓ Fuzzy semantic matching
- ✓ Single-file changes
- ✓ Generalization across codebase

### When to Add Graph Traversal
- ✗ (Before) Multi-file impact analysis
- ✗ (Before) Transitive dependency queries
- ✗ (Before) Cross-file validation
- ✓ (After with additions) All of above

### When NOT to Adopt Full DeepCode
- Hologram's HDC resonance is already semantic-aware
- Full graph infrastructure would duplicate functionality
- FactStore S-P-O triples already capture edges
- Thin traversal layer is sufficient

---

## Recommendation

**IMPLEMENT the 3 minimal additions.**

1. **Week 1:** Add CodeDependencyGraph (~80 LOC)
2. **Week 1:** Enhance encoder metadata (~20 LOC)
3. **Week 2:** Add MultiFilePatchValidator (~60 LOC)
4. **Week 2:** Integrate into CodeGenerator (~20 LOC)

**Total:** ~180 LOC, ~8 hours, zero refactoring, zero breaking changes.

**Result:** Multi-file impact analysis capability without compromising Hologram's HDC semantic learning advantage.

---

## Files Reference

| Document | Purpose |
|----------|---------|
| `DEEPCODE_ANALYSIS.md` | Full technical analysis (30 pages) |
| `DEEPCODE_TECHNICAL_APPENDIX.md` | Execution traces & comparisons (20 pages) |
| `DEEPCODE_EXEC_SUMMARY.md` | This document (2 pages) |

**Start here:** DEEPCODE_EXEC_SUMMARY.md (this file)
**Deep dive:** DEEPCODE_ANALYSIS.md (Part 5: Specific Recommendations)
**Implementation reference:** DEEPCODE_TECHNICAL_APPENDIX.md (Phased Implementation)
