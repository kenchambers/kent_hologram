# SWE Implementation Roadmap - Action Items

Based on comprehensive code review of the SWE module. See `SWE_CODE_REVIEW.md` for detailed analysis.

**Document Date:** 2025-12-16
**Priority Level:** HIGH - Blocks competitive SWE-bench evaluation

---

## Critical Issues (Must Fix)

### 1. CodeResonator Integration (P0 - BLOCKS REASONING)

**Status:** Implemented but unused
**Impact:** System cannot do generalization beyond rote memorization
**File:** `src/hologram/swe/generator.py`

**Current State:**
```python
def __init__(self, ..., resonator: CodeResonator, ...):
    self._resonator = resonator  # Stored but never used

def generate(self, task: SWETask, ...):
    # CodeResonator is never called here
    # Falls back to simple dictionary lookup instead
```

**Action:**
1. Create **structured issue vectors** that match patch vector structure
   - Current: `bundle(term_1, term_2, ...)` - unstructured
   - Needed: `bind(action_vec, role_action) + bind(target_vec, role_target) + ...`

2. **Enable resonator in generation path** (lines 136-157)
   ```python
   # Current fallback path:
   if used_learned_pattern and memory_pattern is not None:
       operation, target_file, location, content = memory_pattern
   else:
       # Template fallback - unused resonator could help here!

   # Proposed:
   if used_learned_pattern and memory_pattern is not None:
       operation, target_file, location, content = memory_pattern
   else:
       # Try resonator on unstructured issue vector
       factorizations = self._resonator.resonate_topk(issue_vec, k=5)
       best = factorizations[0]
       operation = best.operation
       target_file = best.file
       location = best.location
       content = generate_from_factorization(best)
   ```

3. Test with new integration flow before enabling benchmarking

**Effort:** 4-6 hours
**Risk:** Medium - resonator expects structured input, current vectors are unstructured

---

### 2. Fix Fake Pattern Learning (P0 - DATA QUALITY)

**Status:** Hardcoded values, not actual learning
**Impact:** All learned patterns are incorrect
**File:** `src/hologram/swe/generator.py:212-258`

**Current Implementation:**
```python
# Lines 251-256 - HARDCODED!
self._pattern_cache[label] = (
    "modify_line",  # ← Always the same, never analyzes diff
    changed_files[0],
    "line_1",       # ← Always line 1, ignores actual location
    task.code_after.get(changed_files[0], "")[:100],  # ← Truncated
)
```

**Action:**
1. **Implement basic diff analysis**
   ```python
   def _extract_pattern_from_task(self, task: SWETask):
       """Analyze code_before/code_after to extract pattern details."""

       # For each changed file:
       for filename in task.code_before:
           if filename not in task.code_after:
               operation = "delete_file"
               continue

           before = task.code_before[filename]
           after = task.code_after[filename]

           # Simple heuristics:
           if len(after) > len(before):
               # Lines added
               added_lines = extract_added_lines(before, after)
               if any('import' in line for line in added_lines):
                   operation = "add_import"
               elif any('def ' in line for line in added_lines):
                   operation = "add_function"
               else:
                   operation = "add_line"
           elif len(after) < len(before):
               operation = "delete_line"
           else:
               # Same length, content changed
               operation = "modify_line"

           # Find actual location of change
           location = find_first_changed_line(before, after)

           # Store full content (no truncation)
           content = after  # or diff chunk if storing space is concern
   ```

2. **Test on sample tasks** to verify operations are correct

3. **Measure improvement:** Learning accuracy before/after

**Effort:** 6-8 hours
**Risk:** Low - incremental improvement on existing logic

---

### 3. Fix State Fragmentation (P0 - PERSISTENCE)

**Status:** Pattern cache lost on restart
**Impact:** All learned knowledge is lost when process restarts
**File:** `src/hologram/swe/generator.py:92-93`

**Current Problem:**
```python
# Memory storage (persistent if saved):
self._memory.consolidate([fact], epochs=20, batch_size=8)

# Pattern storage (in-memory only - LOST ON RESTART):
self._pattern_cache: Dict[str, Tuple[str, str, str, str]] = {}
```

**Action:**
1. **Option A: Store patterns in NeuralMemory**
   ```python
   # Encode pattern tuple as vector
   pattern_vec = self._encoder.encode_patch(CodePatch(
       file=target_file,
       operation=operation,
       location=location,
       content=content,
   ))

   # Store in memory instead of cache
   fact = ConsolidationFact(
       key_vector=issue_vec,
       value_index=0,
       value_label=label,
       # NEW: Store encoded pattern
       pattern_vector=pattern_vec,
   )
   ```

2. **Option B: Add persistence to cache**
   ```python
   def save_patterns(self, filepath: str):
       """Save pattern cache to disk."""
       import json
       with open(filepath, 'w') as f:
           json.dump(self._pattern_cache, f)

   def load_patterns(self, filepath: str):
       """Load pattern cache from disk."""
       import json
       with open(filepath) as f:
           self._pattern_cache = json.load(f)
   ```

3. **Add invariant checking**
   ```python
   def verify_consistency(self) -> bool:
       """Verify memory and cache stay in sync."""
       # For each label in cache, verify it's stored in memory
       # For each label in memory, verify cache entry exists
       return is_consistent
   ```

**Effort:** 2-3 hours (Option B is simpler)
**Risk:** Low - both approaches are isolated changes

---

## High Priority Issues (Significant Improvement)

### 4. Generate Executable Code (P1 - OUTPUT QUALITY)

**Status:** Template placeholders are not executable
**Impact:** Generated code never works
**File:** `src/hologram/swe/types.py:35-46`, `src/hologram/swe/generator.py:158-168`

**Current Templates:**
```python
PATCH_TEMPLATES = {
    "add_line": "# Added: {content}",  # ← Comment, not code!
    "add_function": "def {location}():\n    pass",  # ← Stub with no body
    ...
}
```

**Action:**
1. **Replace templates with syntax-aware generation**
   ```python
   def generate_patch_content(self, operation: str, context: str, suggestion: str) -> str:
       """Generate actual code, not comment stubs."""
       if operation == "add_line":
           # Ensure proper indentation
           indent = guess_indent(context)
           return indent + suggestion.strip()
       elif operation == "add_function":
           # Generate function with pass body (not empty)
           return f"def {location}():\n    pass"
       # ... etc
   ```

2. **Add syntax validation**
   ```python
   import ast
   try:
       ast.parse(generated_code)
   except SyntaxError:
       # Fall back to safe comment
       return f"# TODO: {generated_code}"
   ```

3. **Test code generation** produces valid Python

**Effort:** 4-6 hours
**Risk:** Medium - careful indentation handling required

---

### 5. Multi-File Support (P1 - REAL-WORLD TASKS)

**Status:** Single file, single patch only
**Impact:** Cannot solve 40%+ of SWE-bench tasks
**File:** `src/hologram/swe/generator.py:138-177`

**Current Code:**
```python
# Lines 138-140 - Takes ONLY first file
target_file = list(task.code_before.keys())[0] if task.code_before else "unknown.py"

# Lines 171-177 - Always exactly one patch
patches = []
patch = CodePatch(...)
patches.append(patch)
return PatchResult(patches=patches, ...)  # len=1 always
```

**Action:**
1. **Generate multiple patches** (one per changed file)
   ```python
   patches = []
   for filename in task.code_before:
       # Determine change type for this file
       operation = determine_operation(task.code_before[filename], task.code_after.get(filename))

       patch = CodePatch(
           file=filename,
           operation=operation,
           location=find_location(operation, task.code_before[filename]),
           content=generate_patch_content(operation, ...),
       )
       patches.append(patch)

   return PatchResult(patches=patches, ...)
   ```

2. **Update tests** to verify multiple patches are generated

3. **Update benchmark** to evaluate all patches, not just first

**Effort:** 6-8 hours
**Risk:** Medium - requires understanding all file interactions

---

## Medium Priority Issues (Correctness & Maintainability)

### 6. Add Code Context Understanding (P2 - CODE REASONING)

**Status:** First 100 characters only
**Impact:** System ignores 99%+ of available context
**File:** `src/hologram/swe/encoder.py:70-105`, `src/hologram/swe/generator.py:150-156`

**Current Limitation:**
```python
# Lines 150-156
snippet = file_content[:100].replace('\n', ' ').strip()  # ← Only 100 chars!
content = f"# Based on: {snippet}..."
```

**Action:**
1. **Parse code structure**
   ```python
   import ast

   def extract_code_structure(self, code: str) -> dict:
       """Parse code to extract functions, classes, imports."""
       tree = ast.parse(code)

       functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
       classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
       imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.Import)]

       return {
           "functions": functions,
           "classes": classes,
           "imports": imports,
           "total_lines": len(code.split('\n')),
       }
   ```

2. **Use structure in generation**
   ```python
   structure = self.extract_code_structure(task.code_before[target_file])

   # Pass to generator for context-aware decisions
   if operation == "add_function" and "utils" in target_file:
       # Add near other utility functions
       location = structure["functions"][-1]
   ```

3. **Encode structural context** in issue vector

**Effort:** 6-8 hours
**Risk:** Medium - requires careful AST handling

---

### 7. Improve Verification Logic (P2 - CONSISTENCY)

**Status:** Different metrics for different paths
**Impact:** Inconsistent decision-making
**File:** `src/hologram/swe/generator.py:186-189`

**Current Logic:**
```python
if used_learned_pattern:
    verification_passed = memory_confidence >= confidence_threshold  # 0.3
else:
    verification_passed = verification_score > 0.1  # Different threshold!
```

**Action:**
1. **Unify verification metric**
   ```python
   # Always use same metric
   verification_passed = verification_score >= confidence_threshold
   ```

2. **Track confidence sources**
   ```python
   confidence_source = "memory" if used_learned_pattern else "vector_similarity"

   return PatchResult(
       patches=patches,
       confidence=verification_score,
       verification_passed=verification_passed,
       confidence_source=confidence_source,  # For debugging
   )
   ```

3. **Add logging/tracing** for debugging

**Effort:** 1-2 hours
**Risk:** Low - straightforward refactoring

---

## Low Priority Issues (Code Quality)

### 8. Test Coverage (P3 - QUALITY GATES)

**Status:** Tests don't verify code correctness
**Impact:** Silent failures in generated code
**File:** `tests/swe/test_generator.py`, `tests/swe/test_integration.py`

**Current Tests:**
```python
def test_full_pipeline(self, shared_components):
    # Only checks that result is returned, not correctness
    assert isinstance(result, PatchResult)
    assert len(result.patches) > 0
    # No assertion that patches are executable!
```

**Action:**
1. **Add code validity tests**
   ```python
   def test_generated_patches_are_valid_python(self, generator, sample_task):
       """Verify generated patches are valid Python."""
       import ast

       result = generator.generate(sample_task)

       for patch in result.patches:
           try:
               ast.parse(patch.content)
           except SyntaxError as e:
               pytest.fail(f"Generated patch has syntax error: {e}")
   ```

2. **Add diff correctness tests**
   ```python
   def test_learned_patterns_match_ground_truth(self, generator, sample_task):
       """Verify learned patterns match the actual code_after."""
       generator.learn_from_task(sample_task)

       # Verify pattern cache was populated correctly
       # Verify pattern matches ground truth
   ```

3. **Add benchmark integration tests**

**Effort:** 4-6 hours
**Risk:** Low - additive tests only

---

### 9. Remove Dead Code (P3 - CLEANUP)

**Status:** CodeResonator and circuit_observer unused
**Impact:** Confusing codebase
**File:** `src/hologram/swe/generator.py`

**Action (after resonator is integrated):**
1. Either use circuit_observer or remove parameter
2. Clean up unused imports
3. Remove FIX_BUG comments (lines 123, 130, 250, 264)

**Effort:** 1 hour
**Risk:** None - cleanup only

---

## Implementation Sequence

### Phase 1: Fix Critical Issues (Week 1)
1. ✓ Understand current architecture (DONE)
2. [ ] Implement proper diff analysis for learning (Issue #2)
3. [ ] Add pattern persistence (Issue #3)
4. [ ] Enable CodeResonator integration (Issue #1)

**Checkpoint:** Run benchmark, verify improvement

### Phase 2: Improve Quality (Week 2)
5. [ ] Generate executable code (Issue #4)
6. [ ] Add multi-file support (Issue #5)
7. [ ] Improve verification logic (Issue #7)

**Checkpoint:** Run benchmark on sample tasks, measure accuracy improvement

### Phase 3: Add Intelligence (Week 3)
8. [ ] Add code structure understanding (Issue #6)
9. [ ] Improve test coverage (Issue #8)
10. [ ] Clean up dead code (Issue #9)

**Checkpoint:** Prepare for SWE-bench evaluation

---

## Success Metrics

### After Phase 1 (Critical Fixes)
- [ ] CodeResonator is called in generation path
- [ ] Learned patterns match actual code_after
- [ ] Pattern cache persists across sessions
- [ ] Benchmark accuracy improves from ~2% to ~5-8%

### After Phase 2 (Output Quality)
- [ ] All generated patches are valid Python syntax
- [ ] Multi-file tasks generate multiple patches
- [ ] Verification metric is consistent
- [ ] Benchmark accuracy: 8-12%

### After Phase 3 (Code Intelligence)
- [ ] System understands code structure (AST parsing)
- [ ] Context is used in generation decisions
- [ ] Test coverage > 80%
- [ ] Benchmark accuracy: 12-20% (approaching baseline)

---

## Estimated Effort

| Phase | Hours | Risk | Blocker |
|-------|-------|------|---------|
| Phase 1 | 16-18 | Medium | No |
| Phase 2 | 14-18 | Medium | No |
| Phase 3 | 10-14 | Low | No |
| **Total** | **40-50** | **Medium** | **No** |

---

## Risk Mitigation

1. **CodeResonator integration risk:**
   - Mitigate: Create structured issue vectors before enabling
   - Mitigate: Test resonator separately with ground-truth vectors
   - Mitigate: Keep template fallback as safety net

2. **Pattern learning risk:**
   - Mitigate: Test diff analysis on real SWE-bench samples
   - Mitigate: Start with simple heuristics, improve incrementally
   - Mitigate: Verify against ground truth before benchmarking

3. **Multi-file support risk:**
   - Mitigate: Implement for 2-file tasks first
   - Mitigate: Test on SWE-bench tasks with known file counts
   - Mitigate: Verify patch ordering and dependencies

---

## Questions for Team

1. **Vector Domain:** Why are issue vectors unstructured but patch vectors structured? Should we change encode_issue to output structured vectors?

2. **Resonator Purpose:** Was CodeResonator always intended to be used, or is it exploratory code that should be removed?

3. **Learning Goals:** What accuracy target are we aiming for? (5%? 10%? 20%?)

4. **Context Strategy:** Should the system use full code context or stick with lightweight summarization?

5. **Benchmark:** When should we start evaluating on SWE-bench vs sample tasks?

---

**Report Generated:** 2025-12-16
**For Questions:** Review SWE_CODE_REVIEW.md for detailed analysis
**Next Steps:** Begin Phase 1 implementation
