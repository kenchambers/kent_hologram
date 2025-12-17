# DeepCode Analysis: Technical Appendix

## Execution Traces: How Current Code Flows

### Trace 1: Single-File Patch Generation (Current Hologram)

**File:** `src/hologram/swe/generator.py`, lines 95-210

```
INPUT: SWETask
  ├─ issue_text: "The encode_issue() function signature changed, needs updating"
  ├─ code_before: {
  │    "encoder.py": "def encode_issue(self, issue_text: str) -> Tensor: ...",
  │    "generator.py": "issue_vec = self._encoder.encode_issue(task.issue_text)"
  │  }
  └─ code_after: {
       "encoder.py": "def encode_issue(self, issue_text: str, context: dict) -> Tensor: ..."
     }

EXECUTION FLOW:

Step 1: Encode issue (Line 116)
  issue_vec = self._encoder.encode_issue(task.issue_text)

  In encoder.py:encode_issue() [Lines 107-128]:
    ├─ words = ["signature", "changed", "needs", "updating"]
    ├─ key_terms = ["signature", "changed", "needs", "updating"]  (filtered)
    ├─ term_vecs = [encode("term_signature"), encode("term_changed"), ...]
    └─ return bundle(*term_vecs)  ← Single HDC vector representing issue

Step 2: Register files (Line 118-120)
  for filepath in task.code_before.keys():
    self._encoder.register_file(filepath)

  Result: self._encoder._file_vectors = {
    "encoder.py": <vector from seed>,
    "generator.py": <vector from seed>
  }

Step 3: Query memory for learned patterns (Line 128-134)
  memory_label, memory_confidence = self._memory.query(issue_vec)

  In consolidation_manager.py:query() [Lines 329-387]:
    ├─ Query HDC: resonance(issue_vec, [all_candidate_vectors])
    ├─ Query Neural: neural_memory.query(issue_vec)
    └─ return ("swe::task_123::encoder.py", 0.82)  ← Confidence from neural

  Confidence check (Line 131):
    if memory_confidence (0.82) >= confidence_threshold (0.3):  ✓ PASS
      memory_pattern = self.get_learned_pattern("swe::task_123::encoder.py")

Step 4: Retrieve learned pattern (Line 132)
  memory_pattern = ("modify_function", "encoder.py", "line_42", "def encode_issue(...)")

Step 5: Create patch (Line 144-176)
  operation = "modify_function"
  target_file = "encoder.py"
  location = "line_42"
  content = "def encode_issue(self, issue_text: str, context: dict) -> Tensor: ..."

  patch = CodePatch(
    file="encoder.py",
    operation="modify_function",
    location="line_42",
    content="def encode_issue(...)",
  )
  patches = [patch]

Step 6: Verify patch (Line 179-189)
  patch_vec = self._encoder.encode_patch(patch)

  In encoder.py:encode_patch() [Lines 70-105]:
    ├─ op_vec = encode("operation_modify_function")
    ├─ file_vec = _get_or_create_file_vector("encoder.py") = self._file_vectors["encoder.py"]
    ├─ loc_type = "function_name"
    ├─ loc_vec = bind(encode("location_function_name"), encode("loc_line_42"))
    └─ observation = bundle(
         bind(op_vec, role_operation),
         bind(file_vec, role_file),
         bind(loc_vec, role_location)
       )

  verification_score = cosine(issue_vec, patch_vec) = 0.78
  verification_passed = (0.78 > 0.1) = True

Step 7: Report (Line 192-199)
  self._circuit_observer.observe(
    items=["modify_function", "encoder.py", "line_42"],
    success=True,
    confidence=0.82,
    context="code_generation"
  )

OUTPUT: PatchResult
  ├─ patches: [CodePatch(file="encoder.py", operation="modify_function", ...)]
  ├─ confidence: 0.82
  ├─ verification_passed: True
  └─ factorization: {"operation": "modify_function", "file": "encoder.py", "location": "line_42"}

═══════════════════════════════════════════════════════════════════════════════

CRITICAL GAP EXPOSED:

  ✗ NO PATCHES FOR generator.py
    Trace shows: patches = [CodePatch(file="encoder.py", ...)]
    Missing: CodePatch(file="generator.py", operation="modify_line", ...)

    Why? Line 139:
      target_file = list(task.code_before.keys())[0]  ← Picks FIRST file only!

    The system generates a patch for encoder.py (the change source),
    but has NO MECHANISM to identify that generator.py also needs updating.

    This is because CodeGenerator has NO ACCESS TO:
    - Call graph (who calls encode_issue?)
    - File dependencies (which files import encoder.py?)
    - Transitive callers (what breaks if signature changes?)

═══════════════════════════════════════════════════════════════════════════════
```

---

### Trace 2: Attempting Transitive Query with Current FactStore

**File:** `src/hologram/memory/fact_store.py`

```
SCENARIO: "Find all functions that call encode_issue()"

FACTS STORED (S-P-O triples via add_fact):
  Fact 1: ("encode_issue", "defined_in", "encoder.py")
  Fact 2: ("CodeGenerator.generate", "calls", "encode_issue")
  Fact 3: ("CodeGenerator", "uses", "encoder.py")
  Fact 4: ("test_generator", "calls", "CodeGenerator.generate")

QUERY: "What calls encode_issue?"

Using query_subject() [Lines 335-409]:

  Query: query_subject("calls", "encode_issue")

  In _memory.resonance() [Line 405]:
    ├─ Encodes: "calls" → p_vec, "encode_issue" → o_vec
    ├─ Creates key: bind(p_vec, o_vec) = bind(calls_vec, encode_issue_vec)
    ├─ Builds candidates: [encode_vec(s) for s in all_subjects]
    │    candidates = [encode("encode_issue"), encode("CodeGenerator.generate"),
    │                  encode("CodeGenerator"), encode("test_generator")]
    ├─ Resonance: similarities = memory.resonance(key, candidates)
    │    similarities = [0.15, 0.92, 0.22, 0.18]  ← Best match
    └─ return subject_list[1] = "CodeGenerator.generate", confidence = 0.92

RESULT: ("CodeGenerator.generate", 0.92)  ← Direct match via reverse query

═════════════════════════════════════════════════════════════════════════════

FOLLOW-UP QUERY: "Find all functions that call CodeGenerator.generate()"

PROBLEM:
  - Fact 4 stores: ("test_generator", "calls", "CodeGenerator.generate")
  - But query_subject needs the exact object string match
  - String: "CodeGenerator.generate" vs Fact object: "CodeGenerator.generate"
  - Should match, but resonance picks best candidate via similarity

Using query_subject() again [Lines 335-409]:

  Query: query_subject("calls", "CodeGenerator.generate")

  Candidates: [encode("encode_issue"), encode("CodeGenerator.generate"),
               encode("CodeGenerator"), encode("test_generator")]

  resonance(key, candidates) = [0.18, 0.15, 0.11, 0.88]

  RESULT: ("test_generator", 0.88)  ← Found!

═════════════════════════════════════════════════════════════════════════════

MULTI-HOP CHAIN:

  Hop 1: "What calls encode_issue?"
    → query_subject("calls", "encode_issue")
    → ("CodeGenerator.generate", 0.92)

  Hop 2: "What calls CodeGenerator.generate?"
    → query_subject("calls", "CodeGenerator.generate")
    → ("test_generator", 0.88)

  Hop 3: "What calls test_generator?"
    → query_subject("calls", "test_generator")
    → FAIL: No such fact exists
    → return ("", 0.0)

CONFIDENCE DECAY:
  encode_issue ──0.92──> CodeGenerator.generate ──0.88──> test_generator
  Combined confidence: 0.92 * 0.88 = 0.81

  For 4+ hops:
    0.92 * 0.88 * 0.85 * 0.80 = 0.55  ← Gets unreliable

═════════════════════════════════════════════════════════════════════════════

WHY GRAPH TRAVERSAL WINS:

  Explicit graph:
    encode_issue ─calls──> CodeGenerator.generate ─calls──> test_generator
    Traversal: [encode_issue, CodeGenerator.generate, test_generator]
    Confidence: 1.0 (exact edges, no resonance)
    Time: O(n) BFS, not O(n²) resonance
```

---

### Trace 3: CodeResonator Factorization (What It Captures)

**File:** `src/hologram/swe/code_resonator.py`

```
INPUT: Code observation vector (bundled representation of a code change)

FACTORIZATION PROCESS [Lines 105-116]:

  observation = encode(changed code snippet)

  In TransformationResonator.resonate():
    Uses ALS (Alternating Least Squares) to decompose:
    observation ≈ bundle(operation_vec, file_vec, location_vec)

    Iterates:
      ├─ Iteration 1: Solve for action, keep target/modifier fixed
      ├─ Iteration 2: Solve for target, keep action/modifier fixed
      ├─ Iteration 3: Solve for modifier, keep action/target fixed
      ├─ ... repeat until convergence
      └─ Return: CodeFactorization with all three slots

OUTPUT: CodeFactorization

  ┌─ operation: "modify_function"        ← What type of change
  ├─ file: "encoder.py"                  ← Which file
  ├─ location: "line_42"                 ← Where in file
  ├─ operation_vec: <tensor>
  ├─ file_vec: <tensor>
  ├─ location_vec: <tensor>
  ├─ iterations: 12
  ├─ converged: True
  └─ confidence: {
       "operation": 0.95,
       "file": 0.88,
       "location": 0.76
     }

═════════════════════════════════════════════════════════════════════════════

WHAT THIS CAPTURES:

  ✓ Changes ARE factorized into 3 dimensions
  ✓ Each dimension has confidence score
  ✓ Reconstruction error shows goodness-of-fit

WHAT THIS LOSES:

  ✗ NO mapping from file → other files that depend on it
  ✗ NO connection between slots
      E.g., "If location (line_42) changes, what other locations break?"

  ✗ NO context about what changed in the content
      location = "line_42" is just a position, not "function signature changed"

  ✗ NO semantic relationship between factorization slots
      operation="modify_function" + file="encoder.py"
      Should trigger check: "What calls functions in encoder.py?"
      But factorization has no way to express this

═════════════════════════════════════════════════════════════════════════════

EXAMPLE WHERE SLOTS ARE INDEPENDENT:

  Change 1:
    operation="modify_function", file="encoder.py", location="line_42"
    Confidence: {operation: 0.95, file: 0.88, location: 0.76}

  Change 2:
    operation="add_line", file="generator.py", location="line_99"
    Confidence: {operation: 0.92, file: 0.85, location: 0.68}

  ALS decomposes each change independently.
  There's NO CONSTRAINT that says:
    "If encoder.py:line_42 changed signature,
     then generator.py:line_99 (a call site) MUST also change"

  This cross-slot, cross-file dependency is MISSING.

═════════════════════════════════════════════════════════════════════════════
```

---

## Architecture Comparison: Resonance vs Graph

### Comparison 1: Answering "Who calls encode()?"

```
═════════════════════════════════════════════════════════════════════════════
PURE HDC RESONANCE (Current Hologram)
═════════════════════════════════════════════════════════════════════════════

Data Storage:
  FactStore: S-P-O triples in HDC
  ("CodeGenerator.generate", "calls", "encode_issue")

  Stored as:
    key = bind(encode_subject, encode_predicate)
    value = encode_object

    key = bind(encode("CodeGenerator.generate"), encode("calls"))
    value = encode("encode_issue")

Query:
  fs.query_subject("calls", "encode_issue")

  Execution:
    ├─ p_vec = encode("calls")
    ├─ o_vec = encode("encode_issue")
    ├─ query_key = bind(p_vec, o_vec)
    ├─ candidates = [encode(s) for s in all_subjects]
    ├─ sims = memory.resonance(query_key, candidates)
    │   resonance involves unbinding: sims = cosine(unbind(query_key, candidate), subject)
    └─ return candidates[argmax(sims)]

Accuracy: HIGH (0.92 confidence for exact match)
Time: O(n) where n = number of facts
Scalability: ~100 facts before resonance interference becomes problematic
Space: Single 10000-dim vector (small)

═════════════════════════════════════════════════════════════════════════════
EXPLICIT GRAPH (DeepCode Style)
═════════════════════════════════════════════════════════════════════════════

Data Storage:
  Graph: Adjacency list
  encode_issue → [CodeGenerator.generate]  (1 edge)

  Stored as:
    edges = {
      "encode_issue": ["CodeGenerator.generate"],
      "CodeGenerator.generate": ["test_generator", "main"]
    }

Query:
  graph.get_reverse_edges("encode_issue")

  Execution:
    ├─ Look up key: "encode_issue"
    ├─ Return reverse adjacency list
    └─ Result: ["CodeGenerator.generate"]

Accuracy: 100% (exact edges)
Time: O(1) lookup + O(k) where k = number of callers (very small)
Scalability: Unlimited
Space: Node + k edges per node

═════════════════════════════════════════════════════════════════════════════
HYBRID (Proposed Minimal Addition)
═════════════════════════════════════════════════════════════════════════════

Data Storage:
  SAME as resonance: FactStore S-P-O triples

Query:
  graph = CodeDependencyGraph(fact_store)
  result = graph.get_transitive_dependents("encode_issue", max_depth=5)

  Execution:
    ├─ visited = {}, queue = [("encode_issue", 0)]
    ├─ Loop:
    │    current = "encode_issue", depth = 0
    │    facts = fact_store.get_facts_by_object("encode_issue")
    │    dependents = [f.subject for f in facts if f.predicate == "calls"]
    │    for dep in dependents:
    │      queue.append((dep, 1))
    │      visited.add(dep)
    └─ Return visited

Accuracy: HIGH (reuses FactStore's S-P-O facts)
Time: O(n) BFS where n = total facts traversed (no resonance per hop)
Scalability: Better than pure resonance (no confidence decay)
Space: FactStore overhead (same as resonance) + thin traversal code

═════════════════════════════════════════════════════════════════════════════
```

---

### Comparison 2: Multi-File Impact Analysis

```
═════════════════════════════════════════════════════════════════════════════
SCENARIO: "If encode_issue() signature changes, what breaks?"
═════════════════════════════════════════════════════════════════════════════

RESONANCE APPROACH (Current):

  Step 1: Get direct callers
    fs.query_subject("calls", "encode_issue")
    → "CodeGenerator.generate" (0.92 confidence)

  Step 2: Find file containing CodeGenerator.generate
    fs.query("CodeGenerator.generate", "defined_in")
    → "generator.py" (0.88 confidence)

    ✓ Works if fact exists, ✗ Confidence loss at each hop

  Step 3: ???
    No way to know "CodeGenerator.generate's signature constraint"
    Stored fact: ("CodeGenerator.generate", "calls", "encode_issue")
    Missing: ("encode_issue", "signature", "(self, issue_text: str, context: dict)")

    Can't verify: "Will this call site break?"

═════════════════════════════════════════════════════════════════════════════

GRAPH APPROACH (Proposed):

  Step 1: Get direct callers (via traversal)
    graph.get_transitive_dependents("encode_issue")
    → {"CodeGenerator.generate"}  (exact set)

  Step 2: Find files containing these functions
    for func in dependents:
      file = graph.get_function_file(func)
      affected_files.add(file)
    → {"generator.py"}

  Step 3: Validate signature compatibility
    old_sig = ("self", "issue_text: str") → Tensor
    new_sig = ("self", "issue_text: str", "context: dict") → Tensor

    for call_site in graph.get_call_sites("CodeGenerator.generate", "encode_issue"):
      if call_site.passes_args(old_sig) and not new_sig.compatible(old_sig):
        errors.append(f"Breaking change in {call_site.file}:{call_site.line}")

Result:
  Affected files: {generator.py}
  Errors: ["Breaking change in generator.py:99 - missing 'context' argument"]

═════════════════════════════════════════════════════════════════════════════
```

---

## Confidence Decay Analysis

```
═════════════════════════════════════════════════════════════════════════════
RESONANCE CONFIDENCE AT N HOPS
═════════════════════════════════════════════════════════════════════════════

Assumption: Each resonance query returns confidence ≈ 0.85-0.95 for exact match

Hop 1: encode_issue → CodeGenerator.generate
  query_subject("calls", "encode_issue") → (result, 0.92)

Hop 2: CodeGenerator.generate → ??
  query_subject("calls", "CodeGenerator.generate") → (result, 0.88)

Hop 3: ?? → ??
  query_subject("calls", ...) → (result, 0.84)

Hop 4: ?? → ??
  query_subject("calls", ...) → (result, 0.78)

Hop 5: ?? → ??
  query_subject("calls", ...) → (result, 0.68)

CUMULATIVE CONFIDENCE LOSS:

  After hop 1: 0.92
  After hop 2: 0.92 * 0.88 = 0.81
  After hop 3: 0.81 * 0.84 = 0.68
  After hop 4: 0.68 * 0.78 = 0.53
  After hop 5: 0.53 * 0.68 = 0.36  ← Unreliable threshold (assume 0.5+)

IMPLICATION:
  ✓ 1-2 hops: Reliable (>0.80 confidence)
  ⚠ 3-4 hops: Marginal (0.50-0.80)
  ✗ 5+ hops: Unreliable (<0.50)

Real code often has 4-8 hop chains:
  main() → handler() → processor() → encoder() → helper()

═════════════════════════════════════════════════════════════════════════════
GRAPH TRAVERSAL: NO DECAY
═════════════════════════════════════════════════════════════════════════════

Transitive closure: BFS on exact edges
  main() ──called_by──> handler()
  handler() ──called_by──> processor()
  processor() ──called_by──> encoder()
  encoder() ──called_by──> helper()

Result: {main, handler, processor, encoder, helper}
Confidence: 1.0 (or "exact" if you want to distinguish)
Time: O(n) where n = edges traversed

═════════════════════════════════════════════════════════════════════════════
```

---

## Recommendation: Phased Implementation

### Phase 1: Add Dependency Graph (Week 1)

```python
# src/hologram/swe/dependency_graph.py (NEW, ~80 lines)

class CodeDependencyGraph:
    def __init__(self, fact_store):
        self._fs = fact_store

    def get_transitive_dependents(self, function: str, max_depth: int = 5) -> Set[str]:
        """BFS traversal of fact triples"""
        # Implementation: ~30 lines

    def get_files_affected(self, function: str) -> Dict[str, List[str]]:
        """Map functions to files they're in"""
        # Implementation: ~20 lines

    def get_import_chain(self, from_file: str, to_file: str) -> Optional[List[str]]:
        """Find import path between files"""
        # Implementation: ~30 lines
```

**Integration:**
```python
# In src/hologram/swe/generator.py, line 122:

graph = CodeDependencyGraph(self._fact_store)
affected_files = graph.get_files_affected(changed_function)
```

---

### Phase 2: Enhance Metadata Storage (Week 1)

```python
# In src/hologram/swe/encoder.py, add:

def extract_function_signature(code: str, location: str) -> Optional[str]:
    # Extract (param_types) -> return_type
    # Implementation: ~20 lines

# In src/hologram/swe/generator.py, modify learn_from_task():

# Store function-file relationships
for file in changed_files:
    functions = extract_functions_defined(code_after[file])
    for func in functions:
        fs.add_fact(func, "defined_in", file)
        fs.add_fact(file, "defines", func)
```

---

### Phase 3: Add Multi-File Validator (Week 2)

```python
# src/hologram/swe/validator.py (NEW, ~60 lines)

class MultiFilePatchValidator:
    def validate_patch_set(self, patches: List[CodePatch]) -> Tuple[bool, List[str]]:
        """Check for breaking changes across files"""
        # Implementation: ~60 lines
```

**Integration:**
```python
# In src/hologram/swe/generator.py, line 186:

validator = MultiFilePatchValidator(self._fact_store, self._encoder)
is_valid, errors = validator.validate_patch_set(patches)

if not is_valid:
    verification_passed = False
    return PatchResult(..., verification_passed=False, errors=errors)
```

---

## Summary: What Changes and What Doesn't

### Files UNCHANGED:
- `src/hologram/memory/fact_store.py` - Keep S-P-O storage
- `src/hologram/consolidation/manager.py` - Keep neural consolidation
- `src/hologram/swe/encoder.py` - Minimal enhancement (metadata extraction)
- `src/hologram/swe/code_resonator.py` - No changes

### Files MODIFIED:
- `src/hologram/swe/generator.py` - Add graph usage, multi-file logic (~20 lines)
- `src/hologram/swe/encoder.py` - Add metadata extraction (~20 lines)

### Files CREATED:
- `src/hologram/swe/dependency_graph.py` - NEW (~80 lines)
- `src/hologram/swe/validator.py` - NEW (~60 lines)

### Total Code Addition: ~180 lines (all new/additive, no refactoring)

