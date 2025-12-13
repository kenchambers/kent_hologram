# Technical Addendum: Critical Code-Level Concerns

**Status**: Supplementary analysis for implementation team
**Date**: 2025-12-13

---

## A. MultiStepResonator Implementation Concerns

### A1. Subtraction in HDC Space (CRITICAL)

**Current Code Location**: MULTI_STEP_IMPLEMENTATION.md, Phase 2 (line 226)

```python
residue = residue - positioned_recon
```

**The Problem**:

In hyperdimensional computing, subtraction is **not algebraically well-defined** in the same way as vector subtraction. The deflation algorithm requires:

```
residue_old = (step1_positioned + step2_positioned + noise)
residue_new = residue_old - step1_positioned
            = step2_positioned + noise  (ideally)
```

**But in HDC**:
- Vectors are unit-normalized for similarity measures
- Subtraction `u - v` doesn't preserve HDC properties
- May create vectors outside valid hypervector space

**Recommended Fix**:

Instead of subtraction, use **weighted bundling with inverse**:

```python
# CURRENT (PROBLEMATIC):
residue = residue - positioned_recon

# RECOMMENDED (SAFE):
# Create "inverse" reconstruction
inverse_recon = positioned_recon * (-1.0)  # Negate all elements
residue = bundle(residue, inverse_recon)
# This preserves HDC properties while "subtracting" signal
```

**Validation**:
```python
def test_deflation_subtraction():
    v1 = torch.randn(10000)
    v1 = v1 / torch.norm(v1)  # Unit normalize

    v2 = Operations.permute(v1, 100)
    v3 = Operations.permute(v1, 200)

    bundled = Operations.bundle(v2, v3)

    # Method 1: Direct subtraction (PROBLEMATIC)
    residue_1 = bundled - v2

    # Method 2: Inverse bundling (RECOMMENDED)
    inv_v2 = -v2
    residue_2 = Operations.bundle(bundled, inv_v2)

    # Method 2 should preserve orthogonality better
    unbind_result = Operations.unbind(residue_2, permute(v1, 200))
    # Should be closer to v1 than Method 1
```

**Action Required**: Review deflation mathematics before implementation

---

### A2. Residue Norm Threshold (CRITICAL)

**Current Code Location**: MULTI_STEP_IMPLEMENTATION.md, Phase 2 (line 199)

```python
residue_norm = torch.norm(proposed_step)
if residue_norm < self._residue_threshold:
    break
```

**The Problem**:

This check happens **after** unbinding but **before** factorization. The logic is correct, but implementation detail is subtle.

**Question**: Should threshold be on `proposed_step` or on `residue`?

```python
# Current interpretation:
proposed_step = unbind(residue, position_key)
residue_norm = ||proposed_step||  # Checking proposed, not residue

# Alternative interpretation:
residue_norm = ||residue||  # Checking original residue

# Which is correct?
```

**Answer**: CHECKING PROPOSED IS CORRECT, but needs clarification

The `proposed_step` is the extracted signal after position unbinding. If it's noisy (low norm), there's no valid step to extract. Checking `residue` directly would miss valid steps.

**Implementation Detail**:

```python
def resonate_sequence(self, observation, max_steps=None):
    residue = observation.clone()

    for step_idx in range(max_steps):
        shift = (step_idx + 1) * self._position_encoding_stride

        # This unbinding isolates step at this position
        proposed_step = self._unbind_positioned_step(residue, shift)

        # Energy in extracted step tells us if valid signal exists
        step_energy = torch.norm(proposed_step)

        # Energy below threshold = no signal at this position
        if step_energy < self._residue_threshold:
            break  # CORRECT

        # Continue with ALS factorization...
```

**Recommendation**: Document this clearly in code comment

---

### A3. Convergence vs. Oscillation Detection

**Current Code Location**: transform_resonator.py, lines 202-213

```python
# Existing convergence check
if self._check_convergence(a, t, m, a_prev, t_prev, m_prev):
    return TransformResult(...)

# Existing oscillation check
state = (a_word, t_word, m_word)
history.append(state)
if self._detect_oscillation(history):
    return TransformResult(...)
```

**For MultiStep, Need to Add**:

```python
# When is a step considered "extracted" in sequence?
# Option 1: When convergence achieved (current)
# Option 2: When quality exceeds threshold (different)
# Option 3: When residue norm falls below threshold (different)

# These give DIFFERENT results!
```

**Recommendation**: For sequence extraction, use **quality threshold** not convergence

```python
def _extract_single_step(self, residue, shift):
    """
    Extract and factorize single step from residue.

    Returns:
        (TransformResult, quality)  or  (None, 0.0) if quality too low
    """
    proposed_step = self._unbind_positioned_step(residue, shift)

    # Run ALS to convergence OR max iterations
    result = self.resonate(proposed_step)  # Uses existing ALS

    # Verify: how well does reconstruction match proposal?
    quality = self.verify_factorization(proposed_step, result)

    if quality >= self._sequence_quality_threshold:
        return result, quality
    else:
        return None, quality  # Step not valid
```

This is **already in the MULTI_STEP_IMPLEMENTATION.md** (line 207), so implementation is correct. Just needs careful testing.

---

### A4. Position Encoding Stride Selection

**Issue**: How to choose `_position_encoding_stride = 100`?

**Current Assumption**: Stride of 100 positions gives sufficient separation

**Mathematical Concern**:

For 10,000-dimensional vectors with random permutation:
- Shift by 100: Orthogonality ≈ 0.0 (good)
- Shift by 1: Orthogonality ≈ high (bad interference)

**Required Validation**:

```python
def test_permutation_orthogonality():
    """Validate that permutation stride achieves separation."""
    v = torch.randn(10000)

    v1_pos = Operations.permute(v, shifts=100)
    v2_pos = Operations.permute(v, shifts=200)

    orthogonality = Similarity.cosine(v1_pos, v2_pos)
    # Should be ≈ 0.0 (orthogonal)
    # If > 0.3, interference likely

    assert orthogonality < 0.1, "Stride too small, signals interfere"
```

**Recommendation**: Make stride **configurable and validated**

```python
class MultiStepResonator:
    def __init__(self, ..., position_encoding_stride=100):
        self._position_encoding_stride = position_encoding_stride
        self._validate_stride()  # NEW

    def _validate_stride(self):
        """Ensure stride provides sufficient signal separation."""
        ref_vec = self._codebook.encode("__STRIDE_TEST__")
        v1 = Operations.permute(ref_vec, self._position_encoding_stride)
        v2 = Operations.permute(ref_vec, 2 * self._position_encoding_stride)

        orthogonality = Similarity.cosine(v1, v2)
        if orthogonality > 0.2:
            raise ValueError(
                f"Stride {self._position_encoding_stride} insufficient "
                f"(orthogonality={orthogonality:.3f}, need < 0.1)"
            )
```

---

## B. RelationalEncoder Implementation Concerns

### B1. Relation Vocabulary Hardcoding

**Location**: NEW FILE (proposed)

**Issue**: Only 4 relations hardcoded: adjacency, same_color, same_shape, containment

```python
RELATION_VOCABULARY = [
    "adjacency",
    "same_color",
    "same_shape",
    "containment"
]
```

**Problem**: How to detect each relation algorithmically?

**Example - Adjacency**:
```python
def detect_adjacency(obj1: Object, obj2: Object) -> bool:
    """Are two objects adjacent (4-connected)?"""
    # Need to check if bounding boxes touch
    # But what about diagonal adjacency? (8-connected)
    # Current code doesn't specify
```

**Recommendation**: Implement with clear semantics

```python
class RelationalEncoder:
    @staticmethod
    def detect_adjacency(obj1: Object, obj2: Object, connectivity=4) -> float:
        """
        Detect adjacency between objects.

        Args:
            connectivity: 4 (cardinal) or 8 (including diagonals)

        Returns:
            Confidence score [0.0, 1.0]
        """
        # Check if any pixels touch
        for (r1, c1) in obj1.pixels:
            for (r2, c2) in obj2.pixels:
                dr = abs(r1 - r2)
                dc = abs(c1 - c2)

                if connectivity == 4:
                    if (dr == 1 and dc == 0) or (dr == 0 and dc == 1):
                        return 1.0
                elif connectivity == 8:
                    if max(dr, dc) == 1:
                        return 1.0

        return 0.0
```

### B2. Hard Cap of 30 Relations (DESIGN RISK)

**Issue**: What happens if more than 30 relations detected?

**Current Plan**: Just cap at 30 (from INNOVATION_ANALYSIS.md)

**Problem**: Which 30 to keep? Random? First 30? By confidence?

**Recommendation**: Sort by confidence and keep top 30

```python
def encode_object_relations(self, obj: Object, context: Grid, max_relations=30) -> torch.Tensor:
    """
    Encode all salient relations for an object.

    Args:
        obj: Object to analyze
        context: Full grid for finding related objects
        max_relations: Maximum relations to encode (default 30)

    Returns:
        Bundled relation vector
    """
    # Detect all objects in context
    all_objects = self.detector.detect(context)
    other_objects = [o for o in all_objects if o != obj]

    # Compute relation scores
    relations_with_scores = []
    for other_obj in other_objects:
        adj_score = self.detect_adjacency(obj, other_obj)
        if adj_score > 0:
            relations_with_scores.append(("adjacency", adj_score))

        color_score = 1.0 if obj.color == other_obj.color else 0.0
        if color_score > 0:
            relations_with_scores.append(("same_color", color_score))

        # ... more relations

    # Sort by confidence and take top 30
    relations_with_scores.sort(key=lambda x: x[1], reverse=True)
    top_relations = relations_with_scores[:max_relations]

    # Encode
    relation_vecs = []
    for rel_name, rel_score in top_relations:
        rel_vec = self._codebook.encode(rel_name)
        # Weight by confidence
        rel_vec = rel_vec * rel_score
        relation_vecs.append(rel_vec)

    if not relation_vecs:
        return self._codebook.encode("no_relations")

    return Operations.bundle(*relation_vecs)
```

---

## C. HierarchicalSalienceResonator: Implementation Risk

### C1. Gating Function Design (CRITICAL REDESIGN NEEDED)

**Current Proposal** (from INNOVATION_ANALYSIS.md):

```python
# Phase 2: Compute gate weight α
α = confidence_margin(result1)  # [0, 1]

# Phase 4: Augment observation
augmented_obs = observation + (α * relation_vec)
```

**Problem**: What does `observation + (α * relation_vec)` mean in HDC space?

**Issue 1**: Different magnitudes

```python
observation: ||o|| = 1.0 (normalized)
relation_vec: ||r|| = 1.0 (normalized)

observation + α * relation_vec:
  = o + 0.3 * r (for example)
  # Magnitude = √(||o||² + 2*(o·r)*α + ||r||²*α²)
  #           ≈ √(1 + 0 + 0.09) ≈ 1.04 (NOT NORMALIZED)
```

This creates a vector outside the valid hypervector space!

### C2. Recommended Redesign (REQUIRED)

**Use multiplicative gating instead of additive**:

```python
class HierarchicalSalienceResonator:

    def resonate_with_saliency(self, observation, relations) -> TransformResult:
        """
        Two-phase resonation with saliency augmentation.

        Phase 1: Primary resonation
        Phase 2-4: Saliency-based augmentation via multiplicative gating
        Phase 5-6: Re-resonation and fusion
        """
        # Phase 1: Standard factorization
        result1 = self.resonate(observation)

        if result1.confidence > self._phase_1_threshold:
            return result1  # Fast path

        # Phase 2: Compute gate strength
        gate_strength = result1.min_confidence  # [0, 1]

        # Phase 3: Extract salient relations
        relations_vec = self._encode_relations(observation)  # (normalized)

        # Phase 4: MULTIPLICATIVE augmentation (mathematically sound)
        # Instead of: augmented = obs + gate_strength * rel
        # Use: augmented = bundle(obs, bind(rel, gate_vec))

        gate_vec = self._codebook.encode(f"gate_{int(gate_strength*10)}")
        saliency_signal = Operations.bind(relations_vec, gate_vec)
        augmented_obs = Operations.bundle(observation, saliency_signal)
        # augmented_obs is still unit-normalized ✓

        # Phase 5: Re-resonate
        result2 = self.resonate(augmented_obs)

        # Phase 6: SELECTION-based fusion (not averaging)
        return self._fuse_results_selective(result1, result2, gate_strength)

    def _fuse_results_selective(self, result1, result2, gate_strength):
        """
        Fuse results using gate strength as decision criterion.

        This ensures output is always vocabulary-constrained.
        """
        if gate_strength < 0.25:
            # Saliency didn't help much
            return result1
        elif gate_strength > 0.75:
            # Saliency was crucial
            return result2
        else:
            # Intermediate case: select by confidence
            return result1 if result1.confidence > result2.confidence else result2
```

**Key Differences**:
- Uses `bind()` instead of `+` (mathematically valid in HDC)
- Uses `bundle()` to combine signals (valid superposition)
- Maintains unit normalization throughout
- Selection-based fusion (always valid result, never invalid interpolation)

---

## D. Vocabulary Expansion: Margin Degradation Testing

### D1. Test Plan

**Location**: tests/arc/test_vocabulary_expansion.py (NEW)

```python
def test_cleanup_margin_degradation():
    """
    Verify that larger vocabulary doesn't break cleanup operation.
    """
    encoder = ObjectEncoder(fractal_space, codebook)

    # Current modifiers: 24 items
    current_mods = encoder._modifier_vectors
    assert len(current_mods) == 24

    # Extended modifiers: 37 items (after additions)
    # Simulate by creating new ones
    new_mods = current_mods.copy()
    for i in range(13):
        new_mods[f"new_modifier_{i}"] = codebook.encode(f"new_modifier_{i}")

    # Create noisy proposal
    proposal = codebook.encode("test_proposal") + 0.1 * torch.randn(10000)

    # Cleanup with current vocab
    similarities_current = Similarity.cosine_batch(proposal,
                                                    torch.stack(list(current_mods.values())))
    margin_current = (torch.topk(similarities_current, 2).values[0] -
                      torch.topk(similarities_current, 2).values[1]).item()

    # Cleanup with extended vocab
    similarities_extended = Similarity.cosine_batch(proposal,
                                                     torch.stack(list(new_mods.values())))
    margin_extended = (torch.topk(similarities_extended, 2).values[0] -
                       torch.topk(similarities_extended, 2).values[1]).item()

    # Margin should degrade but stay above threshold
    degradation_ratio = margin_extended / margin_current

    print(f"Margin degradation: {margin_current:.3f} → {margin_extended:.3f} "
          f"({degradation_ratio:.1%})")

    assert degradation_ratio > 0.8, \
        f"Margin degraded too much: {degradation_ratio:.1%}"
    assert margin_extended > 0.05, \
        f"Margin too small: {margin_extended:.3f}"
```

### D2. Expected Behavior

**Hypothesis**: 24 → 37 items (54% increase) causes ~15% margin reduction

```
Current (24 mods):
  Average spacing: ~0.12
  Margin (typical): ~0.10

Extended (37 mods):
  Average spacing: ~0.08
  Margin (typical): ~0.085 (15% reduction)

This is acceptable: Still > 0.05 threshold
```

---

## E. Testing Strategy: What NOT to Miss

### E1. Critical Path Tests

```python
# MUST PASS before shipping
def test_no_hallucination_guarantee():
    """
    Verify that ALL outputs from solver are vocabulary items.
    """
    solver = HolographicARCSolver(iterative=True)

    # Generate 100 random tasks
    for _ in range(100):
        task = generate_random_arc_task()
        result = solver.solve(task)

        if result.transformation is not None:
            # Verify each component is in vocabulary
            assert result.transformation.action in ACTIONS
            assert result.transformation.target in TARGETS
            assert result.transformation.modifier in MODIFIERS
```

### E2. Regression Tests

```python
# MUST NOT DEGRADE single-step performance
def test_backward_compatibility():
    """
    Single-step tasks should still work with new architecture.
    """
    # Load existing single-step test cases
    tasks = load_single_step_benchmarks()

    solver_old = HolographicARCSolver()  # Old behavior
    solver_new = HolographicARCSolver(iterative=True)  # New behavior

    for task in tasks:
        result_old = solver_old.solve(task)
        result_new = solver_new.solve(task)

        # Should match or improve
        assert accuracy(result_new) >= 0.95 * accuracy(result_old)
```

### E3. Edge Case Tests

```python
# MUST HANDLE gracefully
def test_edge_cases():
    """
    Test boundary conditions that could break algorithm.
    """
    # Empty grid
    grid_empty = Grid.from_list([[0]])
    objects_empty = detector.detect(grid_empty)
    assert objects_empty == []

    # Single large object (all non-background)
    grid_full = Grid.from_list([[1]*30]*30)
    objects_full = detector.detect(grid_full)
    assert len(objects_full) == 1

    # High noise observation (residue norm very high)
    noise_obs = torch.randn(10000)  # Not a clean observation
    result = resonator.resonate_sequence(noise_obs)
    assert result.num_steps <= 1  # Should quit early

    # K-step detection with K > max_steps
    # (Should only extract max_steps and stop)
    assert result.num_steps <= resonator._max_sequence_steps
```

---

## F. Performance Monitoring

### F1. Profiling Points

```python
@timer("deflation_algorithm")
def resonate_sequence(self, observation):
    # Track time for each step extraction
    for step_idx in range(max_steps):
        with timer(f"step_{step_idx}_unbind"):
            proposed = self._unbind_positioned_step(residue, shift)

        with timer(f"step_{step_idx}_als"):
            result = self.resonate(proposed)

        with timer(f"step_{step_idx}_verify"):
            quality = self.verify_factorization(proposed, result)
```

### F2. Expected Performance

```
Single-step ALS:     ~100-200ms per task
Multi-step (K=2):    ~200-400ms per task (+1-2x)
Multi-step (K=3):    ~300-600ms per task (+2-3x)
Multi-step (K=5):    ~500-1000ms per task (+3-5x)

Acceptable latency budget: <2 seconds per task
```

---

## G. Integration Checklist: Code Review Points

Before merging any changes, verify:

### G1. Mathematics Checklist

- [ ] No direct subtraction in HDC space (use inverse bundling)
- [ ] All vectors normalized to unit norm
- [ ] Position encoding stride validated (orthogonality < 0.1)
- [ ] Bundling capacity respected (< 50 items per bundle)
- [ ] Cleanup always selects from vocabulary

### G2. Architecture Checklist

- [ ] No breaking changes to public APIs
- [ ] Single-step path still functional (backward compatible)
- [ ] Error handling for edge cases (empty grids, high noise)
- [ ] Configuration parameters well-documented
- [ ] Fallback mechanisms in place

### G3. Testing Checklist

- [ ] No-hallucination guarantee verified for all new code
- [ ] Regression tests pass (single-step tasks)
- [ ] Edge case tests pass (empty, full, noisy, high-K)
- [ ] Performance benchmarks acceptable (< 2s per task)
- [ ] Integration tests on real ARC tasks (expect +25% improvement)

---

## H. Known Limitations & Future Work

### H1. Current Scope Limitations

1. **Counting Operations**: Proposed but not implemented
   - Cardinality estimation via norm (quick win)
   - Test on counting-specific ARC tasks

2. **Symmetry Primitives**: Limited implementation
   - Currently: diagonal_main, diagonal_anti, point_center
   - Could expand: rotational_2, rotational_4, glide_reflection

3. **Fill Patterns**: Basic implementation
   - Current: fill_solid, fill_checkerboard, fill_border, fill_interior
   - Could add: fill_gradient, fill_random, fill_sparse

### H2. Future Research Directions

1. **Continuous Parameter Learning**: Instead of discrete modifiers
   - Learn distribution over angle values (not just 90_degrees)
   - Learn distribution over translation distances (not just up/down)

2. **Attention Mechanism**: Select salient relations dynamically
   - Learn which relations matter for specific tasks
   - Adaptive relation selection instead of fixed 30

3. **Meta-Learning**: Train on large ARC dataset
   - Learn good initialization for ALS from examples
   - Improve convergence speed and accuracy

---

**End of Technical Addendum**
