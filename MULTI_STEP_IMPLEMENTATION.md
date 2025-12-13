# Multi-Step Resonator: Detailed Implementation Guide

**Objective**: Transform single-step Resonator into sequence-factorizing system
**Status**: Design Phase (Ready for Implementation)
**Expected Timeline**: 2-3 weeks

---

## Problem Statement

### Current Limitation
```python
# Current system (transform_resonator.py):
observation_bundle = encoder.bundle_training_observations(pairs)
result = resonator.resonate(observation_bundle)
# Output: TransformResult(action='tile', target='all', modifier='by_pattern')
#         ^ SINGLE STEP ONLY

# Applied as:
output_grid = executor.execute(
    action=result.action,
    target=result.target,
    modifier=result.modifier,
    objects=objects,
    grid=test_input
)
```

### Real ARC Requirement
```
Example Task: Tile pattern 3×3 into 3×3 grid, then recolor red cells to blue

Step 1: Tile(all, by_pattern)     → Creates 3×3 repeated pattern
Step 2: Recolor(red, to_blue)     → Changes red pixels in result

Current system: Can detect ONE of these → Wrong output
Proposed system: Detects BOTH in order → Correct output
```

---

## Mathematical Foundation

### Sequence Encoding with Permutation

**Key Insight**: Use position encoding (permutation) to keep steps orthogonal

```
For K-step sequence, encode as:
    O_seq = Σ(t=1 to K) Π^t( bind(bind(A_t ⊗ role_A, T_t ⊗ role_T), M_t ⊗ role_M) )

Where:
    Π = Permutation operator (circular shift in vector space)
    Π^t = Permute by (t × SHIFT_STRIDE) positions
    SHIFT_STRIDE = Position encoding stride (e.g., 100)

    A_t, T_t, M_t = Action, Target, Modifier at step t
    role_A, role_T, role_M = Role vectors (already exist in codebase)
```

**Why This Works**:
1. Each step is at different position in permutation space
2. Permutation is invertible via torchhd.permute()
3. Orthogonality means: unbind(O_seq, Π^t) ≈ raw_step_t + noise
4. Noise is low because Π^t vectors have high distance from others

### Deflation Algorithm

```
ALGORITHM: ExtractSequenceSteps(observation_bundle, max_steps=5)

    steps = []
    residue = observation_bundle.clone()
    shift_stride = 100  # Position encoding stride

    for step_idx in range(max_steps):
        shift = (step_idx + 1) * shift_stride

        # 1. Unbind position encoding
        position_vec = permute(reference_vec, shift)
        proposed_step = unbind(residue, position_vec)

        # 2. Solve for (A, T, M) via ALS
        (a, t, m, a_word, t_word, m_word) = ALS_factorize(proposed_step)

        # 3. Verify quality
        reconstructed = bundle(
            bind(a, role_A),
            bind(t, role_T),
            bind(m, role_M)
        )
        quality = cosine_similarity(proposed_step, reconstructed)

        # 4. Check confidence
        if quality < QUALITY_THRESHOLD:
            break  # No more valid steps

        steps.append(TransformResult(a_word, t_word, m_word, confidence=quality))

        # 5. Deflate residue
        positioned_reconstruction = permute(reconstructed, shift)
        residue = residue - positioned_reconstruction

    return steps, residue
```

**Convergence Criteria**:
- Step quality < threshold (default: 0.6)
- Residue norm falls below threshold (default: 0.1)
- Max steps reached (default: 5)

---

## Implementation Strategy

### Phase 1: New Dataclass (50 lines)

```python
# File: src/hologram/arc/transform_resonator.py (ADD AT TOP)

@dataclass
class SequenceTransformResult:
    """Result of multi-step sequence factorization."""
    steps: List[TransformResult]
    sequence: List[Tuple[str, str, str]]  # [(action1, target1, mod1), ...]

    # Quality metrics
    num_steps: int
    total_quality: float  # Min quality across all steps
    converged: bool
    final_residue_norm: float

    # Per-step details
    step_qualities: List[float]
    step_confidences: List[Dict[str, float]]

    def __str__(self) -> str:
        result = f"SequenceTransform(n_steps={self.num_steps}, converged={self.converged})\n"
        for i, (step, quality) in enumerate(zip(self.steps, self.step_qualities)):
            result += f"  Step {i+1}: {step.action}({step.target}, {step.modifier}) [quality={quality:.3f}]\n"
        return result

    def as_execution_plan(self) -> List[Tuple[str, str, str]]:
        """Return sequence as (action, target, modifier) tuples for executor."""
        return [(s.action, s.target, s.modifier) for s in self.steps]
```

### Phase 2: Extend TransformationResonator (300 lines)

```python
# File: src/hologram/arc/transform_resonator.py (ADD TO CLASS)

class TransformationResonator:
    # ... existing code ...

    def __init__(self, ...):
        # ... existing init ...

        # New: Position encoding configuration
        self._position_encoding_stride = 100
        self._max_sequence_steps = 5
        self._sequence_quality_threshold = 0.6
        self._residue_threshold = 0.1

    def resonate_sequence(
        self,
        observation: torch.Tensor,
        max_steps: int = None,
    ) -> SequenceTransformResult:
        """
        Factorize observation into sequence of (A,T,M) transformations.

        Uses deflation algorithm: iteratively extract and subtract steps.

        Args:
            observation: Bundled observation vector
            max_steps: Maximum steps to extract (default: self._max_sequence_steps)

        Returns:
            SequenceTransformResult with all steps, qualities, confidence metrics
        """
        if max_steps is None:
            max_steps = self._max_sequence_steps

        steps = []
        step_qualities = []
        step_confidences = []

        residue = observation.clone()

        for step_idx in range(max_steps):
            # 1. Compute position encoding for this step
            shift = (step_idx + 1) * self._position_encoding_stride

            # 2. Unbind position encoding from residue
            proposed_step = self._unbind_positioned_step(residue, shift)

            # 3. Check if residue has enough energy
            residue_norm = torch.norm(proposed_step)
            if residue_norm < self._residue_threshold:
                break

            # 4. Factorize proposed step into (A, T, M) via ALS
            step_result = self.resonate(proposed_step)

            # 5. Verify quality
            quality = self.verify_factorization(proposed_step, step_result)
            step_qualities.append(quality)
            step_confidences.append(step_result.confidence)

            # 6. Check if quality acceptable
            if quality < self._sequence_quality_threshold:
                break

            steps.append(step_result)

            # 7. Reconstruct and deflate residue
            reconstructed = self._ops.bundle(
                self._ops.bind(step_result.action_vec, self._role_action),
                self._ops.bind(step_result.target_vec, self._role_target),
                self._ops.bind(step_result.modifier_vec, self._role_modifier),
            )

            # Subtract positioned reconstruction
            positioned_recon = self._apply_position_encoding(reconstructed, shift)
            residue = residue - positioned_recon

        # Build result
        return SequenceTransformResult(
            steps=steps,
            sequence=[(s.action, s.target, s.modifier) for s in steps],
            num_steps=len(steps),
            total_quality=min(step_qualities) if step_qualities else 0.0,
            converged=len(steps) < max_steps,
            final_residue_norm=float(torch.norm(residue).item()),
            step_qualities=step_qualities,
            step_confidences=step_confidences,
        )

    def _unbind_positioned_step(
        self,
        residue: torch.Tensor,
        shift: int,
    ) -> torch.Tensor:
        """
        Extract step at given position encoding.

        Args:
            residue: Current residue vector
            shift: Position shift amount

        Returns:
            Proposed step vector (may be noisy)
        """
        # Create inverse position encoding
        # Use action role as reference for permutation
        position_key = Operations.permute(self._role_action, shifts=shift)

        # Unbind: residue with position key
        proposed = self._ops.unbind(residue, position_key)

        return proposed

    def _apply_position_encoding(
        self,
        vector: torch.Tensor,
        shift: int,
    ) -> torch.Tensor:
        """
        Apply position encoding (permutation) to vector.

        Args:
            vector: Vector to encode
            shift: Position shift amount

        Returns:
            Position-encoded vector
        """
        return Operations.permute(vector, shifts=shift)

    # Optional: Utility to detect likely sequence length
    def estimate_sequence_length(self, observation: torch.Tensor) -> int:
        """
        Estimate K (number of steps) without full solving.

        Uses heuristic: plot residue norm vs steps, detect inflection.
        """
        residue = observation.clone()
        norms = [float(torch.norm(residue).item())]

        for _ in range(self._max_sequence_steps):
            # Try to extract step (don't verify, just estimate)
            step = self.resonate(residue)

            # Rough reconstruction
            recon = self._ops.bundle(
                self._ops.bind(step.action_vec, self._role_action),
                self._ops.bind(step.target_vec, self._role_target),
                self._ops.bind(step.modifier_vec, self._role_modifier),
            )

            residue = residue - recon
            norms.append(float(torch.norm(residue).item()))

        # Simple heuristic: count until norm drops below threshold
        for i, norm in enumerate(norms):
            if norm < self._residue_threshold:
                return i

        return self._max_sequence_steps
```

### Phase 3: Extend Solver (200 lines)

```python
# File: src/hologram/arc/solver.py (MODIFY solve() METHOD)

class HolographicARCSolver:
    # ... existing code ...

    def solve(self, task: ARCTask) -> SolverResult:
        """
        Attempt to solve an ARC task using multi-step resonator.
        """
        # ... existing task signature and cache code ...

        # Bundle observations from training pairs
        observations = []
        for pair in task.training:
            obs = self._observe_training_pair(pair)
            if obs is not None:
                observations.append(obs)

        if not observations:
            return SolverResult(
                output=None,
                transformation=None,
                confidence=0.0,
                from_cache=False,
                message="No valid observations",
            )

        observation_bundle = self._ops.bundle(*observations)

        # TRY MULTI-STEP FIRST (new behavior)
        sequence_result = self._resonator.resonate_sequence(observation_bundle)

        if sequence_result.num_steps > 0:
            # Multi-step succeeded
            combined_confidence = sequence_result.total_quality

            if combined_confidence >= self._confidence_threshold:
                # Apply sequence
                output = self._apply_sequence(sequence_result, task.test_input)

                # Consolidate
                self._store_skill_sequence(task_sig_vec, sequence_result)

                return SolverResult(
                    output=output,
                    transformation=sequence_result.steps[0],  # First step for logging
                    confidence=combined_confidence,
                    from_cache=False,
                    message=f"Multi-step ({sequence_result.num_steps}): {sequence_result}",
                )

        # FALLBACK: Single-step (legacy behavior)
        single_result = self._resonator.resonate(observation_bundle)
        verification = self._resonator.verify_factorization(
            observation_bundle, single_result
        )
        combined_confidence = min(single_result.min_confidence, verification)

        if combined_confidence >= self._confidence_threshold:
            output = self._apply_transformation(single_result, task.test_input)
            self._store_skill(task_sig_vec, single_result)

            return SolverResult(
                output=output,
                transformation=single_result,
                confidence=combined_confidence,
                from_cache=False,
                message=f"Single-step: {single_result}",
            )

        # No solution found
        return SolverResult(
            output=None,
            transformation=None,
            confidence=0.0,
            from_cache=False,
            message="No solution (single or multi-step)",
        )

    def _apply_sequence(
        self,
        sequence: SequenceTransformResult,
        test_input: Grid,
    ) -> Grid:
        """Apply sequence of transformations in order."""
        current_grid = test_input
        current_objects = self._detector.detect(current_grid)

        for step in sequence.steps:
            # Apply this transformation
            current_grid = self._executor.execute(
                action=step.action,
                target=step.target,
                modifier=step.modifier,
                objects=current_objects,
                grid=current_grid,
            )
            # Re-detect objects for next step
            current_objects = self._detector.detect(current_grid)

        return current_grid

    def _store_skill_sequence(
        self,
        task_sig: torch.Tensor,
        sequence: SequenceTransformResult,
    ) -> None:
        """Store multi-step skill in neural memory."""
        label = "_".join(
            f"{s.action}_{s.target}_{s.modifier}"
            for s in sequence.steps
        )

        # Store in cache (for later retrieval if seen again)
        # Note: Current NeuralMemory is designed for single transforms
        # TODO: Enhance NeuralMemory to handle sequences

        # For now: Store in simple dict
        if not hasattr(self, '_skill_sequence_cache'):
            self._skill_sequence_cache = {}

        self._skill_sequence_cache[label] = sequence
```

### Phase 4: Testing (200 lines)

```python
# File: tests/arc/test_multi_step_resonator.py (NEW FILE)

import pytest
import torch
from hologram.arc.transform_resonator import (
    TransformationResonator,
    SequenceTransformResult
)
from hologram.arc.solver import HolographicARCSolver, create_simple_task
from hologram.core.vector_space import VectorSpace
from hologram.core.codebook import Codebook
from hologram.arc.encoder import ObjectEncoder
from hologram.core.fractal import FractalSpace


class TestMultiStepResonator:
    """Test multi-step sequence factorization."""

    @pytest.fixture
    def resonator(self):
        """Create test resonator."""
        space = VectorSpace(dimensions=10000)
        fractal = FractalSpace(dimensions=10000)
        codebook = Codebook(space)
        encoder = ObjectEncoder(fractal, codebook)
        return TransformationResonator(encoder, codebook)

    def test_single_step_unchanged(self, resonator):
        """Verify single-step behavior still works."""
        # Create simple single-step observation
        space = resonator._codebook._space

        # Encode: tile action
        action_vec = resonator._action_vectors["tile"]
        target_vec = resonator._target_vectors["all_objects"]
        modifier_vec = resonator._modifier_vectors["by_pattern"]

        observation = torch.mean(torch.stack([
            action_vec, target_vec, modifier_vec
        ]), dim=0)

        # Single-step resonate
        result = resonator.resonate(observation)

        assert result.action == "tile"
        assert result.target == "all_objects"
        assert result.modifier == "by_pattern"
        assert result.converged

    def test_two_step_sequence(self, resonator):
        """Test factorization of two-step sequence."""
        space = resonator._codebook._space

        # Create two steps
        step1_action = resonator._action_vectors["tile"]
        step1_target = resonator._target_vectors["all_objects"]
        step1_modifier = resonator._modifier_vectors["by_pattern"]

        step2_action = resonator._action_vectors["recolor"]
        step2_target = resonator._target_vectors["red"]
        step2_modifier = resonator._modifier_vectors["to_blue"]

        # Position-encode each step
        shift_stride = resonator._position_encoding_stride
        step1_encoded = torch.mean(torch.stack([step1_action, step1_target, step1_modifier]), dim=0)
        step1_positioned = Operations.permute(step1_encoded, shifts=shift_stride)

        step2_encoded = torch.mean(torch.stack([step2_action, step2_target, step2_modifier]), dim=0)
        step2_positioned = Operations.permute(step2_encoded, shifts=2*shift_stride)

        # Bundle to create sequence observation
        sequence_obs = Operations.bundle(step1_positioned, step2_positioned)

        # Extract sequence
        result = resonator.resonate_sequence(sequence_obs)

        assert result.num_steps >= 1, f"Expected at least 1 step, got {result.num_steps}"
        assert result.converged or result.num_steps < resonator._max_sequence_steps

    def test_sequence_with_noise(self, resonator):
        """Test sequence extraction with noise."""
        space = resonator._codebook._space

        # Create clean two-step sequence
        step1 = torch.randn(10000)
        step2 = torch.randn(10000)

        # Position-encode
        step1_pos = Operations.permute(step1, shifts=100)
        step2_pos = Operations.permute(step2, shifts=200)

        sequence = Operations.bundle(step1_pos, step2_pos)

        # Add noise
        noise = torch.randn(10000) * 0.1
        noisy_sequence = sequence + noise

        # Should still extract something
        result = resonator.resonate_sequence(noisy_sequence)
        assert isinstance(result, SequenceTransformResult)

    def test_sequence_result_properties(self, resonator):
        """Test SequenceTransformResult has expected properties."""
        # Create minimal sequence
        space = resonator._codebook._space
        obs = torch.randn(10000)

        result = resonator.resonate_sequence(obs)

        assert hasattr(result, 'steps')
        assert hasattr(result, 'sequence')
        assert hasattr(result, 'num_steps')
        assert hasattr(result, 'total_quality')
        assert hasattr(result, 'step_qualities')
        assert len(result.steps) == len(result.step_qualities)
        assert len(result.sequence) == len(result.steps)

class TestMultiStepSolver:
    """Test multi-step solving on complete solver."""

    def test_solver_tiling_task(self):
        """Test solver on simple 2-step tiling + recolor task."""
        # Create task: 3×3 pattern → tile into 3×3 grid
        task = create_simple_task(
            train_inputs=[
                [[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]
            ],
            train_outputs=[
                [[1, 2, 3, 1, 2, 3, 1, 2, 3],
                 [4, 5, 6, 4, 5, 6, 4, 5, 6],
                 [7, 8, 9, 7, 8, 9, 7, 8, 9],
                 [1, 2, 3, 1, 2, 3, 1, 2, 3],
                 [4, 5, 6, 4, 5, 6, 4, 5, 6],
                 [7, 8, 9, 7, 8, 9, 7, 8, 9],
                 [1, 2, 3, 1, 2, 3, 1, 2, 3],
                 [4, 5, 6, 4, 5, 6, 4, 5, 6],
                 [7, 8, 9, 7, 8, 9, 7, 8, 9]]
            ],
            test_input=[
                [1, 0, 2],
                [3, 0, 4],
                [5, 0, 6]
            ]
        )

        solver = HolographicARCSolver(dimensions=10000)
        result = solver.solve(task)

        # Should at least attempt to solve
        # (May not be perfectly correct, but should recognize multi-step)
        assert result is not None
```

---

## Integration Checklist

### Before Implementation
- [ ] Review existing TransformationResonator thoroughly
- [ ] Understand position encoding via permute()
- [ ] Design test cases for 1-step, 2-step, 3-step sequences
- [ ] Plan performance profiling strategy

### Implementation
- [ ] Create SequenceTransformResult dataclass
- [ ] Add position encoding utilities to Operations
- [ ] Implement deflation algorithm
- [ ] Extend TransformationResonator with resonate_sequence()
- [ ] Modify solver.solve() to try multi-step first
- [ ] Create MultiStepExecutor pattern

### Testing
- [ ] Unit tests for deflation algorithm
- [ ] Integration tests on synthetic 2-step, 3-step tasks
- [ ] Regression tests on single-step tasks
- [ ] ARC-AGI-2 benchmark tests (measure improvement)

### Documentation
- [ ] Update docstrings in transform_resonator.py
- [ ] Add algorithm explanation comments
- [ ] Document position encoding strategy
- [ ] Create usage examples

---

## Performance Considerations

### Computational Cost

**Single-step**: ~O(A_max × T_max × M_max × iterations) = O(11 × 9 × 24 × 50) ≈ 120K operations

**K-step sequence**: ~O(K × Single-step) = O(5 × 120K) = ~600K operations per task

**Estimate**: +5-10x latency for multi-step solving (but only when needed)

### Optimization Opportunities

1. **Early termination**: Stop if quality drops below threshold
2. **Residue caching**: Don't recompute on failed steps
3. **Parallel step extraction**: Extract all K steps simultaneously (optional)
4. **Pruning**: Skip unlikely vocabulary items based on prior probabilities

---

## Fallback & Robustness

### Graceful Degradation

```python
# If multi-step fails → automatically try single-step
if sequence_result.num_steps == 0:
    # Try single-step (existing code)
    single_result = self._resonator.resonate(observation_bundle)
    return apply_single_step(single_result)
```

### Failure Modes

| Scenario | Symptom | Recovery |
|----------|---------|----------|
| K_detected > K_actual | Extra steps fail quality check | Algorithm stops automatically |
| Residue noise too high | Can't extract valid step | Early termination |
| Position encoding interference | Steps confuse each other | Increase SHIFT_STRIDE |
| Single-step observation | Works with multi-step too | Single step returned (K=1) |

---

## Success Metrics

After implementation:

1. **Correctness**: 80%+ accuracy on synthetic 2-step tasks
2. **Performance**: +25% improvement on ARC-AGI-2 training set
3. **No regression**: Single-step tasks still work
4. **Robustness**: Graceful handling of 1-step, 2-step, 3+ step tasks
5. **Code quality**: >80% test coverage

---

## Next Steps

1. **Review & Validate**: Share this design with stakeholders
2. **Implement Phase 1-2**: SequenceTransformResult + deflation algorithm
3. **Validate Theoretically**: Test on synthetic sequences before ARC
4. **Integrate with Solver**: Modify solve() to use multi-step
5. **Benchmark**: Measure accuracy improvement on ARC-AGI-2

---

**End of Implementation Guide**
