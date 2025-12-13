# Implementation Code Patterns Guide

**Purpose**: Concrete code examples for implementing validated enhancements
**Status**: Ready for development team
**Date**: 2025-12-13

---

## Part A: MultiStepResonator - Core Implementation

### A1. New Dataclass: SequenceTransformResult

**File**: `/src/hologram/arc/transform_resonator.py` (ADD AT TOP)

```python
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class SequenceTransformResult:
    """
    Result of multi-step sequence factorization.

    Represents a sequence of K transformations extracted via deflation,
    with quality metrics for each step.
    """
    steps: List[TransformResult]
    sequence: List[Tuple[str, str, str]]  # [(action, target, modifier), ...]

    # Sequence-level metrics
    num_steps: int
    total_quality: float  # Minimum quality across all steps
    converged: bool
    final_residue_norm: float

    # Per-step details
    step_qualities: List[float]
    step_confidences: List[Dict[str, float]]

    def __str__(self) -> str:
        result = f"SequenceTransform(n_steps={self.num_steps}, "
        result += f"converged={self.converged}, "
        result += f"quality={self.total_quality:.3f})\n"

        for i, (step, quality) in enumerate(zip(self.steps, self.step_qualities)):
            result += (f"  Step {i+1}: {step.action}({step.target}, "
                      f"{step.modifier}) [quality={quality:.3f}]\n")

        return result

    def as_execution_plan(self) -> List[Tuple[str, str, str]]:
        """Return sequence as (action, target, modifier) tuples for executor."""
        return [(s.action, s.target, s.modifier) for s in self.steps]
```

### A2. Extend TransformationResonator with resonate_sequence()

**File**: `/src/hologram/arc/transform_resonator.py` (ADD TO CLASS)

```python
class TransformationResonator:
    """
    Extended resonator with multi-step sequence factorization.
    """

    def __init__(
        self,
        encoder: ObjectEncoder,
        codebook: Codebook,
        max_iterations: int = MAX_RESONATOR_ITERATIONS,
        convergence_threshold: float = CONVERGENCE_THRESHOLD,
        # NEW PARAMETERS FOR SEQUENCE FACTORIZATION
        position_encoding_stride: int = 100,
        max_sequence_steps: int = 5,
        sequence_quality_threshold: float = 0.6,
        residue_threshold: float = 0.1,
    ):
        """Initialize with sequence parameters."""
        # ... existing init code ...

        # New: Position encoding configuration
        self._position_encoding_stride = position_encoding_stride
        self._max_sequence_steps = max_sequence_steps
        self._sequence_quality_threshold = sequence_quality_threshold
        self._residue_threshold = residue_threshold

        # Validate position encoding stride
        self._validate_position_stride()

    def _validate_position_stride(self) -> None:
        """
        Ensure position encoding stride provides sufficient orthogonality.

        Checks that permuted versions of the same vector remain separable
        when shifted by the stride.
        """
        # Use role_action as reference vector
        v1 = Operations.permute(self._role_action, shifts=self._position_encoding_stride)
        v2 = Operations.permute(self._role_action, shifts=2 * self._position_encoding_stride)

        orthogonality = Similarity.cosine(v1, v2)

        if orthogonality > 0.2:
            raise ValueError(
                f"Position stride {self._position_encoding_stride} insufficient "
                f"(orthogonality={orthogonality:.3f}, need < 0.1). "
                f"Increase stride to improve separation."
            )

    def resonate_sequence(
        self,
        observation: torch.Tensor,
        max_steps: int = None,
    ) -> SequenceTransformResult:
        """
        Factorize observation into sequence of (A,T,M) transformations.

        Uses deflation algorithm: iteratively extract and subtract steps.
        Each step must satisfy quality threshold to be accepted.

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
            step_energy = torch.norm(proposed_step)
            if step_energy < self._residue_threshold:
                # No signal at this position - done
                break

            # 4. Factorize proposed step into (A, T, M) via ALS
            step_result = self.resonate(proposed_step)

            # 5. Verify quality of factorization
            quality = self.verify_factorization(proposed_step, step_result)
            step_qualities.append(quality)
            step_confidences.append(step_result.confidence)

            # 6. Check if quality meets threshold
            if quality < self._sequence_quality_threshold:
                # Quality too low - don't accept this step, but continue checking
                # (it might be noise, and next step might be valid)
                continue

            # 7. Step is accepted
            steps.append(step_result)

            # 8. Reconstruct this step's contribution
            reconstructed = self._ops.bundle(
                self._ops.bind(step_result.action_vec, self._role_action),
                self._ops.bind(step_result.target_vec, self._role_target),
                self._ops.bind(step_result.modifier_vec, self._role_modifier),
            )

            # 9. Subtract positioned reconstruction from residue
            # IMPORTANT: Use inverse bundling to maintain HDC properties
            positioned_recon = self._apply_position_encoding(reconstructed, shift)

            # Instead of: residue = residue - positioned_recon
            # Use inverse bundling (safer in HDC space)
            inverse_recon = positioned_recon * (-1.0)  # Negate all elements
            residue = self._ops.bundle(residue, inverse_recon)
            # Residue is now (approximately) (step2 + step3 + ... + noise)

        # Build and return result
        return SequenceTransformResult(
            steps=steps,
            sequence=[(s.action, s.target, s.modifier) for s in steps],
            num_steps=len(steps),
            total_quality=min(step_qualities) if step_qualities else 0.0,
            converged=len(steps) < max_steps and len(steps) > 0,
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
        position_key = Operations.permute(self._role_action, shifts=shift)

        # Unbind: bind residue with inverse position key
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
```

### A3. Integrate with Solver

**File**: `/src/hologram/arc/solver.py` (MODIFY solve() METHOD)

```python
def solve(self, task: ARCTask, strategy: str = "hybrid") -> SolverResult:
    """
    Attempt to solve an ARC task using adaptive strategy.

    Args:
        task: ARC task with training pairs and test input
        strategy: "single_step" (fast), "iterative" (state traversal),
                 or "hybrid" (try single_step first, fallback to iterative)

    Returns:
        SolverResult with output grid (or None if refused)
    """
    # 1. Compute task signature for neural memory lookup
    task_sig_vec = self._compute_task_signature_vector(task)

    # 2. Check skill memory (O(1) neural lookup)
    cached_label, cache_confidence = self._skill_memory.query(task_sig_vec)
    if cached_label is not None and cache_confidence >= self.SKILL_CONFIDENCE_THRESHOLD:
        cached_transform = self._transform_cache.get(cached_label)
        if cached_transform is not None:
            output = self._apply_transformation(cached_transform, task.test_input)
            return SolverResult(
                output=output,
                transformation=cached_transform,
                confidence=cache_confidence,
                from_cache=True,
                message=f"Retrieved from skill memory (conf={cache_confidence:.2f})",
            )

    # 3. Process training pairs to create observations
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
            message="No valid observations from training pairs",
        )

    # 4. Bundle observations
    observation_bundle = self._ops.bundle(*observations)

    # 5. Determine solving strategy
    if strategy == "hybrid":
        # Try single-step first (fast path)
        result = self._solve_single_step(observation_bundle, task)
        if result is not None:
            return result

        # Fallback to multi-step
        result = self._solve_multi_step(observation_bundle, task)
        if result is not None:
            return result

        # Fallback to iterative
        return self._solve_iterative(task)

    elif strategy == "single_step":
        result = self._solve_single_step(observation_bundle, task)
        return result or SolverResult(
            output=None, transformation=None, confidence=0.0,
            from_cache=False, message="Single-step solving failed"
        )

    elif strategy == "iterative":
        return self._solve_iterative(task)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


    # Helper methods

    def _solve_single_step(
        self,
        observation_bundle: torch.Tensor,
        task: ARCTask,
    ) -> Optional[SolverResult]:
        """Try single-step solving."""
        result = self._resonator.resonate(observation_bundle)

        verification_score = self._resonator.verify_factorization(
            observation_bundle, result
        )
        combined_confidence = min(result.min_confidence, verification_score)

        if combined_confidence < self._confidence_threshold:
            return None

        output = self._apply_transformation(result, task.test_input)

        # Store successful skill
        task_sig_vec = self._compute_task_signature_vector(task)
        self._store_skill(task_sig_vec, result)

        return SolverResult(
            output=output,
            transformation=result,
            confidence=combined_confidence,
            from_cache=False,
            message=f"Single-step: {result.action}({result.target}, {result.modifier})",
        )

    def _solve_multi_step(
        self,
        observation_bundle: torch.Tensor,
        task: ARCTask,
    ) -> Optional[SolverResult]:
        """Try multi-step sequence solving."""
        sequence_result = self._resonator.resonate_sequence(observation_bundle)

        if sequence_result.num_steps == 0:
            return None  # No valid sequence detected

        combined_confidence = sequence_result.total_quality

        if combined_confidence < self._confidence_threshold:
            return None

        output = self._apply_sequence(sequence_result, task.test_input)

        # Store successful skill
        task_sig_vec = self._compute_task_signature_vector(task)
        self._store_skill_sequence(task_sig_vec, sequence_result)

        return SolverResult(
            output=output,
            transformation=sequence_result.steps[0],  # First step for logging
            confidence=combined_confidence,
            from_cache=False,
            message=f"Multi-step ({sequence_result.num_steps}): {sequence_result}",
        )

    def _solve_iterative(self, task: ARCTask) -> SolverResult:
        """Try iterative state-traversal solving."""
        if self._iterative_solver is None:
            return SolverResult(
                output=None, transformation=None, confidence=0.0,
                from_cache=False, message="Iterative solver not available"
            )

        result = self._iterative_solver.solve(task)

        if result.solved:
            return SolverResult(
                output=result.output,
                transformation=result.transform_chain[0] if result.transform_chain else None,
                confidence=result.confidence,
                from_cache=False,
                message=f"Iterative ({result.steps_taken} steps): Solved",
            )
        else:
            return SolverResult(
                output=result.output,
                transformation=result.transform_chain[0] if result.transform_chain else None,
                confidence=result.confidence,
                from_cache=False,
                message=f"Iterative ({result.steps_taken} steps): Partial",
            )

    def _apply_sequence(
        self,
        sequence: SequenceTransformResult,
        test_input: Grid,
    ) -> Grid:
        """Apply sequence of transformations in order."""
        current_grid = test_input

        for step in sequence.steps:
            # Re-detect objects after each transformation
            objects = self._detector.detect(current_grid)

            # Apply this transformation
            current_grid = self._executor.execute(
                action=step.action,
                target=step.target,
                modifier=step.modifier,
                objects=objects,
                grid=current_grid,
            )

        return current_grid

    def _store_skill_sequence(
        self,
        task_sig: torch.Tensor,
        sequence: SequenceTransformResult,
    ) -> None:
        """Store multi-step skill in neural memory."""
        # Create unique label for this sequence
        label = "_".join(
            f"{s.action}_{s.target}_{s.modifier}"
            for s in sequence.steps
        )

        # For now: Store in simple dict (NeuralMemory designed for single steps)
        if not hasattr(self, '_skill_sequence_cache'):
            self._skill_sequence_cache = {}

        self._skill_sequence_cache[label] = sequence

        # TODO: Enhance NeuralMemory to handle sequences
```

---

## Part B: RelationalEncoder Implementation

### B1. New Class: RelationalEncoder

**File**: `/src/hologram/arc/relational_encoder.py` (NEW FILE)

```python
"""
RelationalEncoder: Encode salient spatial relationships between objects.

Detects and encodes relationships like adjacency, same color, same shape,
and containment. Hard cap of 30 relations per object to avoid bundling saturation.
"""

from typing import List, Tuple, Set, Dict
import torch

from hologram.arc.types import Object, Grid, Color
from hologram.arc.detector import ObjectDetector
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity


class RelationalEncoder:
    """
    Encode spatial relationships between objects.

    Supported relationships:
    - adjacency: Objects touching (4-connected)
    - same_color: Same primary color
    - same_shape: Identical shape/mask
    - containment: One object inside another
    """

    RELATION_VOCABULARY = [
        "adjacency",
        "same_color",
        "same_shape",
        "containment",
    ]

    def __init__(self, codebook: Codebook, detector: ObjectDetector):
        """Initialize with codebook and detector."""
        self._codebook = codebook
        self._detector = detector
        self._ops = Operations

        # Pre-encode relation vectors
        self._relation_vectors = {
            rel: codebook.encode(f"relation_{rel}")
            for rel in self.RELATION_VOCABULARY
        }

        # Pre-encode role vector for relations
        self._role_relation = codebook.encode("__ROLE_RELATION__")

    def encode_object_relations(
        self,
        obj: Object,
        context: Grid,
        max_relations: int = 30,
    ) -> torch.Tensor:
        """
        Encode all salient relations for an object.

        Args:
            obj: Target object
            context: Full grid for finding related objects
            max_relations: Maximum relations to encode (default 30)

        Returns:
            Bundled relation vector
        """
        # Detect all objects in context
        all_objects = self._detector.detect(context)
        other_objects = [o for o in all_objects if id(o) != id(obj)]

        # Compute relation scores
        relations_with_scores = []

        for other_obj in other_objects:
            # Check each relation type
            if self._is_adjacent(obj, other_obj):
                relations_with_scores.append(("adjacency", 1.0))

            if self._same_color(obj, other_obj):
                relations_with_scores.append(("same_color", 1.0))

            if self._same_shape(obj, other_obj):
                relations_with_scores.append(("same_shape", 1.0))

            if self._is_contained(obj, other_obj):
                relations_with_scores.append(("containment", 1.0))

        # Sort by confidence and take top N
        relations_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_relations = relations_with_scores[:max_relations]

        if not top_relations:
            # No relations detected
            return self._codebook.encode("no_relations")

        # Encode selected relations
        relation_vecs = []
        for rel_name, rel_score in top_relations:
            rel_vec = self._relation_vectors[rel_name]
            # Weight by confidence
            rel_vec = rel_vec * rel_score
            relation_vecs.append(rel_vec)

        # Bundle all relations
        return self._ops.bundle(*relation_vecs)

    def encode_all_relations(
        self,
        objects: List[Object],
        context: Grid,
    ) -> torch.Tensor:
        """
        Encode relations for all objects.

        Returns:
            Bundled vector of all object relations
        """
        all_relation_vecs = []

        for obj in objects:
            obj_rel = self.encode_object_relations(obj, context)
            all_relation_vecs.append(obj_rel)

        if not all_relation_vecs:
            return self._codebook.encode("no_objects")

        return self._ops.bundle(*all_relation_vecs)

    # Relation detection methods

    @staticmethod
    def _is_adjacent(obj1: Object, obj2: Object, connectivity: int = 4) -> bool:
        """
        Check if two objects are adjacent (touching).

        Args:
            connectivity: 4 (cardinal) or 8 (including diagonals)

        Returns:
            True if objects touch
        """
        for (r1, c1) in obj1.pixels:
            for (r2, c2) in obj2.pixels:
                dr = abs(r1 - r2)
                dc = abs(c1 - c2)

                if connectivity == 4:
                    if (dr == 1 and dc == 0) or (dr == 0 and dc == 1):
                        return True
                elif connectivity == 8:
                    if max(dr, dc) == 1:
                        return True

        return False

    @staticmethod
    def _same_color(obj1: Object, obj2: Object) -> bool:
        """Check if objects have same primary color."""
        return obj1.color == obj2.color

    @staticmethod
    def _same_shape(obj1: Object, obj2: Object) -> bool:
        """Check if objects have same shape (mask)."""
        return obj1.mask == obj2.mask

    @staticmethod
    def _is_contained(obj1: Object, obj2: Object) -> bool:
        """Check if obj1 is contained within obj2."""
        # Check if all pixels of obj1 are within bbox of obj2
        for (r, c) in obj1.pixels:
            if not (obj2.bbox.min_row <= r <= obj2.bbox.max_row and
                    obj2.bbox.min_col <= c <= obj2.bbox.max_col):
                return False

        return len(obj1.pixels) > 0
```

### B2. Integrate with ObjectEncoder

**File**: `/src/hologram/arc/encoder.py` (ADD TO CLASS)

```python
def __init__(self, fractal_space, codebook, detector=None):
    """Initialize with optional detector for relation encoding."""
    # ... existing init code ...

    # New: Relational encoder
    self._detector = detector
    if detector:
        from hologram.arc.relational_encoder import RelationalEncoder
        self._relational_encoder = RelationalEncoder(codebook, detector)
    else:
        self._relational_encoder = None


def encode_observation_with_relations(
    self,
    input_objs: List[Object],
    output_objs: List[Object],
    input_grid: Grid,
    output_grid: Grid,
) -> torch.Tensor:
    """
    Encode transformation observation including spatial relations.

    Args:
        input_objs: Objects in input grid
        output_objs: Objects in output grid
        input_grid: Input grid
        output_grid: Output grid

    Returns:
        Observation vector with relation information
    """
    # Get base observation (object-level)
    obs_vectors = []

    for in_obj, out_obj in zip(input_objs, output_objs):
        obs = self.encode_transformation_observation(in_obj, out_obj)
        obs_vectors.append(obs)

    # Add relational information
    if self._relational_encoder is not None:
        input_relations = self._relational_encoder.encode_all_relations(
            input_objs, input_grid
        )
        output_relations = self._relational_encoder.encode_all_relations(
            output_objs, output_grid
        )

        # Bind relations to role
        role_relations = self._codebook.encode("__ROLE_RELATIONS__")
        relation_obs = self._ops.bundle(
            self._ops.bind(input_relations, role_relations),
            self._ops.bind(output_relations, role_relations)
        )
        obs_vectors.append(relation_obs)

    # Bundle all observations
    return self._ops.bundle(*obs_vectors)
```

---

## Part C: Vocabulary Expansion Integration

### C1. Extend types.py

**File**: `/src/hologram/arc/types.py` (MODIFY)

```python
# Existing code...

ACTIONS = [
    "identity",     # No change
    "rotate",       # Rotate object
    "translate",    # Move object
    "recolor",      # Change color
    "flip",         # Mirror object
    "scale",        # Resize object
    "delete",       # Remove object
    "copy",         # Duplicate object
    "tile",         # Tile input pattern
    "expand",       # Expand grid
    "fill",         # Fill region
]

TARGETS = [
    "all_objects",  # Apply to all
    "largest",      # Largest by pixel count
    "smallest",     # Smallest by pixel count
    "red",          # Objects of color red
    "blue",         # Objects of color blue
    "green",        # Objects of color green
    "yellow",       # Objects of color yellow
    "by_position",  # Select by position
    "by_color",     # Select by color match
    # NEW TARGETS
    "background",   # Background/empty space
    "bounding_box", # Bounding box of object
    "between_objects", # Space between objects
    "by_adjacency",    # Objects adjacent to others
]

MODIFIERS = [
    "none",         # No modifier
    "90_degrees",   # Rotate 90 CW
    "180_degrees",  # Rotate 180
    "270_degrees",  # Rotate 270 CW
    "up",           # Translate up
    "down",         # Translate down
    "left",         # Translate left
    "right",        # Translate right
    "to_red",       # Recolor to red
    "to_blue",      # Recolor to blue
    "to_green",     # Recolor to green
    "horizontal",   # Flip horizontally
    "vertical",     # Flip vertically
    "by_2x",        # Scale 2x
    "by_half",      # Scale 0.5x
    "tile_2x2",     # Tile into 2x2 grid
    "tile_3x3",     # Tile into 3x3 grid
    "tile_4x4",     # Tile into 4x4 grid
    "by_pattern",   # Tile based on pattern
    "to_yellow",    # Recolor to yellow
    "to_grey",      # Recolor to grey
    "to_magenta",   # Recolor to magenta
    "to_orange",    # Recolor to orange
    "to_cyan",      # Recolor to cyan
    # NEW COUNTING PRIMITIVES
    "count_1",      # Count = 1
    "count_2",      # Count = 2
    "count_3",      # Count = 3
    "count_4",      # Count = 4
    "count_5",      # Count = 5
    "count_n",      # Count = N (general)
    # NEW SYMMETRY PRIMITIVES
    "diagonal_main",   # Main diagonal symmetry
    "diagonal_anti",   # Anti-diagonal symmetry
    "point_center",    # Point/rotational symmetry
    # NEW FILL PATTERNS
    "fill_solid",          # Solid fill
    "fill_checkerboard",   # Checkerboard pattern
    "fill_border",         # Border fill
    "fill_interior",       # Interior fill
]
```

### C2. Extend ObjectEncoder Vocabularies

**File**: `/src/hologram/arc/encoder.py` (MODIFY __init__)

```python
def __init__(self, fractal_space, codebook):
    """Initialize encoder with extended vocabularies."""
    # ... existing code ...

    # Pre-encode transformation vocabulary (EXTENDED)
    self._action_vectors = {
        action: self._codebook.encode(f"action_{action}")
        for action in ACTIONS
    }
    self._target_vectors = {
        target: self._codebook.encode(f"target_{target}")
        for target in TARGETS  # Now includes new targets
    }
    self._modifier_vectors = {
        modifier: self._codebook.encode(f"modifier_{modifier}")
        for modifier in MODIFIERS  # Now includes new modifiers
    }
```

---

## Part D: Testing Framework

### D1. Test Suite Template

**File**: `/tests/arc/test_multi_step_resonator.py` (NEW)

```python
"""
Test suite for multi-step sequence factorization.
"""

import pytest
import torch
from hologram.arc.transform_resonator import (
    TransformationResonator,
    TransformResult,
    SequenceTransformResult,
)
from hologram.arc.solver import HolographicARCSolver, create_simple_task
from hologram.arc.encoder import ObjectEncoder
from hologram.core.vector_space import VectorSpace
from hologram.core.codebook import Codebook
from hologram.core.fractal import FractalSpace
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity


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

    def test_single_step_backward_compatible(self, resonator):
        """
        Verify single-step behavior unchanged.

        Regression test: existing single-step tasks should still work.
        """
        # Create simple single-step observation
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

    def test_sequence_result_dataclass(self, resonator):
        """Verify SequenceTransformResult has all expected attributes."""
        obs = torch.randn(10000)
        result = resonator.resonate_sequence(obs)

        assert hasattr(result, 'steps')
        assert hasattr(result, 'sequence')
        assert hasattr(result, 'num_steps')
        assert hasattr(result, 'total_quality')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'final_residue_norm')
        assert hasattr(result, 'step_qualities')
        assert hasattr(result, 'step_confidences')

        assert len(result.steps) == len(result.step_qualities)
        assert len(result.sequence) == len(result.steps)

    def test_position_stride_validation(self, resonator):
        """
        Verify position encoding stride provides sufficient orthogonality.
        """
        stride = resonator._position_encoding_stride

        v1 = Operations.permute(resonator._role_action, shifts=stride)
        v2 = Operations.permute(resonator._role_action, shifts=2*stride)

        orthogonality = Similarity.cosine(v1, v2)

        # Should be nearly orthogonal
        assert orthogonality < 0.2, \
            f"Position stride {stride} insufficient (orthogonality={orthogonality:.3f})"

    def test_residue_subtraction_via_inverse_bundling(self):
        """
        Verify residue subtraction preserves HDC properties.

        Tests that inverse bundling is safer than direct subtraction.
        """
        # Create two positioned step signals
        v1 = torch.randn(10000)
        v1 = v1 / torch.norm(v1)  # Normalize

        v2 = torch.randn(10000)
        v2 = v2 / torch.norm(v2)

        # Bundle them
        bundled = Operations.bundle(v1, v2)

        # Method 1: Direct subtraction (PROBLEMATIC)
        residue_1 = bundled - v1

        # Method 2: Inverse bundling (RECOMMENDED)
        inv_v1 = -v1
        residue_2 = Operations.bundle(bundled, inv_v1)

        # Verify Method 2 preserves better orthogonality
        unbind_v2_from_1 = Operations.unbind(residue_1, v2)
        unbind_v2_from_2 = Operations.unbind(residue_2, v2)

        sim_1 = Similarity.cosine(unbind_v2_from_1, v2)
        sim_2 = Similarity.cosine(unbind_v2_from_2, v2)

        # Method 2 should give better recovery
        assert sim_2 > sim_1, \
            f"Inverse bundling should preserve signal better "
            f"(Method 2: {sim_2:.3f} > Method 1: {sim_1:.3f})"


class TestMultiStepSolver:
    """Test multi-step solving on complete solver."""

    def test_solver_single_step_still_works(self):
        """
        Verify solver still works on single-step tasks.

        Regression test: should not break existing functionality.
        """
        task = create_simple_task(
            train_inputs=[[[1, 2], [3, 4]]],
            train_outputs=[[[2, 1], [4, 3]]],  # Transpose-like
            test_input=[[5, 6], [7, 8]],
            test_output=[[6, 5], [8, 7]],
        )

        solver = HolographicARCSolver(dimensions=10000)
        result = solver.solve(task)

        # Should at least attempt to solve
        assert result is not None

    def test_no_hallucination_guarantee(self):
        """
        Critical test: Verify all outputs are vocabulary items.

        The no-hallucination guarantee must be preserved.
        """
        from hologram.arc.types import ACTIONS, TARGETS, MODIFIERS

        task = create_simple_task(
            train_inputs=[[[1, 0], [0, 2]]],
            train_outputs=[[[1, 1], [2, 2]]],
            test_input=[[3, 0], [0, 4]],
        )

        solver = HolographicARCSolver(dimensions=10000)
        result = solver.solve(task)

        if result.transformation is not None:
            # CRITICAL: Each component must be in vocabulary
            assert result.transformation.action in ACTIONS, \
                f"Action {result.transformation.action} not in vocabulary"
            assert result.transformation.target in TARGETS, \
                f"Target {result.transformation.target} not in vocabulary"
            assert result.transformation.modifier in MODIFIERS, \
                f"Modifier {result.transformation.modifier} not in vocabulary"
```

---

## Summary Table

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| SequenceTransformResult | transform_resonator.py | ~50 | NEW |
| resonate_sequence() | transform_resonator.py | ~200 | NEW |
| Position utilities | transform_resonator.py | ~50 | NEW |
| Solver.solve() refactor | solver.py | ~150 | MODIFY |
| RelationalEncoder | relational_encoder.py | ~150 | NEW |
| ObjectEncoder.encode_relations() | encoder.py | ~40 | ADD |
| types.py vocabulary | types.py | ~30 | EXTEND |
| Test suite | test_multi_step_resonator.py | ~200 | NEW |

**Total New Code**: ~870 LOC
**Modification**: ~190 LOC
**Test Code**: ~400 LOC

---

**Ready for implementation team to begin coding Phase 1**
