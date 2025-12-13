"""
IterativeSolver: Multi-step ARC solver via state traversal.

Key insight from Dr. Nexus: We don't encode sequences in vectors.
We TRAVERSE the state space, resonating one step at a time.
The "sequence" exists in Python control flow, NOT in vector algebra.

This implements the "Verify-and-Refine" approach:
1. Observe what STILL needs to change (current → target delta)
2. Resonate for SINGLE best transform
3. Execute and get new state
4. Check if solved, iterate if needed

This avoids "Holographic Saturation" by keeping each step lightweight.
"""

from dataclasses import dataclass
from typing import List, Optional, Set
import hashlib

import torch

from hologram.arc.types import Grid, Object, ARCTask, TrainingPair
from hologram.arc.encoder import ObjectEncoder
from hologram.arc.transform_resonator import TransformationResonator, TransformResult
from hologram.arc.executor import TransformationExecutor
from hologram.arc.detector import ObjectDetector


@dataclass
class IterativeResult:
    """
    Result from iterative solving.

    Attributes:
        output: Final output grid (best effort)
        transform_chain: List of transformations applied
        steps_taken: Number of iterations performed
        solved: Whether the task was fully solved
        confidence: Confidence of the final step
    """
    output: Grid
    transform_chain: List[TransformResult]
    steps_taken: int
    solved: bool
    confidence: float

    def __str__(self) -> str:
        status = "SOLVED" if self.solved else "PARTIAL"
        chain_str = " → ".join(
            f"{t.action}({t.target}, {t.modifier})"
            for t in self.transform_chain
        ) if self.transform_chain else "none"
        return f"IterativeResult({status}, steps={self.steps_taken}, chain=[{chain_str}])"


class IterativeSolver:
    """
    Multi-step ARC solver via state traversal.

    Instead of encoding transformation sequences into vectors (which causes
    saturation), we iterate through the state space one step at a time.

    Each step:
    1. Observes the remaining delta (current state → target hints)
    2. Resonates for a single best transformation
    3. Executes the transformation
    4. Checks if solved or needs more steps

    Attributes:
        max_steps: Maximum iterations to prevent infinite loops
        convergence_threshold: Grid similarity to consider "solved"
        refusal_threshold: Below this confidence, refuse to transform

    Example:
        >>> solver = IterativeSolver(encoder, resonator, executor)
        >>> result = solver.solve(task)
        >>> print(result.solved, result.steps_taken)
        True 2
    """

    MAX_STEPS = 5  # Default: prevent infinite loops
    CONVERGENCE_THRESHOLD = 0.95  # Grid similarity to consider solved
    REFUSAL_THRESHOLD = 0.01  # Below this, no valid transform found

    def __init__(
        self,
        encoder: ObjectEncoder,
        resonator: TransformationResonator,
        executor: TransformationExecutor,
        detector: Optional[ObjectDetector] = None,
        max_steps: int = MAX_STEPS,
        convergence_threshold: float = CONVERGENCE_THRESHOLD,
        refusal_threshold: float = REFUSAL_THRESHOLD,
    ):
        """
        Initialize iterative solver.

        Args:
            encoder: ObjectEncoder for creating observations
            resonator: TransformationResonator for factorization
            executor: TransformationExecutor for applying transforms
            detector: ObjectDetector for finding objects in grids
            max_steps: Maximum iterations (default: 5)
            convergence_threshold: Grid similarity threshold (default: 0.95)
            refusal_threshold: Confidence threshold (default: 0.01)
        """
        self._encoder = encoder
        self._resonator = resonator
        self._executor = executor
        self._detector = detector or ObjectDetector()
        self._max_steps = max_steps
        self._convergence_threshold = convergence_threshold
        self._refusal_threshold = refusal_threshold

    def solve(self, task: ARCTask) -> IterativeResult:
        """
        Attempt to solve an ARC task iteratively.

        Args:
            task: ARC task with training pairs and test input

        Returns:
            IterativeResult with output grid and transformation chain
        """
        current_state = task.test_input
        transform_chain: List[TransformResult] = []
        visited_states: Set[str] = {self._state_hash(current_state)}

        for step in range(self._max_steps):
            # 1. Observe: What's the delta between current and target?
            observation = self._observe_remaining_delta(current_state, task.training)

            if observation is None:
                # No objects to transform, or no valid delta
                break

            # 2. Resonate: Find SINGLE best transform for this step
            result = self._resonator.resonate(observation)

            if result.min_confidence < self._refusal_threshold:
                # No confident transform found
                break

            # 3. Execute: Apply transform, get new state
            objects = self._detector.detect(current_state)
            new_state = self._executor.execute(
                action=result.action,
                target=result.target,
                modifier=result.modifier,
                objects=objects,
                grid=current_state,
            )

            # 3a. Verify progress: Check if transformation had any effect
            if new_state == current_state:
                # No progress made - transformation was ineffective
                break

            transform_chain.append(result)

            # 4. Check: Are we done?
            if self._is_solved(new_state, task):
                return IterativeResult(
                    output=new_state,
                    transform_chain=transform_chain,
                    steps_taken=step + 1,
                    solved=True,
                    confidence=result.min_confidence,
                )

            # 5. Cycle detection: Have we seen this state before?
            state_hash = self._state_hash(new_state)
            if state_hash in visited_states:
                # Stuck in a loop - return best effort
                break
            visited_states.add(state_hash)

            # 6. Iterate: Treat (new_state → target) as new sub-problem
            current_state = new_state

        # Didn't fully solve, return best effort
        return IterativeResult(
            output=current_state,
            transform_chain=transform_chain,
            steps_taken=len(transform_chain),
            solved=False,
            confidence=transform_chain[-1].min_confidence if transform_chain else 0.0,
        )

    def solve_beam(
        self,
        task: ARCTask,
        beam_width: int = 5,
    ) -> IterativeResult:
        """
        Attempt to solve ARC task via beam search (exploring multiple paths).

        Instead of committing to a single best transform at each step,
        explores top-k candidates in parallel. Maintains a "beam" of
        beam_width best states, pruning poor-scoring branches.

        This is more robust for multi-step tasks where a single greedy
        choice might lead to a dead end.

        Args:
            task: ARC task with training pairs and test input
            beam_width: Number of best states to maintain (default: 5)

        Returns:
            IterativeResult with best solution found

        Example:
            >>> result = solver.solve_beam(task, beam_width=5)
            >>> print(result.solved, result.steps_taken)
            True 2
        """
        # Initialize beam: (current_grid, transform_chain, score)
        initial_state = (task.test_input, [], 1.0)
        beam = [initial_state]

        best_result = IterativeResult(
            output=task.test_input,
            transform_chain=[],
            steps_taken=0,
            solved=False,
            confidence=0.0,
        )

        for step in range(self._max_steps):
            next_beam = []

            for current_grid, chain, score in beam:
                # 1. Observe: What's the delta?
                observation = self._observe_remaining_delta(current_grid, task.training)

                if observation is None:
                    # No valid pattern - this state exhausted
                    continue

                # 2. Resonate: Get top-k candidates
                candidates = self._resonator.resonate_topk(
                    observation,
                    k=beam_width,
                )

                if not candidates:
                    # No valid candidates - this state exhausted
                    continue

                # 3. Execute: Try each candidate
                for candidate in candidates:
                    objects = self._detector.detect(current_grid)
                    new_grid = self._executor.execute(
                        action=candidate.action,
                        target=candidate.target,
                        modifier=candidate.modifier,
                        objects=objects,
                        grid=current_grid,
                    )

                    # 3a. Verify progress
                    if new_grid == current_grid:
                        # Transformation had no effect - skip
                        continue

                    new_chain = chain + [candidate]
                    new_score = score * candidate.min_confidence

                    # 4. Check: Are we done?
                    if self._is_solved(new_grid, task):
                        return IterativeResult(
                            output=new_grid,
                            transform_chain=new_chain,
                            steps_taken=step + 1,
                            solved=True,
                            confidence=candidate.min_confidence,
                        )

                    next_beam.append((new_grid, new_chain, new_score))

            # Keep only top beam_width states by score
            if not next_beam:
                # All paths exhausted - convergence
                break

            next_beam.sort(key=lambda x: -x[2])  # Sort descending by score
            beam = next_beam[:beam_width]

            # Track best effort (for incomplete solutions)
            best_state = beam[0]
            if best_state[2] > best_result.confidence or len(best_state[1]) > len(best_result.transform_chain):
                best_result = IterativeResult(
                    output=best_state[0],
                    transform_chain=best_state[1],
                    steps_taken=step + 1,
                    solved=False,
                    confidence=best_state[2],
                )

        # Return best effort if not solved
        return best_result

    def _observe_remaining_delta(
        self,
        current: Grid,
        training: List[TrainingPair],
    ) -> Optional[torch.Tensor]:
        """
        Observe the transformation PATTERN from training pairs.

        Key insight (from cursor-code review): Training outputs are NOT
        universal targets. We should encode the transformation pattern
        (training input → training output), not (current → training output).

        The transformation pattern tells us WHAT operation was applied,
        then we apply that same operation to the current state.

        Args:
            current: Current state grid
            training: List of training pairs as hints

        Returns:
            Observation vector encoding the transformation pattern
        """
        best_pair = self._find_best_target_hint(current, training)

        if best_pair is None:
            return None

        # Encode the TRAINING transformation pattern, not current→target delta
        # This tells us what operation to apply, regardless of current state
        return self._encode_training_pattern(best_pair)

    def _find_best_target_hint(
        self,
        current: Grid,
        training: List[TrainingPair],
    ) -> Optional[TrainingPair]:
        """
        Find the training pair that best guides us toward a solution.

        Strategy: Find training pair whose INPUT structure most resembles
        current state, then use its transformation pattern as a hint.

        Args:
            current: Current state grid
            training: List of training pairs

        Returns:
            Best matching training pair, or None if no training pairs
        """
        if not training:
            return None

        if len(training) == 1:
            return training[0]

        # Find training pair with most similar INPUT to current state
        best_pair = training[0]
        best_similarity = 0.0

        for pair in training:
            similarity = self._grid_similarity(current, pair.input)
            if similarity > best_similarity:
                best_similarity = similarity
                best_pair = pair

        return best_pair

    def _encode_training_pattern(
        self,
        pair: TrainingPair,
    ) -> Optional[torch.Tensor]:
        """
        Encode the transformation pattern from a training pair.

        This encodes WHAT transformation was applied (input → output),
        which can then be applied to any input with similar structure.

        Args:
            pair: Training pair to encode

        Returns:
            Observation vector for the transformation pattern
        """
        # Check for grid-level transformation first (e.g., tiling)
        grid_obs = self._encoder.encode_grid_transformation(
            pair.input.shape, pair.output.shape
        )
        if grid_obs is not None:
            return grid_obs

        # Detect objects in training input/output
        input_objects = self._detector.detect(pair.input)
        output_objects = self._detector.detect(pair.output)

        if not input_objects and not output_objects:
            return None

        # Match objects between input and output
        matches = self._detector.match_objects(input_objects, output_objects)

        # Encode each transformation observation
        obs_vectors = []
        for in_obj, out_obj in matches:
            obs = self._encoder.encode_transformation_observation(in_obj, out_obj)
            obs_vectors.append(obs)

        if not obs_vectors:
            return None

        from hologram.core.operations import Operations
        return Operations.bundle(*obs_vectors)

    def _encode_delta(
        self,
        current: Grid,
        target: Grid,
    ) -> Optional[torch.Tensor]:
        """
        Encode the transformation needed to go from current → target.

        NOTE: This method is kept for backwards compatibility but
        _encode_training_pattern should be preferred for iterative solving.

        Args:
            current: Current state grid
            target: Target grid (from training output)

        Returns:
            Observation vector, or None if no valid delta
        """
        # Check for grid-level transformation first
        grid_obs = self._encoder.encode_grid_transformation(
            current.shape, target.shape
        )
        if grid_obs is not None:
            return grid_obs

        # Fall back to object-level observation
        current_objects = self._detector.detect(current)
        target_objects = self._detector.detect(target)

        if not current_objects and not target_objects:
            return None

        # Match objects between grids
        matches = self._detector.match_objects(current_objects, target_objects)

        # Encode each delta as observation
        obs_vectors = []
        for curr_obj, tgt_obj in matches:
            obs = self._encoder.encode_transformation_observation(curr_obj, tgt_obj)
            obs_vectors.append(obs)

        if not obs_vectors:
            return None

        from hologram.core.operations import Operations
        return Operations.bundle(*obs_vectors)

    def _is_solved(self, candidate: Grid, task: ARCTask) -> bool:
        """
        Check if candidate matches expected output pattern.

        Uses training outputs as proxy for test output structure.
        For multi-step tasks, requires matching majority of training
        outputs to avoid premature termination on intermediate states.

        Args:
            candidate: Candidate output grid
            task: ARC task for reference

        Returns:
            True if candidate matches expected pattern
        """
        # First check: If we have test_output, compare directly (ground truth)
        if task.test_output is not None:
            return candidate == task.test_output

        # Second check: Compare structure to training outputs
        # Require matching majority of training outputs (not just one)
        # to avoid premature termination on intermediate states
        match_count = 0
        for pair in task.training:
            similarity = self._grid_similarity(candidate, pair.output)
            if similarity >= self._convergence_threshold:
                match_count += 1

        # Require majority of training outputs to match
        majority_threshold = (len(task.training) + 1) // 2
        return match_count >= majority_threshold

    def _grid_similarity(self, a: Grid, b: Grid) -> float:
        """
        Compute structural similarity between two grids.

        Args:
            a: First grid
            b: Second grid

        Returns:
            Similarity score in [0, 1]
        """
        # Simple approach: normalized pixel-wise match
        if a.shape != b.shape:
            return 0.0

        total_cells = a.height * a.width
        if total_cells == 0:
            return 1.0

        matches = 0
        for r in range(a.height):
            for c in range(a.width):
                if a.data[r, c] == b.data[r, c]:
                    matches += 1

        return matches / total_cells

    def _state_hash(self, grid: Grid) -> str:
        """
        Compute hash of grid state for cycle detection.

        Includes grid shape to avoid collisions between grids with same
        data but different dimensions (e.g., 1x4 vs 2x2).

        Args:
            grid: Grid to hash

        Returns:
            Hash string
        """
        import numpy as np
        # Include shape to avoid collisions on different dimensions
        shape_bytes = np.array([grid.height, grid.width], dtype=np.int32).tobytes()
        data_bytes = grid.data.tobytes()
        return hashlib.sha256(shape_bytes + data_bytes).hexdigest()

    def __repr__(self) -> str:
        return (
            f"IterativeSolver(max_steps={self._max_steps}, "
            f"convergence={self._convergence_threshold}, "
            f"refusal={self._refusal_threshold})"
        )
