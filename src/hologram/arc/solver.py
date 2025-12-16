"""
HolographicARCSolver: Main solver integrating all ARC components.

Orchestrates the full pipeline:
1. Skill memory lookup (O(1) if seen before)
2. Object detection
3. Transformation observation
4. Resonator factorization
5. Verification
6. Execution and consolidation

Maintains the no-hallucination guarantee by only outputting
transformations that match the vocabulary.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING
import torch

from hologram.arc.types import Grid, Object, ARCTask, TrainingPair
from hologram.arc.detector import ObjectDetector
from hologram.arc.encoder import ObjectEncoder
from hologram.arc.transform_resonator import TransformationResonator, TransformResult
from hologram.arc.executor import TransformationExecutor
from typing import Literal

from hologram.arc.iterative_solver import IterativeSolver, IterativeResult
from hologram.arc.search_verifier import SearchVerifier, VerificationStats
from hologram.consolidation.neural_memory import NeuralMemory, ConsolidationFact
from hologram.cognition.metacognition import MetacognitiveState, MetacognitiveMood
from hologram.core.fractal import FractalSpace
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity
from hologram.core.vector_space import VectorSpace
from hologram.introspection import SelfImprovementManager

if TYPE_CHECKING:
    from hologram.introspection import CircuitObserver


@dataclass
class SolverResult:
    """Result of solving an ARC task."""
    output: Optional[Grid]
    transformation: Optional[TransformResult]
    confidence: float
    from_cache: bool
    message: str


class HolographicARCSolver:
    """
    Main ARC solver using holographic reasoning.

    This solver attempts to solve ARC tasks by:
    1. Detecting objects in training pairs
    2. Encoding transformation observations holographically
    3. Using the Resonator to factorize into (ACTION, TARGET, MODIFIER)
    4. Verifying the factorization matches observations
    5. Applying the transformation to the test input

    The system maintains the no-hallucination guarantee because
    the Resonator can only output vocabulary items.

    Attributes:
        detector: Object detector
        encoder: Object encoder
        resonator: Transformation resonator
        executor: Transformation executor
        confidence_threshold: Minimum confidence to return result

    Example:
        >>> solver = HolographicARCSolver()
        >>> result = solver.solve(task)
        >>> if result.output:
        ...     print("Solved with", result.transformation)
    """

    # Lower threshold because bundling multiple observations creates interference
    # but the Resonator can still extract the dominant pattern
    DEFAULT_CONFIDENCE_THRESHOLD = 0.005

    # Minimum confidence for neural memory skill lookup
    # Phase 1.2: Lowered from 0.7 to 0.6 for analogical transfer (try similar tasks)
    SKILL_CONFIDENCE_THRESHOLD = 0.6

    def __init__(
        self,
        dimensions: int = 10000,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iterative: bool = True,
        max_steps: int = 5,
        strategy: Literal["resonator", "search_verify"] = "search_verify",
        search_k: int = 20,
        search_slot_k: int = 5,
        isolate_memory: bool = False,
        enable_self_improvement: bool = True,
        self_improvement_path: Optional[str] = None,
    ):
        """
        Initialize solver with all components.

        Args:
            dimensions: HDC vector dimensions
            confidence_threshold: Minimum confidence for output
            iterative: Use iterative multi-step solving (default: True)
            max_steps: Max steps for iterative solving (default: 5)
            strategy: Solving strategy - "resonator" (single-shot) or "search_verify" (candidate search)
            search_k: Number of top candidates to generate for search_verify (default: 20)
            search_slot_k: Top-k per slot for candidate generation (default: 5)
            isolate_memory: If True, disable skill memory (for benchmark). Default: False.
                           When False (default), memory persists and hologram gets smarter.
                           When True, fresh solver without cross-task learning (for honest eval).
            enable_self_improvement: Enable circuit observer for self-improvement (default: True)
            self_improvement_path: Path to persist learned patterns (default: ./data/arc_learned_patterns.json)
        """
        # Core components
        self._space = VectorSpace(dimensions=dimensions)
        self._fractal_space = FractalSpace(dimensions=dimensions)
        self._codebook = Codebook(self._space)

        # ARC-specific components
        self._detector = ObjectDetector()
        self._encoder = ObjectEncoder(self._fractal_space, self._codebook)
        self._resonator = TransformationResonator(self._encoder, self._codebook)
        self._executor = TransformationExecutor(self._detector)

        self._confidence_threshold = confidence_threshold
        self._ops = Operations
        self._iterative = iterative
        self._strategy = strategy
        self._search_k = search_k
        self._search_slot_k = search_slot_k
        self._isolate_memory = isolate_memory
        self._dimensions = dimensions  # Store for clear_cache()

        # Search verifier for search_verify strategy
        if strategy == "search_verify":
            self._verifier = SearchVerifier(
                executor=self._executor,
                detector=self._detector,
            )

        # Self-improvement manager (learns from transformation outcomes)
        self._self_improvement: Optional[SelfImprovementManager] = None
        if enable_self_improvement and not isolate_memory:
            # Default persistence path for learned patterns
            if self_improvement_path is None:
                self_improvement_path = "./data/arc_learned_patterns.json"
            # Ensure directory exists
            Path(self_improvement_path).parent.mkdir(parents=True, exist_ok=True)
            self._self_improvement = SelfImprovementManager(
                persist_path=self_improvement_path
            )

        # Iterative solver for multi-step reasoning
        if iterative:
            self._iterative_solver = IterativeSolver(
                encoder=self._encoder,
                resonator=self._resonator,
                executor=self._executor,
                detector=self._detector,
                max_steps=max_steps,
                circuit_observer=self._self_improvement.observer if self._self_improvement else None,
            )

        # Skill memory using NeuralMemory for O(1) lookup
        # Only create if not isolated (for progressive learning)
        if not isolate_memory:
            self._skill_memory = NeuralMemory(
                input_dim=dimensions,
                hidden_dim=256,
                initial_vocab_size=100,  # Start small, will auto-expand
            )
        else:
            self._skill_memory = None

        # Cache transform results by label for reconstruction after neural lookup
        self._transform_cache: dict[str, TransformResult] = {}

    def solve(self, task: ARCTask) -> SolverResult:
        """
        Attempt to solve an ARC task.

        Args:
            task: ARC task with training pairs and test input

        Returns:
            SolverResult with output grid (or None if refused)
        """
        # 1. Compute task signature as HDC vector for neural lookup
        task_sig_vec = self._compute_task_signature_vector(task)

        # 2. Check skill memory (O(1) neural lookup) - only if not isolated
        cached_label = None
        cache_confidence = 0.0
        if self._skill_memory is not None:
            cached_label, cache_confidence = self._skill_memory.query(task_sig_vec)
        if cached_label is not None and cache_confidence >= self.SKILL_CONFIDENCE_THRESHOLD:
            cached_transform = self._transform_cache.get(cached_label)
            if cached_transform is not None:
                output = self._apply_transformation(
                    cached_transform, task.test_input
                )
                return SolverResult(
                    output=output,
                    transformation=cached_transform,
                    confidence=cache_confidence,
                    from_cache=True,
                    message=f"Retrieved from skill memory (conf={cache_confidence:.2f})",
                )

        # 3. Use search+verify strategy if enabled
        if self._strategy == "search_verify":
            return self._solve_search_verify(task, task_sig_vec)

        # 4. Use iterative solver for multi-step reasoning if enabled
        if self._iterative:
            iter_result = self._iterative_solver.solve(task)

            # Get the last transform from chain for consolidation
            last_transform = (
                iter_result.transform_chain[-1]
                if iter_result.transform_chain
                else None
            )

            # Store skill whenever we have a valid transformation
            # (consistent with single-shot behavior, which stores on any output)
            if last_transform is not None:
                self._store_skill(task_sig_vec, last_transform)

            return SolverResult(
                output=iter_result.output,
                transformation=last_transform,
                confidence=iter_result.confidence,
                from_cache=False,
                message=f"Iterative: {iter_result.steps_taken} steps, solved={iter_result.solved}",
            )

        # 5. Fallback: Single-shot solving (original logic)
        # Process training pairs
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

        # 5. Resonate to find transformation
        result = self._resonator.resonate(observation_bundle)

        # 6. Verify factorization
        verification_score = self._resonator.verify_factorization(
            observation_bundle, result
        )

        # Combine verification with resonator confidence
        combined_confidence = min(result.min_confidence, verification_score)

        # 7. Check confidence threshold
        if combined_confidence < self._confidence_threshold:
            return SolverResult(
                output=None,
                transformation=result,
                confidence=combined_confidence,
                from_cache=False,
                message=f"Confidence {combined_confidence:.3f} below threshold {self._confidence_threshold}",
            )

        # 8. Apply transformation
        output = self._apply_transformation(result, task.test_input)

        # 9. Consolidate successful transformation into skill memory
        self._store_skill(task_sig_vec, result)

        return SolverResult(
            output=output,
            transformation=result,
            confidence=combined_confidence,
            from_cache=False,
            message=f"Solved: {result.action}({result.target}, {result.modifier})",
        )

    def _compute_task_signature_vector(self, task: ARCTask) -> torch.Tensor:
        """
        Compute a signature vector for the task (for neural memory lookup).

        Args:
            task: ARC task

        Returns:
            HDC signature vector
        """
        # Create signature by encoding task structure
        sig_parts = []
        for i, pair in enumerate(task.training):
            # Encode input shape
            shape_str = f"shape_{pair.input.shape[0]}x{pair.input.shape[1]}"
            shape_vec = self._codebook.encode(shape_str)

            # Encode a sample of the input data for uniqueness
            sample = tuple(pair.input.data.flatten()[:10].tolist())
            sample_vec = self._space.random_vector(seed=hash(sample) % (2**31))

            # Bind shape and sample, then bundle with index
            pair_sig = self._ops.bind(shape_vec, sample_vec)
            sig_parts.append(pair_sig)

        if sig_parts:
            return self._ops.bundle(*sig_parts)
        else:
            return self._space.random_vector(seed=0)

    def _compute_adaptive_k(self, mood: MetacognitiveMood) -> int:
        """
        Compute search width (k) based on metacognitive mood.

        When CONFUSED, widen search to explore more candidates.
        When CONFIDENT, narrow search for efficiency.

        Args:
            mood: Current metacognitive mood

        Returns:
            Adaptive k value (capped at 100)
        """
        base_k = self._search_k  # Default: 20
        if mood == MetacognitiveMood.CONFUSED:
            k = int(base_k * 2.5)  # 50 - widen search
        elif mood == MetacognitiveMood.CURIOUS:
            k = int(base_k * 1.5)  # 30 - moderate expansion
        elif mood == MetacognitiveMood.CONFIDENT:
            k = int(base_k * 0.5)  # 10 - narrow search
        else:
            k = base_k  # 20 (NEUTRAL/ANXIOUS)
        return min(k, 100)  # Cap to prevent explosion

    def _observe_training_pair(
        self,
        pair: TrainingPair,
    ) -> Optional[torch.Tensor]:
        """
        Create observation vector from a training pair.

        PRIORITY: Grid-level transformations (tiling) take precedence.
        If a grid-level transformation is detected (e.g., dimensional change),
        return that observation directly WITHOUT bundling with object-level
        observations to preserve signal strength.

        Args:
            pair: Training pair (input → output)

        Returns:
            Observation vector, or None if no objects detected
        """
        # First, check for grid-level transformations (tiling, etc.)
        # These take priority because they involve the whole grid
        grid_obs = self._encoder.encode_grid_transformation(
            pair.input.shape,
            pair.output.shape,
        )
        if grid_obs is not None:
            # Grid-level transformation detected - return it directly
            # without bundling with object observations to preserve signal
            return grid_obs

        # No grid-level transformation - fall back to object-level
        obs_vectors = []

        # Detect objects
        input_objects = self._detector.detect(pair.input)
        output_objects = self._detector.detect(pair.output)

        # Match objects between input and output
        if input_objects or output_objects:
            matches = self._detector.match_objects(input_objects, output_objects)

            # Encode each match as observation
            for in_obj, out_obj in matches:
                obs = self._encoder.encode_transformation_observation(in_obj, out_obj)
                obs_vectors.append(obs)

        if not obs_vectors:
            return None

        # Bundle all observations from this pair
        return self._ops.bundle(*obs_vectors)

    def _apply_transformation(
        self,
        transform: TransformResult,
        test_input: Grid,
    ) -> Grid:
        """
        Apply transformation to test input.

        Args:
            transform: Transformation result from resonator
            test_input: Test input grid

        Returns:
            Transformed grid
        """
        # Detect objects in test input
        objects = self._detector.detect(test_input)

        # Apply transformation
        return self._executor.execute(
            action=transform.action,
            target=transform.target,
            modifier=transform.modifier,
            objects=objects,
            grid=test_input,
        )

    def _store_skill(self, task_sig: torch.Tensor, result: TransformResult) -> None:
        """
        Store a learned skill in neural memory.

        Does nothing if isolate_memory=True (for benchmark mode).

        Args:
            task_sig: Task signature vector
            result: Transformation result to store
        """
        # Skip storage if memory is isolated (benchmark mode)
        if self._skill_memory is None:
            return

        # Create unique label for this transformation
        label = f"{result.action}_{result.target}_{result.modifier}"

        # Store in transform cache for reconstruction
        self._transform_cache[label] = result

        # Consolidate into neural memory
        fact = ConsolidationFact(
            key_vector=task_sig,
            value_index=0,  # Will be assigned by NeuralMemory
            value_label=label,
        )
        self._skill_memory.consolidate([fact], epochs=20, batch_size=8)

    def _solve_search_verify(
        self,
        task: ARCTask,
        task_sig_vec: torch.Tensor,
    ) -> SolverResult:
        """
        Solve using search+verify strategy with metacognitive feedback.

        Implements the UltraThink loop:
        1. BOSS: Initialize metacognitive state
        2. PROPOSER: Generate candidates with adaptive k
        3. VERIFIER: Check against training pairs
        4. LEARNER: Update mood from partial scores
        5. Retry with widened search if needed

        Args:
            task: ARC task to solve
            task_sig_vec: Task signature vector for skill memory

        Returns:
            SolverResult with verified solution or refusal
        """
        # Build observation bundle from training pairs
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

        # Bundle observations
        observation_bundle = self._ops.bundle(*observations)

        # BOSS: Initialize metacognitive state for this task
        metacog = MetacognitiveState(self._codebook)
        max_attempts = 3
        best_partial_score = 0.0
        best_candidate = None
        stats: Optional[VerificationStats] = None  # Track best stats for diagnostics

        for attempt in range(max_attempts):
            # PROPOSER: Compute adaptive k based on mood
            k = self._compute_adaptive_k(metacog.mood)

            # Generate top-k candidates via resonator
            candidates = self._resonator.resonate_topk(
                observation_bundle,
                k=k,
                slot_k=self._search_slot_k,
            )

            if not candidates:
                # No candidates, update mood and retry
                metacog.update_from_confidence(0.0)
                continue

            # VERIFIER: Verify candidates and get stats
            current_stats = self._verifier.verify_candidates_with_stats(
                candidates, task.training
            )

            # Track best result across attempts (keep the stats with best score)
            if current_stats.best_partial_score > best_partial_score:
                best_partial_score = current_stats.best_partial_score
                best_candidate = candidates[0] if candidates else None
                stats = current_stats  # Preserve best stats for diagnostics

            # LEARNER: Update mood from verification results
            metacog.update_from_confidence(current_stats.best_partial_score)

            # Check for verified solution
            if current_stats.verified_transform is not None:
                # EXECUTOR: Apply verified transformation
                output = self._apply_transformation(
                    current_stats.verified_transform, task.test_input
                )

                # Store successful skill
                self._store_skill(task_sig_vec, current_stats.verified_transform)

                return SolverResult(
                    output=output,
                    transformation=current_stats.verified_transform,
                    confidence=1.0,  # Verified = perfect confidence
                    from_cache=False,
                    message=f"Verified (attempt {attempt + 1}, mood={metacog.mood.value}): "
                            f"{current_stats.verified_transform.action}({current_stats.verified_transform.target}, "
                            f"{current_stats.verified_transform.modifier})",
                )

            # No verified solution yet - loop will retry with updated mood
            # CONFUSED mood will widen search on next attempt

        # All attempts exhausted - return refusal with best partial info
        # Use diagnostic message from stats if available
        diagnostic = f"best partial: {best_partial_score:.0%}"
        if stats and stats.best_partial_transform:
            diagnostic = stats.diagnostic_message()

        return SolverResult(
            output=None,
            transformation=stats.best_partial_transform if stats else best_candidate,
            confidence=best_partial_score,
            from_cache=False,
            message=f"None of {max_attempts} attempts succeeded ({diagnostic}, final mood: {metacog.mood.value})",
        )

    def clear_cache(self) -> None:
        """Clear the skill cache and reset neural memory."""
        self._transform_cache.clear()
        # Reinitialize neural memory (only if not isolated)
        if not self._isolate_memory:
            self._skill_memory = NeuralMemory(
                input_dim=self._dimensions,
                hidden_dim=256,
                initial_vocab_size=100,
            )

    def evaluate_on_task(self, task: ARCTask) -> Tuple[bool, str]:
        """
        Evaluate solver on a task with known ground truth.

        Args:
            task: ARC task with test_output

        Returns:
            Tuple of (correct, message)
        """
        if task.test_output is None:
            return False, "No ground truth available"

        result = self.solve(task)

        if result.output is None:
            return False, f"Refused: {result.message}"

        correct = result.output == task.test_output
        return correct, result.message

    def get_improvement_report(self) -> str:
        """
        Get a report of what the solver has learned from self-improvement.

        Returns:
            Human-readable report of learned patterns, or message if disabled.
        """
        if self._self_improvement is None:
            return "Self-improvement is disabled (isolate_memory=True or enable_self_improvement=False)"
        return self._self_improvement.get_improvement_report()

    def get_improvement_stats(self) -> dict:
        """
        Get statistics about self-improvement learning.

        Returns:
            Dictionary with learning statistics, or empty dict if disabled.
        """
        if self._self_improvement is None:
            return {}
        return self._self_improvement.get_statistics()

    def save_learned_patterns(self) -> None:
        """Save learned patterns to disk (if self-improvement is enabled)."""
        if self._self_improvement is not None:
            self._self_improvement.save()


def create_simple_task(
    train_inputs: List[List[List[int]]],
    train_outputs: List[List[List[int]]],
    test_input: List[List[int]],
    test_output: Optional[List[List[int]]] = None,
    task_id: str = "custom",
) -> ARCTask:
    """
    Create an ARC task from Python lists.

    Convenience function for testing.

    Args:
        train_inputs: List of training input grids
        train_outputs: List of training output grids
        test_input: Test input grid
        test_output: Expected test output (optional)
        task_id: Task identifier

    Returns:
        ARCTask instance

    Example:
        >>> task = create_simple_task(
        ...     train_inputs=[[[0,1],[1,0]], [[0,2],[2,0]]],
        ...     train_outputs=[[[1,0],[0,1]], [[2,0],[0,2]]],
        ...     test_input=[[0,3],[3,0]],
        ...     test_output=[[3,0],[0,3]],
        ... )
    """
    training = []
    for inp, out in zip(train_inputs, train_outputs):
        training.append(TrainingPair(
            input=Grid.from_list(inp),
            output=Grid.from_list(out),
        ))

    return ARCTask(
        task_id=task_id,
        training=training,
        test_input=Grid.from_list(test_input),
        test_output=Grid.from_list(test_output) if test_output else None,
    )
