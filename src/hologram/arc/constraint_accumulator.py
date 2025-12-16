"""
ConstraintAccumulator: Learn from transformation failures in iterative solving.

Key insight: Instead of blindly re-searching the same vocabulary each iteration,
track what's been tried and bias the search toward variations of partial successes.

This implements Phase 1.1 of the ARC improvement plan:
- Track tried transformations with their partial match scores
- Build penalty vectors to suppress repeated failures
- Suggest promising search spaces based on "neighbors" of partial successes
"""

from dataclasses import dataclass
from typing import Set, List, Tuple, Optional, FrozenSet, TYPE_CHECKING
import torch

from hologram.arc.transform_resonator import TransformResult

if TYPE_CHECKING:
    from hologram.introspection.circuit_observer import CircuitObserver


@dataclass(frozen=True)
class TransformSignature:
    """
    Immutable signature of a transformation for tracking.

    Uses (action, target, modifier) tuple as unique identifier.
    Frozen and hashable for use in sets.
    """
    action: str
    target: str
    modifier: str

    @classmethod
    def from_result(cls, result: TransformResult) -> "TransformSignature":
        """Create signature from TransformResult."""
        return cls(
            action=result.action,
            target=result.target,
            modifier=result.modifier,
        )

    def __str__(self) -> str:
        return f"{self.action}({self.target}, {self.modifier})"


class ConstraintAccumulator:
    """
    Accumulate constraints from transformation attempts.

    Tracks:
    - Tried transformations (to avoid re-trying)
    - Partial successes (transformations that partially matched)
    - Failure patterns (to bias search away from similar failures)

    Usage:
        >>> accumulator = ConstraintAccumulator()
        >>> # After trying a transformation
        >>> accumulator.record_attempt(transform, partial_score=0.3)
        >>> # Get penalty for next search
        >>> penalty = accumulator.get_penalty_vector(all_action_vecs, all_target_vecs, all_modifier_vecs)
        >>> # Adjust resonator search to avoid penalties

    Attributes:
        _tried_signatures: Set of transformation signatures attempted
        _partial_successes: List of (signature, score) for non-zero partial matches
        _failure_threshold: Score below this is considered a failure (default: 0.1)
    """

    def __init__(self, failure_threshold: float = 0.1):
        """
        Initialize constraint accumulator.

        Args:
            failure_threshold: Partial score below this is a "failure" to avoid
        """
        self._tried_signatures: Set[TransformSignature] = set()
        self._partial_successes: List[Tuple[TransformSignature, float]] = []
        self._failure_threshold = failure_threshold
        # Self-improvement integration
        self._circuit_observer: Optional['CircuitObserver'] = None

    def set_circuit_observer(self, observer: 'CircuitObserver') -> None:
        """
        Attach circuit observer for self-improvement tracking.

        Args:
            observer: CircuitObserver instance to receive observations
        """
        self._circuit_observer = observer

    def record_attempt(
        self,
        result: TransformResult,
        partial_score: float,
    ) -> None:
        """
        Record a transformation attempt and its partial success.

        Args:
            result: The transformation that was tried
            partial_score: How well it matched (0.0 = total failure, 1.0 = perfect)
        """
        sig = TransformSignature.from_result(result)
        self._tried_signatures.add(sig)

        # Track partial successes (anything above failure threshold)
        if partial_score > self._failure_threshold:
            self._partial_successes.append((sig, partial_score))
            # Keep list sorted by score (best first)
            self._partial_successes.sort(key=lambda x: -x[1])

        # Report to circuit observer for self-improvement learning
        if self._circuit_observer is not None:
            confidence = getattr(result, 'min_confidence', partial_score)
            self._circuit_observer.observe(
                items=[result.action, result.target, result.modifier],
                success=(partial_score > 0.5),  # Threshold for "success"
                confidence=confidence,
                context="arc_transformation"
            )

    def has_tried(self, result: TransformResult) -> bool:
        """Check if a transformation has already been attempted."""
        sig = TransformSignature.from_result(result)
        return sig in self._tried_signatures

    def get_best_partial(self) -> Optional[Tuple[TransformSignature, float]]:
        """Get the best partial success so far, or None if none."""
        if not self._partial_successes:
            return None
        return self._partial_successes[0]

    def get_transformation_prior(self, result: TransformResult) -> float:
        """
        Get prior probability based on historical success.

        Uses the circuit observer to query learned success rates
        for this transformation combination.

        Args:
            result: Transformation to query

        Returns:
            Prior probability [0.0, 1.0], or 0.5 if no observer
        """
        if self._circuit_observer is None:
            return 0.5  # Neutral prior when not learning

        return self._circuit_observer.get_prior([
            result.action,
            result.target,
            result.modifier
        ])

    def get_penalty_vector(
        self,
        action_vectors: torch.Tensor,  # (n_actions, dim)
        target_vectors: torch.Tensor,  # (n_targets, dim)
        modifier_vectors: torch.Tensor,  # (n_modifiers, dim)
        action_names: List[str],
        target_names: List[str],
        modifier_names: List[str],
        penalty_strength: float = 0.3,
    ) -> torch.Tensor:
        """
        Build penalty vector to bias search away from failures.

        Creates a vector that, when subtracted from candidate scores,
        reduces likelihood of re-trying failed transformations.

        Args:
            action_vectors: Stacked action vocabulary vectors
            target_vectors: Stacked target vocabulary vectors
            modifier_vectors: Stacked modifier vocabulary vectors
            action_names: Action names corresponding to vectors
            target_names: Target names corresponding to vectors
            modifier_names: Modifier names corresponding to vectors
            penalty_strength: How much to penalize (0.0 = no penalty, 1.0 = max penalty)

        Returns:
            Penalty vector (dim,) to subtract from resonator scores
        """
        if not self._tried_signatures:
            # No failures to avoid yet
            return torch.zeros(action_vectors.shape[1])

        # Build penalty as bundle of failed transformation vectors
        from hologram.core.operations import Operations

        penalty_parts = []

        for sig in self._tried_signatures:
            # Find vectors for this signature's components
            try:
                action_idx = action_names.index(sig.action)
                target_idx = target_names.index(sig.target)
                modifier_idx = modifier_names.index(sig.modifier)

                action_vec = action_vectors[action_idx]
                target_vec = target_vectors[target_idx]
                modifier_vec = modifier_vectors[modifier_idx]

                # Combine components (same way encoder does)
                failed_transform = Operations.bundle(action_vec, target_vec, modifier_vec)
                penalty_parts.append(failed_transform)

            except ValueError:
                # Signature component not in vocabulary (shouldn't happen)
                continue

        if not penalty_parts:
            return torch.zeros(action_vectors.shape[1])

        # Bundle all failures and scale by penalty strength
        penalty = Operations.bundle(*penalty_parts)
        return penalty * penalty_strength

    def suggest_search_bias(
        self,
        action_vectors: torch.Tensor,
        target_vectors: torch.Tensor,
        modifier_vectors: torch.Tensor,
        action_names: List[str],
        target_names: List[str],
        modifier_names: List[str],
        bias_strength: float = 0.5,
    ) -> Optional[torch.Tensor]:
        """
        Suggest a search bias toward "neighbors" of partial successes.

        If we had transformations that partially worked, bias the search
        toward similar transformations (same action, different modifier, etc.)

        Args:
            action_vectors: Stacked action vocabulary vectors
            target_vectors: Stacked target vocabulary vectors
            modifier_vectors: Stacked modifier vocabulary vectors
            action_names: Action names corresponding to vectors
            target_names: Target names corresponding to vectors
            modifier_names: Modifier names corresponding to vectors
            bias_strength: How much to bias (0.0 = no bias, 1.0 = max bias)

        Returns:
            Bias vector (dim,) to add to observation, or None if no partial successes
        """
        best_partial = self.get_best_partial()
        if best_partial is None:
            return None

        sig, score = best_partial

        # Build bias by bundling "neighboring" transformations
        # Neighbors = same action/target, different modifier (or vice versa)
        from hologram.core.operations import Operations

        bias_parts = []

        try:
            # Get vectors for the partial success
            action_idx = action_names.index(sig.action)
            target_idx = target_names.index(sig.target)
            modifier_idx = modifier_names.index(sig.modifier)

            action_vec = action_vectors[action_idx]
            target_vec = target_vectors[target_idx]
            modifier_vec = modifier_vectors[modifier_idx]

            # Add the partial success itself (weighted by its score)
            partial_vec = Operations.bundle(action_vec, target_vec, modifier_vec)
            bias_parts.append(partial_vec * score)

            # Add variations (same action+target, all modifiers)
            for mod_idx, mod_name in enumerate(modifier_names):
                if mod_name != sig.modifier:  # Don't add the same one twice
                    variant = Operations.bundle(
                        action_vec,
                        target_vec,
                        modifier_vectors[mod_idx]
                    )
                    bias_parts.append(variant * (score * 0.5))  # Weaker than original

        except ValueError:
            # Couldn't find component in vocabulary
            return None

        if not bias_parts:
            return None

        # Bundle all biases and scale by bias strength
        bias = Operations.bundle(*bias_parts)
        return bias * bias_strength

    def reset(self) -> None:
        """Clear all accumulated constraints (for new task)."""
        self._tried_signatures.clear()
        self._partial_successes.clear()

    def __len__(self) -> int:
        """Return number of transformations tried."""
        return len(self._tried_signatures)

    def __str__(self) -> str:
        """String representation for debugging."""
        best = self.get_best_partial()
        best_str = f"{best[0]} ({best[1]:.1%})" if best else "none"
        return (
            f"ConstraintAccumulator(tried={len(self)}, "
            f"best_partial={best_str})"
        )
