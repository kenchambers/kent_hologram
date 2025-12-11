"""
Dreamer: Iterative exploration loop for creative generation.

Replaces the rigid "Refuse on Low Confidence" policy with an iterative
"Dreaming" loop that injects noise to find creative associations.

When confidence is low, instead of refusing, the Dreamer:
1. Injects Gaussian noise into the thought vector (simulating "mind wandering")
2. Re-runs the Resonator to find new associations
3. Repeats until confidence threshold met or max iterations reached

This enables creative exploration and associative thinking.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from hologram.core.resonator import Resonator, ResonatorResult
from hologram.core.similarity import Similarity


@dataclass
class DreamResult:
    """Result of dreaming exploration.

    Attributes:
        thought_vector: Final thought vector (possibly modified with noise)
        resonator_result: Resonator result from final iteration
        iterations: Number of dreaming iterations performed
        best_confidence: Best confidence achieved across all iterations
        converged: Whether confidence threshold was met
    """
    thought_vector: torch.Tensor
    resonator_result: ResonatorResult
    iterations: int
    best_confidence: float
    converged: bool

    def __str__(self) -> str:
        status = "converged" if self.converged else "max_iter"
        return (
            f"DreamResult({status} @ {self.iterations} iter, "
            f"best_conf={self.best_confidence:.3f})"
        )


class Dreamer:
    """
    Iterative exploration loop for creative generation.

    When confidence is low, injects noise and re-explores rather than
    refusing. This enables creative associations and "mind wandering"
    behavior.

    Attributes:
        _resonator: Resonator for thought factorization
        _max_iterations: Maximum dreaming iterations
        _confidence_threshold: Target confidence threshold
        _noise_scale: Scale of Gaussian noise injection
        _noise_decay: Decay factor for noise scale over iterations

    Example:
        >>> dreamer = Dreamer(resonator, max_iterations=10)
        >>> result = dreamer.dream(thought_vector, nouns, verbs)
        >>> if result.converged:
        >>>     print(f"Found solution: {result.resonator_result}")
    """

    def __init__(
        self,
        resonator: Resonator,
        max_iterations: int = 10,
        confidence_threshold: float = 0.5,
        noise_scale: float = 0.1,
        noise_decay: float = 0.9,
    ):
        """
        Initialize dreamer.

        Args:
            resonator: Resonator for thought factorization
            max_iterations: Maximum dreaming iterations (default: 10)
            confidence_threshold: Target confidence threshold (default: 0.5)
            noise_scale: Initial scale of Gaussian noise (default: 0.1)
            noise_decay: Decay factor for noise scale (default: 0.9)
        """
        self._resonator = resonator
        self._max_iterations = max_iterations
        self._confidence_threshold = confidence_threshold
        self._noise_scale = noise_scale
        self._noise_decay = noise_decay

    def dream(
        self,
        thought_vector: torch.Tensor,
        noun_vocabulary: List[str],
        verb_vocabulary: List[str],
    ) -> DreamResult:
        """
        Perform dreaming exploration loop.

        Iteratively injects noise and re-runs Resonator until confidence
        threshold is met or max iterations reached.

        Args:
            thought_vector: Initial thought vector to explore
            noun_vocabulary: Noun vocabulary for Resonator
            verb_vocabulary: Verb vocabulary for Resonator

        Returns:
            DreamResult with final thought vector and resonator result

        Example:
            >>> result = dreamer.dream(thought_vec, nouns, verbs)
            >>> if result.converged:
            >>>     print("Found creative solution!")
        """
        current_thought = thought_vector.clone()
        best_result: Optional[ResonatorResult] = None
        best_confidence = 0.0
        best_thought = current_thought.clone()
        current_noise_scale = self._noise_scale

        for iteration in range(self._max_iterations):
            # Run Resonator on current thought
            result = self._resonator.resonate(
                current_thought,
                noun_vocabulary,
                verb_vocabulary,
            )

            # Calculate overall confidence (average of slot confidences)
            avg_confidence = sum(result.confidence.values()) / len(result.confidence)

            # Track best result
            if avg_confidence > best_confidence:
                best_confidence = avg_confidence
                best_result = result
                best_thought = current_thought.clone()

            # Check if we've reached threshold
            if avg_confidence >= self._confidence_threshold:
                return DreamResult(
                    thought_vector=best_thought,
                    resonator_result=best_result,
                    iterations=iteration + 1,
                    best_confidence=best_confidence,
                    converged=True,
                )

            # Inject noise for next iteration (mind wandering)
            noise = torch.randn_like(current_thought) * current_noise_scale
            current_thought = self._normalize(current_thought + noise)

            # Decay noise scale
            current_noise_scale *= self._noise_decay

        # Return best result found
        return DreamResult(
            thought_vector=best_thought,
            resonator_result=best_result or result,
            iterations=self._max_iterations,
            best_confidence=best_confidence,
            converged=False,
        )

    def _normalize(self, vector: torch.Tensor) -> torch.Tensor:
        """Normalize vector to unit length."""
        norm = torch.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def __repr__(self) -> str:
        return (
            f"Dreamer(max_iter={self._max_iterations}, "
            f"threshold={self._confidence_threshold:.2f}, "
            f"noise_scale={self._noise_scale:.2f})"
        )
