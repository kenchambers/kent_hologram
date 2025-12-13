"""
TransformationResonator: Extended Resonator for ARC transformation factorization.

Given an observation bundle (what changed between input and output),
factorizes it into (ACTION, TARGET, MODIFIER) using Alternating Least Squares.

Key property: Can ONLY output vocabulary items = NO HALLUCINATION.

The system cannot invent new transformations - it can only recognize
patterns that match the predefined vocabulary.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from hologram.arc.encoder import ObjectEncoder
from hologram.config.constants import (
    CONVERGENCE_THRESHOLD,
    MAX_RESONATOR_ITERATIONS,
    OSCILLATION_WINDOW,
)
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity


@dataclass
class TransformResult:
    """
    Result of transformation factorization.

    Attributes:
        action: Best matching action (e.g., "rotate", "translate")
        target: Best matching target (e.g., "all_objects", "red")
        modifier: Best matching modifier (e.g., "90_degrees", "up")
        action_vec: Action hypervector
        target_vec: Target hypervector
        modifier_vec: Modifier hypervector
        iterations: Number of ALS iterations
        converged: Whether resonance converged
        confidence: Per-slot confidence margins
    """
    action: str
    target: str
    modifier: str
    action_vec: torch.Tensor
    target_vec: torch.Tensor
    modifier_vec: torch.Tensor
    iterations: int
    converged: bool
    confidence: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "converged" if self.converged else "max_iter"
        return (
            f"Transform({self.action}({self.target}, {self.modifier}), "
            f"{status} @ {self.iterations} iter, "
            f"conf=[A:{self.confidence.get('action', 0):.2f}, "
            f"T:{self.confidence.get('target', 0):.2f}, "
            f"M:{self.confidence.get('modifier', 0):.2f}])"
        )

    def as_gene(self) -> torch.Tensor:
        """
        Convert to a single "gene" vector for skill memory.

        Returns:
            Bound vector representing this transformation
        """
        return Operations.bind(
            self.action_vec,
            Operations.bind(self.target_vec, self.modifier_vec)
        )

    @property
    def min_confidence(self) -> float:
        """Minimum confidence across all slots."""
        if not self.confidence:
            return 0.0
        return min(self.confidence.values())


class TransformationResonator:
    """
    Extended Resonator for (ACTION, TARGET, MODIFIER) factorization.

    Given an observation bundle (the bundled transformation observations
    from training pairs), uses ALS to find the best factorization into
    action, target, and modifier components.

    Unlike the base Resonator which factorizes thoughts into (S, V, O),
    this factorizes transformations into (ACTION, TARGET, MODIFIER).

    Attributes:
        _encoder: ObjectEncoder for vocabulary access
        _codebook: Codebook for role vectors
        _max_iterations: Maximum ALS iterations
        _convergence_threshold: Similarity for convergence

    Example:
        >>> resonator = TransformationResonator(encoder, codebook)
        >>> result = resonator.resonate(observation_bundle)
        >>> print(result.action, result.target, result.modifier)
        'rotate' 'all_objects' '90_degrees'
    """

    def __init__(
        self,
        encoder: ObjectEncoder,
        codebook: Codebook,
        max_iterations: int = MAX_RESONATOR_ITERATIONS,
        convergence_threshold: float = CONVERGENCE_THRESHOLD,
    ):
        """
        Initialize transformation resonator.

        Args:
            encoder: ObjectEncoder with vocabulary
            codebook: Codebook for role vectors
            max_iterations: Max ALS iterations (default: 100)
            convergence_threshold: Similarity for convergence (default: 0.95)
        """
        self._encoder = encoder
        self._codebook = codebook
        self._ops = Operations
        self._max_iterations = max_iterations
        self._convergence_threshold = convergence_threshold

        # Pre-encode role vectors for ATM (Action-Target-Modifier)
        self._role_action = self._codebook.encode("__ROLE_ACTION__")
        self._role_target = self._codebook.encode("__ROLE_TARGET__")
        self._role_modifier = self._codebook.encode("__ROLE_MODIFIER__")

        # Get vocabularies
        self._action_names, self._action_vectors = encoder.get_action_vocabulary()
        self._target_names, self._target_vectors = encoder.get_target_vocabulary()
        self._modifier_names, self._modifier_vectors = encoder.get_modifier_vocabulary()

    def resonate(self, observation: torch.Tensor) -> TransformResult:
        """
        Main ALS loop to factorize observation into (ACTION, TARGET, MODIFIER).

        Args:
            observation: Bundled transformation observation vector

        Returns:
            TransformResult with factorized components
        """
        # Initialize with superposition of all candidates (maximum uncertainty)
        a = self._superpose(self._action_vectors)   # Action
        t = self._superpose(self._target_vectors)   # Target
        m = self._superpose(self._modifier_vectors) # Modifier

        # Track state history for oscillation detection
        history: List[Tuple[str, str, str]] = []

        for iteration in range(self._max_iterations):
            a_prev, t_prev, m_prev = a.clone(), t.clone(), m.clone()

            # Solve for Action
            a, a_word, a_conf = self._solve_for_slot(
                observation, self._role_action,
                self._action_names, self._action_vectors,
                self._ops.bundle(
                    self._ops.bind(t, self._role_target),
                    self._ops.bind(m, self._role_modifier)
                )
            )

            # Solve for Target
            t, t_word, t_conf = self._solve_for_slot(
                observation, self._role_target,
                self._target_names, self._target_vectors,
                self._ops.bundle(
                    self._ops.bind(a, self._role_action),
                    self._ops.bind(m, self._role_modifier)
                )
            )

            # Solve for Modifier
            m, m_word, m_conf = self._solve_for_slot(
                observation, self._role_modifier,
                self._modifier_names, self._modifier_vectors,
                self._ops.bundle(
                    self._ops.bind(a, self._role_action),
                    self._ops.bind(t, self._role_target)
                )
            )

            # Check convergence
            if self._check_convergence(a, t, m, a_prev, t_prev, m_prev):
                return TransformResult(
                    action=a_word, target=t_word, modifier=m_word,
                    action_vec=a, target_vec=t, modifier_vec=m,
                    iterations=iteration + 1,
                    converged=True,
                    confidence={"action": a_conf, "target": t_conf, "modifier": m_conf}
                )

            # Track history and check for oscillation
            state = (a_word, t_word, m_word)
            history.append(state)

            if self._detect_oscillation(history):
                return TransformResult(
                    action=a_word, target=t_word, modifier=m_word,
                    action_vec=a, target_vec=t, modifier_vec=m,
                    iterations=iteration + 1,
                    converged=False,
                    confidence={"action": a_conf, "target": t_conf, "modifier": m_conf}
                )

        # Max iterations reached - get final words
        _, a_word, a_conf = self._cleanup_with_confidence(
            a, self._action_names, self._action_vectors
        )
        _, t_word, t_conf = self._cleanup_with_confidence(
            t, self._target_names, self._target_vectors
        )
        _, m_word, m_conf = self._cleanup_with_confidence(
            m, self._modifier_names, self._modifier_vectors
        )

        return TransformResult(
            action=a_word, target=t_word, modifier=m_word,
            action_vec=a, target_vec=t, modifier_vec=m,
            iterations=self._max_iterations,
            converged=False,
            confidence={"action": a_conf, "target": t_conf, "modifier": m_conf}
        )

    def verify_factorization(
        self,
        observation: torch.Tensor,
        result: TransformResult,
    ) -> float:
        """
        Verify that factorization matches observation.

        Reconstructs the observation from (A, T, M) and measures
        cosine similarity with the original.

        Args:
            observation: Original observation vector
            result: Factorization result

        Returns:
            Cosine similarity between reconstruction and original
        """
        # Reconstruct: bind(bind(A, role_A), bundle(...))
        reconstructed = self._ops.bundle(
            self._ops.bind(result.action_vec, self._role_action),
            self._ops.bind(result.target_vec, self._role_target),
            self._ops.bind(result.modifier_vec, self._role_modifier),
        )

        return Similarity.cosine(observation, reconstructed)

    def resonate_topk(
        self,
        observation: torch.Tensor,
        k: int = 20,
        slot_k: int = 5,
    ) -> List[TransformResult]:
        """
        Generate top-k candidate transformations via ALS + Cartesian product.

        First runs ALS to convergence (or max iterations), then extracts
        top-k candidates for each slot and generates combinations, re-scoring
        each via verify_factorization.

        Args:
            observation: Bundled transformation observation vector
            k: Maximum number of candidates to return
            slot_k: Top-k candidates to consider per slot (default: 5)

        Returns:
            List of TransformResult candidates, sorted by verification score
        """
        # Run ALS to get converged vectors (reuse existing logic)
        best_result = self.resonate(observation)
        
        # Extract converged vectors from best result
        a_converged = best_result.action_vec
        t_converged = best_result.target_vec
        m_converged = best_result.modifier_vec

        # Get top-k candidates for each slot
        action_candidates = self._get_topk_candidates(
            a_converged, self._action_names, self._action_vectors, slot_k
        )
        target_candidates = self._get_topk_candidates(
            t_converged, self._target_names, self._target_vectors, slot_k
        )
        modifier_candidates = self._get_topk_candidates(
            m_converged, self._modifier_names, self._modifier_vectors, slot_k
        )

        # Generate Cartesian product and score each combination
        candidates = []
        for a_name, a_vec, a_conf in action_candidates:
            for t_name, t_vec, t_conf in target_candidates:
                for m_name, m_vec, m_conf in modifier_candidates:
                    # Create TransformResult for this combination
                    result = TransformResult(
                        action=a_name,
                        target=t_name,
                        modifier=m_name,
                        action_vec=a_vec,
                        target_vec=t_vec,
                        modifier_vec=m_vec,
                        iterations=best_result.iterations,
                        converged=best_result.converged,
                        confidence={
                            "action": a_conf,
                            "target": t_conf,
                            "modifier": m_conf,
                        },
                    )
                    
                    # Score via verification
                    verification_score = self.verify_factorization(observation, result)
                    candidates.append((verification_score, result))

        # Sort by verification score (descending) and return top-k
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [result for _, result in candidates[:k]]

    def _get_topk_candidates(
        self,
        proposal: torch.Tensor,
        vocabulary: List[str],
        vocab_vectors: torch.Tensor,
        k: int,
    ) -> List[Tuple[str, torch.Tensor, float]]:
        """
        Get top-k vocabulary candidates for a proposal vector.

        Args:
            proposal: Proposal vector
            vocabulary: List of vocabulary strings
            vocab_vectors: Stacked vocabulary vectors
            k: Number of top candidates to return

        Returns:
            List of (word, vector, confidence) tuples, sorted by similarity
        """
        similarities = Similarity.cosine_batch(proposal, vocab_vectors)
        topk_indices = torch.topk(similarities, min(k, len(vocabulary))).indices

        candidates = []
        for idx in topk_indices:
            idx_int = int(idx.item())
            word = vocabulary[idx_int]
            vec = vocab_vectors[idx_int]
            conf = float(similarities[idx_int].item())
            candidates.append((word, vec, conf))

        return candidates

    def _solve_for_slot(
        self,
        target: torch.Tensor,
        role: torch.Tensor,
        vocabulary: List[str],
        vocab_vectors: torch.Tensor,
        other_contributions: torch.Tensor,
    ) -> Tuple[torch.Tensor, str, float]:
        """
        Solve for single slot given others.

        Args:
            target: Full target/observation vector
            role: Role vector for this slot
            vocabulary: List of word strings
            vocab_vectors: Pre-encoded vocabulary vectors
            other_contributions: Sum of other (word ⊗ role) contributions

        Returns:
            Tuple of (cleaned vector, word string, confidence margin)
        """
        # Isolate this slot: T - (other contributions)
        remains = target - other_contributions

        # Unbind with role to get raw proposal
        proposal = self._ops.unbind(remains, role)

        # Cleanup to nearest vocabulary item
        return self._cleanup_with_confidence(proposal, vocabulary, vocab_vectors)

    def _cleanup_with_confidence(
        self,
        proposal: torch.Tensor,
        vocabulary: List[str],
        vocab_vectors: torch.Tensor,
    ) -> Tuple[torch.Tensor, str, float]:
        """
        Cleanup operation with confidence calculation.

        Args:
            proposal: Noisy proposal vector
            vocabulary: List of word strings
            vocab_vectors: Pre-encoded vocabulary vectors

        Returns:
            Tuple of (best matching vector, word, confidence margin)
        """
        similarities = Similarity.cosine_batch(proposal, vocab_vectors)

        # Find best match
        best_idx = int(torch.argmax(similarities).item())
        best_word = vocabulary[best_idx]
        best_vec = vocab_vectors[best_idx]

        # Calculate confidence as margin between top-2
        if len(vocabulary) >= 2:
            top2 = torch.topk(similarities, min(2, len(similarities)))
            margin = float((top2.values[0] - top2.values[1]).item())
        else:
            margin = 1.0

        return best_vec, best_word, margin

    def _superpose(self, vectors: torch.Tensor) -> torch.Tensor:
        """Create superposition of all vectors (maximum uncertainty)."""
        return torch.mean(vectors, dim=0)

    def _check_convergence(
        self,
        a: torch.Tensor, t: torch.Tensor, m: torch.Tensor,
        a_prev: torch.Tensor, t_prev: torch.Tensor, m_prev: torch.Tensor,
    ) -> bool:
        """Check if solution has converged."""
        sim_a = Similarity.cosine(a, a_prev)
        sim_t = Similarity.cosine(t, t_prev)
        sim_m = Similarity.cosine(m, m_prev)

        return (
            sim_a >= self._convergence_threshold and
            sim_t >= self._convergence_threshold and
            sim_m >= self._convergence_threshold
        )

    def _detect_oscillation(
        self,
        history: List[Tuple[str, str, str]],
        window: int = OSCILLATION_WINDOW,
    ) -> bool:
        """Detect if solver is oscillating between states."""
        if len(history) < window:
            return False

        recent = history[-window:]
        return len(recent) != len(set(recent))

    def __repr__(self) -> str:
        return (
            f"TransformationResonator(max_iter={self._max_iterations}, "
            f"threshold={self._convergence_threshold}, "
            f"actions={len(self._action_names)}, "
            f"targets={len(self._target_names)}, "
            f"modifiers={len(self._modifier_names)})"
        )
