"""
Resonator: Iterative factorization network for HDC.

Implements Alternating Least Squares (ALS) in hyperdimensional space to
decompose a composite "thought" vector into its constituent concepts.

Given a thought vector T, the Resonator finds (subject, verb, object) from
the item memory such that:
    T ≈ (subject ⊗ R_subj) ⊕ (verb ⊗ R_verb) ⊕ (object ⊗ R_obj)

This replaces LLM "decoding" with algebraic constraint solving.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch

from hologram.config.constants import (
    CONVERGENCE_THRESHOLD,
    MAX_RESONATOR_ITERATIONS,
    OSCILLATION_WINDOW,
)
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity


@dataclass
class ResonatorResult:
    """Result of resonator convergence.

    Attributes:
        subject: Best subject hypervector
        verb: Best verb hypervector
        object: Best object hypervector
        subject_word: Matched subject word from vocabulary
        verb_word: Matched verb word from vocabulary
        object_word: Matched object word from vocabulary
        iterations: Number of iterations until convergence
        converged: Whether the resonator reached stable state
        confidence: Per-slot confidence margins (gap between top-2 matches)
    """
    subject: torch.Tensor
    verb: torch.Tensor
    object: torch.Tensor
    subject_word: str
    verb_word: str
    object_word: str
    iterations: int
    converged: bool
    confidence: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "converged" if self.converged else "max_iter"
        return (
            f"ResonatorResult({self.subject_word} {self.verb_word} "
            f"{self.object_word}, {status} @ {self.iterations} iter)"
        )


class Resonator:
    """
    Alternating Least Squares solver for HDC factorization.

    Given a composite thought vector T, iteratively refines guesses for
    (subject, verb, object) until convergence or max iterations.

    The algorithm:
    1. Initialize guesses as superposition of all candidates (noise)
    2. For each iteration:
       a. Solve for subject: isolate subject component, cleanup to nearest word
       b. Solve for verb: use new subject, cleanup to nearest word
       c. Solve for object: use new subject+verb, cleanup to nearest word
       d. Check convergence: if (x, y, z) unchanged from previous, stop

    Attributes:
        _codebook: Codebook for encoding concepts and roles
        _max_iterations: Maximum ALS iterations
        _convergence_threshold: Similarity threshold for convergence
        _ops: Operations class for HDC operations

    Example:
        >>> codebook = Codebook(VectorSpace())
        >>> resonator = Resonator(codebook)
        >>> thought = create_thought_vector(...)
        >>> result = resonator.resonate(thought, nouns, verbs)
        >>> print(f"{result.subject_word} {result.verb_word} {result.object_word}")
    """

    def __init__(
        self,
        codebook: Codebook,
        operations: Type[Operations] = Operations,
        max_iterations: int = MAX_RESONATOR_ITERATIONS,
        convergence_threshold: float = CONVERGENCE_THRESHOLD,
    ):
        """
        Initialize resonator.

        Args:
            codebook: Codebook for encoding concepts
            operations: Operations class (default: Operations)
            max_iterations: Max ALS iterations (default: 100)
            convergence_threshold: Similarity for convergence (default: 0.95)
        """
        self._codebook = codebook
        self._ops = operations
        self._max_iterations = max_iterations
        self._convergence_threshold = convergence_threshold

        # Pre-encode role vectors
        self._role_subject = self._codebook.get_role("SUBJECT")
        self._role_verb = self._codebook.get_role("VERB")
        self._role_object = self._codebook.get_role("OBJECT")

    def resonate(
        self,
        thought: torch.Tensor,
        noun_vocabulary: List[str],
        verb_vocabulary: List[str],
    ) -> ResonatorResult:
        """
        Main ALS loop to factorize thought vector.

        Args:
            thought: Composite thought vector to factorize
            noun_vocabulary: List of noun strings for subject/object
            verb_vocabulary: List of verb strings

        Returns:
            ResonatorResult with factorized components and metadata

        Example:
            >>> result = resonator.resonate(thought, ["cat", "fish"], ["eats", "chases"])
            >>> print(result.subject_word, result.verb_word, result.object_word)
        """
        # Encode vocabularies
        noun_vectors = self._codebook.encode_batch(noun_vocabulary)
        verb_vectors = self._codebook.encode_batch(verb_vocabulary)

        # Initialize with superposition of all candidates (maximum uncertainty)
        x = self._superpose(noun_vectors)  # Subject
        y = self._superpose(verb_vectors)  # Verb
        z = self._superpose(noun_vectors)  # Object

        # Track state history for oscillation detection
        history: List[Tuple[str, str, str]] = []

        for iteration in range(self._max_iterations):
            x_prev, y_prev, z_prev = x.clone(), y.clone(), z.clone()

            # Solve for Subject
            x, x_word, x_conf = self._solve_for_slot(
                thought, self._role_subject, noun_vocabulary, noun_vectors,
                self._ops.bundle(
                    self._ops.bind(y, self._role_verb),
                    self._ops.bind(z, self._role_object)
                ),
                iteration
            )

            # Solve for Verb
            y, y_word, y_conf = self._solve_for_slot(
                thought, self._role_verb, verb_vocabulary, verb_vectors,
                self._ops.bundle(
                    self._ops.bind(x, self._role_subject),
                    self._ops.bind(z, self._role_object)
                ),
                iteration
            )

            # Solve for Object
            z, z_word, z_conf = self._solve_for_slot(
                thought, self._role_object, noun_vocabulary, noun_vectors,
                self._ops.bundle(
                    self._ops.bind(x, self._role_subject),
                    self._ops.bind(y, self._role_verb)
                ),
                iteration
            )

            # Check convergence
            if self._check_convergence(x, y, z, x_prev, y_prev, z_prev):
                return ResonatorResult(
                    subject=x, verb=y, object=z,
                    subject_word=x_word, verb_word=y_word, object_word=z_word,
                    iterations=iteration + 1,
                    converged=True,
                    confidence={"subject": x_conf, "verb": y_conf, "object": z_conf}
                )

            # Track history and check for oscillation
            state = (x_word, y_word, z_word)
            history.append(state)

            if self._detect_oscillation(history):
                # Break oscillation by keeping current best
                return ResonatorResult(
                    subject=x, verb=y, object=z,
                    subject_word=x_word, verb_word=y_word, object_word=z_word,
                    iterations=iteration + 1,
                    converged=False,
                    confidence={"subject": x_conf, "verb": y_conf, "object": z_conf}
                )

        # Get final words if not converged
        _, x_word, x_conf = self._cleanup_with_confidence(x, noun_vocabulary, noun_vectors)
        _, y_word, y_conf = self._cleanup_with_confidence(y, verb_vocabulary, verb_vectors)
        _, z_word, z_conf = self._cleanup_with_confidence(z, noun_vocabulary, noun_vectors)

        return ResonatorResult(
            subject=x, verb=y, object=z,
            subject_word=x_word, verb_word=y_word, object_word=z_word,
            iterations=self._max_iterations,
            converged=False,
            confidence={"subject": x_conf, "verb": y_conf, "object": z_conf}
        )

    def _solve_for_slot(
        self,
        target: torch.Tensor,
        role: torch.Tensor,
        vocabulary: List[str],
        vocab_vectors: torch.Tensor,
        other_contributions: torch.Tensor,
        iteration: int = 0,
    ) -> Tuple[torch.Tensor, str, float]:
        """
        Solve for single slot given others with optional soft cleanup.

        Isolates the slot component by subtracting other contributions,
        then unbinds with role and cleans up to nearest vocabulary item.

        Args:
            target: Full target/thought vector
            role: Role vector for this slot (R_subj, R_verb, R_obj)
            vocabulary: List of word strings
            vocab_vectors: Pre-encoded vocabulary vectors
            other_contributions: Sum of other (word ⊗ role) contributions
            iteration: Current iteration number for annealing schedule

        Returns:
            Tuple of (cleaned vector, word string, confidence margin)
        """
        # Isolate this slot: T - (other contributions)
        remains = target - other_contributions

        # Unbind with role to get raw proposal
        proposal = self._ops.unbind(remains, role)

        # Annealing: soft early, hard late
        if iteration < self._max_iterations // 2:
            # Soft cleanup during exploration
            temperature = 0.5 * (1.0 - iteration / (self._max_iterations // 2)) + 0.1
            soft_vec = self._cleanup_soft(proposal, vocab_vectors, temperature)
            # Still need word and confidence for tracking
            _, word, conf = self._cleanup_with_confidence(proposal, vocabulary, vocab_vectors)
            return soft_vec, word, conf
        else:
            # Hard cleanup during exploitation
            return self._cleanup_with_confidence(proposal, vocabulary, vocab_vectors)

    def _cleanup_with_confidence(
        self,
        proposal: torch.Tensor,
        vocabulary: List[str],
        vocab_vectors: torch.Tensor,
    ) -> Tuple[torch.Tensor, str, float]:
        """
        Cleanup operation with confidence calculation.

        Finds nearest vocabulary item and calculates confidence as the
        margin between the top-2 matches.

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
            margin = 1.0  # Only one option

        return best_vec, best_word, margin

    def _cleanup_soft(
        self,
        proposal: torch.Tensor,
        vocab_vectors: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Soft cleanup: weighted combination of vocabulary vectors.

        Preserves uncertainty by using softmax instead of argmax.
        Temperature controls sharpness (lower = sharper).

        Args:
            proposal: Noisy proposal vector
            vocab_vectors: Vocabulary matrix (V, D)
            temperature: Softmax temperature (default: 0.1)

        Returns:
            Soft-cleaned vector (weighted combination)
        """
        similarities = Similarity.cosine_batch(proposal, vocab_vectors)
        weights = torch.softmax(similarities / temperature, dim=0)
        soft_vec = torch.einsum('v,vd->d', weights, vocab_vectors)

        # Normalize to unit length
        norm = torch.norm(soft_vec)
        if norm > 1e-6:
            soft_vec = soft_vec / norm

        return soft_vec

    def _superpose(self, vectors: torch.Tensor) -> torch.Tensor:
        """Create superposition of all vectors (maximum uncertainty)."""
        return torch.mean(vectors, dim=0)

    def _check_convergence(
        self,
        x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
        x_prev: torch.Tensor, y_prev: torch.Tensor, z_prev: torch.Tensor,
    ) -> bool:
        """Check if solution has converged (unchanged from previous)."""
        sim_x = Similarity.cosine(x, x_prev)
        sim_y = Similarity.cosine(y, y_prev)
        sim_z = Similarity.cosine(z, z_prev)

        return (
            sim_x >= self._convergence_threshold and
            sim_y >= self._convergence_threshold and
            sim_z >= self._convergence_threshold
        )

    def _detect_oscillation(
        self,
        history: List[Tuple[str, str, str]],
        window: int = OSCILLATION_WINDOW,
    ) -> bool:
        """
        Detect if solver is oscillating between states.

        Checks if any state in the history appears twice within the window.

        Args:
            history: List of (subject, verb, object) state tuples
            window: Number of recent states to check

        Returns:
            True if oscillation detected
        """
        if len(history) < window:
            return False

        recent = history[-window:]
        return len(recent) != len(set(recent))

    def __repr__(self) -> str:
        return (
            f"Resonator(max_iter={self._max_iterations}, "
            f"threshold={self._convergence_threshold})"
        )
