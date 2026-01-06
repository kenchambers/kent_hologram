"""
Analogical Reasoning: Solve proportional analogies using HDC vector arithmetic.

Implements proportional analogy solving via hyperdimensional vector arithmetic:
- A is to B as C is to ??? (solve for D)
- Two methods: multiplicative (bind-based) and additive

In the Bentov model, analogies represent relational structure transformations:
the "relation" from A→B can be extracted and applied to other domains.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity


@dataclass
class AnalogyResult:
    """Result of analogy completion.

    Attributes:
        answer: The completed term (D in A:B :: C:D)
        confidence: Similarity score to nearest vocabulary item [0, 1]
        reasoning: Human-readable reasoning string
    """
    answer: str
    confidence: float
    reasoning: str


class AnalogyEngine:
    """
    Solve proportional analogies using HDC vector arithmetic.

    Core principle: A:B :: C:D solves via vector arithmetic:
    - Multiplicative: D = C ⊗ (B ⊗ inv(A))
      (bind C with the relation that transforms A→B)
    - Additive: D = B - A + C
      (vector offset in embedding space)

    Both methods leverage the key insight that HDC binding encodes
    reversible transformations that generalize across domains.

    Examples:
        >>> engine = AnalogyEngine(codebook, vocabulary)
        >>> result = engine.solve("Paris", "France", "Tokyo")
        >>> print(result.answer)  # "Japan"
        >>> result = engine.solve("king", "man", "queen")
        >>> print(result.answer)  # "woman"
    """

    def __init__(
        self,
        codebook: Codebook,
        vocabulary: List[str],
    ):
        """
        Initialize analogy engine.

        Args:
            codebook: Codebook for encoding concepts
            vocabulary: List of vocabulary words for cleanup
        """
        self._codebook = codebook
        self._vocabulary = vocabulary
        self._vocab_vectors: Optional[torch.Tensor] = None

    def solve(
        self,
        a: str,
        b: str,
        c: str,
        method: str = "multiplicative",
    ) -> AnalogyResult:
        """
        Solve: A is to B as C is to ???

        Finds the best completion D such that the relation A→B
        matches the relation C→D.

        Args:
            a: First term (source concept)
            b: Second term (transformed concept)
            c: Third term (source in new domain)
            method: "multiplicative" (bind-based) or "additive" (vector arithmetic)

        Returns:
            AnalogyResult with answer, confidence, and reasoning

        Example:
            >>> result = engine.solve("Paris", "France", "Tokyo", method="multiplicative")
            >>> print(f"{a}:{b} :: {c}:{result.answer}")
            Paris:France :: Tokyo:Japan
        """
        a_vec = self._codebook.encode(a)
        b_vec = self._codebook.encode(b)
        c_vec = self._codebook.encode(c)

        if method == "multiplicative":
            # D = C ⊗ (B ⊗ inv(A))
            # Extract relation R = B ⊗ inv(A) (what transforms A to B)
            # Apply it: D = C ⊗ R (apply same relation to C)
            relation = Operations.bind(b_vec, Operations.inverse(a_vec))
            d_vec = Operations.bind(c_vec, relation)
        elif method == "additive":
            # D = B - A + C (vector offset in embedding)
            d_vec = b_vec - a_vec + c_vec
            # Normalize to unit length
            norm = torch.norm(d_vec)
            if norm > 1e-6:
                d_vec = d_vec / norm
            else:
                # Edge case: cancellation. Fallback to multiplicative
                relation = Operations.bind(b_vec, Operations.inverse(a_vec))
                d_vec = Operations.bind(c_vec, relation)
        else:
            raise ValueError(f"Unknown method: {method}")

        answer, confidence = self._cleanup_to_vocab(d_vec)

        return AnalogyResult(
            answer=answer,
            confidence=confidence,
            reasoning=f"{a}:{b} :: {c}:{answer}",
        )

    def extract_relation(self, a: str, b: str) -> torch.Tensor:
        """
        Extract relation vector representing the transformation A→B.

        The relation vector encodes the semantic transformation from
        concept A to concept B. This can be reused to transform other
        concepts in the same way.

        Args:
            a: Source concept
            b: Target concept

        Returns:
            Relation vector (hypervector encoding A→B transformation)

        Example:
            >>> capital_rel = engine.extract_relation("Paris", "France")
            >>> # This relation encodes: city → country
            >>> # Can be applied to other cities
        """
        a_vec = self._codebook.encode(a)
        b_vec = self._codebook.encode(b)
        # Relation: what multiplied by A gives B?
        # R = B ⊗ inv(A)
        return Operations.bind(b_vec, Operations.inverse(a_vec))

    def apply_relation(
        self,
        relation: torch.Tensor,
        source: str,
    ) -> AnalogyResult:
        """
        Apply a stored relation vector to a new source.

        Takes a previously extracted relation and applies it to
        transform a new source concept, returning the analogous result.

        Args:
            relation: Relation vector (from extract_relation)
            source: Source concept to transform

        Returns:
            AnalogyResult with transformed concept

        Example:
            >>> capital_rel = engine.extract_relation("Paris", "France")
            >>> result = engine.apply_relation(capital_rel, "Tokyo")
            >>> print(result.answer)  # "Japan"
        """
        source_vec = self._codebook.encode(source)
        result_vec = Operations.bind(source_vec, relation)
        answer, confidence = self._cleanup_to_vocab(result_vec)

        return AnalogyResult(
            answer=answer,
            confidence=confidence,
            reasoning=f"apply_relation({source}) → {answer}",
        )

    def _cleanup_to_vocab(self, query: torch.Tensor) -> Tuple[str, float]:
        """
        Find nearest vocabulary item using cosine similarity.

        Performs nearest-neighbor cleanup: finds the vocabulary word
        whose hypervector is most similar to the query vector.

        Args:
            query: Query hypervector

        Returns:
            Tuple of (best_word, confidence_score)
        """
        if self._vocab_vectors is None:
            self._vocab_vectors = self._codebook.encode_batch(self._vocabulary)

        similarities = Similarity.cosine_batch(query, self._vocab_vectors)
        best_idx = int(torch.argmax(similarities).item())
        confidence = float(similarities[best_idx].item())

        return self._vocabulary[best_idx], confidence
