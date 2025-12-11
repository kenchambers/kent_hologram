"""
Similarity: Functions for measuring vector similarity (resonance).

In the Bentov holographic model, similarity = resonance strength.
Higher similarity means the query vector "vibrates" more strongly
with the memory pattern.
"""

import torch
import torchhd


class Similarity:
    """
    Similarity functions for resonance detection.

    All similarity computations are deterministic (no randomness),
    which is critical for the "bounded hallucination" property.

    The primary metric is cosine similarity, which measures the
    angle between vectors independent of their magnitude.
    """

    @staticmethod
    def cosine(query: torch.Tensor, memory: torch.Tensor) -> float:
        """
        Cosine similarity between two vectors.

        Fully deterministic - same inputs always produce same output.
        This is critical for zero-hallucination: we can mathematically
        prove whether a fact is in memory.

        Returns value in [-1, 1] where:
        - 1.0: Perfect match (same direction)
        - 0.0: Orthogonal (no relation)
        - -1.0: Opposite direction

        Args:
            query: Query hypervector
            memory: Memory hypervector

        Returns:
            Cosine similarity in range [-1, 1]

        Example:
            >>> sim = Similarity()
            >>> v1 = torch.randn(10000)
            >>> v2 = v1  # Same vector
            >>> sim.cosine(v1, v2)
            1.0
        """
        result = torchhd.cosine_similarity(query, memory)
        # torchhd returns a tensor, extract scalar
        if isinstance(result, torch.Tensor):
            if result.dim() == 0:  # Scalar tensor
                return float(result.item())
            else:
                # Should not happen for single vectors, but handle gracefully
                return float(result[0].item())
        return float(result)

    @staticmethod
    def cosine_batch(
        query: torch.Tensor,
        candidates: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch cosine similarity against multiple candidates.

        Efficiently computes similarity between one query and many
        candidate vectors. This is the core of the "cleanup" operation.

        Args:
            query: Single query vector of shape (dimensions,)
            candidates: Batch of vectors of shape (n_candidates, dimensions)

        Returns:
            Tensor of shape (n_candidates,) with similarity scores

        Example:
            >>> sim = Similarity()
            >>> query = torch.randn(10000)
            >>> candidates = torch.randn(100, 10000)  # 100 possible answers
            >>> scores = sim.cosine_batch(query, candidates)
            >>> best_idx = torch.argmax(scores)
            >>> # candidates[best_idx] is the closest match
        """
        return torchhd.cosine_similarity(query, candidates)

    @staticmethod
    def resonance_strength(similarity: float) -> float:
        """
        Convert raw similarity to resonance strength [0, 1].

        Cosine similarity ranges from [-1, 1], but for resonance
        interpretation we map it to [0, 1] where:
        - 0.0: No resonance (orthogonal or opposite)
        - 1.0: Maximum resonance (perfect match)

        Args:
            similarity: Cosine similarity in range [-1, 1]

        Returns:
            Resonance strength in range [0, 1]

        Example:
            >>> sim = Similarity()
            >>> sim.resonance_strength(1.0)  # Perfect match
            1.0
            >>> sim.resonance_strength(0.0)  # Orthogonal
            0.5
            >>> sim.resonance_strength(-1.0)  # Opposite
            0.0
        """
        # Map [-1, 1] to [0, 1]
        return (similarity + 1.0) / 2.0

    @staticmethod
    def above_threshold(similarity: float, threshold: float) -> bool:
        """
        Check if similarity exceeds confidence threshold.

        This is the decision boundary for "respond" vs "refuse".

        Args:
            similarity: Cosine similarity value
            threshold: Minimum acceptable similarity

        Returns:
            True if similarity >= threshold

        Example:
            >>> sim = Similarity()
            >>> sim.above_threshold(0.85, threshold=0.6)
            True
            >>> sim.above_threshold(0.45, threshold=0.6)
            False
        """
        return similarity >= threshold

    @staticmethod
    def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> int:
        """
        Hamming distance for binary vectors.

        Note: This is primarily for BSC (Binary Spatter Code) VSA model.
        For MAP model (our default), cosine similarity is preferred.

        Args:
            a: First binary vector
            b: Second binary vector

        Returns:
            Number of differing bits

        Example:
            >>> sim = Similarity()
            >>> a = torch.tensor([1, 0, 1, 0])
            >>> b = torch.tensor([1, 1, 1, 0])
            >>> sim.hamming_distance(a, b)
            1
        """
        return int(torch.sum(a != b).item())

    @staticmethod
    def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        """
        Euclidean (L2) distance between vectors.

        Note: For high-dimensional spaces, cosine similarity is usually
        more meaningful than Euclidean distance due to the curse of
        dimensionality. This is provided for completeness.

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            L2 distance

        Example:
            >>> sim = Similarity()
            >>> a = torch.zeros(10000)
            >>> b = torch.ones(10000)
            >>> dist = sim.euclidean_distance(a, b)
            >>> # dist ≈ sqrt(10000) ≈ 100
        """
        return float(torch.norm(a - b, p=2).item())
