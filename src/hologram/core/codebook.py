"""
Codebook: Deterministic hypervector generation.

Maps concepts (strings) to reproducible random hypervectors using
hash-seeded random generation. This ensures that the same concept
always maps to the same vector, which is critical for consistency.

In the Bentov model, each concept is a "pebble" - a unique disturbance
pattern that can be dropped into the water (memory).
"""

import hashlib
from typing import Dict

import torch

from hologram.core.vector_space import VectorSpace


class Codebook:
    """
    Deterministic hypervector generation for concepts.

    Maps strings to deterministic random hypervectors. The mapping is
    reproducible: the same concept always produces the same vector,
    but different concepts produce orthogonal vectors.

    This implements the "pebble" metaphor from Bentov's model - each
    concept creates a unique, deterministic disturbance pattern.

    Attributes:
        _space: The VectorSpace configuration
        _cache: Memoization cache for generated vectors

    Example:
        >>> space = VectorSpace(dimensions=10000)
        >>> codebook = Codebook(space)
        >>> v1 = codebook.encode("apple")
        >>> v2 = codebook.encode("apple")
        >>> torch.allclose(v1, v2)  # Same concept = same vector
        True
        >>> v3 = codebook.encode("orange")
        >>> cosine_sim(v1, v3)  # Different concepts ≈ orthogonal
        ~0.0
    """

    def __init__(self, space: VectorSpace):
        """
        Initialize codebook with a vector space.

        Args:
            space: VectorSpace defining dimensionality and dtype
        """
        self._space = space
        self._cache: Dict[str, torch.Tensor] = {}

    def encode(self, concept: str) -> torch.Tensor:
        """
        Generate deterministic hypervector for a concept.

        Uses: hash(concept) -> seed -> torch.randn(seed) -> vector
        Results are cached for efficiency.

        Args:
            concept: String representing the concept to encode

        Returns:
            Deterministic hypervector of shape (dimensions,)

        Example:
            >>> codebook = Codebook(VectorSpace())
            >>> france = codebook.encode("France")
            >>> france.shape
            torch.Size([10000])
        """
        if concept not in self._cache:
            seed = self._hash_to_seed(concept)
            self._cache[concept] = self._space.random_vector(seed)
        return self._cache[concept]

    def encode_batch(self, concepts: list[str]) -> torch.Tensor:
        """
        Batch encode multiple concepts.

        Args:
            concepts: List of concept strings

        Returns:
            Tensor of shape (len(concepts), dimensions)

        Example:
            >>> codebook = Codebook(VectorSpace())
            >>> vectors = codebook.encode_batch(["apple", "orange", "banana"])
            >>> vectors.shape
            torch.Size([3, 10000])
        """
        vectors = [self.encode(concept) for concept in concepts]
        return torch.stack(vectors)

    def get_positional(self, position: int) -> torch.Tensor:
        """
        Get positional vector for sequence encoding.

        Returns vector for position marker __POS_{position}__.
        Used by SequenceEncoder to encode word order.

        Args:
            position: Integer position in sequence (0-indexed)

        Returns:
            Positional hypervector

        Example:
            >>> codebook = Codebook(VectorSpace())
            >>> pos0 = codebook.get_positional(0)
            >>> pos1 = codebook.get_positional(1)
            >>> # pos0 and pos1 are orthogonal markers for different positions
        """
        return self.encode(f"__POS_{position}__")

    def get_role(self, role: str) -> torch.Tensor:
        """
        Get grammatical role vector.

        Returns vector for role marker __ROLE_{role}__ (e.g., SUBJECT, VERB, OBJECT).
        Used for grammar template-based generation.

        Args:
            role: Grammatical role name (e.g., "SUBJECT", "VERB", "OBJECT")

        Returns:
            Role hypervector

        Example:
            >>> codebook = Codebook(VectorSpace())
            >>> subj = codebook.get_role("SUBJECT")
            >>> verb = codebook.get_role("VERB")
            >>> # subj and verb are orthogonal markers for different roles
        """
        return self.encode(f"__ROLE_{role}__")

    def clear_cache(self) -> None:
        """
        Clear the memoization cache.

        Useful for memory management in long-running systems.
        Cleared vectors can be regenerated deterministically.
        """
        self._cache.clear()

    def cache_size(self) -> int:
        """
        Get number of cached vectors.

        Returns:
            Number of concepts currently in cache
        """
        return len(self._cache)

    def _hash_to_seed(self, concept: str) -> int:
        """
        Convert concept string to deterministic seed.

        Uses SHA-256 hash for uniform distribution and collision resistance.
        Hash is truncated to 32 bits for use as torch random seed.

        Args:
            concept: Input string

        Returns:
            Integer seed in range [0, 2^32-1]
        """
        hash_bytes = hashlib.sha256(concept.encode('utf-8')).hexdigest()
        # Take first 8 hex chars (32 bits) and convert to int
        seed = int(hash_bytes[:8], 16)
        return seed

    def __repr__(self) -> str:
        return (
            f"Codebook(dimensions={self._space.dimensions}, "
            f"cached={self.cache_size()})"
        )
