"""
SemanticCodebook: Semantic-aware hypervector generation.

Extends Codebook with semantic embeddings to allow semantically similar
concepts (e.g., "dog" and "puppy") to be close in vector space, rather
than orthogonal as with random vectors.

Uses sentence-transformers to get semantic embeddings, then projects
them into the hyperdimensional space using a fixed random projection.
"""

from typing import Dict, Optional

import torch

from hologram.core.codebook import Codebook
from hologram.core.vector_space import VectorSpace


class SemanticCodebook(Codebook):
    """
    Semantic-aware codebook using embeddings + projection.

    Maps concepts to hypervectors that preserve semantic similarity:
    - "dog" and "puppy" will be close in vector space
    - "France" and "Paris" will be close
    - Still deterministic: same concept → same vector

    Uses sentence-transformers/all-MiniLM-L6-v2 (384d) and projects
    to HDC space (10000d) via a fixed random projection matrix.

    Attributes:
        _space: The VectorSpace configuration
        _cache: Memoization cache for generated vectors
        _embedding_model: Loaded sentence transformer model (lazy)
        _projection_matrix: Fixed projection matrix (384d → 10000d)

    Example:
        >>> space = VectorSpace(dimensions=10000)
        >>> codebook = SemanticCodebook(space)
        >>> dog = codebook.encode("dog")
        >>> puppy = codebook.encode("puppy")
        >>> cosine_sim(dog, puppy)  # Should be > 0.5 (semantically similar)
        ~0.7
        >>> cat = codebook.encode("cat")
        >>> cosine_sim(dog, cat)  # Less similar but still > 0.0
        ~0.3
    """

    def __init__(self, space: VectorSpace, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic codebook.

        Args:
            space: VectorSpace defining dimensionality and dtype
            model_name: Sentence transformer model name (default: all-MiniLM-L6-v2)

        Raises:
            ImportError: If sentence-transformers not installed
        """
        super().__init__(space)
        self._model_name = model_name
        self._embedding_model: Optional[object] = None
        self._projection_matrix: Optional[torch.Tensor] = None
        
        # Check for dependency at initialization so container can catch it
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def encode(self, concept: str) -> torch.Tensor:
        """
        Generate semantic hypervector for a concept.

        Uses semantic embeddings + projection to HDC space.
        Results are cached for efficiency.

        Args:
            concept: String representing the concept to encode

        Returns:
            Semantic hypervector of shape (dimensions,)

        Example:
            >>> codebook = SemanticCodebook(VectorSpace())
            >>> france = codebook.encode("France")
            >>> france.shape
            torch.Size([10000])
        """
        if concept not in self._cache:
            # Get semantic embedding
            embedding = self._get_embedding(concept)

            # Project to HDC space
            hdc_vector = self._project_embedding(embedding)

            # Normalize to unit length (standard for HDC)
            norm = torch.norm(hdc_vector)
            if norm > 0:
                hdc_vector = hdc_vector / norm

            self._cache[concept] = hdc_vector

        return self._cache[concept]

    def _get_embedding(self, concept: str) -> torch.Tensor:
        """
        Get semantic embedding for a concept.

        Uses sentence-transformers to get 384d embedding.

        Args:
            concept: Input string

        Returns:
            Embedding tensor of shape (384,)
        """
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self._model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

        # Get embedding (returns numpy array, convert to torch)
        embedding_np = self._embedding_model.encode(concept, convert_to_numpy=True)
        return torch.tensor(embedding_np, dtype=self._space.dtype)

    def _project_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Project semantic embedding to HDC space.

        Uses a fixed, seeded random projection matrix to map 384d → 10000d.
        The projection matrix is deterministic (same seed always).

        Args:
            embedding: Semantic embedding tensor of shape (384,)

        Returns:
            Projected vector of shape (dimensions,)
        """
        if self._projection_matrix is None:
            # Create fixed projection matrix using seed
            # Seed chosen to be deterministic but different from concept hashing
            gen = torch.Generator().manual_seed(0xCAFEBABE)  # Fixed seed
            self._projection_matrix = torch.randn(
                self._space.dimensions,
                embedding.shape[0],  # 384
                dtype=self._space.dtype,
                generator=gen
            )

        # Project: (10000, 384) @ (384,) = (10000,)
        return torch.matmul(self._projection_matrix, embedding)

    def __repr__(self) -> str:
        return (
            f"SemanticCodebook(dimensions={self._space.dimensions}, "
            f"model={self._model_name}, cached={self.cache_size()})"
        )
