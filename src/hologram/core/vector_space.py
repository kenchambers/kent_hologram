"""
VectorSpace: Configuration for hyperdimensional vector space.

The VectorSpace is an immutable configuration object that defines the
dimensionality and properties of the hypervector space. It's shared across
all components to ensure dimensional consistency.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VectorSpace:
    """
    Immutable configuration for hyperdimensional space (CPU-only).

    All hypervectors in the system must belong to the same VectorSpace
    to ensure they can be compared and combined correctly.

    Attributes:
        dimensions: Number of dimensions in the hypervector space (default: 10000)
        dtype: PyTorch data type for vectors (default: float32)

    Example:
        >>> space = VectorSpace(dimensions=10000)
        >>> v1 = space.empty_vector()
        >>> v2 = space.random_vector(seed=42)
        >>> v1.shape[0] == v2.shape[0] == 10000
        True
    """

    dimensions: int = 10000
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.dimensions < 100:
            raise ValueError(f"Dimensions must be >= 100, got {self.dimensions}")
        if self.dimensions > 100000:
            raise ValueError(f"Dimensions must be <= 100000, got {self.dimensions}")

    def empty_vector(self) -> torch.Tensor:
        """
        Create a zero-initialized hypervector.

        Returns:
            Tensor of shape (dimensions,) filled with zeros.
        """
        return torch.zeros(self.dimensions, dtype=self.dtype)

    def random_vector(self, seed: int) -> torch.Tensor:
        """
        Create a random hypervector with deterministic seed.

        This produces a reproducible random vector using a seeded generator.
        Same seed always produces the same vector.
        
        CRITICAL: Returns unit vector (norm=1) to ensure consistent magnitude
        with Operations.bind/bundle results.

        Args:
            seed: Integer seed for random number generation.

        Returns:
            Tensor of shape (dimensions,) with values from N(0, 1/D).
            Norm is exactly 1.0.

        Example:
            >>> space = VectorSpace(dimensions=100)
            >>> v1 = space.random_vector(seed=42)
            >>> torch.norm(v1)
            tensor(1.)
        """
        gen = torch.Generator().manual_seed(seed)
        vec = torch.randn(self.dimensions, dtype=self.dtype, generator=gen)
        
        # Normalize to unit length
        norm = torch.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
            
        return vec

    def validate_vector(self, vector: torch.Tensor) -> None:
        """
        Validate that a vector belongs to this space.

        Args:
            vector: Tensor to validate.

        Raises:
            ValueError: If vector has wrong shape or dtype.
        """
        if vector.dim() != 1:
            raise ValueError(f"Vector must be 1D, got shape {vector.shape}")
        if vector.shape[0] != self.dimensions:
            raise ValueError(
                f"Vector has {vector.shape[0]} dimensions, "
                f"expected {self.dimensions}"
            )
        if vector.dtype != self.dtype:
            raise ValueError(
                f"Vector has dtype {vector.dtype}, expected {self.dtype}"
            )

    def __repr__(self) -> str:
        return f"VectorSpace(dimensions={self.dimensions}, dtype={self.dtype})"
