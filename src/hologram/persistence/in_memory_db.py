"""
InMemoryDatabase: Test implementation of VectorDatabase protocol.

Provides a simple, fast in-memory vector store for unit testing and
development. Implements the full VectorDatabase protocol without disk I/O.

Useful for:
- Testing client code that uses VectorDatabase
- Benchmarking logic without Faiss overhead
- Validation that protocol is complete
- Development before committing to persistence

Does NOT implement persistence (save/load are no-ops), suitable only for
testing and short-lived sessions.
"""

from typing import Dict, List, Tuple, Optional
import torch
import numpy as np


class InMemoryDatabase:
    """
    In-memory vector database for testing.

    Implements VectorDatabase protocol entirely in memory.
    No persistence (save/load are no-ops).

    Attributes:
        dimensions: Vector dimensionality
        vectors: List of stored vectors
        metadata: Dict mapping ID → metadata
        _id_counter: Next ID to assign

    Example:
        >>> db = InMemoryDatabase(dimensions=10000)
        >>> vec = torch.randn(10000)
        >>> vec_id = db.store(vec, {"data": "test"})
        >>>
        >>> query = vec + torch.randn(10000) * 0.1
        >>> results = db.query(query, k=1)
        >>> results[0][0] == vec_id  # Should retrieve same vector
        True
    """

    def __init__(self, dimensions: int):
        """
        Initialize empty in-memory database.

        Args:
            dimensions: Vector dimensionality
        """
        if dimensions < 1:
            raise ValueError(f"Dimensions must be >= 1, got {dimensions}")

        self.dimensions = dimensions
        self.vectors: List[np.ndarray] = []
        self.metadata: Dict[int, dict] = {}
        self._id_counter = 0

    def store(self, vector: torch.Tensor, metadata: dict) -> int:
        """
        Store vector in memory.

        Args:
            vector: Hypervector, shape (dimensions,) or (1, dimensions)
            metadata: Context dictionary

        Returns:
            Integer ID assigned to this vector
        """
        # Validate metadata
        if not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be dict, got {type(metadata).__name__}"
            )

        # Handle shape
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        elif vector.dim() != 2 or vector.shape[0] != 1:
            raise ValueError(
                f"Vector must have shape (d,) or (1, d), got {vector.shape}"
            )

        # Validate dimensions
        if vector.shape[1] != self.dimensions:
            raise ValueError(
                f"Vector dimensionality {vector.shape[1]} != "
                f"expected {self.dimensions}"
            )

        # Convert to numpy and normalize
        vec_np = vector.cpu().detach().numpy().astype('float32').flatten()
        vec_norm = np.linalg.norm(vec_np)

        if vec_norm == 0:
            raise ValueError(
                "Cannot normalize zero vector (all dimensions are 0)"
            )

        vec_np = vec_np / vec_norm  # L2 normalization

        # Store
        self.vectors.append(vec_np)
        current_id = self._id_counter
        self.metadata[current_id] = metadata.copy()
        self._id_counter += 1

        return current_id

    def query(
        self,
        vector: torch.Tensor,
        k: int = 5
    ) -> List[Tuple[int, float, dict]]:
        """
        Find k nearest neighbors (brute force similarity search).

        Args:
            vector: Query hypervector
            k: Number of neighbors to return

        Returns:
            List of (id, similarity, metadata) tuples, sorted by similarity
        """
        # Validate
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        if k > len(self.vectors):
            raise ValueError(
                f"k={k} exceeds total vectors {len(self.vectors)}"
            )

        # Handle shape
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        elif vector.dim() != 2 or vector.shape[0] != 1:
            raise ValueError(
                f"Vector must have shape (d,) or (1, d), got {vector.shape}"
            )

        # Validate dimensions
        if vector.shape[1] != self.dimensions:
            raise ValueError(
                f"Vector dimensionality {vector.shape[1]} != "
                f"expected {self.dimensions}"
            )

        # Normalize query
        vec_np = vector.cpu().detach().numpy().astype('float32').flatten()
        vec_norm = np.linalg.norm(vec_np)

        if vec_norm == 0:
            raise ValueError(
                "Cannot normalize zero vector (all dimensions are 0)"
            )

        vec_np = vec_np / vec_norm

        # Compute cosine similarity with all vectors
        similarities = []
        for idx, stored_vec in enumerate(self.vectors):
            sim = float(np.dot(vec_np, stored_vec))
            similarities.append((idx, sim))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k with metadata
        results = []
        for idx, sim in similarities[:k]:
            results.append((idx, sim, self.metadata[idx]))

        return results

    def save(self) -> None:
        """
        No-op for in-memory database.

        Raises:
            RuntimeError: Always (persistence not supported)
        """
        raise RuntimeError(
            "InMemoryDatabase does not support persistence. "
            "Use FaissAdapter for persistent storage."
        )

    def load(self) -> None:
        """
        No-op for in-memory database.

        Raises:
            RuntimeError: Always (persistence not supported)
        """
        raise RuntimeError(
            "InMemoryDatabase does not support persistence. "
            "Use FaissAdapter for persistent storage."
        )

    @property
    def vector_count(self) -> int:
        """Return total vectors stored."""
        return len(self.vectors)

    def __repr__(self) -> str:
        return (
            f"InMemoryDatabase(dims={self.dimensions}, "
            f"vectors={self.vector_count})"
        )
