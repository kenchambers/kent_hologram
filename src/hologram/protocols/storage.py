"""
VectorDatabase Protocol: Abstract interface for vector storage.

Defines the contract that all vector storage implementations must follow.
Enables swapping between different backends (Faiss, In-memory, SQLite, etc.)
without changing calling code.

This protocol establishes the "storage boundary" in the resonant cavity
architecture, separating vector indexing concerns from query logic.
"""

from typing import Protocol, Tuple, List, Dict, Optional
import torch


class VectorDatabase(Protocol):
    """
    Abstract protocol for persistent vector storage.

    Implementations must:
    1. Store vectors with associated metadata
    2. Support similarity-based retrieval
    3. Guarantee atomic persistence
    4. Validate consistency on load
    5. Handle errors gracefully

    Properties:
    - Deterministic: Same vector always gets same results
    - Persistent: Data survives reload from disk
    - Consistent: Metadata always matches index state
    - Validated: Consistency checked on load
    """

    def store(self, vector: torch.Tensor, metadata: Dict) -> int:
        """
        Store a vector with associated metadata.

        Args:
            vector: Hypervector of shape (dimensions,) or (1, dimensions)
            metadata: Dictionary of context (timestamp, source, etc.)

        Returns:
            Unique integer ID assigned to this vector

        Raises:
            ValueError: If vector shape doesn't match dimensions
            TypeError: If metadata is not a dict

        Example:
            >>> db = FaissAdapter(10000, "/data/index")
            >>> vec = torch.randn(10000)
            >>> vec_id = db.store(vec, {"source": "memory_trace", "epoch": 1})
            >>> # Later: vec_id can be used to retrieve metadata
        """
        ...

    def query(
        self,
        vector: torch.Tensor,
        k: int = 5
    ) -> List[Tuple[int, float, Dict]]:
        """
        Find k nearest neighbors to a query vector.

        Args:
            vector: Query hypervector of shape (dimensions,) or (1, dimensions)
            k: Number of neighbors to return (default: 5)

        Returns:
            List of (id, similarity_score, metadata) tuples
            Sorted by similarity (highest first)

        Raises:
            ValueError: If k > total vectors or k < 1
            ValueError: If vector shape doesn't match dimensions
            RuntimeError: If index not initialized (load not called)

        Properties:
        - Similarity score in range [-1.0, +1.0] (cosine similarity)
        - Higher scores = more similar
        - Results deterministic (same query → same results)
        - Results sorted descending by similarity

        Example:
            >>> db = FaissAdapter(10000, "/data/index")
            >>> db.load()  # Load persisted vectors
            >>> query = torch.randn(10000)
            >>> results = db.query(query, k=5)
            >>> for id, sim, meta in results:
            ...     print(f"ID {id}: similarity={sim:.3f}, source={meta['source']}")
        """
        ...

    def save(self) -> None:
        """
        Persist the index and all metadata to disk.

        Guarantees:
        - Atomic: Either complete save or no write (exception on failure)
        - Consistent: Metadata always matches index count
        - Recoverable: load() can restore exact state
        - Idempotent: Multiple saves with same state are safe

        Implementation must:
        1. Create directory if needed
        2. Save index file (e.g., "index.faiss")
        3. Save metadata with ID counter (CRITICAL!)
        4. Verify consistency before returning
        5. Raise on any failure (don't silently fail)

        Raises:
            IOError: If unable to write to persist_path
            RuntimeError: If internal state is inconsistent

        Example:
            >>> db = FaissAdapter(10000, "/data/index")
            >>> db.store(vec1, {"data": "a"})
            >>> db.store(vec2, {"data": "b"})
            >>> db.save()  # Persists all vectors and metadata
            >>> # Can safely shut down; state will be restored on load()
        """
        ...

    def load(self) -> None:
        """
        Restore index and metadata from persistent storage.

        Guarantees:
        - Idempotent: Can load multiple times, second load is no-op
        - Consistent: Validates metadata matches index before returning
        - Complete: Restores exact state including ID counter
        - Recoverable: Works even if process crashed after partial save

        Implementation must:
        1. Load index file from disk
        2. Load metadata and restore ID counter
        3. Validate consistency (check no gaps, no collisions)
        4. Raise on any inconsistency (don't return partial state)
        5. Leave system ready for store/query operations

        Raises:
            FileNotFoundError: If persistence files don't exist
            ValueError: If metadata/index mismatch detected
            RuntimeError: If persistence files are corrupted

        Example:
            >>> db = FaissAdapter(10000, "/data/index")
            >>> db.load()  # Restores all previous vectors
            >>> # Can immediately query or add new vectors
        """
        ...


class InMemoryDatabase(Protocol):
    """
    Optional: In-memory implementation for testing.

    Useful for:
    - Unit testing without disk I/O
    - Development iteration
    - Benchmarking
    - Validation of interface contract

    Must implement VectorDatabase protocol exactly.
    """
    ...
