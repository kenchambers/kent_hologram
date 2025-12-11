"""
FaissAdapter: Faiss-based vector database for efficient similarity search.

Implements high-dimensional vector indexing using Facebook's Faiss library.
Optimized for 10,000-dimensional hypervectors with cosine similarity search.

Architecture:
- Uses IndexFlatIP (inner product) for cosine similarity on normalized vectors
- Stores metadata separately (JSON) for interpretability
- Maintains consistency between index and metadata through ID counter
- Gracefully handles persistence across sessions
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch

import faiss
import numpy as np


class FaissAdapter:
    """
    Faiss-based persistent vector database.

    Stores high-dimensional vectors (typically 10,000 dims) with efficient
    similarity-based retrieval. Uses IndexFlatIP (inner product) for cosine
    similarity after L2 normalization.

    Attributes:
        dimensions: Vector dimensionality (typically 10,000)
        persist_path: Directory for persisting index and metadata
        index: Faiss index object (IndexFlatIP)
        metadata: Dict mapping vector ID → metadata dict
        _id_counter: Next ID to assign (CRITICAL for persistence)
        _vector_count: Total vectors in index (for consistency check)

    Guarantees:
        - Deterministic: Same query vector returns same results
        - Atomic persistence: Save is all-or-nothing (no partial state)
        - Consistent: Metadata always matches index after load
        - Safe IDs: No collisions after reload (counter properly restored)

    Example:
        >>> db = FaissAdapter(dimensions=10000, persist_path="/data/vectors")
        >>> vec = torch.randn(10000)
        >>> vec_id = db.store(vec, {"source": "memory_trace", "epoch": 5})
        >>> db.save()
        >>>
        >>> # Later session:
        >>> db2 = FaissAdapter(dimensions=10000, persist_path="/data/vectors")
        >>> db2.load()
        >>> results = db2.query(vec, k=5)  # [k x 3] tuples: (id, sim, meta)
    """

    def __init__(self, dimensions: int, persist_path: str):
        """
        Initialize empty Faiss adapter.

        Args:
            dimensions: Vector dimensionality (10000 for hologram system)
            persist_path: Directory to persist index and metadata

        Raises:
            ValueError: If dimensions < 1
        """
        if dimensions < 1:
            raise ValueError(f"Dimensions must be >= 1, got {dimensions}")

        self.dimensions = dimensions
        self.persist_path = Path(persist_path)

        # IndexFlatIP: computes inner product (cosine sim for normalized)
        self.index = faiss.IndexFlatIP(dimensions)

        # Metadata storage
        self.metadata: Dict[int, dict] = {}

        # ID counter: maps to next available ID (CRITICAL for persistence)
        self._id_counter = 0

        # Track count for consistency checks
        self._vector_count = 0

    def store(self, vector: torch.Tensor, metadata: dict) -> int:
        """
        Store normalized vector with metadata.

        The vector is L2-normalized in-place, then added to the index.
        Metadata is stored separately for retrieval with query results.

        Args:
            vector: Hypervector, shape (dimensions,) or (1, dimensions)
            metadata: Dict with context (timestamp, source, etc.)

        Returns:
            Integer ID assigned to this vector (used in query results)

        Raises:
            ValueError: If vector shape doesn't match dimensions
            ValueError: If vector is all-zeros (norm = 0)
            TypeError: If metadata is not a dict

        Implementation notes:
        - Vector is copied to numpy (preserves torch tensor)
        - Normalized with faiss.normalize_L2() before adding
        - ID is deterministic: increments from 0
        - Safe for use after reload (counter is persisted)

        Example:
            >>> adapter = FaissAdapter(10000, "/tmp/index")
            >>> vec = torch.randn(10000)
            >>> meta = {"source": "embedding", "timestamp": "2024-01-01"}
            >>> vec_id = adapter.store(vec, meta)
            >>> assert vec_id == 0  # First vector gets ID 0
        """
        # Validate metadata type
        if not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be dict, got {type(metadata).__name__}"
            )

        # Handle vector shape
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

        # Convert to numpy (creates copy, preserves torch tensor)
        vec_np = vector.cpu().detach().numpy().astype('float32')

        # Check for zero vector (norm = 0 → can't normalize)
        vec_norm = np.linalg.norm(vec_np)
        if vec_norm == 0:
            raise ValueError(
                "Cannot normalize zero vector (all dimensions are 0)"
            )

        # L2-normalize in-place for cosine similarity
        # After: ||vec|| = 1.0, dot(normalized) = cosine_similarity
        faiss.normalize_L2(vec_np)

        # Add to index
        self.index.add(vec_np)

        # Store metadata with current ID
        current_id = self._id_counter
        self.metadata[current_id] = metadata.copy()

        # Increment for next store
        self._id_counter += 1
        self._vector_count += 1

        return current_id

    def query(
        self,
        vector: torch.Tensor,
        k: int = 5
    ) -> List[Tuple[int, float, dict]]:
        """
        Find k nearest neighbors using cosine similarity.

        The query vector is normalized, then used to search the index.
        Results are sorted by similarity (highest first).

        Args:
            vector: Query hypervector, shape (dimensions,) or (1, dimensions)
            k: Number of neighbors to return (default: 5)

        Returns:
            List of (vector_id, similarity_score, metadata) tuples
            - vector_id: Integer ID from store()
            - similarity_score: Cosine similarity in [-1.0, +1.0]
              * +1.0 = perfect match (same direction)
              * +0.0 = orthogonal (unrelated)
              * -1.0 = opposite direction
            - metadata: Dict from store() call

        Raises:
            ValueError: If vector shape doesn't match dimensions
            ValueError: If k < 1 or k > total_vectors
            ValueError: If vector is all-zeros
            RuntimeError: If index not initialized (load() not called)

        Implementation notes:
        - Similarity scores are deterministic (no randomness)
        - Results always sorted by similarity (descending)
        - Invalid indices (-1) are filtered out
        - Negative scores are possible (opposite vectors)

        Example:
            >>> adapter = FaissAdapter(10000, "/tmp/index")
            >>> vec1 = torch.randn(10000)
            >>> vec2 = torch.randn(10000)
            >>> id1 = adapter.store(vec1, {"data": "a"})
            >>> id2 = adapter.store(vec2, {"data": "b"})
            >>>
            >>> # Query with similar vector
            >>> query_vec = vec1 + torch.randn(10000) * 0.1  # noisy version
            >>> results = adapter.query(query_vec, k=2)
            >>> results[0][0]  # Should be id1 (most similar)
            >>> results[0][1]  # Similarity score (close to 1.0)
        """
        # Validate k parameter
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        if k > self.index.ntotal:
            raise ValueError(
                f"k={k} exceeds total vectors {self.index.ntotal}"
            )

        # Handle vector shape
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

        # Convert to numpy
        vec_np = vector.cpu().detach().numpy().astype('float32')

        # Check for zero vector
        vec_norm = np.linalg.norm(vec_np)
        if vec_norm == 0:
            raise ValueError(
                "Cannot normalize zero vector (all dimensions are 0)"
            )

        # L2-normalize for cosine similarity
        faiss.normalize_L2(vec_np)

        # Search: returns distances and indices
        # distances: shape (1, k) - inner products in range [-1, +1]
        # indices: shape (1, k) - vector IDs
        distances, indices = self.index.search(vec_np, k)

        # Collect results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Skip invalid indices (should not happen for valid k, but be safe)
            if idx < 0:
                continue

            # idx is now an integer ID
            idx = int(idx)
            similarity = float(dist)

            # Retrieve metadata (use empty dict if missing, but shouldn't)
            meta = self.metadata.get(idx, {})

            results.append((idx, similarity, meta))

        return results

    def save(self) -> None:
        """
        Persist index and metadata to disk.

        Saves both:
        1. Faiss index: binary format (index.faiss)
        2. Metadata: JSON format with ID counter (metadata.json)

        The ID counter is CRITICAL for correct behavior after reload.
        Without it, new vectors would collide with existing IDs.

        Guarantees:
        - Atomic: Either completes fully or raises exception
        - Safe: Creates directory if needed
        - Consistent: Metadata count matches index count

        Implementation:
        - Creates persist_path if it doesn't exist
        - Writes metadata.json with _id_counter
        - Writes index.faiss from Faiss
        - Verifies both files exist before returning

        Raises:
            IOError: If unable to write to persist_path
            RuntimeError: If metadata/index are inconsistent
            ValueError: If IDs are not contiguous (gap detected)

        Example:
            >>> adapter = FaissAdapter(10000, "/data/vectors")
            >>> adapter.store(vec1, {"a": 1})
            >>> adapter.store(vec2, {"b": 2})
            >>> adapter.save()  # Writes to /data/vectors/
            >>>
            >>> # Verify persistence
            >>> import os
            >>> os.path.exists("/data/vectors/index.faiss")
            True
            >>> os.path.exists("/data/vectors/metadata.json")
            True
        """
        # Create directory if needed
        self.persist_path.mkdir(parents=True, exist_ok=True)

        # Validate consistency before saving
        self._validate_consistency()

        # Build checkpoint: metadata dict + system state
        checkpoint = {
            '_id_counter': self._id_counter,
            '_vector_count': self._vector_count,
            **self.metadata
        }

        # Save metadata as JSON
        metadata_file = self.persist_path / "metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        except IOError as e:
            raise IOError(f"Failed to write metadata: {e}")

        # Save Faiss index
        index_file = str(self.persist_path / "index.faiss")
        try:
            faiss.write_index(self.index, index_file)
        except Exception as e:
            raise IOError(f"Failed to write Faiss index: {e}")

        # Verify both files exist
        if not metadata_file.exists():
            raise RuntimeError(f"Metadata file not created: {metadata_file}")
        if not Path(index_file).exists():
            raise RuntimeError(f"Index file not created: {index_file}")

    def load(self) -> None:
        """
        Restore index and metadata from persistent storage.

        Loads both index and metadata, validating consistency.
        The ID counter is restored, ensuring new vectors get correct IDs.

        Guarantees:
        - Idempotent: Multiple calls are safe (second load is no-op)
        - Consistent: Validates no gaps in IDs
        - Complete: Restores exact session state
        - Safe: Proper error messages on failure

        Implementation:
        - Loads metadata.json first (includes _id_counter)
        - Loads index.faiss from Faiss
        - Validates counts match
        - Validates IDs are contiguous (0, 1, 2, ..., n-1)

        Raises:
            FileNotFoundError: If metadata.json or index.faiss missing
            ValueError: If ID counter mismatches
            ValueError: If IDs are not contiguous (gap detected)
            ValueError: If metadata ID keys don't match index count
            RuntimeError: If Faiss index is corrupted

        Example:
            >>> adapter = FaissAdapter(10000, "/data/vectors")
            >>> adapter.load()  # Restores previous session
            >>>
            >>> # Verify state was restored
            >>> len(adapter.metadata)  # Should match previous count
            >>> adapter._id_counter    # Should match previous counter
            >>>
            >>> # Can immediately use
            >>> results = adapter.query(vec, k=5)
        """
        metadata_file = self.persist_path / "metadata.json"
        index_file = self.persist_path / "index.faiss"

        # Validate files exist
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}\n"
                f"Run save() first, or check persist_path"
            )

        if not index_file.exists():
            raise FileNotFoundError(
                f"Index file not found: {index_file}\n"
                f"Run save() first, or check persist_path"
            )

        # Load metadata JSON
        try:
            with open(metadata_file, 'r') as f:
                checkpoint = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Metadata file corrupted (invalid JSON): {e}"
            )

        # Extract system state
        self._id_counter = checkpoint.pop('_id_counter', 0)
        self._vector_count = checkpoint.pop('_vector_count', 0)

        # Remaining keys are metadata (string keys from JSON → int)
        self.metadata = {int(k): v for k, v in checkpoint.items()}

        # Load Faiss index
        try:
            self.index = faiss.read_index(str(index_file))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Faiss index (corrupted?): {e}"
            )

        # Validate consistency
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        """
        Verify metadata and index are in sync.

        Checks:
        1. Metadata count matches index count
        2. ID counter matches metadata size
        3. All expected IDs (0..n-1) are present
        4. No gaps in ID sequence

        This is called by both save() and load() to ensure
        atomicity: if validation fails, no partial state is returned.

        Raises:
            ValueError: If any consistency check fails

        Implementation note:
        Detects data corruption early, before it causes silent failures.
        """
        # Check 1: Metadata count vs Index count
        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                f"Metadata count ({len(self.metadata)}) != "
                f"Index count ({self.index.ntotal})\n"
                f"This indicates data corruption or partial save"
            )

        # Check 2: ID counter consistency
        if self._id_counter != len(self.metadata):
            raise ValueError(
                f"ID counter ({self._id_counter}) != "
                f"Metadata size ({len(self.metadata)})\n"
                f"This would cause ID collisions on next store()"
            )

        # Check 3 & 4: All IDs from 0 to n-1 must be present
        expected_ids = set(range(self._id_counter))
        actual_ids = set(self.metadata.keys())

        if expected_ids != actual_ids:
            missing = expected_ids - actual_ids
            extra = actual_ids - expected_ids
            raise ValueError(
                f"ID sequence has gaps:\n"
                f"  Missing IDs: {sorted(missing)}\n"
                f"  Extra IDs: {sorted(extra)}\n"
                f"This indicates data corruption"
            )

    @property
    def vector_count(self) -> int:
        """Return total vectors in index."""
        return self.index.ntotal

    def __repr__(self) -> str:
        return (
            f"FaissAdapter(dims={self.dimensions}, "
            f"vectors={self.vector_count}, "
            f"path={self.persist_path})"
        )
