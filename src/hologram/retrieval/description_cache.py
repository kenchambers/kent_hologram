"""
LayerDescriptionCache: FAISS-based semantic search over layer descriptions.

Pattern: FaissAdapter (semantic search)

Stores layer descriptions as vectors for O(log n) semantic routing.
"""

from dataclasses import dataclass
from typing import List

import torch

from hologram.core.codebook import Codebook
from hologram.persistence.faiss_adapter import FaissAdapter


@dataclass
class LayerMatch:
    """
    Result of layer description matching.
    
    Attributes:
        layer_id: Layer identifier
        description: Layer description
        similarity: Similarity score
    """
    layer_id: str
    description: str
    similarity: float


class LayerDescriptionCache:
    """
    FAISS-based cache for layer description search.
    
    Pattern: FaissAdapter for O(log n) semantic matching
    
    Methods:
        add_layer: Add layer description to cache
        find_matching_layers: Find layers by semantic similarity
        remove_layer: Remove layer from cache
    """
    
    def __init__(
        self,
        dimensions: int,
        codebook: Codebook,
        persist_path: str,
        use_hnsw: bool = True,
    ):
        """
        Initialize description cache.
        
        Args:
            dimensions: Vector dimensionality
            codebook: Codebook for encoding descriptions
            persist_path: Path for persistence
            use_hnsw: Use HNSW for fast search
        """
        self._codebook = codebook
        self._faiss = FaissAdapter(
            dimensions=dimensions,
            persist_path=persist_path,
            use_hnsw=use_hnsw,
        )
        # Map FAISS ID → (layer_id, description)
        self._id_mapping: dict[int, tuple[str, str]] = {}
    
    def add_layer(
        self,
        layer_id: str,
        description: str,
        description_vec: torch.Tensor = None,
    ) -> None:
        """
        Add layer description to cache.
        
        Args:
            layer_id: Layer identifier
            description: Layer description text
            description_vec: Pre-computed description vector (optional)
        """
        # Encode description if not provided
        if description_vec is None:
            description_vec = self._codebook.encode(description)
        
        # Store in FAISS
        faiss_id = self._faiss.store(
            description_vec,
            {"layer_id": layer_id, "description": description}
        )
        
        # Track mapping
        self._id_mapping[faiss_id] = (layer_id, description)
    
    def find_matching_layers(
        self,
        query_vec: torch.Tensor,
        top_k: int = 3,
    ) -> List[LayerMatch]:
        """
        Find matching layers by semantic similarity.

        Args:
            query_vec: Query vector
            top_k: Number of matches to return

        Returns:
            List of LayerMatch objects
        """
        if self._faiss.vector_count == 0:
            return []

        # Dimension check: ensure query vector matches index dimensions
        expected_dim = self._faiss.dimensions
        query_dim = query_vec.shape[-1] if query_vec.dim() > 0 else 0
        if query_dim != expected_dim:
            # Dimension mismatch - return empty to avoid FAISS crash
            return []

        # Limit k to available vectors
        k = min(top_k, self._faiss.vector_count)

        # Query FAISS
        results = self._faiss.query(query_vec, k=k)
        
        # Convert to LayerMatch objects
        matches = []
        for faiss_id, similarity, metadata in results:
            layer_id = metadata.get("layer_id", "")
            description = metadata.get("description", "")
            
            matches.append(LayerMatch(
                layer_id=layer_id,
                description=description,
                similarity=similarity,
            ))
        
        return matches
    
    def remove_layer(self, layer_id: str) -> bool:
        """
        Remove layer from cache.
        
        Note: FAISS doesn't support deletion, so this is a no-op.
        In production, would rebuild index periodically.
        
        Args:
            layer_id: Layer to remove
        
        Returns:
            False (not implemented for FAISS)
        """
        # FAISS doesn't support deletion efficiently
        # Would need to rebuild index to actually remove
        return False
    
    def save(self) -> None:
        """Save cache to disk."""
        self._faiss.save()
    
    def load(self) -> None:
        """Load cache from disk."""
        self._faiss.load()

        # Rebuild ID mapping directly from FAISS metadata (already loaded)
        self._id_mapping = {}
        for faiss_id, metadata in self._faiss.metadata.items():
            layer_id = metadata.get("layer_id", "")
            description = metadata.get("description", "")
            if layer_id:
                self._id_mapping[faiss_id] = (layer_id, description)
    
    @property
    def layer_count(self) -> int:
        """Get number of cached layers."""
        return self._faiss.vector_count
