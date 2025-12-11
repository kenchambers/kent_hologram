"""
FractalSpace: Holographic vector space with fractal expansion.

Implements the "Fractal Shard" protocol where every vector is a deterministic
expansion of a small "DNA" seed. This creates true holographic properties:
- Any fragment contains the whole (lower resolution)
- Robust to corruption and slicing
- Mathematically recoverable from shards

Inspired by Bentov's holographic consciousness model: "If you cut a hologram
in half, you don't get half a rose. You get the whole rose, just fuzzier."
"""

import torch
from typing import Optional

from hologram.core.vector_space import VectorSpace


class FractalSpace(VectorSpace):
    """
    VectorSpace where every vector is a fractal expansion of a DNA seed.
    
    Instead of generating random 10,000-dim vectors directly, we:
    1. Generate a 64-dim "DNA" seed vector
    2. Expand it to 10,000 dims using deterministic rotation matrices
    3. Each 64-dim block contains the "whole" concept (rotated)
    
    This creates true holographic properties: any fragment can recover
    the original DNA, enabling graceful degradation instead of garbage.
    
    Attributes:
        dimensions: Total dimensions (default: 10000)
        dtype: PyTorch data type (default: float32)
        dna_dimensions: Size of DNA seed (default: 64)
        _rotation_matrices: Cached rotation matrices for expansion
    
    Example:
        >>> space = FractalSpace(dimensions=10000)
        >>> v1 = space.random_vector(seed=42)
        >>> v1.shape
        torch.Size([10000])
        >>> # Extract block 78 using helper (recommended)
        >>> shard = space.extract_block(v1, block_index=78)
        >>> # Recover DNA from shard
        >>> recovered = space.recover_dna(shard, block_index=78)
        >>> # recovered ≈ original DNA (cosine similarity > 0.9)
    """
    
    dna_dimensions: int = 64
    
    def __init__(self, dimensions: int = 10000, dtype: torch.dtype = torch.float32, dna_dimensions: int = 64):
        """
        Initialize FractalSpace.
        
        Args:
            dimensions: Total vector dimensions (default: 10000)
            dtype: PyTorch data type (default: float32)
            dna_dimensions: Size of DNA seed (default: 64)
        """
        # Call parent __init__ manually since we're overriding __post_init__
        object.__setattr__(self, 'dimensions', dimensions)
        object.__setattr__(self, 'dtype', dtype)
        object.__setattr__(self, 'dna_dimensions', dna_dimensions)
        
        # Validate
        if dimensions < 100:
            raise ValueError(f"Dimensions must be >= 100, got {dimensions}")
        if dimensions > 100000:
            raise ValueError(f"Dimensions must be <= 100000, got {dimensions}")
        if dna_dimensions < 8:
            raise ValueError(f"DNA dimensions must be >= 8, got {dna_dimensions}")
        if dna_dimensions > dimensions:
            raise ValueError(f"DNA dimensions ({dna_dimensions}) must be <= total dimensions ({dimensions})")
        
        # Cache for rotation matrices (computed lazily)
        self._rotation_matrices: dict[int, torch.Tensor] = {}
    
    def random_vector(self, seed: int) -> torch.Tensor:
        """
        Generate fractal vector from seed.
        
        Creates a 64-dim DNA seed, then expands it to full dimensions
        using deterministic rotation matrices. This ensures:
        - Same seed → same vector (deterministic)
        - Any 64-dim block can recover the DNA
        - Cosine similarity preserved (rotations are orthogonal)
        
        Args:
            seed: Integer seed for random number generation
            
        Returns:
            Fractal hypervector of shape (dimensions,)
            Norm is exactly 1.0 (unit vector)
        """
        # Step 1: Generate DNA seed (64-dim unit vector)
        gen = torch.Generator().manual_seed(seed)
        dna = torch.randn(self.dna_dimensions, dtype=self.dtype, generator=gen)
        
        # Normalize DNA to unit length
        norm = torch.norm(dna)
        if norm > 1e-6:
            dna = dna / norm
        else:
            # Fallback: use identity vector if norm is too small
            dna = torch.ones(self.dna_dimensions, dtype=self.dtype)
            dna = dna / torch.norm(dna)
        
        # Step 2: Expand DNA to full dimensions using rotations
        return self._expand(dna)
    
    def _expand(self, dna: torch.Tensor) -> torch.Tensor:
        """
        Expand DNA vector to full dimensions via rotation matrices.
        
        Divides the full space into blocks of size dna_dimensions.
        Block 0 is the original DNA.
        Block k is R_k @ DNA (where R_k is a deterministic rotation matrix).
        
        This creates "deterministic echoes" - each block contains the whole
        concept, just rotated into different subspaces.
        
        Args:
            dna: DNA seed vector of shape (dna_dimensions,)
            
        Returns:
            Expanded vector of shape (dimensions,)
        """
        n_blocks = (self.dimensions + self.dna_dimensions - 1) // self.dna_dimensions
        blocks = []
        
        for k in range(n_blocks):
            if k == 0:
                # Block 0 is the original DNA (no rotation)
                blocks.append(dna)
            else:
                # Block k is rotated DNA
                rotation_matrix = self._get_rotation_matrix(k)
                rotated = rotation_matrix @ dna
                blocks.append(rotated)
        
        # Concatenate all blocks
        full = torch.cat(blocks)
        
        # Trim to exact dimensions (in case of rounding)
        full = full[:self.dimensions]
        
        # Normalize to unit length
        norm = torch.norm(full)
        if norm > 1e-6:
            full = full / norm
        else:
            # Fallback: use identity
            full = torch.ones(self.dimensions, dtype=self.dtype)
            full = full / torch.norm(full)
        
        return full
    
    def _get_rotation_matrix(self, block_index: int) -> torch.Tensor:
        """
        Get deterministic rotation matrix for a block index.
        
        Uses a seeded random generator to create orthogonal rotation matrices.
        Same block_index always produces the same rotation matrix.
        
        Args:
            block_index: Index of the block (1, 2, 3, ...)
            
        Returns:
            Rotation matrix of shape (dna_dimensions, dna_dimensions)
            Property: R^T @ R = I (orthogonal)
        """
        if block_index in self._rotation_matrices:
            return self._rotation_matrices[block_index]
        
        # Generate deterministic rotation matrix
        # Use block_index as seed for reproducibility
        gen = torch.Generator().manual_seed(block_index + 1000000)  # Offset to avoid seed collision
        
        # Generate random matrix
        A = torch.randn(self.dna_dimensions, self.dna_dimensions, dtype=self.dtype, generator=gen)
        
        # QR decomposition to get orthogonal matrix
        Q, R = torch.linalg.qr(A)
        
        # Ensure determinant is +1 (proper rotation, not reflection)
        if torch.det(Q) < 0:
            Q[:, -1] *= -1
        
        self._rotation_matrices[block_index] = Q
        return Q
    
    def recover_dna(self, shard: torch.Tensor, block_index: int) -> torch.Tensor:
        """
        Recover DNA from any block (the "holographic" property).
        
        Given a 64-dim shard from block k, applies the inverse rotation
        to recover the original DNA. This is the key property: any
        fragment contains the whole concept.
        
        Args:
            shard: Vector shard of shape (dna_dimensions,) or larger
            block_index: Which block this shard came from (0 = no rotation)
            
        Returns:
            Recovered DNA vector of shape (dna_dimensions,)
            
        Example:
            >>> space = FractalSpace()
            >>> v = space.random_vector(seed=42)
            >>> shard = space.extract_block(v, block_index=78)  # Recommended
            >>> dna = space.recover_dna(shard, block_index=78)
            >>> # dna ≈ original DNA used to generate v
        """
        # Extract exactly dna_dimensions from shard
        if shard.shape[0] > self.dna_dimensions:
            shard = shard[:self.dna_dimensions]
        elif shard.shape[0] < self.dna_dimensions:
            # Pad with zeros if shard is too small
            padding = torch.zeros(self.dna_dimensions - shard.shape[0], dtype=self.dtype)
            shard = torch.cat([shard, padding])
        
        if block_index == 0:
            # Block 0 is unrotated DNA
            return shard
        
        # Apply inverse rotation: DNA = R^T @ shard
        rotation_matrix = self._get_rotation_matrix(block_index)
        # For orthogonal matrices, inverse = transpose
        recovered = rotation_matrix.T @ shard
        
        # Normalize
        norm = torch.norm(recovered)
        if norm > 1e-6:
            recovered = recovered / norm
        
        return recovered

    def extract_block(self, vector: torch.Tensor, block_index: int) -> torch.Tensor:
        """
        Extract a DNA-sized block from a vector.

        Use this method to correctly extract blocks for DNA recovery.
        Block k spans indices [k * dna_dimensions, (k+1) * dna_dimensions).

        Args:
            vector: Full vector of shape (dimensions,)
            block_index: Which block to extract (0-indexed)

        Returns:
            Block of shape (dna_dimensions,) or smaller for truncated last block

        Example:
            >>> space = FractalSpace(dimensions=10000)
            >>> v = space.random_vector(seed=42)
            >>> block = space.extract_block(v, block_index=78)
            >>> dna = space.recover_dna(block, block_index=78)
        """
        start = block_index * self.dna_dimensions
        end = min(start + self.dna_dimensions, len(vector))
        return vector[start:end]

    def recover_dna_from_multiple_shards(
        self,
        shards: list[torch.Tensor],
        block_indices: list[int]
    ) -> torch.Tensor:
        """
        Recover DNA from multiple shards (averaging reduces noise).
        
        This implements the "fuzzy rose" effect: if you have multiple
        fragments, averaging the recovered DNA vectors gives a cleaner
        reconstruction than any single shard.
        
        Args:
            shards: List of vector shards
            block_indices: List of block indices corresponding to shards
            
        Returns:
            Averaged recovered DNA vector of shape (dna_dimensions,)
        """
        if not shards:
            raise ValueError("Must provide at least one shard")
        
        recovered_list = []
        for shard, block_idx in zip(shards, block_indices):
            dna = self.recover_dna(shard, block_idx)
            recovered_list.append(dna)
        
        # Average the recovered DNA vectors
        averaged = torch.stack(recovered_list).mean(dim=0)
        
        # Normalize
        norm = torch.norm(averaged)
        if norm > 1e-6:
            averaged = averaged / norm
        
        return averaged
    
    def __repr__(self) -> str:
        return (
            f"FractalSpace(dimensions={self.dimensions}, "
            f"dna_dimensions={self.dna_dimensions}, "
            f"dtype={self.dtype})"
        )
