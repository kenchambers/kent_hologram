"""
Operations: Wrapper for torchhd operations with correct API usage.

CRITICAL: torchhd does NOT have an unbind() function.
Unbinding is performed via: bind(composite, inverse(key))

This module wraps torchhd operations with semantic naming that matches
the Bentov holographic memory model.
"""

import torch
import torchhd


class Operations:
    """
    HDC operations wrapper providing correct torchhd API usage.

    This class wraps torchhd functions with semantic names matching the
    holographic memory metaphor:
    - bind: Combines two concepts (creating association)
    - bundle: Superimposes multiple vectors (creating interference pattern)
    - unbind: Extracts value from composite (resonance extraction)
    - inverse: Gets the inverse vector for unbinding
    - permute: Circular shift for sequence encoding

    All methods are static - this is a stateless utility class.
    """

    @staticmethod
    def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Binding operation: combines two concepts.

        In the Bentov model, this is like dropping two pebbles simultaneously -
        their ripples combine into a unique pattern that looks nothing like
        either pebble alone.

        Properties:
        - Dissimilar: bind(a, b) is orthogonal to both a and b
        - Reversible: unbind(bind(a, b), a) ≈ b
        - Commutative (for MAP): bind(a, b) = bind(b, a)

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Bound vector representing the association a↔b
        """
        result = torchhd.bind(a, b)
        
        # CRITICAL: Normalize result to prevent magnitude drift (vanishing/exploding gradients)
        # When using float embeddings, element-wise multiplication shrinks norm.
        norm = torch.norm(result)
        if norm > 1e-6:
            result = result / norm
            
        return result

    @staticmethod
    def bundle(*vectors: torch.Tensor) -> torch.Tensor:
        """
        Bundling operation: superimposes multiple vectors.

        In the Bentov model, this is the "water surface" - all the ripples
        from many pebbles interfere and create a holographic pattern that
        contains all of them.

        Properties:
        - Similarity preserving: bundle(a, b) is similar to both a and b
        - Capacity limited: too many vectors → noise
        - Order independent: bundle(a, b) = bundle(b, a)

        Args:
            *vectors: Variable number of hypervectors to bundle

        Returns:
            Bundled vector (superposition of all inputs)
        """
        if len(vectors) == 0:
            raise ValueError("Cannot bundle zero vectors")
        if len(vectors) == 1:
            return vectors[0]

        # torchhd.bundle only takes 2 arguments, so we need to bundle pairwise
        result = vectors[0]
        for v in vectors[1:]:
            result = torchhd.bundle(result, v)

        # CRITICAL: Normalize result to prevent magnitude drift
        # Bundling increases norm by sqrt(N). We want unit vectors.
        norm = torch.norm(result)
        if norm > 1e-6:
            result = result / norm

        return result

    @staticmethod
    def unbind(composite: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Unbinding operation: extracts value from composite.

        CRITICAL: torchhd.unbind() does NOT exist!
        Unbinding is performed via: bind(composite, inverse(key))

        In the Bentov model, this is "resonance" - you vibrate the water
        surface with the key frequency, and it resonates back the value.

        Given: composite = bind(key, value)
        Then: unbind(composite, key) ≈ value

        Args:
            composite: Composite vector (result of bind)
            key: Key vector to unbind

        Returns:
            Extracted value vector (may be noisy)

        Example:
            >>> ops = Operations()
            >>> key = torch.randn(10000)
            >>> value = torch.randn(10000)
            >>> composite = ops.bind(key, value)
            >>> recovered = ops.unbind(composite, key)
            >>> # recovered ≈ value (cosine similarity > 0.9)
        """
        # CORRECT API: unbind = bind with inverse
        return torchhd.bind(composite, torchhd.inverse(key))

    @staticmethod
    def inverse(vector: torch.Tensor) -> torch.Tensor:
        """
        Get the inverse of a vector for unbinding.

        The inverse has the property that:
        bind(vector, inverse(vector)) ≈ identity

        For MAP VSA: inverse is the vector itself
        For FHRR: inverse is the complex conjugate

        Args:
            vector: Input hypervector

        Returns:
            Inverse hypervector

        Example:
            >>> ops = Operations()
            >>> v = torch.randn(10000)
            >>> inv_v = ops.inverse(v)
            >>> identity = ops.bind(v, inv_v)
            >>> # identity should be close to a neutral element
        """
        return torchhd.inverse(vector)

    @staticmethod
    def permute(vector: torch.Tensor, shifts: int) -> torch.Tensor:
        """
        Permutation: circular shift for sequence encoding.

        Used to encode position in sequences by shifting the vector
        by different amounts for different positions.

        Args:
            vector: Input hypervector
            shifts: Number of positions to shift (can be negative)

        Returns:
            Permuted hypervector

        Example:
            >>> ops = Operations()
            >>> word = torch.randn(10000)
            >>> # Position 0: no shift
            >>> pos0 = word
            >>> # Position 1: shift by 1
            >>> pos1 = ops.permute(word, 1)
            >>> # pos0 and pos1 represent the same word at different positions
        """
        return torchhd.permute(vector, shifts=shifts)

    @staticmethod
    def cleanup(query: torch.Tensor, codebook: torch.Tensor) -> int:
        """
        Cleanup: find closest vector in codebook.

        This is the critical "snap to grid" operation that prevents
        hallucination - we can only return vectors that exist in the
        codebook, not arbitrary fabricated vectors.

        Args:
            query: Noisy query vector
            codebook: Tensor of shape (n_vectors, dimensions) containing
                     all valid vectors

        Returns:
            Index of closest vector in codebook

        Example:
            >>> ops = Operations()
            >>> codebook = torch.randn(100, 10000)  # 100 known concepts
            >>> noisy_query = codebook[5] + torch.randn(10000) * 0.1
            >>> idx = ops.cleanup(noisy_query, codebook)
            >>> # idx should be 5 (or close) despite noise
        """
        similarities = torchhd.cosine_similarity(query, codebook)
        return int(torch.argmax(similarities).item())
