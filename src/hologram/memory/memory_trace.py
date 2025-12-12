"""
MemoryTrace: Holographic interference pattern storage.

Implements the Bentov "water surface" metaphor where:
- Each fact creates a "ripple" (bound key-value pair)
- All ripples superimpose into a single interference pattern
- The pattern encodes all facts holographically
- Query "vibrates" the surface to extract resonant patterns

This is the core of the holographic storage system.
"""

from typing import Optional

import torch

from hologram.config.constants import (
    SURPRISE_THRESHOLD,
    SURPRISE_LEARNING_RATE,
    SURPRISE_MOMENTUM_DECAY,
    SURPRISE_DECAY,
)
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity
from hologram.core.vector_space import VectorSpace


class MemoryTrace:
    """
    Holographic memory trace using interference patterns.

    Stores multiple key-value pairs in a single bundled vector through
    superposition. Facts can be retrieved by unbinding with the key vector.

    Properties:
    - Holographic: Information distributed across all dimensions
    - Capacity-limited: Too many facts → increased noise
    - Graceful degradation: Noise lowers confidence, doesn't cause wrong answers
    - Survives corruption: Partial damage doesn't destroy all information

    Attributes:
        _space: VectorSpace configuration
        _trace: The bundled memory vector (the "water surface")
        _fact_count: Number of stored facts
        _momentum: Exponential moving average of recent facts (Titans-inspired)
        _momentum_decay: Decay factor for momentum tracking

    Example:
        >>> space = VectorSpace(dimensions=10000)
        >>> trace = MemoryTrace(space)
        >>> key = torch.randn(10000)
        >>> value = torch.randn(10000)
        >>> trace.store(key, value)
        >>> retrieved = trace.query(key)
        >>> cosine_similarity(retrieved, value) > 0.9  # High similarity
        True
    """

    def __init__(
        self,
        space: VectorSpace,
        initial_trace: Optional[torch.Tensor] = None,
    ):
        """
        Initialize memory trace, optionally with existing trace vector.

        Args:
            space: VectorSpace defining dimensionality
            initial_trace: Optional pre-existing trace vector (for consolidation decay)
        """
        self._space = space
        if initial_trace is not None:
            space.validate_vector(initial_trace)
            self._trace = initial_trace.clone()
        else:
            self._trace = space.empty_vector()
        self._fact_count = 0
        self._momentum = space.empty_vector()  # Titans momentum tracking
        self._momentum_decay = SURPRISE_MOMENTUM_DECAY

    def store(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """
        Store a key-value pair in memory (legacy method).

        Delegates to store_with_surprise() for backward compatibility.
        This method maintains the original API while using surprise-gated
        learning under the hood.

        Args:
            key: Key hypervector
            value: Value hypervector

        Example:
            >>> trace = MemoryTrace(VectorSpace())
            >>> france = torch.randn(10000)
            >>> paris = torch.randn(10000)
            >>> trace.store(france, paris)
        """
        self.store_with_surprise(key, value)

    def store_with_surprise(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        learning_rate: float = SURPRISE_LEARNING_RATE,
        surprise_threshold: float = SURPRISE_THRESHOLD
    ) -> float:
        """
        Store with Titans-inspired surprise gating.

        Before bundling a new fact, measures how "surprising" (novel) it is
        relative to existing memory. Only updates memory if surprise exceeds
        threshold, preventing duplicate encoding and noise accumulation.

        Args:
            key: Key hypervector (e.g., bind(subject, predicate))
            value: Value hypervector (e.g., object encoding)
            learning_rate: Base learning rate for updates (default: SURPRISE_LEARNING_RATE)
            surprise_threshold: Minimum surprise to trigger learning (default: SURPRISE_THRESHOLD)

        Returns:
            Surprise score (0.0 = known, 1.0 = completely novel)

        Example:
            >>> trace = MemoryTrace(VectorSpace())
            >>> key = torch.randn(10000)
            >>> value = torch.randn(10000)
            >>> surprise1 = trace.store_with_surprise(key, value)
            >>> surprise2 = trace.store_with_surprise(key, value)  # Duplicate
            >>> assert surprise1 > 0.5  # First time: novel
            >>> assert surprise2 < 0.1  # Second time: known
        """
        self._space.validate_vector(key)
        self._space.validate_vector(value)

        # Create fact vector
        fact = Operations.bind(key, value)

        # Calculate surprise
        if self._fact_count == 0:
            surprise = 1.0
            momentum_surprise = 1.0
        else:
            # Current surprise: how different from memory?
            memory_sim = Similarity.cosine(self._trace, fact)
            surprise = 1.0 - max(0.0, memory_sim)

            # Momentum surprise: how different from recent learning direction?
            if torch.norm(self._momentum) > 1e-6:
                momentum_sim = Similarity.cosine(self._momentum, fact)
                momentum_surprise = 1.0 - max(0.0, momentum_sim)
            else:
                momentum_surprise = surprise

        # Combined surprise (Titans uses both instant and momentum)
        combined_surprise = 0.7 * surprise + 0.3 * momentum_surprise

        # Gate update
        if combined_surprise < surprise_threshold:
            return combined_surprise

        # Warm-up: reduce learning rate for first 10 facts
        if self._fact_count < 10:
            learning_rate = learning_rate * (self._fact_count + 1) / 10

        # Update momentum (exponential moving average of recent facts)
        self._momentum = self._momentum_decay * self._momentum + (1 - self._momentum_decay) * fact

        # Weighted update
        update_strength = combined_surprise * learning_rate

        if self._fact_count == 0:
            self._trace = fact
        else:
            weighted_fact = fact * update_strength
            self._trace = Operations.bundle(self._trace, weighted_fact)

            # Normalize to prevent drift
            norm = torch.norm(self._trace)
            if norm > 1e-6:
                self._trace = self._trace / norm

        self._fact_count += 1
        return combined_surprise

    def query(self, key: torch.Tensor) -> torch.Tensor:
        """
        Query memory with a key to extract associated value.

        Unbinds the key from the trace, returning a (potentially noisy)
        approximation of the original value. This is the "resonance"
        operation - the key vibrates the memory surface.

        Args:
            key: Query key hypervector

        Returns:
            Retrieved value vector (may be noisy)

        Example:
            >>> trace = MemoryTrace(VectorSpace())
            >>> key = torch.randn(10000)
            >>> value = torch.randn(10000)
            >>> trace.store(key, value)
            >>> retrieved = trace.query(key)
            >>> # retrieved ≈ value (cosine similarity > 0.9)
        """
        self._space.validate_vector(key)
        return Operations.unbind(self._trace, key)

    def resonance(
        self,
        key: torch.Tensor,
        candidates: torch.Tensor
    ) -> torch.Tensor:
        """
        Get resonance strengths against candidate values.

        This is the core retrieval operation: unbind the key, then measure
        similarity to all possible values (the "cleanup" step).

        Args:
            key: Query key vector
            candidates: Tensor of shape (n_candidates, dimensions)

        Returns:
            Tensor of shape (n_candidates,) with similarity scores

        Example:
            >>> trace = MemoryTrace(VectorSpace())
            >>> key = torch.randn(10000)
            >>> value = torch.randn(10000)
            >>> trace.store(key, value)
            >>> candidates = torch.stack([value, torch.randn(10000), torch.randn(10000)])
            >>> scores = trace.resonance(key, candidates)
            >>> torch.argmax(scores).item()  # Should be 0 (the correct value)
            0
        """
        result = self.query(key)
        return Similarity.cosine_batch(result, candidates)

    def corrupt(self, noise_ratio: float) -> "MemoryTrace":
        """
        Create corrupted copy for testing graceful degradation.

        Adds Gaussian noise proportional to noise_ratio. Used to test
        that the holographic property allows partial recovery even
        when the memory is damaged.

        Args:
            noise_ratio: Ratio of noise to signal (0.0 = no noise, 1.0 = 100% noise)

        Returns:
            New MemoryTrace with corrupted trace vector

        Example:
            >>> trace = MemoryTrace(VectorSpace())
            >>> # Store facts...
            >>> noisy_trace = trace.corrupt(noise_ratio=0.3)
            >>> # noisy_trace should still retrieve facts, with lower confidence
        """
        corrupted = MemoryTrace(self._space)
        noise = torch.randn_like(self._trace) * noise_ratio
        corrupted._trace = self._trace + noise
        corrupted._fact_count = self._fact_count
        return corrupted

    def merge(self, other: "MemoryTrace") -> "MemoryTrace":
        """
        Merge two memory traces via bundling.

        Creates a new trace containing facts from both inputs.
        This demonstrates the holographic property - memories can be
        combined without loss (up to capacity limits).

        Args:
            other: Another MemoryTrace to merge

        Returns:
            New MemoryTrace containing both sets of facts

        Example:
            >>> trace1 = MemoryTrace(VectorSpace())
            >>> trace2 = MemoryTrace(VectorSpace())
            >>> # Store different facts in each...
            >>> merged = trace1.merge(trace2)
            >>> # merged contains all facts from both traces
        """
        if self._space.dimensions != other._space.dimensions:
            raise ValueError("Cannot merge traces from different vector spaces")

        merged = MemoryTrace(self._space)
        merged._trace = Operations.bundle(self._trace, other._trace)
        merged._fact_count = self._fact_count + other._fact_count
        return merged

    @property
    def trace_vector(self) -> torch.Tensor:
        """
        Get the raw trace vector for persistence.

        Returns:
            The bundled memory vector
        """
        return self._trace

    @property
    def fact_count(self) -> int:
        """
        Get number of stored facts.

        Returns:
            Number of facts bundled into trace
        """
        return self._fact_count

    @property
    def saturation_estimate(self) -> float:
        """
        Rough estimate of capacity usage.

        Based on heuristic: capacity ∝ √dimensions
        This is UNPROVEN and requires empirical validation.

        Returns:
            Estimated saturation ratio (0.0 = empty, 1.0 = saturated)

        Example:
            >>> trace = MemoryTrace(VectorSpace(dimensions=10000))
            >>> # Store 50 facts...
            >>> trace.saturation_estimate
            0.5  # 50 / sqrt(10000) = 50 / 100 = 0.5
        """
        capacity_estimate = self._space.dimensions ** 0.5
        return self._fact_count / capacity_estimate

    def forget(self, decay: float = SURPRISE_DECAY) -> None:
        """
        Apply forgetting (weight decay) to enable bounded memory.

        Titans insight: Active forgetting prevents memory saturation.
        Call periodically (e.g., every N store operations).

        Args:
            decay: Retention factor (0.99 = 1% forgetting per call)

        Example:
            >>> trace = MemoryTrace(VectorSpace())
            >>> # Store many facts...
            >>> trace.forget()  # Apply 1% decay
        """
        self._trace = self._trace * decay
        # Re-normalize after decay
        norm = torch.norm(self._trace)
        if norm > 1e-6:
            self._trace = self._trace / norm

    def __repr__(self) -> str:
        return (
            f"MemoryTrace(facts={self._fact_count}, "
            f"saturation={self.saturation_estimate:.2%})"
        )
