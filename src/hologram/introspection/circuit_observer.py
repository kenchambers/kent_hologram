"""
CircuitObserver: Track which 'circuits' (vocabulary items, transformations)
activate and whether they lead to success.

Inspired by sparse circuits research: we track activation patterns and
success rates at the vocabulary level, enabling pruning and reinforcement.

Key concept: A "circuit" is any vocabulary item or transformation pattern
that can be activated and observed. By tracking which circuits succeed,
we can:
1. Prune consistently failing circuits
2. Reinforce consistently succeeding circuits
3. Discover co-activation patterns (circuits that succeed together)
4. Build context-specific success patterns
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time
import threading
import ast
import math


@dataclass
class ActivationRecord:
    """Record of a single activation event."""
    items: List[str]  # Vocabulary items that activated
    success: bool
    confidence: float
    timestamp: float = field(default_factory=time.time)
    context: Optional[str] = None  # e.g., "arc_solver", "fact_query"


class CircuitObserver:
    """
    Observes which 'circuits' (transformation patterns, query paths,
    vocabulary items) are used and whether they succeed.

    Thread-safe via RLock. Uses exponential decay for recency weighting.

    Key methods:
    - observe(): Record an activation and its outcome
    - get_success_rate(): Get historical success rate for an item
    - suggest_pruning(): Get items that consistently fail
    - suggest_reinforcement(): Get items that consistently succeed
    - get_co_activations(): Get items that succeed together

    Example:
        >>> observer = CircuitObserver()
        >>> observer.observe(
        ...     items=["rotate", "largest", "90_degrees"],
        ...     success=True,
        ...     confidence=0.85,
        ...     context="arc_solver"
        ... )
        >>> observer.get_success_rate("rotate")
        0.85
        >>> pruning_candidates = observer.suggest_pruning(threshold=0.2)
        >>> reinforcement_candidates = observer.suggest_reinforcement(threshold=0.8)
    """

    def __init__(self,
                 min_observations: int = 5,
                 recency_weight: float = 0.95,
                 max_history: int = 10000):
        """
        Initialize circuit observer.

        Args:
            min_observations: Minimum observations before making suggestions
            recency_weight: Decay factor for old observations (0-1, higher = stronger recency)
            max_history: Maximum activation records to keep
        """
        self._lock = threading.RLock()

        # Tracking structures with weighted counts
        self._activation_counts: Dict[str, float] = defaultdict(float)
        self._success_counts: Dict[str, float] = defaultdict(float)

        # Co-activation tracking: (item1, item2) -> weighted count
        # Only track pairs where both succeeded
        self._co_activations: Dict[Tuple[str, str], float] = defaultdict(float)

        # Per-context statistics: context -> {item -> (count, success_count)}
        self._context_stats: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(
            lambda: defaultdict(lambda: (0.0, 0.0))
        )

        # Recent history for persistence and trend analysis
        self._history: List[ActivationRecord] = []

        # Configuration
        self._min_observations = min_observations
        self._recency_weight = recency_weight
        self._max_history = max_history

    def observe(self,
                items: List[str],
                success: bool,
                confidence: float,
                context: Optional[str] = None) -> None:
        """
        Record an activation and its outcome.

        Thread-safe. Uses exponential decay: newer observations are weighted
        higher than older ones.

        Args:
            items: List of items/circuits that activated
            success: Whether the activation led to success
            confidence: Confidence in the outcome [0, 1]
            context: Optional context string (e.g., "arc_solver", "fact_query")

        Example:
            >>> observer.observe(
            ...     items=["pattern_a", "pattern_b"],
            ...     success=True,
            ...     confidence=0.9,
            ...     context="arc_solver"
            ... )
        """
        with self._lock:
            # Create activation record
            record = ActivationRecord(
                items=items,
                success=success,
                confidence=confidence,
                timestamp=time.time(),
                context=context
            )
            self._history.append(record)

            # Trim history if it exceeds max length
            if len(self._history) > self._max_history:
                self._history.pop(0)

            # Weight for this observation (confidence-weighted)
            weight = confidence

            # Track activation and success for each item
            for item in items:
                self._activation_counts[item] += weight

                if success:
                    self._success_counts[item] += weight

                # Track context-specific statistics
                if context:
                    old_count, old_success = self._context_stats[context][item]
                    self._context_stats[context][item] = (
                        old_count + weight,
                        old_success + (weight if success else 0.0)
                    )

            # Track co-activations (only for successful activations)
            if success:
                for i, item1 in enumerate(items):
                    for item2 in items[i+1:]:
                        key = tuple(sorted([item1, item2]))
                        self._co_activations[key] += weight

    def get_success_rate(self, item: str, context: Optional[str] = None) -> float:
        """
        Get success rate for an item (0.5 if unknown).

        Thread-safe. Returns weighted success rate across all observations.

        Args:
            item: Item to query
            context: Optional context filter (if provided, only considers that context)

        Returns:
            Success rate [0, 1]. Returns 0.5 if item has never been observed
            (neutral prior).
        """
        with self._lock:
            if context:
                # Context-specific success rate
                count, success = self._context_stats[context].get(item, (0.0, 0.0))
                if count < self._min_observations:
                    return 0.5
                return success / count if count > 0 else 0.5

            # Global success rate
            total = self._activation_counts.get(item, 0.0)
            if total < self._min_observations:
                return 0.5

            successes = self._success_counts.get(item, 0.0)
            return successes / total if total > 0 else 0.5

    def get_confidence_adjusted_rate(self, item: str) -> float:
        """
        Get success rate weighted by confidence of observations.

        Thread-safe. Items observed with high confidence have more influence
        on the final rate.

        Args:
            item: Item to query

        Returns:
            Confidence-adjusted success rate [0, 1]
        """
        with self._lock:
            total = self._activation_counts.get(item, 0.0)
            if total < self._min_observations:
                return 0.5

            successes = self._success_counts.get(item, 0.0)
            return successes / total if total > 0 else 0.5

    def suggest_pruning(self, threshold: float = 0.2) -> List[Tuple[str, float]]:
        """
        Get items that consistently fail, with their success rates.

        Thread-safe. Items with success rate below threshold are candidates
        for pruning from future searches.

        Args:
            threshold: Success rate below which to consider for pruning

        Returns:
            List of (item, success_rate) tuples, sorted by success rate
        """
        with self._lock:
            candidates = []

            for item, total in self._activation_counts.items():
                if total < self._min_observations:
                    continue

                rate = self._success_counts.get(item, 0.0) / total if total > 0 else 0.0
                if rate < threshold:
                    candidates.append((item, rate))

            # Sort by success rate (worst first)
            candidates.sort(key=lambda x: x[1])
            return candidates

    def suggest_reinforcement(self, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Get items that consistently succeed, with their success rates.

        Thread-safe. Items with success rate above threshold are candidates
        for reinforcement (higher search priority).

        Args:
            threshold: Success rate above which to consider for reinforcement

        Returns:
            List of (item, success_rate) tuples, sorted by success rate (best first)
        """
        with self._lock:
            candidates = []

            for item, total in self._activation_counts.items():
                if total < self._min_observations:
                    continue

                rate = self._success_counts.get(item, 0.0) / total if total > 0 else 0.0
                if rate >= threshold:
                    candidates.append((item, rate))

            # Sort by success rate (best first)
            candidates.sort(key=lambda x: -x[1])
            return candidates

    def get_co_activations(self, item: str, min_count: int = 3) -> List[Tuple[str, float]]:
        """
        Get items that frequently succeed together with this item.

        Thread-safe. Returns items that were activated together with the
        query item in successful operations.

        Args:
            item: Query item
            min_count: Minimum weighted count for inclusion

        Returns:
            List of (item, co_activation_score) tuples, sorted by score
        """
        with self._lock:
            pairs = []

            for (item1, item2), count in self._co_activations.items():
                if count < min_count:
                    continue

                # Find the other item in the pair
                other = item2 if item1 == item else item1 if item2 == item else None

                if other is not None:
                    pairs.append((other, count))

            # Sort by co-activation score (highest first)
            pairs.sort(key=lambda x: -x[1])
            return pairs

    def get_prior(self, items: List[str]) -> float:
        """
        Get combined prior probability for a set of items.

        Thread-safe. Computes the geometric mean of individual success rates,
        boosted by co-activation strength.

        Args:
            items: List of items to evaluate

        Returns:
            Prior probability [0, 1]
        """
        with self._lock:
            if not items:
                return 0.5

            # Start with geometric mean of success rates
            rates = [self.get_success_rate(item) for item in items]

            # Geometric mean
            product = 1.0
            for rate in rates:
                product *= rate
            geometric_mean = product ** (1.0 / len(rates)) if product > 0 else 0.5

            # Boost by co-activation strength
            if len(items) > 1:
                co_act_scores = []
                for i, item1 in enumerate(items):
                    for item2 in items[i+1:]:
                        key = tuple(sorted([item1, item2]))
                        co_act_scores.append(self._co_activations.get(key, 0.0))

                if co_act_scores:
                    avg_co_act = sum(co_act_scores) / len(co_act_scores)
                    max_observed = max(self._co_activations.values()) if self._co_activations else 1.0
                    co_act_boost = 0.5 + 0.5 * (avg_co_act / max_observed) if max_observed > 0 else 0.5
                    geometric_mean = geometric_mean * co_act_boost

            return min(geometric_mean, 1.0)

    def decay_old_observations(self) -> None:
        """
        Apply recency decay to all observations.

        Thread-safe. Reduces the weight of older observations to focus on
        recent patterns. Useful when calling periodically to adapt
        to changing conditions.

        Multiply all weighted counts by recency_weight, emphasizing recent
        observations.
        """
        with self._lock:
            factor = self._recency_weight

            for item in self._activation_counts:
                self._activation_counts[item] *= factor
                self._success_counts[item] *= factor

            for context in self._context_stats:
                for item in self._context_stats[context]:
                    count, success = self._context_stats[context][item]
                    self._context_stats[context][item] = (count * factor, success * factor)

            for key in self._co_activations:
                self._co_activations[key] *= factor

    def get_observation_count(self, item: str) -> float:
        """Get the weighted observation count for an item."""
        with self._lock:
            return self._activation_counts.get(item, 0.0)

    def get_stats_summary(self) -> Dict[str, int]:
        """Get summary statistics about observations."""
        with self._lock:
            return {
                "unique_items": len(self._activation_counts),
                "total_observations": len(self._history),
                "unique_contexts": len(self._context_stats),
                "co_activation_pairs": len(self._co_activations),
            }

    def reset(self) -> None:
        """Clear all observations (start fresh)."""
        with self._lock:
            self._activation_counts.clear()
            self._success_counts.clear()
            self._co_activations.clear()
            self._context_stats.clear()
            self._history.clear()

    def state_dict(self) -> dict:
        """
        Get state for persistence.

        Thread-safe. Returns a JSON-serializable dictionary containing
        all observations and statistics.

        Returns:
            Dictionary with state ready for json.dump()
        """
        with self._lock:
            return {
                "activation_counts": dict(self._activation_counts),
                "success_counts": dict(self._success_counts),
                "co_activations": {
                    str(k): v for k, v in self._co_activations.items()
                },
                "context_stats": {
                    ctx: {
                        item: {
                            "count": count,
                            "success": success
                        }
                        for item, (count, success) in items.items()
                    }
                    for ctx, items in self._context_stats.items()
                },
                "history": [
                    {
                        "items": r.items,
                        "success": r.success,
                        "confidence": r.confidence,
                        "timestamp": r.timestamp,
                        "context": r.context
                    }
                    for r in self._history
                ],
                "config": {
                    "min_observations": self._min_observations,
                    "recency_weight": self._recency_weight,
                    "max_history": self._max_history,
                }
            }

    def load_state_dict(self, state: dict) -> None:
        """Load state from persistence."""
        with self._lock:
            # Load counts
            self._activation_counts = defaultdict(float, state.get("activation_counts", {}))
            self._success_counts = defaultdict(float, state.get("success_counts", {}))

            # Load co-activations (convert string keys back to tuples)
            co_act_dict = state.get("co_activations", {})
            self._co_activations = defaultdict(float)
            for k, v in co_act_dict.items():
                try:
                    key = ast.literal_eval(k)
                    if isinstance(key, tuple):
                        self._co_activations[key] = v
                except (ValueError, SyntaxError):
                    continue

            # Load context stats
            self._context_stats = defaultdict(lambda: defaultdict(lambda: (0.0, 0.0)))
            for ctx, items_dict in state.get("context_stats", {}).items():
                for item, stats in items_dict.items():
                    self._context_stats[ctx][item] = (
                        stats.get("count", 0.0),
                        stats.get("success", 0.0)
                    )

            # Load history
            self._history = []
            for record_dict in state.get("history", []):
                self._history.append(ActivationRecord(
                    items=record_dict.get("items", []),
                    success=record_dict.get("success", False),
                    confidence=record_dict.get("confidence", 0.5),
                    timestamp=record_dict.get("timestamp", time.time()),
                    context=record_dict.get("context")
                ))

            # Load config
            config = state.get("config", {})
            self._min_observations = config.get("min_observations", 5)
            self._recency_weight = config.get("recency_weight", 0.95)
            self._max_history = config.get("max_history", 10000)

    def __len__(self) -> int:
        """Return number of unique items observed."""
        with self._lock:
            return len(self._activation_counts)

    def __repr__(self) -> str:
        """String representation for debugging."""
        with self._lock:
            return (
                f"CircuitObserver("
                f"items={len(self._activation_counts)}, "
                f"observations={len(self._history)}, "
                f"co_activation_pairs={len(self._co_activations)})"
            )
