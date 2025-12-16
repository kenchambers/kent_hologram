"""
PatternAnalyzer: Discover patterns in circuit activations.

Finds:
- Clusters of items that succeed together
- Context-specific success patterns
- Trending improvements/degradations
"""

from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
from statistics import mean

if TYPE_CHECKING:
    from .circuit_observer import CircuitObserver


@dataclass
class Pattern:
    """A discovered pattern in activations."""
    items: Tuple[str, ...]
    success_rate: float
    observation_count: int
    contexts: List[str]
    trend: float  # Positive = improving, negative = degrading


class PatternAnalyzer:
    """
    Analyzes CircuitObserver data to discover actionable patterns.

    Example:
        >>> observer = CircuitObserver()
        >>> analyzer = PatternAnalyzer(observer)
        >>> patterns = analyzer.find_success_patterns(min_items=2)
        >>> for p in patterns:
        ...     print(f"{p.items}: {p.success_rate:.0%}")
    """

    def __init__(self, observer: 'CircuitObserver'):
        self._observer = observer

    def find_success_patterns(self,
                              min_items: int = 1,
                              min_success_rate: float = 0.7,
                              min_observations: int = 5) -> List[Pattern]:
        """
        Find item combinations that consistently succeed.

        Args:
            min_items: Minimum items in pattern (1 for single items, 2+ for pairs)
            min_success_rate: Minimum success rate threshold
            min_observations: Minimum observation count

        Returns:
            List of Pattern objects meeting the criteria
        """
        patterns = []

        # Get all items with sufficient observations
        candidate_items = []
        for item in self._observer._activation_counts:
            count = self._observer.get_observation_count(item)
            rate = self._observer.get_success_rate(item)
            if count >= min_observations and rate >= min_success_rate:
                candidate_items.append((item, rate))

        # For single items
        if min_items <= 1:
            for item, rate in candidate_items:
                patterns.append(Pattern(
                    items=(item,),
                    success_rate=rate,
                    observation_count=self._observer.get_observation_count(item),
                    contexts=self._get_item_contexts(item),
                    trend=self.compute_trend(item)
                ))

        # For multi-item combinations
        if min_items <= 2:
            for item1, rate1 in candidate_items:
                co_acts = self._observer.get_co_activations(item1, min_count=min_observations)
                for item2, co_act_score in co_acts:
                    # Check if item2 also meets success threshold
                    item2_rate = self._observer.get_success_rate(item2)
                    if item2_rate >= min_success_rate:
                        # Combined success is the co-activation rate
                        max_count = max(
                            self._observer._activation_counts.get(item1, 1.0),
                            self._observer._activation_counts.get(item2, 1.0)
                        )
                        combined_rate = min(co_act_score / max_count, 1.0)

                        if combined_rate >= min_success_rate:
                            patterns.append(Pattern(
                                items=tuple(sorted([item1, item2])),
                                success_rate=combined_rate,
                                observation_count=int(co_act_score),
                                contexts=self._get_items_contexts(item1, item2),
                                trend=self._compute_pair_trend(item1, item2)
                            ))

        # Remove duplicates
        seen = set()
        unique_patterns = []
        for p in patterns:
            key = p.items
            if key not in seen:
                seen.add(key)
                unique_patterns.append(p)

        return unique_patterns

    def find_failure_patterns(self,
                              min_items: int = 1,
                              max_success_rate: float = 0.3,
                              min_observations: int = 5) -> List[Pattern]:
        """
        Find item combinations that consistently fail.

        Args:
            min_items: Minimum items in pattern
            max_success_rate: Maximum success rate to be considered failure
            min_observations: Minimum observation count

        Returns:
            List of Pattern objects meeting the criteria
        """
        patterns = []

        # Get all items with sufficient observations and low success rate
        for item in self._observer._activation_counts:
            count = self._observer.get_observation_count(item)
            rate = self._observer.get_success_rate(item)
            if count >= min_observations and rate <= max_success_rate:
                patterns.append(Pattern(
                    items=(item,),
                    success_rate=rate,
                    observation_count=count,
                    contexts=self._get_item_contexts(item),
                    trend=self.compute_trend(item)
                ))

        return patterns

    def get_context_recommendations(self, context: str) -> Dict[str, float]:
        """
        Get item weight adjustments for a specific context.

        Args:
            context: Context string to analyze

        Returns:
            Dict mapping item -> weight multiplier (>1 = boost, <1 = penalize)
        """
        recommendations = {}

        # Find items that perform well in this context
        for item in self._observer._activation_counts:
            rate = self._observer.get_success_rate(item, context=context)
            count = self._observer.get_observation_count(item)

            if count < 5:
                continue

            # Weight adjustment: high performers get boosted, low performers downweighted
            if rate >= 0.8:
                # Strong success: boost by 1.5x
                recommendations[item] = 1.5
            elif rate <= 0.2:
                # Strong failure: downweight by 0.5x
                recommendations[item] = 0.5
            elif rate < 0.5:
                # Mediocre: slight penalty
                recommendations[item] = 0.8

        return recommendations

    def compute_trend(self, item: str, window: int = 20) -> float:
        """
        Compute recent trend for an item's success rate.

        Positive trend = improving, negative = degrading.

        Args:
            item: Item to analyze
            window: Number of recent records to consider

        Returns:
            Trend value (positive = improving, negative = degrading)
        """
        # Get recent records for this item
        recent = []
        for record in self._observer._history[-window:]:
            if item in record.items:
                recent.append(1.0 if record.success else 0.0)

        if len(recent) < 3:
            return 0.0

        # Compute trend as difference between recent and older
        mid = len(recent) // 2
        recent_avg = mean(recent[mid:]) if recent[mid:] else 0.5
        older_avg = mean(recent[:mid]) if recent[:mid] else 0.5

        return recent_avg - older_avg

    def _compute_pair_trend(self, item1: str, item2: str) -> float:
        """Compute trend for a pair of items."""
        trend1 = self.compute_trend(item1)
        trend2 = self.compute_trend(item2)
        return (trend1 + trend2) / 2.0

    def _get_item_contexts(self, item: str) -> List[str]:
        """Get all contexts where an item was observed."""
        contexts = []
        for ctx in self._observer._context_stats:
            if item in self._observer._context_stats[ctx]:
                contexts.append(ctx)
        return contexts

    def _get_items_contexts(self, item1: str, item2: str) -> List[str]:
        """Get contexts where both items were observed."""
        contexts1 = set(self._get_item_contexts(item1))
        contexts2 = set(self._get_item_contexts(item2))
        return list(contexts1 & contexts2)

    def get_improvement_report(self) -> str:
        """
        Generate human-readable report of learned patterns.

        Returns:
            Formatted string with analysis summary
        """
        stats = self._observer.get_stats_summary()

        report = f"""
Circuit Introspection Report
=============================

Observations Summary:
  - Total observations: {stats['total_observations']}
  - Unique circuits: {stats['unique_items']}
  - Context types: {stats['unique_contexts']}
  - Co-activation pairs: {stats['co_activation_pairs']}

Top Performers (>80% success rate):
"""

        reinforcement = self._observer.suggest_reinforcement(threshold=0.8)
        if reinforcement:
            for item, rate in reinforcement[:5]:
                count = self._observer.get_observation_count(item)
                report += f"  - {item}: {rate:.0%} ({count:.0f} obs)\n"
        else:
            report += "  (none yet)\n"

        report += "\nCandidates for Pruning (<20% success rate):\n"
        pruning = self._observer.suggest_pruning(threshold=0.2)
        if pruning:
            for item, rate in pruning[:5]:
                count = self._observer.get_observation_count(item)
                report += f"  - {item}: {rate:.0%} ({count:.0f} obs)\n"
        else:
            report += "  (none yet)\n"

        # Add trending info
        report += "\nTrending Items:\n"
        improving = []
        degrading = []
        for item in self._observer._activation_counts:
            trend = self.compute_trend(item)
            if trend > 0.1:
                improving.append((item, trend))
            elif trend < -0.1:
                degrading.append((item, trend))

        if improving:
            improving.sort(key=lambda x: -x[1])
            report += "  Improving:\n"
            for item, trend in improving[:3]:
                report += f"    - {item}: +{trend:.0%}\n"

        if degrading:
            degrading.sort(key=lambda x: x[1])
            report += "  Degrading:\n"
            for item, trend in degrading[:3]:
                report += f"    - {item}: {trend:.0%}\n"

        return report
