"""
SelfImprovementManager: Central coordinator for self-improvement.

Manages the lifecycle of the self-improvement system:
- Creates and coordinates CircuitObserver and PatternAnalyzer
- Handles persistence of learned patterns
- Provides reporting and statistics
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import threading

from .circuit_observer import CircuitObserver
from .pattern_analyzer import PatternAnalyzer


class SelfImprovementManager:
    """
    Central manager for Kent Hologram's self-improvement system.

    Coordinates:
    - CircuitObserver for tracking activations
    - PatternAnalyzer for discovering patterns
    - Persistence for saving/loading learned patterns
    - Periodic analysis and reporting

    Example:
        >>> manager = SelfImprovementManager(persist_path="./learned_patterns.json")
        >>>
        >>> # Attach to components
        >>> constraint_accum.set_circuit_observer(manager.observer)
        >>> metacog_loop.set_circuit_observer(manager.observer)
        >>>
        >>> # Later: get statistics
        >>> stats = manager.get_statistics()
        >>> print(f"Total observations: {stats['total_observations']}")
        >>>
        >>> # Save learned patterns
        >>> manager.save()

    Attributes:
        _persist_path: Path for saving/loading learned patterns
        _auto_save_interval: Seconds between auto-saves (if enabled)
        _auto_analyze_interval: Number of observations between auto-analysis
        _observer: CircuitObserver instance
        _analyzer: PatternAnalyzer instance
    """

    def __init__(self,
                 persist_path: Optional[str] = None,
                 auto_save_interval: int = 300,  # seconds
                 auto_analyze_interval: int = 100):  # observations
        """
        Initialize self-improvement manager.

        Args:
            persist_path: Path for saving/loading patterns (None = no persistence)
            auto_save_interval: Seconds between auto-saves (0 = disabled)
            auto_analyze_interval: Observations between auto-analysis (0 = disabled)
        """
        self._persist_path = Path(persist_path) if persist_path else None
        self._auto_save_interval = auto_save_interval
        self._auto_analyze_interval = auto_analyze_interval

        self._observer = CircuitObserver()
        self._analyzer = PatternAnalyzer(self._observer)

        self._observation_count = 0
        self._last_save_time = time.time()
        self._lock = threading.RLock()

        # Load existing patterns if available
        if self._persist_path and self._persist_path.exists():
            self.load()

    @property
    def observer(self) -> CircuitObserver:
        """
        Get the circuit observer for attaching to components.

        Returns:
            CircuitObserver instance
        """
        return self._observer

    @property
    def analyzer(self) -> PatternAnalyzer:
        """
        Get the pattern analyzer.

        Returns:
            PatternAnalyzer instance
        """
        return self._analyzer

    def save(self) -> None:
        """
        Save learned patterns to disk.

        Serializes the observer's state to JSON for persistence.
        Thread-safe.
        """
        if self._persist_path is None:
            return

        with self._lock:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "observer": self._observer.state_dict(),
                "observation_count": self._observation_count,
                "last_save_time": self._last_save_time,
            }
            with open(self._persist_path, 'w') as f:
                json.dump(state, f, indent=2)

            self._last_save_time = time.time()

    def load(self) -> None:
        """
        Load learned patterns from disk.

        Deserializes the observer's state from JSON.
        Thread-safe.
        """
        if self._persist_path is None or not self._persist_path.exists():
            return

        with self._lock:
            with open(self._persist_path, 'r') as f:
                state = json.load(f)

            self._observer.load_state_dict(state.get("observer", {}))
            self._observation_count = state.get("observation_count", 0)
            self._last_save_time = state.get("last_save_time", time.time())

    def increment_observation_count(self) -> None:
        """
        Increment observation counter.

        Called by components after each observation.
        Triggers auto-save and auto-analysis if thresholds are met.
        """
        with self._lock:
            self._observation_count += 1

            # Auto-save if interval elapsed
            if (self._auto_save_interval > 0 and
                time.time() - self._last_save_time >= self._auto_save_interval):
                self.save()

            # Auto-analyze if observation threshold met
            if (self._auto_analyze_interval > 0 and
                self._observation_count % self._auto_analyze_interval == 0):
                # Could trigger analysis here if needed
                pass

    def get_improvement_report(self) -> str:
        """
        Get a report of learned patterns and recommendations.

        Returns:
            Formatted string with analysis results
        """
        return self._analyzer.get_improvement_report()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics about learned patterns.

        Returns:
            Dictionary with statistics including:
            - total_observations: Total observations recorded
            - items_to_prune: Number of items suggested for pruning
            - items_to_reinforce: Number of items suggested for reinforcement
            - top_performers: Best performing item combinations
            - worst_performers: Worst performing item combinations
        """
        with self._lock:
            pruning = self._observer.suggest_pruning()
            reinforcement = self._observer.suggest_reinforcement()

            # Get actual observation count from observer's history, not our counter
            observer_stats = self._observer.get_stats_summary()
            return {
                "total_observations": observer_stats["total_observations"],
                "unique_items": observer_stats["unique_items"],
                "items_to_prune": len(pruning),
                "items_to_reinforce": len(reinforcement),
                "top_performers": reinforcement[:5] if reinforcement else [],
                "worst_performers": pruning[:5] if pruning else [],
                "last_save_time": self._last_save_time,
            }

    def reset(self) -> None:
        """
        Reset all learned patterns.

        Clears the observer's state and resets counters.
        Use with caution - this discards all learned knowledge.
        """
        with self._lock:
            self._observer.reset()
            self._observation_count = 0
            self._last_save_time = time.time()

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"SelfImprovementManager("
            f"observations={stats.get('total_observations', 0)}, "
            f"persist_path={self._persist_path})"
        )
