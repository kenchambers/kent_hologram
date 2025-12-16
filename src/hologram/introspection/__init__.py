"""
Introspection module for self-improvement.

Provides circuit observation and pattern analysis inspired by
sparse circuits research: observe activations, track success,
prune failures, reinforce successes.

Based on OpenAI's sparse circuits philosophy: "observe what activates,
track what works, prune what doesn't."

Components:
- CircuitObserver: Tracks activation patterns and success rates
- PatternAnalyzer: Discovers patterns and provides recommendations
- SelfImprovingMixin: Reusable mixin for self-improvement capabilities
- SelfImprovementManager: Central coordinator for the system

Example:
    >>> from hologram.introspection import SelfImprovementManager
    >>> manager = SelfImprovementManager(persist_path="./learned_patterns.json")
    >>>
    >>> # Attach to components
    >>> constraint_accum.set_circuit_observer(manager.observer)
    >>> metacog_loop.set_circuit_observer(manager.observer)
    >>>
    >>> # Get improvement report
    >>> print(manager.get_improvement_report())
"""

from .circuit_observer import CircuitObserver, ActivationRecord
from .pattern_analyzer import PatternAnalyzer, Pattern
from .mixin import SelfImprovingMixin
from .manager import SelfImprovementManager

__all__ = [
    "CircuitObserver",
    "ActivationRecord",
    "PatternAnalyzer",
    "Pattern",
    "SelfImprovingMixin",
    "SelfImprovementManager",
]
