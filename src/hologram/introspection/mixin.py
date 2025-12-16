"""
SelfImprovingMixin: Add self-improvement capabilities to any component.

This mixin provides a standard interface for components to integrate with
the circuit observation system, enabling automatic pattern learning and
self-improvement.
"""

from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .circuit_observer import CircuitObserver


class SelfImprovingMixin:
    """
    Mixin that adds self-improvement tracking to any component.

    Components that inherit from this mixin can easily report outcomes
    to the circuit observer and query learned priors to make better decisions.

    Usage:
        class MyComponent(SelfImprovingMixin):
            def __init__(self):
                super().__init__()
                self.set_component_context("my_component")

            def do_work(self, action: str, target: str):
                items = [action, target]

                # Check if this combination has historically succeeded
                if not self._should_try_items(items):
                    return None  # Skip low-probability attempts

                # Perform work
                success = self._perform_work(action, target)

                # Report outcome for learning
                self._observe_outcome(items, success, confidence=0.8)

                return result

    Attributes:
        _circuit_observer: Optional CircuitObserver instance
        _component_context: String identifying this component's observation context
    """

    _circuit_observer: Optional['CircuitObserver'] = None
    _component_context: str = "unknown"

    def set_circuit_observer(self, observer: 'CircuitObserver') -> None:
        """
        Attach circuit observer for self-improvement tracking.

        Args:
            observer: CircuitObserver instance to receive observations
        """
        self._circuit_observer = observer

    def set_component_context(self, context: str) -> None:
        """
        Set the context name for this component's observations.

        The context helps the observer distinguish between different
        types of operations (e.g., "arc_transformation" vs "metacognitive_query").

        Args:
            context: String identifier for this component's context
        """
        self._component_context = context

    def _observe_outcome(self,
                         items: List[str],
                         success: bool,
                         confidence: float) -> None:
        """
        Report an outcome to the circuit observer.

        This is the core method for feeding observations into the learning system.
        Components should call this after each significant operation.

        Args:
            items: List of vocabulary items involved in this operation
            success: Whether the operation succeeded
            confidence: Confidence score [0.0, 1.0] for this outcome
        """
        if self._circuit_observer is not None:
            self._circuit_observer.observe(
                items=items,
                success=success,
                confidence=confidence,
                context=self._component_context
            )

    def _get_item_prior(self, items: List[str]) -> float:
        """
        Get prior probability for items based on historical success.

        Queries the observer for the learned success rate of this item combination.
        Useful for deciding whether to attempt an operation.

        Args:
            items: List of vocabulary items to query

        Returns:
            Prior probability [0.0, 1.0], or 0.5 if no observer attached
        """
        if self._circuit_observer is None:
            return 0.5  # Neutral prior when not learning
        return self._circuit_observer.get_prior(items)

    def _should_try_items(self, items: List[str], threshold: float = 0.2) -> bool:
        """
        Check if items have sufficient success rate to try.

        Uses learned priors to filter out low-probability attempts.
        This is a key optimization: avoid repeating known failures.

        Args:
            items: List of vocabulary items to check
            threshold: Minimum prior probability to proceed (default: 0.2)

        Returns:
            True if prior >= threshold, False otherwise
        """
        prior = self._get_item_prior(items)
        return prior >= threshold
