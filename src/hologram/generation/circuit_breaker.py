"""
Simple Circuit Breaker for generation failure detection.

Implements a minimal circuit breaker pattern inspired by MIRAS/Titans research:
tracks generation failures and automatically trips to fall back to templates
when failure threshold is exceeded.

This provides the essential feedback loop for memory systems: detect failure
and adapt behavior without requiring neural gradients (which are not applicable
to HDC systems).
"""

import time
from enum import Enum
from typing import List, Tuple


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal: generation allowed
    OPEN = "open"          # Tripped: use templates only


class SimpleCircuitBreaker:
    """
    Tracks generation failures and trips when threshold exceeded.
    
    Maintains a sliding window of recent generation attempts and trips
    the circuit when failure count exceeds threshold. Automatically resets
    after cooldown period to allow recovery testing.
    
    Attributes:
        failure_threshold: Number of failures needed to trip circuit
        window_size: Number of recent attempts to track
        cooldown_seconds: Seconds to wait before resetting (allowing retry)
        _history: List of (timestamp, failed) tuples
        _tripped_at: Timestamp when circuit was tripped (0.0 if not tripped)
    
    Example:
        >>> breaker = SimpleCircuitBreaker(failure_threshold=3, window_size=10)
        >>> if not breaker.is_open():
        ...     result = generate()
        ...     breaker.record(failed=(result is None))
    """
    
    def __init__(
        self,
        failure_threshold: int = 3,
        window_size: int = 10,
        cooldown_seconds: float = 60.0
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures needed to trip (default: 3)
            window_size: Number of recent attempts to track (default: 10)
            cooldown_seconds: Seconds before resetting after trip (default: 60.0)
        """
        self.failure_threshold = failure_threshold
        self.window_size = window_size
        self.cooldown_seconds = cooldown_seconds
        self._history: List[Tuple[float, bool]] = []  # (timestamp, failed)
        self._tripped_at: float = 0.0
    
    def record(self, failed: bool) -> None:
        """
        Record a generation attempt result.
        
        Args:
            failed: True if generation failed validation, False if succeeded
        """
        now = time.time()
        self._history.append((now, failed))
        # Keep only recent history (sliding window)
        self._history = self._history[-self.window_size:]
    
    def is_open(self) -> bool:
        """
        Check if circuit breaker is open (tripped).
        
        Returns:
            True if circuit is open (generation should be blocked),
            False if circuit is closed (generation allowed)
        """
        now = time.time()
        
        # If tripped, check if cooldown period has passed
        if self._tripped_at > 0:
            if now - self._tripped_at > self.cooldown_seconds:
                # Cooldown expired - reset to allow testing
                self._tripped_at = 0.0
            else:
                # Still in cooldown - circuit remains open
                return True
        
        # Count recent failures
        recent_failures = sum(1 for _, f in self._history if f)
        
        # Trip if threshold exceeded
        if recent_failures >= self.failure_threshold:
            self._tripped_at = now
            return True
        
        return False
    
    def get_state(self) -> CircuitState:
        """
        Get current circuit breaker state.
        
        Returns:
            CircuitState.CLOSED or CircuitState.OPEN
        """
        return CircuitState.OPEN if self.is_open() else CircuitState.CLOSED
    
    def reset(self) -> None:
        """Manually reset circuit breaker (for testing/debugging)."""
        self._tripped_at = 0.0
        self._history = []
