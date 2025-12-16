"""
Metacognitive Layer: Self-monitoring feedback loop.

Implements the "Prefrontal Cortex" metaphor from cognitive neuroscience:
- Observes its own state (confidence, confusion)
- Labels internal states ("I am confused", "I am confident")
- Rewires itself based on observations (adds "Curiosity" when stuck)
- Enables retry loops with modified query vectors

This transforms the system from reactive (query → answer) to adaptive
(query → observe → rewire → retry if needed).
"""

import logging
import inspect
from enum import Enum
from typing import Optional, Callable, Tuple, Any, List, TYPE_CHECKING

import torch

from hologram.core.codebook import Codebook
from hologram.core.operations import Operations

if TYPE_CHECKING:
    from hologram.introspection.circuit_observer import CircuitObserver


logger = logging.getLogger(__name__)


class MetacognitiveMood(Enum):
    """Internal state labels for the system."""
    
    NEUTRAL = "neutral"
    CONFIDENT = "confident"
    CONFUSED = "confused"
    CURIOUS = "curious"
    ANXIOUS = "anxious"


class MetacognitiveState:
    """
    Persistent self-state vector for metacognitive awareness.
    
    Maintains a "mood" vector that persists across queries, allowing
    the system to adapt its behavior based on past experiences.
    
    Attributes:
        self_vector: Persistent state vector (modulated by observations)
        mood: Current mood label
        codebook: Codebook for encoding mood concepts
        confidence_history: Recent confidence scores for trend analysis
    
    Example:
        >>> state = MetacognitiveState(codebook)
        >>> state.update_from_confidence(0.9)  # High confidence
        >>> state.mood
        MetacognitiveMood.CONFIDENT
        >>> state.self_vector  # Updated with confidence signal
    """
    
    def __init__(self, codebook: Codebook, initial_mood: MetacognitiveMood = MetacognitiveMood.NEUTRAL):
        """
        Initialize metacognitive state.
        
        Args:
            codebook: Codebook for encoding mood vectors
            initial_mood: Starting mood (default: NEUTRAL)
        """
        self._codebook = codebook
        self.mood = initial_mood
        
        # Initialize self_vector as neutral state
        self.self_vector = self._codebook.encode("__METACOGNITIVE_NEUTRAL__")
        
        # Track recent confidence for trend analysis
        self.confidence_history: list[float] = []
        self.max_history = 10
    
    def update_from_confidence(self, confidence: float, weight: float = 0.1) -> None:
        """
        Update self-state based on query confidence.
        
        Implements the "labeling" process: observes confidence and updates
        the self_vector accordingly. This is the core metacognitive operation.
        
        Args:
            confidence: Confidence score from query [0, 1]
            weight: How strongly to update (default: 0.1)
        """
        # Track history
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > self.max_history:
            self.confidence_history.pop(0)
        
        # Label the state based on confidence
        if confidence >= 0.8:
            # High confidence → reinforce confidence
            confidence_vec = self._codebook.encode("__METACOGNITIVE_CONFIDENCE__")
            self.self_vector = Operations.bundle(
                self.self_vector,
                confidence_vec * weight
            )
            self.mood = MetacognitiveMood.CONFIDENT
            
        elif confidence <= 0.2:
            # Low confidence → label as confusion, but add curiosity
            confusion_vec = self._codebook.encode("__METACOGNITIVE_CONFUSION__")
            curiosity_vec = self._codebook.encode("__METACOGNITIVE_CURIOSITY__")
            
            # Add confusion (labeling the state)
            self.self_vector = Operations.bundle(
                self.self_vector,
                confusion_vec * weight
            )
            
            # Add curiosity (rewiring for retry)
            # Higher weight for curiosity to enable exploration
            self.self_vector = Operations.bundle(
                self.self_vector,
                curiosity_vec * (weight * 2.0)  # Curiosity stronger than confusion
            )
            
            self.mood = MetacognitiveMood.CONFUSED
            
        elif confidence <= 0.4:
            # Medium-low confidence → anxious but curious
            anxiety_vec = self._codebook.encode("__METACOGNITIVE_ANXIETY__")
            curiosity_vec = self._codebook.encode("__METACOGNITIVE_CURIOSITY__")
            
            self.self_vector = Operations.bundle(
                self.self_vector,
                anxiety_vec * weight
            )
            self.self_vector = Operations.bundle(
                self.self_vector,
                curiosity_vec * weight
            )
            
            self.mood = MetacognitiveMood.ANXIOUS
            
        else:
            # Medium confidence → neutral/curious
            curiosity_vec = self._codebook.encode("__METACOGNITIVE_CURIOSITY__")
            self.self_vector = Operations.bundle(
                self.self_vector,
                curiosity_vec * weight * 0.5
            )
            self.mood = MetacognitiveMood.CURIOUS
        
        # Normalize to prevent drift
        norm = torch.norm(self.self_vector)
        if norm > 1e-6:
            self.self_vector = self.self_vector / norm
    
    def get_confidence_trend(self) -> float:
        """
        Get trend in confidence (positive = improving, negative = degrading).
        
        Returns:
            Trend value: positive if confidence is increasing, negative if decreasing
        """
        if len(self.confidence_history) < 2:
            return 0.0
        
        # Simple linear trend
        recent = self.confidence_history[-5:]  # Last 5 queries
        if len(recent) < 2:
            return 0.0
        
        # Calculate slope
        x = torch.arange(len(recent), dtype=torch.float32)
        y = torch.tensor(recent, dtype=torch.float32)
        
        # Linear regression slope
        mean_x = x.mean()
        mean_y = y.mean()
        
        numerator = ((x - mean_x) * (y - mean_y)).sum()
        denominator = ((x - mean_x) ** 2).sum()
        
        if denominator < 1e-6:
            return 0.0
        
        slope = numerator / denominator
        return float(slope)
    
    def reset(self) -> None:
        """Reset state to neutral."""
        self.self_vector = self._codebook.encode("__METACOGNITIVE_NEUTRAL__")
        self.mood = MetacognitiveMood.NEUTRAL
        self.confidence_history.clear()
    
    def __repr__(self) -> str:
        trend = self.get_confidence_trend()
        trend_str = f"+{trend:.2f}" if trend > 0 else f"{trend:.2f}"
        return (
            f"MetacognitiveState(mood={self.mood.value}, "
            f"history={len(self.confidence_history)}, trend={trend_str})"
        )


class MetacognitiveLoop:
    """
    Metacognitive feedback loop wrapper for queries.
    
    Wraps any query function with metacognitive awareness:
    1. Modifies query using self-state vector
    2. Executes query
    3. Observes confidence
    4. Updates self-state
    5. Retries with rewired state if confidence is low
    
    This implements the "rewiring" mechanism: when stuck, the system
    doesn't give up - it changes its own state and tries again.
    
    Attributes:
        state: MetacognitiveState instance
        max_retries: Maximum retry attempts (default: 2)
        retry_threshold: Confidence threshold for retry (default: 0.3)
    
    Example:
        >>> loop = MetacognitiveLoop(codebook)
        >>> result, confidence = loop.execute_query(
        ...     query_func=lambda q: fact_store.query("France", "capital"),
        ...     query_text="What is the capital of France?"
        ... )
        >>> # If initial query fails, automatically retries with rewired state
    """
    
    def __init__(
        self,
        codebook: Codebook,
        max_retries: int = 2,
        retry_threshold: float = 0.3,
    ):
        """
        Initialize metacognitive loop.
        
        Args:
            codebook: Codebook for encoding
            max_retries: Maximum retry attempts (default: 2)
            retry_threshold: Confidence below which to retry (default: 0.3)
        """
        self.state = MetacognitiveState(codebook)
        self.max_retries = max_retries
        self.retry_threshold = retry_threshold
        # Self-improvement integration
        self._circuit_observer: Optional['CircuitObserver'] = None

    def set_circuit_observer(self, observer: 'CircuitObserver') -> None:
        """
        Attach circuit observer for self-improvement tracking.

        Args:
            observer: CircuitObserver instance to receive observations
        """
        self._circuit_observer = observer

    def _extract_query_items(self, query_text: str) -> List[str]:
        """
        Extract vocabulary items from query text.

        Simple tokenization that filters out very short words.
        More sophisticated extraction could be added in the future.

        Args:
            query_text: Query string to extract from

        Returns:
            List of vocabulary items (words)
        """
        # Simple word extraction: lowercase, filter short words
        words = query_text.lower().split()
        return [w.strip('.,!?;:()[]{}') for w in words if len(w) > 2]
    
    def execute_query(
        self,
        query_func: Callable[[str], Tuple[Any, float]],
        query_text: str,
        context_vector: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, float]:
        """
        Execute query with metacognitive feedback loop.
        
        This is the core "thinking" process:
        1. Contextualize query with self-state
        2. Execute query
        3. Observe confidence
        4. Update self-state
        5. Retry if confidence is low (rewiring)
        
        Args:
            query_func: Function that takes query text and returns (result, confidence)
            query_text: Original query text
            context_vector: Optional context vector to bind with self-state
            
        Returns:
            Tuple of (result, confidence) from best attempt
        """
        best_result = None
        best_confidence = 0.0
        attempt = 0
        
        while attempt <= self.max_retries:
            # Step 1: Contextualize query with self-state
            # The self_vector modulates how we interpret the query
            # This is the "rewiring" - same query, different perspective
            
            modulated_context = None
            if context_vector is not None:
                # Bind context with self-state for richer modulation
                modulated_context = Operations.bind(context_vector, self.state.self_vector)
            
            # Step 2: Execute query
            result, confidence = self._invoke_query(query_func, query_text, modulated_context)
            
            # Step 3: Observe and update state
            self.state.update_from_confidence(confidence)

            # Report to circuit observer for self-improvement learning
            if self._circuit_observer is not None:
                items = self._extract_query_items(query_text)
                self._circuit_observer.observe(
                    items=items,
                    success=(confidence >= self.retry_threshold),
                    confidence=confidence,
                    context="metacognitive_query"
                )

            # Track best result
            if confidence > best_confidence:
                best_result = result
                best_confidence = confidence
            
            # Step 4: Decide whether to retry
            if confidence >= self.retry_threshold:
                # Good enough - return result
                return result, confidence
            
            # Step 5: Retry if we haven't exceeded max_retries
            if attempt < self.max_retries:
                # The self-state has been updated (rewired) with curiosity
                # This will change how the next query is processed
                attempt += 1
                # Continue loop to retry with rewired state
            else:
                # Max retries reached
                break
        
        # Return best result even if below threshold
        return best_result, best_confidence
    
    def contextualize_query(
        self,
        query_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contextualize a query vector with self-state.
        
        Binds the query with self-state to create a modulated query.
        This is the "rewiring" operation: the same query, viewed through
        the lens of current self-awareness.
        
        Args:
            query_vector: Original query vector
            
        Returns:
            Contextualized query vector
        """
        return Operations.bind(query_vector, self.state.self_vector)

    def _invoke_query(
        self,
        query_func: Callable[..., Tuple[Any, float]],
        query_text: str,
        modulated_context: Optional[torch.Tensor],
    ) -> Tuple[Any, float]:
        """
        Call query_func, passing modulated_context when supported.

        This preserves backward compatibility with existing single-argument
        call sites while enabling richer context for functions that opt in.
        """
        sig = inspect.signature(query_func)
        params = sig.parameters

        # Prefer explicit keyword if accepted
        if "context_vector" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            return query_func(query_text, context_vector=modulated_context)

        if "context" in params:
            return query_func(query_text, context=modulated_context)

        # Fallback: if the callable accepts 2 positional-or-keyword args
        positional_params = [
            p for p in params.values() if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional_params) >= 2:
            try:
                return query_func(query_text, modulated_context)
            except TypeError as exc:
                logger.debug("Query function rejected context positional arg: %s", exc)

        # Legacy signature: only query_text
        return query_func(query_text)
    
    def reset(self) -> None:
        """Reset metacognitive state."""
        self.state.reset()
    
    def __repr__(self) -> str:
        return (
            f"MetacognitiveLoop(state={self.state}, "
            f"max_retries={self.max_retries}, "
            f"threshold={self.retry_threshold})"
        )
