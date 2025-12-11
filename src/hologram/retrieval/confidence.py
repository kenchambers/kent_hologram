"""
ConfidenceScorer: Threshold management and confidence evaluation.

Determines when to respond vs. refuse based on resonance strength.
This is critical for the "bounded hallucination" property.
"""

from enum import Enum

from hologram.config.constants import (
    REFUSAL_CONFIDENCE_THRESHOLD,
    RESPONSE_CONFIDENCE_THRESHOLD,
)


class ConfidenceLevel(Enum):
    """Classification of confidence levels."""
    HIGH = "high"           # >= response_threshold: Respond confidently
    MEDIUM = "medium"       # Between thresholds: Hedge/low confidence
    LOW = "low"             # < refusal_threshold but > 0: Very uncertain
    INSUFFICIENT = "insufficient"  # ≈ 0: No information


class ResponseDecision(Enum):
    """Decision on how to respond."""
    RESPOND = "respond"     # Provide answer with confidence
    HEDGE = "hedge"         # Provide answer but express uncertainty
    REFUSE = "refuse"       # Refuse to answer ("I don't know")


class ConfidenceScorer:
    """
    Evaluates confidence and determines response strategy.

    Uses threshold-based decision making to distinguish between:
    - High confidence: Strong resonance, respond confidently
    - Medium confidence: Moderate resonance, hedge
    - Low confidence: Weak resonance, refuse to answer

    This implements the "bounded hallucination" property: when resonance
    is low, we explicitly refuse rather than fabricating an answer.

    Attributes:
        response_threshold: Minimum confidence for confident response
        refusal_threshold: Below this, refuse to answer

    Example:
        >>> scorer = ConfidenceScorer(response_threshold=0.6, refusal_threshold=0.3)
        >>> scorer.evaluate(0.85)
        ResponseDecision.RESPOND
        >>> scorer.evaluate(0.45)
        ResponseDecision.HEDGE
        >>> scorer.evaluate(0.15)
        ResponseDecision.REFUSE
    """

    def __init__(
        self,
        response_threshold: float = RESPONSE_CONFIDENCE_THRESHOLD,
        refusal_threshold: float = REFUSAL_CONFIDENCE_THRESHOLD
    ):
        """
        Initialize confidence scorer.

        Args:
            response_threshold: Minimum confidence for responding (default: 0.6)
            refusal_threshold: Below this, refuse (default: 0.3)

        Raises:
            ValueError: If thresholds invalid
        """
        if not (0.0 <= refusal_threshold <= response_threshold <= 1.0):
            raise ValueError(
                f"Invalid thresholds: refusal={refusal_threshold}, "
                f"response={response_threshold}. "
                "Must satisfy: 0 <= refusal <= response <= 1"
            )

        self.response_threshold = response_threshold
        self.refusal_threshold = refusal_threshold

    def evaluate(self, resonance: float) -> ResponseDecision:
        """
        Evaluate resonance and decide how to respond.

        Args:
            resonance: Cosine similarity score [0, 1]

        Returns:
            ResponseDecision: RESPOND, HEDGE, or REFUSE

        Example:
            >>> scorer = ConfidenceScorer()
            >>> scorer.evaluate(0.85)
            ResponseDecision.RESPOND
        """
        if resonance >= self.response_threshold:
            return ResponseDecision.RESPOND
        elif resonance >= self.refusal_threshold:
            return ResponseDecision.HEDGE
        else:
            return ResponseDecision.REFUSE

    def get_confidence_level(self, resonance: float) -> ConfidenceLevel:
        """
        Classify confidence level.

        Args:
            resonance: Cosine similarity score [0, 1]

        Returns:
            ConfidenceLevel enum

        Example:
            >>> scorer = ConfidenceScorer()
            >>> scorer.get_confidence_level(0.95)
            ConfidenceLevel.HIGH
        """
        if resonance >= self.response_threshold:
            return ConfidenceLevel.HIGH
        elif resonance >= self.refusal_threshold:
            return ConfidenceLevel.MEDIUM
        elif resonance > 0.1:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.INSUFFICIENT

    def should_respond(self, resonance: float) -> bool:
        """
        Check if confidence is high enough to respond.

        Args:
            resonance: Cosine similarity score

        Returns:
            True if should respond (>= response_threshold)

        Example:
            >>> scorer = ConfidenceScorer(response_threshold=0.6)
            >>> scorer.should_respond(0.75)
            True
            >>> scorer.should_respond(0.45)
            False
        """
        return resonance >= self.response_threshold

    def should_refuse(self, resonance: float) -> bool:
        """
        Check if confidence is too low (should refuse).

        Args:
            resonance: Cosine similarity score

        Returns:
            True if should refuse (< refusal_threshold)

        Example:
            >>> scorer = ConfidenceScorer(refusal_threshold=0.3)
            >>> scorer.should_refuse(0.15)
            True
            >>> scorer.should_refuse(0.45)
            False
        """
        return resonance < self.refusal_threshold

    def format_response(
        self,
        answer: str,
        resonance: float,
        include_confidence: bool = True
    ) -> str:
        """
        Format response with appropriate confidence language.

        Args:
            answer: The answer to format
            resonance: Confidence score
            include_confidence: Whether to show confidence score

        Returns:
            Formatted response string

        Example:
            >>> scorer = ConfidenceScorer()
            >>> scorer.format_response("Paris", 0.95)
            'Paris (confidence: 95%)'
            >>> scorer.format_response("Berlin", 0.45)
            'Uncertain: Berlin (confidence: 45%)'
        """
        decision = self.evaluate(resonance)

        if decision == ResponseDecision.REFUSE:
            return "I don't have sufficient information to answer that question."

        conf_str = f" (confidence: {resonance:.0%})" if include_confidence else ""

        if decision == ResponseDecision.RESPOND:
            return f"{answer}{conf_str}"
        else:  # HEDGE
            return f"Uncertain: {answer}{conf_str}"

    def __repr__(self) -> str:
        return (
            f"ConfidenceScorer(response={self.response_threshold:.2f}, "
            f"refusal={self.refusal_threshold:.2f})"
        )
