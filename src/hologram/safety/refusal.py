"""
RefusalPolicy: "I don't know" logic for bounded hallucination.

Implements the core safety mechanism: when resonance is low, refuse to
answer rather than fabricating a response. This is the primary defense
against hallucination.
"""

from dataclasses import dataclass
from typing import Optional

from hologram.retrieval.confidence import ConfidenceScorer, ResponseDecision


@dataclass
class RefusalDecision:
    """
    Result of refusal evaluation.

    Attributes:
        should_refuse: Whether to refuse answering
        reason: Explanation for refusal
        confidence: The confidence score that triggered refusal
        suggestion: Optional suggestion for user (what to ask instead)
    """
    should_refuse: bool
    reason: str
    confidence: float
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        if not self.should_refuse:
            return "No refusal"
        msg = f"Refusal: {self.reason} (confidence: {self.confidence:.2%})"
        if self.suggestion:
            msg += f"\nSuggestion: {self.suggestion}"
        return msg


class RefusalPolicy:
    """
    Refusal policy for zero-hallucination.

    Determines when to refuse answering based on confidence scores.
    Implements the principle: It's better to say "I don't know" than
    to fabricate an answer.

    Key principle: We can only provide answers when we have sufficient
    resonance with stored facts. Low resonance = honest uncertainty.

    Attributes:
        confidence_scorer: ConfidenceScorer for threshold management
        refusal_messages: Templates for different refusal reasons

    Example:
        >>> policy = RefusalPolicy()
        >>> decision = policy.evaluate(answer="Paris", confidence=0.15)
        >>> decision.should_refuse
        True
        >>> decision.reason
        'Insufficient confidence'
    """

    DEFAULT_MESSAGES = {
        "low_confidence": "I don't have sufficient confidence to answer that question.",
        "no_information": "I don't have any information about that topic.",
        "conflicting": "I found conflicting information and cannot provide a definitive answer.",
        "out_of_scope": "That question is outside my knowledge base.",
    }

    def __init__(
        self,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        custom_messages: Optional[dict[str, str]] = None
    ):
        """
        Initialize refusal policy.

        Args:
            confidence_scorer: ConfidenceScorer instance (creates default if None)
            custom_messages: Custom refusal message templates
        """
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.refusal_messages = {**self.DEFAULT_MESSAGES}
        if custom_messages:
            self.refusal_messages.update(custom_messages)

    def evaluate(
        self,
        answer: str,
        confidence: float,
        has_contradictions: bool = False
    ) -> RefusalDecision:
        """
        Evaluate whether to refuse answering.

        Refuses if:
        - Confidence below refusal threshold
        - Contradictions found (even with high confidence)
        - No answer retrieved (empty/None)

        Args:
            answer: The retrieved answer
            confidence: Confidence score [0, 1]
            has_contradictions: Whether contradictions were detected

        Returns:
            RefusalDecision with refusal status and reason

        Example:
            >>> policy = RefusalPolicy()
            >>> policy.evaluate("Paris", 0.85)
            RefusalDecision(should_refuse=False, ...)
            >>> policy.evaluate("Unknown", 0.15)
            RefusalDecision(should_refuse=True, reason='Insufficient confidence', ...)
        """
        # Check for contradictions first (highest priority)
        if has_contradictions:
            return RefusalDecision(
                should_refuse=True,
                reason=self.refusal_messages["conflicting"],
                confidence=confidence
            )

        # Check if answer is empty/None
        if not answer or answer.strip() == "":
            return RefusalDecision(
                should_refuse=True,
                reason=self.refusal_messages["no_information"],
                confidence=0.0
            )

        # Check confidence threshold
        decision = self.confidence_scorer.evaluate(confidence)

        if decision == ResponseDecision.REFUSE:
            return RefusalDecision(
                should_refuse=True,
                reason=self.refusal_messages["low_confidence"],
                confidence=confidence,
                suggestion=self._generate_suggestion(answer, confidence)
            )

        # All checks passed - safe to respond
        return RefusalDecision(
            should_refuse=False,
            reason="",
            confidence=confidence
        )

    def format_refusal(self, decision: RefusalDecision) -> str:
        """
        Format a refusal message for display.

        Args:
            decision: RefusalDecision to format

        Returns:
            Human-readable refusal message

        Example:
            >>> policy = RefusalPolicy()
            >>> decision = RefusalDecision(
            ...     should_refuse=True,
            ...     reason="Low confidence",
            ...     confidence=0.15
            ... )
            >>> policy.format_refusal(decision)
            "I don't have sufficient confidence to answer that question."
        """
        if not decision.should_refuse:
            return ""

        message = decision.reason

        # Add suggestion if available
        if decision.suggestion:
            message += f"\n\n{decision.suggestion}"

        return message

    def _generate_suggestion(
        self,
        attempted_answer: str,
        confidence: float
    ) -> Optional[str]:
        """
        Generate a suggestion for what user could ask instead.

        Args:
            attempted_answer: The answer that had low confidence
            confidence: The confidence score

        Returns:
            Suggestion string or None
        """
        # For now, simple implementation
        # Could be enhanced with vocabulary analysis
        if confidence > 0.1:
            return (
                f"I found a possible answer ('{attempted_answer}') but with low "
                "confidence. Try rephrasing your question or providing more context."
            )
        return None

    def __repr__(self) -> str:
        return f"RefusalPolicy(scorer={self.confidence_scorer})"
