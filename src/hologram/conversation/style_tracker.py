"""
User style tracking via holographic traces.

Tracks user communication style and adapts responses accordingly.
"""

from typing import Optional

import torch

from hologram.config.constants import STYLE_ADAPTATION_MIN_MESSAGES
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity
from hologram.modulation.sesame import SesameModulator, StyleType


class UserStyleTracker:
    """
    Track user's communication style via holographic trace.

    Bundles user messages into a style trace and compares against
    style prototypes to infer preferred communication style.

    Attributes:
        _codebook: Codebook for encoding
        _sesame: SesameModulator with style vectors
        _style_trace: Bundled user message vectors
        _message_count: Number of observed messages

    Example:
        >>> tracker = UserStyleTracker(codebook, sesame)
        >>> tracker.observe("Hey! What's up?")
        >>> tracker.observe("Cool, thanks!")
        >>> style = tracker.get_inferred_style()
        >>> # Returns StyleType.CASUAL
    """

    def __init__(
        self,
        codebook: Codebook,
        sesame: SesameModulator,
        min_messages: int = STYLE_ADAPTATION_MIN_MESSAGES,
    ):
        """
        Initialize style tracker.

        Args:
            codebook: Shared Codebook instance
            sesame: SesameModulator with style prototypes
            min_messages: Minimum messages before inferring style
        """
        self._codebook = codebook
        self._sesame = sesame
        self._min_messages = min_messages
        self._style_trace: Optional[torch.Tensor] = None
        self._message_count: int = 0

    def observe(self, text: str) -> None:
        """
        Observe user message and update style trace.

        Args:
            text: User message text
        """
        # Tokenize and encode
        tokens = self._tokenize(text)
        if not tokens:
            return

        # Create message vector
        token_vecs = [self._codebook.encode(t) for t in tokens]
        message_vec = token_vecs[0]
        for vec in token_vecs[1:]:
            message_vec = Operations.bundle(message_vec, vec)

        # Bundle into style trace
        if self._style_trace is None:
            self._style_trace = message_vec
        else:
            self._style_trace = Operations.bundle(self._style_trace, message_vec)

        self._message_count += 1

    def get_inferred_style(self) -> StyleType:
        """
        Infer user's preferred style from trace.

        Returns:
            Best matching StyleType, or NEUTRAL if insufficient data
        """
        if self._message_count < self._min_messages:
            return StyleType.NEUTRAL

        if self._style_trace is None:
            return StyleType.NEUTRAL

        # Compare against style prototypes
        best_style = StyleType.NEUTRAL
        best_sim = -1.0

        for style in [StyleType.FORMAL, StyleType.CASUAL, StyleType.URGENT]:
            style_vec = self._sesame.get_style_vector(style)
            sim = float(Similarity.cosine(self._style_trace, style_vec))

            if sim > best_sim:
                best_sim = sim
                best_style = style

        # Only return non-neutral if similarity is meaningful
        if best_sim > 0.1:
            return best_style

        return StyleType.NEUTRAL

    def get_style_confidence(self) -> float:
        """
        How confident are we in the inferred style?

        Returns:
            Confidence score (0-1) based on message count and similarity
        """
        if self._message_count < self._min_messages:
            return 0.0

        if self._style_trace is None:
            return 0.0

        # Get similarity to best style
        style = self.get_inferred_style()
        if style == StyleType.NEUTRAL:
            return 0.3  # Low confidence for neutral

        style_vec = self._sesame.get_style_vector(style)
        sim = float(Similarity.cosine(self._style_trace, style_vec))

        # Scale by message count (more messages = higher confidence)
        count_factor = min(self._message_count / 10.0, 1.0)

        return float(sim * count_factor)

    def get_style_scores(self) -> dict:
        """
        Get similarity scores to all styles.

        Returns:
            Dictionary mapping style names to similarity scores
        """
        if self._style_trace is None:
            return {s.value: 0.0 for s in StyleType}

        scores = {}
        for style in StyleType:
            if style == StyleType.NEUTRAL:
                scores[style.value] = 0.0
            else:
                style_vec = self._sesame.get_style_vector(style)
                scores[style.value] = float(
                    Similarity.cosine(self._style_trace, style_vec)
                )

        return scores

    def _tokenize(self, text: str) -> list:
        """Simple tokenization."""
        import re

        text = re.sub(r"[^\w\s']", " ", text.lower())
        return [t for t in text.split() if t]

    def reset(self) -> None:
        """Reset style tracking for new session."""
        self._style_trace = None
        self._message_count = 0

    @property
    def message_count(self) -> int:
        """Number of observed messages."""
        return self._message_count

    def __repr__(self) -> str:
        style = self.get_inferred_style()
        return f"UserStyleTracker(messages={self._message_count}, style={style.value})"
