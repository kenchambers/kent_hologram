"""
SesameModulator: Style modulation and disfluency injection.

The Sesame layer adds human-like texture to generated output through:
1. Style vectors - bias word selection toward formal/casual/urgent styles
2. Disfluency injection - insert "um", "uh", "..." when confidence is low

Named after Sesame for its role in adding "flavor" to output.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Type

import torch

from hologram.config.constants import (
    CREATIVITY_TEMPERATURE,
    DISFLUENCY_THRESHOLD,
    LAMBDA_STYLE,
    STYLE_CASUAL_WORDS,
    STYLE_FORMAL_WORDS,
    STYLE_URGENT_WORDS,
)
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity


class StyleType(Enum):
    """Pre-defined style types."""
    FORMAL = "formal"
    CASUAL = "casual"
    URGENT = "urgent"
    NEUTRAL = "neutral"


class FillerType(Enum):
    """Disfluency filler types."""
    UM = "um"
    UH = "uh"
    PAUSE = "..."


@dataclass
class ModulatedCleanup:
    """Result of modulated cleanup operation.

    Attributes:
        word: Selected word from vocabulary
        score: Combined score (semantic + style)
        semantic_score: Pure semantic match score
        style_score: Style contribution score
    """
    word: str
    score: float
    semantic_score: float
    style_score: float

    def __str__(self) -> str:
        return f"ModulatedCleanup({self.word}, score={self.score:.3f})"


class SesameModulator:
    """
    Style modulation and disfluency injection.

    Provides style vectors that bias word selection without overriding
    semantics, and injects fillers when confidence is low.

    Style influence formula:
    Score = Sim(proposal, word) + lambda * Sim(style, word)

    The lambda parameter controls how much style affects selection.
    Default 0.2 means style nudges but doesn't override semantics.

    Attributes:
        _codebook: Codebook for encoding words
        _ops: Operations class
        _creativity: Creativity temperature (affects variance)
        _style_vectors: Cached style vectors

    Example:
        >>> sesame = SesameModulator(codebook)
        >>> style_vec = sesame.get_style_vector(StyleType.FORMAL)
        >>> result = sesame.modulated_cleanup(proposal, vocab, style_vec)
        >>> if sesame.should_inject_disfluency(0.25):
        ...     filler = sesame.select_filler(0.25)
    """

    def __init__(
        self,
        codebook: Codebook,
        operations: Type[Operations] = Operations,
        creativity_temperature: float = CREATIVITY_TEMPERATURE,
    ):
        """
        Initialize Sesame modulator.

        Args:
            codebook: Codebook for encoding words
            operations: Operations class (default: Operations)
            creativity_temperature: Variance in word selection (default: 0.2)
        """
        self._codebook = codebook
        self._ops = operations
        self._creativity = creativity_temperature
        self._style_vectors: Dict[StyleType, torch.Tensor] = {}

        # Pre-compute style vectors
        self._init_style_vectors()

    def _init_style_vectors(self) -> None:
        """Initialize style vectors from word markers."""
        # Formal style: bundle of formal words
        formal_vecs = [self._codebook.encode(w) for w in STYLE_FORMAL_WORDS]
        self._style_vectors[StyleType.FORMAL] = self._ops.bundle(*formal_vecs)

        # Casual style: bundle of casual words
        casual_vecs = [self._codebook.encode(w) for w in STYLE_CASUAL_WORDS]
        self._style_vectors[StyleType.CASUAL] = self._ops.bundle(*casual_vecs)

        # Urgent style: bundle of urgent words
        urgent_vecs = [self._codebook.encode(w) for w in STYLE_URGENT_WORDS]
        self._style_vectors[StyleType.URGENT] = self._ops.bundle(*urgent_vecs)

        # Neutral: zero vector (no style bias)
        dim = self._style_vectors[StyleType.FORMAL].shape[0]
        self._style_vectors[StyleType.NEUTRAL] = torch.zeros(dim)

    def get_style_vector(self, style: StyleType) -> torch.Tensor:
        """
        Get pre-computed style vector.

        Args:
            style: StyleType enum value

        Returns:
            Style hypervector for the requested style

        Example:
            >>> formal_vec = sesame.get_style_vector(StyleType.FORMAL)
        """
        return self._style_vectors[style]

    def modulated_cleanup(
        self,
        proposal: torch.Tensor,
        vocabulary: List[str],
        style_vector: torch.Tensor,
        lambda_style: float = LAMBDA_STYLE,
    ) -> ModulatedCleanup:
        """
        Cleanup with style weighting.

        Score = Sim(proposal, word) + lambda * Sim(style, word)

        The semantic score dominates, but style can nudge selection
        between similarly-scored candidates.

        Args:
            proposal: Noisy proposal vector from resonator
            vocabulary: List of candidate words
            style_vector: Style modulation vector
            lambda_style: Style weight (default: 0.2)

        Returns:
            ModulatedCleanup with selected word and scores

        Example:
            >>> result = sesame.modulated_cleanup(proposal, vocab, formal_vec)
            >>> print(result.word)  # Might prefer "therefore" over "so"
        """
        # Encode vocabulary
        vocab_vectors = self._codebook.encode_batch(vocabulary)

        # Semantic scores
        semantic_scores = Similarity.cosine_batch(proposal, vocab_vectors)

        # Style scores (how well each word matches the style)
        style_scores = Similarity.cosine_batch(style_vector, vocab_vectors)

        # Combined scores
        combined_scores = semantic_scores + (lambda_style * style_scores)

        # Find best
        best_idx = int(torch.argmax(combined_scores).item())
        best_word = vocabulary[best_idx]

        return ModulatedCleanup(
            word=best_word,
            score=float(combined_scores[best_idx].item()),
            semantic_score=float(semantic_scores[best_idx].item()),
            style_score=float(style_scores[best_idx].item()),
        )

    def select_filler(self, confidence: float) -> FillerType:
        """
        Select appropriate filler based on confidence level.

        Lower confidence -> more prominent disfluency.

        Args:
            confidence: Confidence score [0, 1]

        Returns:
            FillerType enum value

        Example:
            >>> filler = sesame.select_filler(0.15)
            >>> filler
            FillerType.PAUSE  # Very low confidence -> long pause
        """
        if confidence < 0.2:
            return FillerType.PAUSE  # "..." - long pause for very uncertain
        elif confidence < 0.3:
            return FillerType.UM     # "um" - medium uncertainty
        else:
            return FillerType.UH     # "uh" - mild uncertainty

    def should_inject_disfluency(
        self,
        similarity: float,
        threshold: float = DISFLUENCY_THRESHOLD,
    ) -> bool:
        """
        Determine if filler should be injected.

        Args:
            similarity: Current similarity/confidence score
            threshold: Below this, inject disfluency (default: 0.35)

        Returns:
            True if disfluency should be injected

        Example:
            >>> if sesame.should_inject_disfluency(0.25):
            ...     output.append(sesame.select_filler(0.25).value)
        """
        return similarity < threshold

    def get_filler_text(self, filler: FillerType) -> str:
        """Get text representation of filler type."""
        return filler.value

    def __repr__(self) -> str:
        return (
            f"SesameModulator(creativity={self._creativity}, "
            f"styles={list(self._style_vectors.keys())})"
        )
