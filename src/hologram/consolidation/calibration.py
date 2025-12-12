"""
ConfidenceCalibrator: Normalize HDC and Neural confidence scores.

Problem: HDC cosine similarity [0.15, 0.6] and Neural softmax [0.1, 0.99]
are not directly comparable. Winner-take-all selection would be erratic.

Solution: Sigmoid calibration maps both to a shared probability space [0, 1].

Key Design Decisions:
- Sigmoid-based calibration (smooth, bounded, interpretable)
- Separate calibration for HDC and Neural (different raw distributions)
- Configurable thresholds for different use cases
- Unbinding validation gate for HDC algebra preservation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass
class CalibrationResult:
    """Result of confidence calibration and winner selection."""
    winner: Literal["hdc", "neural"]
    hdc_raw: float
    hdc_calibrated: float
    neural_raw: float
    neural_calibrated: float
    margin: float  # Absolute difference between calibrated scores

    @property
    def is_confident(self) -> bool:
        """Whether the winning score indicates high confidence."""
        winning_score = self.hdc_calibrated if self.winner == "hdc" else self.neural_calibrated
        return winning_score > 0.7

    @property
    def is_close_call(self) -> bool:
        """Whether the margin is small (uncertain winner)."""
        return self.margin < 0.1


class ConfidenceCalibrator:
    """
    Calibrates HDC and Neural confidence scores to a shared probability space.

    HDC typically produces cosine similarity in [0.15, 0.6] range.
    Neural softmax tends to be overconfident, producing [0.3, 0.99].

    This calibrator uses sigmoid functions to map both to [0, 1] in a way
    that makes them comparable for winner-take-all selection.

    Args:
        hdc_center: Center point for HDC sigmoid (default: 0.35)
        hdc_scale: Steepness of HDC sigmoid (default: 8.0)
        neural_center: Center point for Neural sigmoid (default: 0.5)
        neural_scale: Steepness of Neural sigmoid (default: 5.0)
        unbind_threshold: Minimum neural confidence for HDC algebra (default: 0.85)
    """

    def __init__(
        self,
        hdc_center: float = 0.35,
        hdc_scale: float = 8.0,
        neural_center: float = 0.5,
        neural_scale: float = 5.0,
        unbind_threshold: float = 0.85,
    ):
        self._hdc_center = hdc_center
        self._hdc_scale = hdc_scale
        self._neural_center = neural_center
        self._neural_scale = neural_scale
        self._unbind_threshold = unbind_threshold

    def calibrate_hdc(self, raw: float) -> float:
        """
        Calibrate HDC cosine similarity to [0, 1].

        HDC typical range: [0.15, 0.6]
        - 0.15 -> ~0.08 (very low confidence)
        - 0.35 -> 0.50 (medium confidence)
        - 0.60 -> ~0.92 (high confidence)

        Args:
            raw: Raw HDC cosine similarity

        Returns:
            Calibrated probability in [0, 1]
        """
        return self._sigmoid(raw, self._hdc_center, self._hdc_scale)

    def calibrate_neural(self, raw: float) -> float:
        """
        Calibrate Neural softmax probability to [0, 1].

        Neural tends to be overconfident, so we use a less steep sigmoid:
        - 0.3 -> ~0.27 (low confidence despite neural thinking 30%)
        - 0.5 -> 0.50 (medium confidence)
        - 0.9 -> ~0.88 (high confidence)

        Args:
            raw: Raw Neural softmax probability

        Returns:
            Calibrated probability in [0, 1]
        """
        return self._sigmoid(raw, self._neural_center, self._neural_scale)

    def _sigmoid(self, x: float, center: float, scale: float) -> float:
        """Apply sigmoid calibration."""
        return 1.0 / (1.0 + math.exp(-scale * (x - center)))

    def pick_winner(
        self,
        hdc_raw: float,
        neural_raw: float,
        require_unbind_safety: bool = False,
    ) -> CalibrationResult:
        """
        Select winner between HDC and Neural based on calibrated confidence.

        Args:
            hdc_raw: Raw HDC cosine similarity
            neural_raw: Raw Neural softmax probability
            require_unbind_safety: If True, reject neural if below unbind threshold

        Returns:
            CalibrationResult with winner selection and all scores
        """
        hdc_cal = self.calibrate_hdc(hdc_raw)
        neural_cal = self.calibrate_neural(neural_raw)

        # Determine initial winner
        if hdc_cal >= neural_cal:
            winner = "hdc"
        else:
            winner = "neural"

        # Apply unbinding safety gate if requested
        if require_unbind_safety and winner == "neural":
            if neural_raw < self._unbind_threshold:
                # Neural too uncertain for HDC algebra - fall back to HDC
                winner = "hdc"

        return CalibrationResult(
            winner=winner,
            hdc_raw=hdc_raw,
            hdc_calibrated=hdc_cal,
            neural_raw=neural_raw,
            neural_calibrated=neural_cal,
            margin=abs(hdc_cal - neural_cal),
        )

    def is_neural_safe_for_unbinding(self, neural_raw: float) -> bool:
        """
        Check if neural confidence is high enough for HDC unbinding operations.

        Neural-retrieved vectors with similarity < threshold break HDC algebra.
        This gate ensures we only use neural results for composition when safe.

        Args:
            neural_raw: Raw neural softmax probability

        Returns:
            True if neural result can be used for unbinding
        """
        return neural_raw >= self._unbind_threshold

    @property
    def unbind_threshold(self) -> float:
        """Get the unbinding safety threshold."""
        return self._unbind_threshold

    @unbind_threshold.setter
    def unbind_threshold(self, value: float) -> None:
        """Set the unbinding safety threshold."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Unbind threshold must be in [0, 1], got {value}")
        self._unbind_threshold = value

    def __repr__(self) -> str:
        return (
            f"ConfidenceCalibrator("
            f"hdc_center={self._hdc_center}, "
            f"neural_center={self._neural_center}, "
            f"unbind_threshold={self._unbind_threshold})"
        )


class AdaptiveCalibrator(ConfidenceCalibrator):
    """
    Calibrator that adapts parameters based on observed distributions.

    Tracks running statistics of HDC and Neural scores to automatically
    adjust calibration parameters. Useful when operating conditions vary.

    Args:
        window_size: Number of samples to track for adaptation
        adaptation_rate: How fast to adapt (0-1, higher = faster)
    """

    def __init__(
        self,
        window_size: int = 100,
        adaptation_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._window_size = window_size
        self._adaptation_rate = adaptation_rate

        # Running statistics
        self._hdc_scores: list[float] = []
        self._neural_scores: list[float] = []

    def observe_hdc(self, raw: float) -> None:
        """Record an HDC score for adaptation."""
        self._hdc_scores.append(raw)
        if len(self._hdc_scores) > self._window_size:
            self._hdc_scores.pop(0)
        self._maybe_adapt_hdc()

    def observe_neural(self, raw: float) -> None:
        """Record a Neural score for adaptation."""
        self._neural_scores.append(raw)
        if len(self._neural_scores) > self._window_size:
            self._neural_scores.pop(0)
        self._maybe_adapt_neural()

    def _maybe_adapt_hdc(self) -> None:
        """Adapt HDC calibration based on observed distribution."""
        if len(self._hdc_scores) < 10:
            return

        # Move center toward median of observed scores
        observed_median = sorted(self._hdc_scores)[len(self._hdc_scores) // 2]
        self._hdc_center += self._adaptation_rate * (observed_median - self._hdc_center)

    def _maybe_adapt_neural(self) -> None:
        """Adapt Neural calibration based on observed distribution."""
        if len(self._neural_scores) < 10:
            return

        # Move center toward median of observed scores
        observed_median = sorted(self._neural_scores)[len(self._neural_scores) // 2]
        self._neural_center += self._adaptation_rate * (observed_median - self._neural_center)
