"""
DivergenceCalculator: Measures drift between target and generated output.

The DivergenceCalculator is the "judge" of the Resonant Cavity - it determines
whether generated output aligns with holographic truth constraints.

It computes weighted similarity between target and generated tensors,
accounting for per-slot confidence, and returns accept/reject decisions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Type

import torch

from hologram.config.constants import (
    DIVERGENCE_ACCEPT_THRESHOLD,
    DIVERGENCE_HARD_THRESHOLD,
    DIVERGENCE_SOFT_THRESHOLD,
)
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity


class DivergenceAction(Enum):
    """Action to take based on divergence calculation."""
    ACCEPT = "accept"                       # Token is good, proceed
    ACCEPT_WITH_CORRECTION = "accept_with_correction"  # Accept but apply correction
    REJECT = "reject"                       # Reject and resample


@dataclass
class DivergenceThresholds:
    """Threshold configuration for divergence decisions.

    Attributes:
        accept: Above this similarity - accept without correction
        soft_reject: Between soft and accept - accept with correction
        hard_reject: Below this - reject and resample
    """
    accept: float = DIVERGENCE_ACCEPT_THRESHOLD
    soft_reject: float = DIVERGENCE_SOFT_THRESHOLD
    hard_reject: float = DIVERGENCE_HARD_THRESHOLD


@dataclass
class DivergenceResult:
    """Result of divergence calculation.

    Attributes:
        similarity: Weighted similarity score [0, 1]
        action: Decision (ACCEPT, ACCEPT_WITH_CORRECTION, REJECT)
        error_signal: Correction vector for feedback (if applicable)
        inject_disfluency: Whether to emit a filler token
        role: Current role being evaluated
    """
    similarity: float
    action: DivergenceAction
    error_signal: Optional[torch.Tensor]
    inject_disfluency: bool
    role: str

    def __str__(self) -> str:
        return (
            f"DivergenceResult({self.action.value}, "
            f"sim={self.similarity:.3f}, role={self.role})"
        )


class DivergenceCalculator:
    """
    Measures semantic drift and determines action.

    Calculates weighted similarity combining:
    - Global alignment: How well does T_generated match T_target overall?
    - Role-specific alignment: How well does the specific slot match?

    The weights are adjusted by per-slot confidence from the Resonator.

    Attributes:
        _codebook: Codebook for role vectors
        _ops: Operations class
        _thresholds: DivergenceThresholds configuration
        _history: List of DivergenceResults for auditing

    Example:
        >>> calc = DivergenceCalculator(codebook)
        >>> result = calc.calculate(target, generated, "SUBJECT", confidence_map)
        >>> if result.action == DivergenceAction.REJECT:
        ...     resample_token()
    """

    def __init__(
        self,
        codebook: Codebook,
        operations: Type[Operations] = Operations,
        thresholds: Optional[DivergenceThresholds] = None,
    ):
        """
        Initialize divergence calculator.

        Args:
            codebook: Codebook for role vectors
            operations: Operations class (default: Operations)
            thresholds: Threshold configuration (default: from constants)
        """
        self._codebook = codebook
        self._ops = operations
        self._thresholds = thresholds or DivergenceThresholds()
        self._history: List[DivergenceResult] = []

    def calculate(
        self,
        target: torch.Tensor,
        generated: torch.Tensor,
        current_role: str,
        confidence_map: Optional[Dict[str, float]] = None,
    ) -> DivergenceResult:
        """
        Calculate divergence and determine action.

        Computes weighted similarity:
        weighted_sim = 0.3 * global_sim + 0.7 * role_sim * slot_confidence

        Args:
            target: Target constraint tensor (T_target)
            generated: Re-encoded generated tensor (T_generated)
            current_role: Role being evaluated ("SUBJECT", "VERB", "OBJECT")
            confidence_map: Per-slot confidence from Resonator (optional)

        Returns:
            DivergenceResult with action and metadata
        """
        confidence_map = confidence_map or {}

        # Global semantic alignment
        global_sim = Similarity.cosine(target, generated)

        # Role-specific alignment
        role_vec = self._codebook.get_role(current_role)
        target_concept = self._ops.unbind(target, role_vec)
        generated_concept = self._ops.unbind(generated, role_vec)
        role_sim = Similarity.cosine(target_concept, generated_concept)

        # Weight by slot confidence (default 0.5 if not provided)
        slot_confidence = confidence_map.get(current_role.lower(), 0.5)

        # Combined weighted similarity
        weighted_sim = (0.3 * global_sim) + (0.7 * role_sim * slot_confidence)

        # Determine action
        action, error_signal = self._determine_action(
            weighted_sim, target, generated
        )

        # Check if disfluency should be injected
        inject_disfluency = weighted_sim < self._thresholds.soft_reject

        result = DivergenceResult(
            similarity=weighted_sim,
            action=action,
            error_signal=error_signal,
            inject_disfluency=inject_disfluency,
            role=current_role,
        )

        self._history.append(result)
        return result

    def _determine_action(
        self,
        similarity: float,
        target: torch.Tensor,
        generated: torch.Tensor,
    ) -> tuple[DivergenceAction, Optional[torch.Tensor]]:
        """
        Determine action based on similarity thresholds.

        Args:
            similarity: Weighted similarity score
            target: Target tensor
            generated: Generated tensor

        Returns:
            Tuple of (action, error_signal)
        """
        if similarity >= self._thresholds.accept:
            return DivergenceAction.ACCEPT, None

        elif similarity >= self._thresholds.soft_reject:
            # Generate mild correction signal
            error_signal = target - generated
            return DivergenceAction.ACCEPT_WITH_CORRECTION, error_signal

        else:
            # Generate strong correction signal
            error_signal = (target - generated) * 2.0
            return DivergenceAction.REJECT, error_signal

    @property
    def divergence_history(self) -> List[DivergenceResult]:
        """Get full history of divergence calculations for auditing."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear history for new generation session."""
        self._history.clear()

    def get_acceptance_rate(self) -> float:
        """Calculate acceptance rate from history."""
        if not self._history:
            return 0.0

        accepted = sum(
            1 for r in self._history
            if r.action in (DivergenceAction.ACCEPT, DivergenceAction.ACCEPT_WITH_CORRECTION)
        )
        return accepted / len(self._history)

    def get_rejection_count(self) -> int:
        """Count rejections in history."""
        return sum(
            1 for r in self._history
            if r.action == DivergenceAction.REJECT
        )

    def __repr__(self) -> str:
        return (
            f"DivergenceCalculator(accept={self._thresholds.accept}, "
            f"soft={self._thresholds.soft_reject}, hard={self._thresholds.hard_reject})"
        )
