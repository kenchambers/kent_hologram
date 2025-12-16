"""
CodeResonator: Factorize code observations into (OPERATION, FILE, LOCATION).

Thin wrapper around TransformationResonator that remaps roles:
- ACTION → OPERATION
- TARGET → FILE
- MODIFIER → LOCATION

Uses composition (not inheritance) following cursor-code review recommendation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch

from hologram.arc.transform_resonator import TransformationResonator, TransformResult
from hologram.swe.encoder import CodeEncoder


@dataclass
class CodeFactorization:
    """
    Result of code observation factorization.

    Maps TransformResult roles to code-specific semantics:
    - action → operation (what change: add_line, modify_function, etc.)
    - target → file (which file to change)
    - modifier → location (where in file: line number, function name)
    """
    operation: str
    file: str
    location: str
    operation_vec: torch.Tensor
    file_vec: torch.Tensor
    location_vec: torch.Tensor
    iterations: int
    converged: bool
    confidence: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_transform_result(cls, result: TransformResult) -> "CodeFactorization":
        """Create from TransformResult by remapping fields."""
        return cls(
            operation=result.action,
            file=result.target,
            location=result.modifier,
            operation_vec=result.action_vec,
            file_vec=result.target_vec,
            location_vec=result.modifier_vec,
            iterations=result.iterations,
            converged=result.converged,
            confidence={
                "operation": result.confidence.get("action", 0.0),
                "file": result.confidence.get("target", 0.0),
                "location": result.confidence.get("modifier", 0.0),
            },
        )

    @property
    def min_confidence(self) -> float:
        """Minimum confidence across all slots."""
        if not self.confidence:
            return 0.0
        return min(self.confidence.values())

    def __str__(self) -> str:
        status = "converged" if self.converged else "max_iter"
        return (
            f"CodeFactorization({self.operation}({self.file}, {self.location}), "
            f"{status} @ {self.iterations} iter, conf={self.min_confidence:.2f})"
        )


class CodeResonator:
    """
    Factorize code observations using TransformationResonator.

    This is a thin wrapper that:
    1. Delegates factorization to TransformationResonator
    2. Remaps result fields to code-specific semantics
    3. Provides code-specific vocabulary access

    Uses composition - contains a TransformationResonator instance.

    Args:
        encoder: CodeEncoder for vocabulary access
        resonator: TransformationResonator for ALS factorization
    """

    def __init__(
        self,
        encoder: CodeEncoder,
        resonator: TransformationResonator,
    ):
        """
        Initialize code resonator.

        Args:
            encoder: CodeEncoder for vocabulary
            resonator: TransformationResonator for factorization
        """
        self._encoder = encoder
        self._resonator = resonator

    def resonate(self, observation: torch.Tensor) -> CodeFactorization:
        """
        Factorize code observation into (OPERATION, FILE, LOCATION).

        Args:
            observation: Bundled code observation vector

        Returns:
            CodeFactorization with code-specific fields
        """
        result = self._resonator.resonate(observation)
        return CodeFactorization.from_transform_result(result)

    def resonate_topk(
        self,
        observation: torch.Tensor,
        k: int = 20,
        slot_k: int = 5,
    ) -> List[CodeFactorization]:
        """
        Generate top-k candidate factorizations.

        Args:
            observation: Code observation vector
            k: Maximum candidates to return
            slot_k: Top-k per slot for Cartesian product

        Returns:
            List of CodeFactorization candidates, sorted by verification score
        """
        results = self._resonator.resonate_topk(observation, k=k, slot_k=slot_k)
        return [CodeFactorization.from_transform_result(r) for r in results]

    def verify_factorization(
        self,
        observation: torch.Tensor,
        factorization: CodeFactorization,
    ) -> float:
        """
        Verify factorization matches observation.

        Args:
            observation: Original observation vector
            factorization: Factorization to verify

        Returns:
            Cosine similarity between reconstruction and original
        """
        # Convert back to TransformResult for verification
        result = TransformResult(
            action=factorization.operation,
            target=factorization.file,
            modifier=factorization.location,
            action_vec=factorization.operation_vec,
            target_vec=factorization.file_vec,
            modifier_vec=factorization.location_vec,
            iterations=factorization.iterations,
            converged=factorization.converged,
            confidence={
                "action": factorization.confidence.get("operation", 0.0),
                "target": factorization.confidence.get("file", 0.0),
                "modifier": factorization.confidence.get("location", 0.0),
            },
        )
        return self._resonator.verify_factorization(observation, result)

    def __repr__(self) -> str:
        return f"CodeResonator(encoder={self._encoder}, resonator={self._resonator})"
