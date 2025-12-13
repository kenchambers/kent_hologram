"""
HierarchicalSalienceResonator: Confidence-gated relation augmentation.

BREAKTHROUGH APPROACH: Treats relations as confidence-gated spatial context
rather than stuffing them into the observation bundle.

Core Insight: Relations are meta-information about TARGET selection
("which objects"), NOT primary transformation semantics ("what transformation").
We augment TARGET, not action/modifier.

Six-Phase Pipeline:
1. Primary Resonation (ATM) - fast path if confident
2. Confidence Gating - compute alpha interpolation weight
3. Spatial Relation Extraction (if alpha > threshold)
4. Salience-Weighted Augmentation - augment target with relation context
5. Re-Resonation - resonate with augmented observation
6. Fusion - blend results based on confidence improvement

Properties:
- Algebraic Purity: Uses only bind/bundle/unbind (no hacks)
- Capacity Efficiency: Dynamic overhead, not permanent
- Self-Tuning: Adapts to task difficulty via confidence gating
- Zero Overhead on Easy Tasks: Fast path exits Phase 1 immediately
- Modular: Wraps existing Resonator, easily testable
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

from hologram.arc.types import Object
from hologram.arc.encoder import ObjectEncoder
from hologram.arc.transform_resonator import TransformationResonator, TransformResult
from hologram.arc.relational_encoder import RelationalEncoder, SalientRelation
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity


@dataclass
class HierarchicalResult:
    """Result from hierarchical resonation."""
    result: TransformResult
    used_relations: bool
    relation_count: int
    alpha: float  # Gate weight used (0.0 = no relations, 1.0 = full relations)
    confidence_improvement: float  # How much relations helped


class HierarchicalSalienceResonator:
    """
    Wraps TransformationResonator with confidence-gated relation augmentation.

    When primary resonation is uncertain (confidence < THRESHOLD_HIGH),
    this resonator extracts spatial relations between objects and uses
    them to augment the target slot, then re-resonates.

    Key Parameters:
        THRESHOLD_HIGH: Above this, skip relations entirely (0.75)
        THRESHOLD_LOW: Below this, use relations fully (0.45)
        IMPROVEMENT_THRESHOLD: Min confidence gain to accept augmented (0.10)
        MAX_RELATIONS: Maximum relations to encode (7)

    Example:
        >>> hier_res = HierarchicalSalienceResonator(encoder, codebook)
        >>> result = hier_res.resonate_with_relations(observation, objects)
        >>> print(result.used_relations, result.confidence_improvement)
    """

    # Confidence thresholds for gating
    THRESHOLD_HIGH = 0.75    # Above: skip relations (fast path)
    THRESHOLD_LOW = 0.45     # Below: use relations fully
    IMPROVEMENT_THRESHOLD = 0.10  # Min gain to accept augmented result
    MAX_RELATIONS = 7        # Respect bundling capacity (7-10 items per 10k)

    def __init__(
        self,
        encoder: ObjectEncoder,
        codebook: Codebook,
        base_resonator: Optional[TransformationResonator] = None,
        threshold_high: float = THRESHOLD_HIGH,
        threshold_low: float = THRESHOLD_LOW,
        improvement_threshold: float = IMPROVEMENT_THRESHOLD,
    ):
        """
        Initialize hierarchical resonator.

        Args:
            encoder: ObjectEncoder for vocabulary
            codebook: Codebook for encoding
            base_resonator: TransformationResonator to wrap (creates new if None)
            threshold_high: Confidence above which to skip relations
            threshold_low: Confidence below which to use relations fully
            improvement_threshold: Min improvement to accept augmented result
        """
        self._encoder = encoder
        self._codebook = codebook
        self._ops = Operations

        # Create or use provided base resonator
        self._base_resonator = base_resonator or TransformationResonator(
            encoder, codebook
        )

        # Create relational encoder
        self._relation_encoder = RelationalEncoder(
            codebook, max_relations=self.MAX_RELATIONS
        )

        # Thresholds
        self._threshold_high = threshold_high
        self._threshold_low = threshold_low
        self._improvement_threshold = improvement_threshold

        # Pre-encode role vectors
        self._role_action = self._codebook.encode("__ROLE_ACTION__")
        self._role_target = self._codebook.encode("__ROLE_TARGET__")
        self._role_modifier = self._codebook.encode("__ROLE_MODIFIER__")

    def resonate(self, observation: torch.Tensor) -> TransformResult:
        """
        Standard resonation (delegates to base resonator).

        Use this for backwards compatibility or when no objects available.

        Args:
            observation: Observation bundle

        Returns:
            TransformResult from base resonator
        """
        return self._base_resonator.resonate(observation)

    def resonate_with_relations(
        self,
        observation: torch.Tensor,
        objects: List[Object],
    ) -> HierarchicalResult:
        """
        Resonate with confidence-gated relation augmentation.

        Six-Phase Pipeline:
        1. Primary resonation - get baseline result
        2. If confident enough, return (fast path)
        3. Compute gate weight alpha based on confidence
        4. Extract spatial relations if alpha > 0.01
        5. Augment target with relation context
        6. Re-resonate and fuse results

        Args:
            observation: Bundled transformation observation
            objects: Detected objects for relation extraction

        Returns:
            HierarchicalResult with final result and metadata
        """
        # Phase 1: Primary resonation (ATM)
        result1 = self._base_resonator.resonate(observation)

        # Fast path: If confident, skip relations entirely
        if result1.min_confidence >= self._threshold_high:
            return HierarchicalResult(
                result=result1,
                used_relations=False,
                relation_count=0,
                alpha=0.0,
                confidence_improvement=0.0,
            )

        # Phase 2: Compute gate weight (smooth interpolation)
        alpha = self._compute_gate_weight(result1.min_confidence)

        # Gate closed - not uncertain enough to benefit from relations
        if alpha < 0.01:
            return HierarchicalResult(
                result=result1,
                used_relations=False,
                relation_count=0,
                alpha=alpha,
                confidence_improvement=0.0,
            )

        # Phase 3: Extract spatial relations
        relations = self._relation_encoder.extract_salient_relations(objects)

        if not relations:
            # No relations found - return primary result
            return HierarchicalResult(
                result=result1,
                used_relations=False,
                relation_count=0,
                alpha=alpha,
                confidence_improvement=0.0,
            )

        # Limit to MAX_RELATIONS
        relations = relations[:self.MAX_RELATIONS]

        # Phase 4: Encode relation context and augment target
        relation_context = self._relation_encoder.encode_relation_context(relations)

        # Augment target with weighted relation context
        # target_augmented = bundle(target1, alpha * relation_context)
        target_augmented = self._ops.bundle(
            result1.target_vec,
            alpha * relation_context,
        )

        # Phase 5: Re-resonate with augmented observation
        augmented_obs = self._build_augmented_observation(
            observation,
            result1.action_vec,
            target_augmented,
            result1.modifier_vec,
        )
        result2 = self._base_resonator.resonate(augmented_obs)

        # Phase 6: Fusion - use augmented only if significantly better
        confidence_improvement = result2.min_confidence - result1.min_confidence

        if confidence_improvement >= self._improvement_threshold:
            # Augmented result is better - use it
            final_result = self._interpolate_results(result1, result2, alpha)
            return HierarchicalResult(
                result=final_result,
                used_relations=True,
                relation_count=len(relations),
                alpha=alpha,
                confidence_improvement=confidence_improvement,
            )

        # No significant improvement - keep original
        return HierarchicalResult(
            result=result1,
            used_relations=False,
            relation_count=len(relations),
            alpha=alpha,
            confidence_improvement=confidence_improvement,
        )

    def _compute_gate_weight(self, confidence: float) -> float:
        """
        Compute gate weight for relation augmentation.

        Smooth interpolation:
        - Above THRESHOLD_HIGH: alpha = 0 (no relations)
        - Below THRESHOLD_LOW: alpha = 1 (full relations)
        - Between: linear interpolation

        Args:
            confidence: Min confidence from primary resonation

        Returns:
            Gate weight in [0.0, 1.0]
        """
        if confidence >= self._threshold_high:
            return 0.0
        if confidence <= self._threshold_low:
            return 1.0

        # Linear interpolation between thresholds
        range_size = self._threshold_high - self._threshold_low
        return (self._threshold_high - confidence) / range_size

    def _build_augmented_observation(
        self,
        original_obs: torch.Tensor,
        action_vec: torch.Tensor,
        target_vec: torch.Tensor,
        modifier_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build augmented observation with new target.

        Reconstructs observation structure:
            bundle(bind(action, role_A), bind(target, role_T), bind(modifier, role_M))

        Args:
            original_obs: Original observation (unused, kept for signature)
            action_vec: Action vector from primary result
            target_vec: Augmented target vector
            modifier_vec: Modifier vector from primary result

        Returns:
            Augmented observation vector
        """
        return self._ops.bundle(
            self._ops.bind(action_vec, self._role_action),
            self._ops.bind(target_vec, self._role_target),
            self._ops.bind(modifier_vec, self._role_modifier),
        )

    def _interpolate_results(
        self,
        result1: TransformResult,
        result2: TransformResult,
        alpha: float,
    ) -> TransformResult:
        """
        Interpolate between primary and augmented results.

        For discrete outputs (action/target/modifier strings), we use
        the result with higher confidence. For vectors, we interpolate.

        Args:
            result1: Primary resonation result
            result2: Augmented resonation result
            alpha: Interpolation weight (0 = result1, 1 = result2)

        Returns:
            Interpolated TransformResult
        """
        # Use result2's discrete outputs since it's the improved result
        # but blend the vectors for smoother representation
        blended_action = (1 - alpha) * result1.action_vec + alpha * result2.action_vec
        blended_target = (1 - alpha) * result1.target_vec + alpha * result2.target_vec
        blended_modifier = (1 - alpha) * result1.modifier_vec + alpha * result2.modifier_vec

        # Blend confidence scores
        blended_confidence = {}
        for key in result1.confidence:
            c1 = result1.confidence.get(key, 0.0)
            c2 = result2.confidence.get(key, 0.0)
            blended_confidence[key] = (1 - alpha) * c1 + alpha * c2

        return TransformResult(
            action=result2.action,  # Use augmented result's discrete values
            target=result2.target,
            modifier=result2.modifier,
            action_vec=blended_action,
            target_vec=blended_target,
            modifier_vec=blended_modifier,
            iterations=result1.iterations + result2.iterations,
            converged=result2.converged,
            confidence=blended_confidence,
        )

    def verify_factorization(
        self,
        observation: torch.Tensor,
        result: TransformResult,
    ) -> float:
        """
        Verify factorization (delegates to base resonator).

        Args:
            observation: Original observation
            result: Factorization result

        Returns:
            Verification score (cosine similarity)
        """
        return self._base_resonator.verify_factorization(observation, result)

    def __repr__(self) -> str:
        return (
            f"HierarchicalSalienceResonator("
            f"threshold_high={self._threshold_high}, "
            f"threshold_low={self._threshold_low}, "
            f"max_relations={self.MAX_RELATIONS})"
        )
