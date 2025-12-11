"""
TargetEncoder: Packages Resonator output into constraint tensors.

The TargetEncoder transforms Resonator results into a generation-ready
format that includes the target tensor, style modulation, and per-slot
confidence information.

This is the bridge between holographic reasoning (Resonator) and
constrained generation (ResonantGenerator).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch

from hologram.config.constants import STYLE_WEIGHT
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.resonator import ResonatorResult


@dataclass
class TargetPackage:
    """Packaged target for generation loop.

    Attributes:
        target_tensor: Composite constraint tensor (T_target)
        confidence_map: Per-slot confidence scores
        style_vector: Style modulation vector (or zeros if no style)
        grammar: Role sequence for generation order
    """
    target_tensor: torch.Tensor
    confidence_map: Dict[str, float]
    style_vector: torch.Tensor
    grammar: List[str] = field(default_factory=lambda: ["SUBJECT", "VERB", "OBJECT"])

    def __str__(self) -> str:
        avg_conf = sum(self.confidence_map.values()) / max(len(self.confidence_map), 1)
        return f"TargetPackage(avg_confidence={avg_conf:.3f}, grammar={self.grammar})"


class TargetEncoder:
    """
    Encodes resonator output into generation-ready constraint tensor.

    Reconstructs target as bundled role-filler pairs and calculates
    per-slot confidence based on the Resonator's convergence quality.

    The target tensor structure:
    T_target = (subject ⊗ R_subj) ⊕ (verb ⊗ R_verb) ⊕ (object ⊗ R_obj) + style

    Attributes:
        _codebook: Codebook for role vectors
        _ops: Operations class
        _style_weight: Weight for style injection (default: 0.1)

    Example:
        >>> encoder = TargetEncoder(codebook)
        >>> package = encoder.encode(resonator_result, style_vector)
        >>> generator.generate(package)
    """

    def __init__(
        self,
        codebook: Codebook,
        operations: Type[Operations] = Operations,
        style_weight: float = STYLE_WEIGHT,
    ):
        """
        Initialize target encoder.

        Args:
            codebook: Codebook for role vectors
            operations: Operations class (default: Operations)
            style_weight: Weight for style injection (default: 0.1)
        """
        self._codebook = codebook
        self._ops = operations
        self._style_weight = style_weight

    def encode(
        self,
        resonator_result: ResonatorResult,
        style_vector: Optional[torch.Tensor] = None,
        grammar: Optional[List[str]] = None,
    ) -> TargetPackage:
        """
        Package resonator output into constraint tensor.

        Args:
            resonator_result: Result from Resonator.resonate()
            style_vector: Optional style modulation vector
            grammar: Optional role sequence (default: ["SUBJECT", "VERB", "OBJECT"])

        Returns:
            TargetPackage ready for generation loop

        Example:
            >>> result = resonator.resonate(thought, nouns, verbs)
            >>> package = encoder.encode(result, style_vec)
        """
        grammar = grammar or ["SUBJECT", "VERB", "OBJECT"]

        # Get role vectors
        r_subj = self._codebook.get_role("SUBJECT")
        r_verb = self._codebook.get_role("VERB")
        r_obj = self._codebook.get_role("OBJECT")

        # Construct target tensor: (S ⊗ R_s) ⊕ (V ⊗ R_v) ⊕ (O ⊗ R_o)
        target_tensor = self._ops.bundle(
            self._ops.bind(resonator_result.subject, r_subj),
            self._ops.bind(resonator_result.verb, r_verb),
            self._ops.bind(resonator_result.object, r_obj),
        )

        # Inject style as weak bias (doesn't override semantics)
        if style_vector is not None:
            target_tensor = self._normalize(
                target_tensor + self._style_weight * style_vector
            )

        # Use confidence from Resonator result
        confidence_map = resonator_result.confidence.copy()

        # Create style vector (zeros if not provided)
        if style_vector is None:
            style_vector = torch.zeros_like(target_tensor)

        return TargetPackage(
            target_tensor=target_tensor,
            confidence_map=confidence_map,
            style_vector=style_vector,
            grammar=grammar,
        )

    def encode_from_words(
        self,
        subject: str,
        verb: str,
        obj: str,
        style_vector: Optional[torch.Tensor] = None,
        confidence_map: Optional[Dict[str, float]] = None,
        grammar: Optional[List[str]] = None,
    ) -> TargetPackage:
        """
        Create target package directly from words.

        Useful for testing or when you have explicit S-V-O words
        rather than a Resonator result.

        Args:
            subject: Subject word
            verb: Verb word
            obj: Object word
            style_vector: Optional style modulation
            confidence_map: Optional per-slot confidence (default: 1.0 for all)
            grammar: Optional role sequence

        Returns:
            TargetPackage ready for generation loop

        Example:
            >>> package = encoder.encode_from_words("cat", "eats", "fish")
        """
        grammar = grammar or ["SUBJECT", "VERB", "OBJECT"]

        # Encode words
        s_vec = self._codebook.encode(subject)
        v_vec = self._codebook.encode(verb)
        o_vec = self._codebook.encode(obj)

        # Get role vectors
        r_subj = self._codebook.get_role("SUBJECT")
        r_verb = self._codebook.get_role("VERB")
        r_obj = self._codebook.get_role("OBJECT")

        # Construct target tensor
        target_tensor = self._ops.bundle(
            self._ops.bind(s_vec, r_subj),
            self._ops.bind(v_vec, r_verb),
            self._ops.bind(o_vec, r_obj),
        )

        # Inject style
        if style_vector is not None:
            target_tensor = self._normalize(
                target_tensor + self._style_weight * style_vector
            )

        # Default confidence
        if confidence_map is None:
            confidence_map = {"subject": 1.0, "verb": 1.0, "object": 1.0}

        # Create style vector
        if style_vector is None:
            style_vector = torch.zeros_like(target_tensor)

        return TargetPackage(
            target_tensor=target_tensor,
            confidence_map=confidence_map,
            style_vector=style_vector,
            grammar=grammar,
        )

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to unit length."""
        norm = torch.norm(tensor)
        if norm > 0:
            return tensor / norm
        return tensor

    def __repr__(self) -> str:
        return f"TargetEncoder(style_weight={self._style_weight})"
