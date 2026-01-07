"""
Jazz Templates: Structure vs Content separation for generation.

Separates the "What" (content/SPO facts) from the "How" (structure/template)
to enable stylistic improvisation while preserving semantic content.

The Jazz metaphor: Content is the melody (what you're saying), Structure
is the rhythm/form (how you're saying it). Together they create the song.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch

from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.generation.cadence_extractor import CadencePattern


class StructureType(Enum):
    """Structural templates for generation styles."""

    NEUTRAL = "neutral"  # Default: no structural bias
    EXPLANATION = "explanation"  # Structured explanation format
    CONVERSATIONAL = "conversational"  # Casual, flowing structure
    FORMAL = "formal"  # Formal, structured format
    POETIC = "poetic"  # Poetic/creative structure
    TECHNICAL = "technical"  # Technical documentation style


class JazzTemplate:
    """
    Template for structural modulation of content.

    Provides structure vectors that can be bound with content vectors
    to create structured outputs. The structure acts as a "form" that
    shapes how content is expressed.

    Formula: Output = Content ⊗ Structure

    Attributes:
        _codebook: Codebook for generating structure vectors
        _structure_type: Type of structure template
        _role_structures: Cached structure vectors per role

    Example:
        >>> codebook = Codebook(VectorSpace())
        >>> template = JazzTemplate(codebook, StructureType.EXPLANATION)
        >>> struct_vec = template.get_structure_vector("SUBJECT")
        >>> # Bind content with structure
        >>> structured = Operations.bind(content_vec, struct_vec)
    """

    def __init__(
        self,
        codebook: Codebook,
        structure_type: StructureType = StructureType.NEUTRAL,
    ):
        """
        Initialize jazz template.

        Args:
            codebook: Codebook for generating structure vectors
            structure_type: Type of structural template to use
        """
        self._codebook = codebook
        self._structure_type = structure_type
        self._role_structures: Dict[str, torch.Tensor] = {}

    def get_structure_vector(self, role: str) -> torch.Tensor:
        """
        Get structure vector for a grammatical role.

        Structure vectors are unique markers that encode how content
        in that role should be expressed. They're deterministic based
        on the structure type and role.

        Args:
            role: Grammatical role ("SUBJECT", "VERB", "OBJECT")

        Returns:
            Structure hypervector for this role

        Example:
            >>> template = JazzTemplate(codebook, StructureType.FORMAL)
            >>> subj_struct = template.get_structure_vector("SUBJECT")
            >>> # Use in generation: bind(content, subj_struct)
        """
        cache_key = f"{self._structure_type.value}_{role}"

        if cache_key not in self._role_structures:
            # Generate structure vector: encode structure type + role
            structure_concept = f"__STRUCTURE_{self._structure_type.value}_{role}__"
            self._role_structures[cache_key] = self._codebook.encode(structure_concept)

        return self._role_structures[cache_key]

    def apply_structure(
        self,
        content_vector: torch.Tensor,
        role: str,
    ) -> torch.Tensor:
        """
        Apply structure template to content vector.

        Binds content with structure to create structured output.

        Args:
            content_vector: Content hypervector (e.g., SPO fact)
            role: Grammatical role for this content

        Returns:
            Structured content vector

        Example:
            >>> template = JazzTemplate(codebook, StructureType.POETIC)
            >>> structured = template.apply_structure(content_vec, "SUBJECT")
        """
        structure_vec = self.get_structure_vector(role)
        return Operations.bind(content_vector, structure_vec)

    def remove_structure(
        self,
        structured_vector: torch.Tensor,
        role: str,
    ) -> torch.Tensor:
        """
        Remove structure from structured vector (unbind).

        Extracts the content vector from a structured vector by unbinding
        the structure component.

        Args:
            structured_vector: Structured vector (content ⊗ structure)
            role: Grammatical role

        Returns:
            Content vector (structure removed)

        Example:
            >>> content = template.remove_structure(structured_vec, "SUBJECT")
        """
        structure_vec = self.get_structure_vector(role)
        return Operations.unbind(structured_vector, structure_vec)

    def get_all_structure_vectors(self) -> Dict[str, torch.Tensor]:
        """
        Get all structure vectors for all roles.

        Returns:
            Dictionary mapping role names to structure vectors
        """
        roles = ["SUBJECT", "VERB", "OBJECT"]
        return {role: self.get_structure_vector(role) for role in roles}

    @property
    def structure_type(self) -> StructureType:
        """Get the structure type."""
        return self._structure_type

    def __repr__(self) -> str:
        return f"JazzTemplate(type={self._structure_type.value}, roles_cached={len(self._role_structures)})"


@dataclass
class ComposedResponse:
    """Result of cadence-based composition."""
    text: str  # Filled template text
    vector: torch.Tensor  # Composed vector
    confidence: float  # Composition confidence


class CadenceJazz(JazzTemplate):
    """
    Extended Jazz for cadence-based composition.

    Instead of fixed structure types, uses learned cadence patterns.
    """

    def compose_with_cadence(
        self,
        content_facts: List[Tuple[str, torch.Tensor]],  # [(fact_text, fact_vector), ...]
        cadence_pattern: CadencePattern,
    ) -> ComposedResponse:
        """
        Compose response by binding content with cadence structure.

        Formula: Response = Cadence_Structure ⊗ Content_Facts

        Args:
            content_facts: Facts retrieved from HDC (0% hallucination)
            cadence_pattern: Structure pattern from Neural Cadence Memory

        Returns:
            ComposedResponse with text and confidence
        """
        # Fill slots in cadence template with content
        filled_template = cadence_pattern.template

        # Replace slot markers with actual content
        slot_index = 0
        for fact_text, _ in content_facts:
            if slot_index == 0:
                # Replace first entity slot
                if "__SLOT_ENTITY__" in filled_template:
                    filled_template = filled_template.replace(
                        "__SLOT_ENTITY__", fact_text, 1
                    )
                    slot_index += 1
            else:
                # Replace subsequent entity slots
                if "__SLOT_ENTITY__" in filled_template:
                    filled_template = filled_template.replace(
                        "__SLOT_ENTITY__", fact_text, 1
                    )

        # Bind content vectors with structure vector
        composed_vector = cadence_pattern.structure_vector
        for _, fact_vec in content_facts:
            composed_vector = Operations.bind(composed_vector, fact_vec)

        # Calculate composition confidence
        confidence = self._calculate_composition_confidence(
            content_facts, cadence_pattern
        )

        return ComposedResponse(
            text=filled_template,
            vector=composed_vector,
            confidence=confidence,
        )

    def _calculate_composition_confidence(
        self,
        content_facts: List[Tuple[str, torch.Tensor]],
        cadence_pattern: CadencePattern,
    ) -> float:
        """
        Calculate confidence for composition.

        Higher confidence when:
        - More facts match slots
        - Structure vector is well-formed

        Args:
            content_facts: List of (text, vector) tuples
            cadence_pattern: Cadence pattern being used

        Returns:
            Confidence score [0, 1]
        """
        if not content_facts:
            return 0.0

        # Count how many slots we can fill
        slot_count = cadence_pattern.template.count("__SLOT_ENTITY__")
        fact_count = len(content_facts)

        # Match ratio: how many slots can be filled
        if slot_count == 0:
            match_ratio = 1.0  # No slots needed
        else:
            match_ratio = min(fact_count / slot_count, 1.0)

        # Structure quality: check if structure vector is non-zero
        structure_norm = torch.norm(cadence_pattern.structure_vector).item()
        structure_quality = min(structure_norm / 100.0, 1.0)  # Normalize

        # Combined confidence
        confidence = (match_ratio * 0.7) + (structure_quality * 0.3)

        return float(confidence)

    def compose_multi_sentence(
        self,
        content_facts: List[List[Tuple[str, torch.Tensor]]],  # Facts per sentence
        cadence_patterns: List[CadencePattern],  # One pattern per sentence
    ) -> ComposedResponse:
        """Compose multiple sentences using cadence patterns."""
        sentences = []
        combined_confidence = 0.0

        for facts, pattern in zip(content_facts, cadence_patterns):
            result = self.compose_with_cadence(facts, pattern)
            sentences.append(result.text)
            combined_confidence += result.confidence

        return ComposedResponse(
            text=". ".join(sentences) + "." if sentences else "",
            vector=cadence_patterns[0].structure_vector if cadence_patterns else self._codebook._space.empty_vector(),
            confidence=combined_confidence / len(cadence_patterns) if cadence_patterns else 0.0,
        )
