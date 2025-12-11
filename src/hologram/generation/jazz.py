"""
Jazz Templates: Structure vs Content separation for generation.

Separates the "What" (content/SPO facts) from the "How" (structure/template)
to enable stylistic improvisation while preserving semantic content.

The Jazz metaphor: Content is the melody (what you're saying), Structure
is the rhythm/form (how you're saying it). Together they create the song.
"""

from enum import Enum
from typing import Dict, Optional

import torch

from hologram.core.codebook import Codebook
from hologram.core.operations import Operations


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
