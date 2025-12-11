"""
Base interfaces for generation components.

Defines the GenerationContext dataclass and Generator protocol that both
ResonantGenerator (HDC-native) and VentriloquistGenerator (SLM-based) implement.

This ensures type safety and allows hybrid routing without breaking changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, TYPE_CHECKING

import torch

# Use TYPE_CHECKING for all imports to avoid circular import issues
# With __future__ annotations, all annotations are strings, so we don't need
# the actual types at class definition time - only for type checking
if TYPE_CHECKING:
    from hologram.conversation.intent import IntentType
    from hologram.modulation.sesame import StyleType
    from hologram.generation.resonant_generator import GenerationResult


@dataclass
class GenerationContext:
    """
    Unified context for generation that supports both HDC and SLM generators.
    
    This dataclass carries all information needed by either generator type:
    - HDC generators use thought_vector
    - SLM generators use query_text, fact_answer, entities
    
    Attributes:
        query_text: Original user query (for SLM)
        thought_vector: HDC thought vector (for ResonantGenerator)
        intent: Classified intent
        fact_answer: Retrieved fact answer (if available)
        entities: List of entity names
        style: Preferred style
        expected_subject: Expected subject for validation
    """
    query_text: str
    thought_vector: Optional[torch.Tensor]
    intent: IntentType  # type: ignore  # Forward reference via __future__ annotations
    fact_answer: Optional[str]
    entities: list[str]
    style: StyleType  # type: ignore  # Forward reference via __future__ annotations
    expected_subject: Optional[str] = None


class Generator(Protocol):
    """
    Protocol for generation components.
    
    Both ResonantGenerator and VentriloquistGenerator implement this interface,
    allowing hybrid routing in ResponseSelector.
    """
    
    def generate_with_validation(
        self,
        context: GenerationContext,
        max_tokens: int = 10,
    ) -> Optional[GenerationResult]:  # Forward reference via __future__ annotations
        """
        Generate response with validation.
        
        Args:
            context: GenerationContext with all necessary information
            max_tokens: Maximum tokens to generate
            
        Returns:
            GenerationResult if validation passes, None otherwise
        """
        ...

