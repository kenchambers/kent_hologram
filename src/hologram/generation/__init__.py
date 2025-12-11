"""
Generation package: Constrained text generation via Resonant Cavity.

This package implements the dual-stream generation system that produces
fluent output while maintaining bounded hallucination properties.

Components:
    - ResonantGenerator: Main orchestration loop
    - GenerationResult: Output with trace and metrics
    - GenerationMetrics: Performance monitoring
"""

from hologram.generation.resonant_generator import (
    GenerationMetrics,
    GenerationResult,
    GenerationTrace,
    ResonantGenerator,
)
from hologram.generation.jazz import JazzTemplate, StructureType
from hologram.generation.dreamer import Dreamer, DreamResult

__all__ = [
    "ResonantGenerator",
    "GenerationResult",
    "GenerationTrace",
    "GenerationMetrics",
    "JazzTemplate",
    "StructureType",
    "Dreamer",
    "DreamResult",
]
