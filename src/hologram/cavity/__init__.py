"""
Cavity package: Resonant cavity components for constrained generation.

This package implements the bidirectional verification loop that ensures
generated output aligns with holographic truth constraints.

Components:
    - TargetEncoder: Packages Resonator output into constraint tensors
    - ReEncoder: Projects tokens back into HDC space for verification
    - DivergenceCalculator: Measures drift and determines accept/reject actions
"""

from hologram.cavity.divergence import (
    DivergenceAction,
    DivergenceCalculator,
    DivergenceResult,
    DivergenceThresholds,
)
from hologram.cavity.re_encoder import ReEncoder
from hologram.cavity.target_encoder import TargetEncoder, TargetPackage

__all__ = [
    "TargetEncoder",
    "TargetPackage",
    "ReEncoder",
    "DivergenceCalculator",
    "DivergenceResult",
    "DivergenceAction",
    "DivergenceThresholds",
]
