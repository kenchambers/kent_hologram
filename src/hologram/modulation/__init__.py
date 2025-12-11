"""
Modulation package: Style and disfluency injection.

This package implements the "Sesame" layer that adds human-like texture
to generated output through style modulation and confidence-based
disfluency injection.

Components:
    - SesameModulator: Style vectors and filler token selection
"""

from hologram.modulation.sesame import (
    FillerType,
    ModulatedCleanup,
    SesameModulator,
    StyleType,
)

__all__ = [
    "SesameModulator",
    "StyleType",
    "FillerType",
    "ModulatedCleanup",
]
