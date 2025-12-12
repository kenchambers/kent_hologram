"""
Neural Consolidation Module for Holographic Memory.

Implements a hybrid HDC + Neural memory system inspired by biological
sleep-dependent memory consolidation. Key components:

- NeuralMemory: Classification-head neural network for O(1) lookup
- ConfidenceCalibrator: Calibrates HDC vs Neural confidence for winner-take-all
- ConsolidationManager: Async background training with double-buffering

The system provides:
- Non-blocking consolidation (async training in background)
- O(1) neural queries (classification head, not reconstruction)
- Confidence calibration (sigmoid normalization for fair comparison)
- Graceful degradation (decay instead of delete)
- HDC algebra preservation (unbinding validation gate)
"""

from hologram.consolidation.calibration import ConfidenceCalibrator
from hologram.consolidation.neural_memory import NeuralMemory
from hologram.consolidation.manager import ConsolidationManager

__all__ = [
    "NeuralMemory",
    "ConfidenceCalibrator",
    "ConsolidationManager",
]
