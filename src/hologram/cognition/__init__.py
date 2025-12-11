"""
Metacognitive Layer: Self-monitoring and adaptive behavior.

Implements a "Prefrontal Cortex" for the Hologram system that:
- Maintains persistent self-state (mood/awareness)
- Observes query confidence
- Labels internal states ("Confident", "Confused", "Curious")
- Rewires itself when stuck, enabling retry loops
"""

from hologram.cognition.metacognition import MetacognitiveState, MetacognitiveLoop

__all__ = ["MetacognitiveState", "MetacognitiveLoop"]
