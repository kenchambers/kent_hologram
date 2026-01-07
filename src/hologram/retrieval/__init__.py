"""
Emergent layer retrieval components for scalable CRAG.
"""

from hologram.retrieval.emergent_layers import (
    EmergentLayerManager,
    SemanticLayer,
)
from hologram.retrieval.layer_description import LayerDescriptionGenerator
from hologram.retrieval.description_cache import LayerDescriptionCache

__all__ = [
    "EmergentLayerManager",
    "SemanticLayer",
    "LayerDescriptionGenerator",
    "LayerDescriptionCache",
]
