"""Core HDC primitives."""

from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.similarity import Similarity
from hologram.core.vector_space import VectorSpace

# Optional semantic codebook (requires sentence-transformers)
try:
    from hologram.core.semantic_codebook import SemanticCodebook
    __all__ = [
        "VectorSpace",
        "Codebook",
        "SemanticCodebook",
        "Operations",
        "Similarity",
    ]
except ImportError:
    __all__ = [
        "VectorSpace",
        "Codebook",
        "Operations",
        "Similarity",
    ]
