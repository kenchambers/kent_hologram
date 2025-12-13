"""
ARC-AGI-2 Holographic Reasoning Module.

A novel approach to ARC that uses HDC algebra instead of traditional
program synthesis. Key components:

1. ObjectDetector - Extract objects from grids via flood-fill
2. ObjectEncoder - Encode objects as fractal DNA vectors
3. TransformationResonator - Factorize observations into (ACTION, TARGET, MODIFIER)
4. TransformationExecutor - Apply transformations to grids
5. HolographicARCSolver - Main solver orchestration
6. IterativeSolver - Multi-step state traversal for complex tasks
7. RelationalEncoder - Salient relationship encoding (Phase 2)
8. HierarchicalSalienceResonator - Confidence-gated relation augmentation
9. HonestBenchmark - Cache-isolated evaluation harness

The system maintains the no-hallucination guarantee by constraining
all outputs to vocabulary items discovered through resonance.
"""

from hologram.arc.types import Grid, Object, BoundingBox, Color, ARCTask, TrainingPair
from hologram.arc.detector import ObjectDetector
from hologram.arc.encoder import ObjectEncoder
from hologram.arc.transform_resonator import TransformationResonator, TransformResult
from hologram.arc.executor import TransformationExecutor
from hologram.arc.solver import HolographicARCSolver, SolverResult, create_simple_task
from hologram.arc.iterative_solver import IterativeSolver, IterativeResult
from hologram.arc.relational_encoder import RelationalEncoder, SalientRelation
from hologram.arc.hierarchical_resonator import HierarchicalSalienceResonator, HierarchicalResult
from hologram.arc.search_verifier import SearchVerifier, VerificationResult
from hologram.arc.benchmark import HonestBenchmark, BenchmarkResult, load_arc_agi_2

__all__ = [
    # Types
    "Grid",
    "Object",
    "BoundingBox",
    "Color",
    "ARCTask",
    "TrainingPair",
    # Components
    "ObjectDetector",
    "ObjectEncoder",
    "TransformationResonator",
    "TransformResult",
    "TransformationExecutor",
    # Phase 1: Iterative Solver
    "IterativeSolver",
    "IterativeResult",
    # Phase 2: Relational Encoding
    "RelationalEncoder",
    "SalientRelation",
    "HierarchicalSalienceResonator",
    "HierarchicalResult",
    # Search+Verify
    "SearchVerifier",
    "VerificationResult",
    # Phase 4: Benchmarking
    "HonestBenchmark",
    "BenchmarkResult",
    "load_arc_agi_2",
    # Solver
    "HolographicARCSolver",
    "SolverResult",
    "create_simple_task",
]
