"""
SWE Module: HDC-Native Code Generation.

Provides software engineering task support with:
- SWETask, CodePatch, PatchResult: Data structures
- CodeEncoder: Maps code/issues to HDC vectors
- CodeResonator: Factorizes observations into (OPERATION, FILE, LOCATION)
- CodeGenerator: Template-based generation with HDC verification

Uses the SAME memory store as ARC (shared ConsolidationManager).
Follows composition pattern for maximum code reuse.
"""

from hologram.swe.types import (
    SWETask,
    CodePatch,
    PatchResult,
    OPERATIONS,
    LOCATION_TYPES,
)
from hologram.swe.encoder import CodeEncoder
from hologram.swe.code_resonator import CodeResonator, CodeFactorization
from hologram.swe.generator import CodeGenerator, GenerationTrace
from hologram.swe.benchmark import HonestCodeBenchmark, BenchmarkResult, TaskResult
from hologram.swe.dependency_graph import CodeDependencyGraph, DependencyResult
from hologram.swe.diff_parser import parse_unified_diff

__all__ = [
    # Types
    "SWETask",
    "CodePatch",
    "PatchResult",
    "OPERATIONS",
    "LOCATION_TYPES",
    # Encoder
    "CodeEncoder",
    # Resonator
    "CodeResonator",
    "CodeFactorization",
    # Generator
    "CodeGenerator",
    "GenerationTrace",
    # Benchmark
    "HonestCodeBenchmark",
    "BenchmarkResult",
    "TaskResult",
    # Dependency Graph
    "CodeDependencyGraph",
    "DependencyResult",
    # Diff Parser
    "parse_unified_diff",
]
