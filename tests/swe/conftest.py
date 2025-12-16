"""
Shared fixtures for SWE tests.

Provides common test fixtures to avoid duplication across test files.
Follows the same pattern as ARC test organization.
"""

import pytest
import torch

from hologram.core.vector_space import VectorSpace
from hologram.core.codebook import Codebook
from hologram.core.fractal import FractalSpace
from hologram.consolidation.neural_memory import NeuralMemory
from hologram.arc.encoder import ObjectEncoder
from hologram.arc.transform_resonator import TransformationResonator
from hologram.container import HologramContainer
from hologram.swe import (
    SWETask,
    CodePatch,
    CodeEncoder,
    CodeResonator,
    CodeGenerator,
)


# =============================================================================
# Core Component Fixtures
# =============================================================================

@pytest.fixture
def dimensions():
    """Standard test dimensions (smaller for faster tests)."""
    return 10000


@pytest.fixture
def vector_space(dimensions):
    """Create VectorSpace for tests."""
    return VectorSpace(dimensions=dimensions)


@pytest.fixture
def fractal_space(dimensions):
    """Create FractalSpace for tests."""
    return FractalSpace(dimensions=dimensions)


@pytest.fixture
def codebook(vector_space):
    """Create Codebook for tests."""
    return Codebook(vector_space)


# =============================================================================
# Encoder Fixtures
# =============================================================================

@pytest.fixture
def arc_encoder(fractal_space, codebook):
    """Create ObjectEncoder for ARC patterns."""
    return ObjectEncoder(fractal_space, codebook)


@pytest.fixture
def code_encoder(fractal_space, codebook):
    """Create CodeEncoder for SWE patterns."""
    return CodeEncoder(fractal_space, codebook)


# =============================================================================
# Resonator Fixtures
# =============================================================================

@pytest.fixture
def transform_resonator(arc_encoder, codebook):
    """Create TransformationResonator."""
    return TransformationResonator(arc_encoder, codebook)


@pytest.fixture
def code_resonator(code_encoder, transform_resonator):
    """Create CodeResonator (wraps TransformationResonator)."""
    return CodeResonator(code_encoder, transform_resonator)


# =============================================================================
# Memory Fixtures
# =============================================================================

@pytest.fixture
def neural_memory(dimensions):
    """Create NeuralMemory for pattern storage."""
    return NeuralMemory(
        input_dim=dimensions,
        hidden_dim=256,
        initial_vocab_size=50,
    )


# =============================================================================
# Generator Fixtures
# =============================================================================

@pytest.fixture
def generator(code_encoder, code_resonator, neural_memory):
    """Create CodeGenerator with all dependencies."""
    return CodeGenerator(
        encoder=code_encoder,
        resonator=code_resonator,
        neural_memory=neural_memory,
    )


@pytest.fixture
def generator_no_memory(code_encoder, code_resonator):
    """Create CodeGenerator without neural memory."""
    return CodeGenerator(
        encoder=code_encoder,
        resonator=code_resonator,
        neural_memory=None,
    )


# =============================================================================
# Container Fixture
# =============================================================================

@pytest.fixture
def container(dimensions):
    """Create HologramContainer for integration tests."""
    return HologramContainer(dimensions=dimensions)


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_patch():
    """Create sample CodePatch for tests."""
    return CodePatch(
        file="utils.py",
        operation="add_line",
        location="42",
        content="x = 1",
    )


@pytest.fixture
def sample_task():
    """Create sample SWETask for tests."""
    return SWETask(
        task_id="test_001",
        repo="test/repo",
        issue_text="Add input validation to process function",
        code_before={"utils.py": "def process(x):\n    return x * 2"},
        code_after={"utils.py": "def process(x):\n    if x is None:\n        raise ValueError\n    return x * 2"},
    )


@pytest.fixture
def sample_task_multifile():
    """Create sample SWETask with multiple files."""
    return SWETask(
        task_id="test_002",
        repo="test/repo",
        issue_text="Add logging to both process and calculate functions",
        code_before={
            "utils.py": "def process(x):\n    return x * 2",
            "math.py": "def calculate(a, b):\n    return a + b",
        },
        code_after={
            "utils.py": "import logging\ndef process(x):\n    logging.info('process')\n    return x * 2",
            "math.py": "import logging\ndef calculate(a, b):\n    logging.info('calc')\n    return a + b",
        },
    )


@pytest.fixture
def sample_tasks():
    """Create list of sample SWE tasks for batch testing."""
    return [
        SWETask(
            task_id="sample_001",
            repo="test/repo",
            issue_text="Add input validation to the process function",
            code_before={"utils.py": "def process(x):\n    return x * 2"},
            code_after={"utils.py": "def process(x):\n    if x is None:\n        raise ValueError('x cannot be None')\n    return x * 2"},
        ),
        SWETask(
            task_id="sample_002",
            repo="test/repo",
            issue_text="Add logging to the calculate function",
            code_before={"math.py": "def calculate(a, b):\n    return a + b"},
            code_after={"math.py": "import logging\n\ndef calculate(a, b):\n    logging.info(f'Calculating {a} + {b}')\n    return a + b"},
        ),
        SWETask(
            task_id="sample_003",
            repo="test/repo",
            issue_text="Fix division by zero in divide function",
            code_before={"math.py": "def divide(a, b):\n    return a / b"},
            code_after={"math.py": "def divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError('Cannot divide by zero')\n    return a / b"},
        ),
    ]
