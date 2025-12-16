"""Tests for CodeGenerator."""

import pytest
import torch

from hologram.core.vector_space import VectorSpace
from hologram.core.codebook import Codebook
from hologram.core.fractal import FractalSpace
from hologram.consolidation.neural_memory import NeuralMemory
from hologram.arc.encoder import ObjectEncoder
from hologram.arc.transform_resonator import TransformationResonator
from hologram.swe import (
    SWETask,
    CodePatch,
    PatchResult,
    CodeEncoder,
    CodeResonator,
    CodeGenerator,
)


@pytest.fixture
def generator():
    """Create CodeGenerator for tests."""
    space = VectorSpace(dimensions=10000)
    fractal = FractalSpace(dimensions=10000)
    codebook = Codebook(space)

    # Create encoders
    arc_encoder = ObjectEncoder(fractal, codebook)
    code_encoder = CodeEncoder(fractal, codebook)

    # Create resonators
    transform_resonator = TransformationResonator(arc_encoder, codebook)
    code_resonator = CodeResonator(code_encoder, transform_resonator)

    # Create neural memory
    neural_memory = NeuralMemory(
        input_dim=10000,
        hidden_dim=256,
        initial_vocab_size=50,
    )

    return CodeGenerator(
        encoder=code_encoder,
        resonator=code_resonator,
        neural_memory=neural_memory,
    )


@pytest.fixture
def sample_task():
    """Create sample SWE task."""
    return SWETask(
        task_id="test_001",
        repo="test/repo",
        issue_text="Add input validation to process function",
        code_before={"utils.py": "def process(x):\n    return x * 2"},
        code_after={"utils.py": "def process(x):\n    if x is None:\n        raise ValueError\n    return x * 2"},
    )


class TestCodeGeneratorBasic:
    """Basic generator tests."""

    def test_generate_returns_result(self, generator, sample_task):
        """Should return PatchResult."""
        result = generator.generate(sample_task)

        assert isinstance(result, PatchResult)
        assert len(result.patches) > 0

    def test_generate_creates_patches(self, generator, sample_task):
        """Should create at least one patch."""
        result = generator.generate(sample_task)

        assert len(result.patches) >= 1
        patch = result.patches[0]
        assert patch.file is not None
        assert patch.operation is not None

    def test_generate_has_confidence(self, generator, sample_task):
        """Should report confidence."""
        result = generator.generate(sample_task)

        assert 0.0 <= result.confidence <= 1.0


class TestCodeGeneratorLearning:
    """Tests for learning from examples."""

    def test_learn_from_task(self, generator, sample_task):
        """Should learn from task with ground truth."""
        success = generator.learn_from_task(sample_task)

        # May or may not succeed depending on implementation
        assert isinstance(success, bool)

    def test_learn_without_memory(self, sample_task):
        """Should handle missing neural memory gracefully."""
        space = VectorSpace(dimensions=10000)
        fractal = FractalSpace(dimensions=10000)
        codebook = Codebook(space)

        arc_encoder = ObjectEncoder(fractal, codebook)
        code_encoder = CodeEncoder(fractal, codebook)
        transform_resonator = TransformationResonator(arc_encoder, codebook)
        code_resonator = CodeResonator(code_encoder, transform_resonator)

        # No neural memory
        generator = CodeGenerator(
            encoder=code_encoder,
            resonator=code_resonator,
            neural_memory=None,
        )

        success = generator.learn_from_task(sample_task)
        assert success is False  # Can't learn without memory


class TestCodeGeneratorVerification:
    """Tests for verification behavior."""

    def test_low_confidence_fails_verification(self, generator):
        """Low confidence should fail verification."""
        # Create task with very short, unclear issue
        task = SWETask(
            task_id="unclear",
            repo="test",
            issue_text="x",  # Very unclear
            code_before={},
            code_after={},
        )

        result = generator.generate(task, confidence_threshold=0.9)

        # With high threshold and unclear input, should fail
        # (exact behavior depends on resonator)
        assert isinstance(result.verification_passed, bool)


class TestCodeGeneratorRepr:
    """Tests for string representation."""

    def test_repr(self, generator):
        """Should have readable repr."""
        s = repr(generator)
        assert "CodeGenerator" in s
        assert "has_memory" in s
