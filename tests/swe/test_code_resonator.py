"""Tests for CodeResonator."""

import pytest
import torch

from hologram.core.vector_space import VectorSpace
from hologram.core.codebook import Codebook
from hologram.core.fractal import FractalSpace
from hologram.arc.encoder import ObjectEncoder
from hologram.arc.transform_resonator import TransformationResonator
from hologram.swe.encoder import CodeEncoder
from hologram.swe.code_resonator import CodeResonator, CodeFactorization
from hologram.swe.types import CodePatch


@pytest.fixture
def resonator():
    """Create CodeResonator for tests."""
    space = VectorSpace(dimensions=10000)
    fractal = FractalSpace(dimensions=10000)
    codebook = Codebook(space)

    # Create ARC encoder for transformation resonator
    arc_encoder = ObjectEncoder(fractal, codebook)

    # Create transformation resonator
    transform_resonator = TransformationResonator(arc_encoder, codebook)

    # Create code encoder
    code_encoder = CodeEncoder(fractal, codebook)

    return CodeResonator(code_encoder, transform_resonator)


@pytest.fixture
def code_encoder():
    """Create CodeEncoder for tests."""
    space = VectorSpace(dimensions=10000)
    fractal = FractalSpace(dimensions=10000)
    codebook = Codebook(space)
    return CodeEncoder(fractal, codebook)


class TestCodeResonatorBasic:
    """Basic resonator tests."""

    def test_resonate_returns_factorization(self, resonator, code_encoder):
        """Should return CodeFactorization."""
        patch = CodePatch(
            file="utils.py",
            operation="add_line",
            location="42",
            content="x = 1",
        )
        observation = code_encoder.encode_patch(patch)

        result = resonator.resonate(observation)

        assert isinstance(result, CodeFactorization)
        assert result.operation is not None
        assert result.file is not None
        assert result.location is not None

    def test_factorization_has_vectors(self, resonator, code_encoder):
        """Factorization should include vectors."""
        patch = CodePatch(
            file="test.py",
            operation="modify_line",
            location="10",
            content="y = 2",
        )
        observation = code_encoder.encode_patch(patch)

        result = resonator.resonate(observation)

        assert result.operation_vec.shape == (10000,)
        assert result.file_vec.shape == (10000,)
        assert result.location_vec.shape == (10000,)

    def test_factorization_has_confidence(self, resonator, code_encoder):
        """Factorization should have confidence scores."""
        patch = CodePatch(
            file="test.py",
            operation="add_function",
            location="def foo",
            content="pass",
        )
        observation = code_encoder.encode_patch(patch)

        result = resonator.resonate(observation)

        assert "operation" in result.confidence
        assert "file" in result.confidence
        assert "location" in result.confidence


class TestCodeResonatorTopK:
    """Tests for top-k factorization."""

    def test_resonate_topk_returns_list(self, resonator, code_encoder):
        """Should return list of factorizations."""
        patch = CodePatch(
            file="test.py",
            operation="add_line",
            location="1",
            content="x",
        )
        observation = code_encoder.encode_patch(patch)

        results = resonator.resonate_topk(observation, k=5)

        assert isinstance(results, list)
        assert len(results) <= 5
        for r in results:
            assert isinstance(r, CodeFactorization)


class TestCodeFactorization:
    """Tests for CodeFactorization dataclass."""

    def test_min_confidence(self):
        """Should compute minimum confidence."""
        fact = CodeFactorization(
            operation="add",
            file="test.py",
            location="1",
            operation_vec=torch.zeros(10000),
            file_vec=torch.zeros(10000),
            location_vec=torch.zeros(10000),
            iterations=10,
            converged=True,
            confidence={"operation": 0.8, "file": 0.6, "location": 0.9},
        )

        assert fact.min_confidence == 0.6

    def test_str_format(self):
        """Should format as readable string."""
        fact = CodeFactorization(
            operation="add",
            file="test.py",
            location="1",
            operation_vec=torch.zeros(10000),
            file_vec=torch.zeros(10000),
            location_vec=torch.zeros(10000),
            iterations=10,
            converged=True,
            confidence={"operation": 0.8, "file": 0.6, "location": 0.9},
        )

        s = str(fact)
        assert "add" in s
        assert "test.py" in s
        assert "converged" in s
