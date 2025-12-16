"""Tests for CodeEncoder."""

import pytest
import torch

from hologram.core.vector_space import VectorSpace
from hologram.core.codebook import Codebook
from hologram.core.fractal import FractalSpace
from hologram.swe.encoder import CodeEncoder
from hologram.swe.types import CodePatch


@pytest.fixture
def encoder():
    """Create CodeEncoder for tests."""
    space = VectorSpace(dimensions=10000)
    fractal = FractalSpace(dimensions=10000)
    codebook = Codebook(space)
    return CodeEncoder(fractal, codebook)


class TestCodeEncoderBasic:
    """Basic encoder tests."""

    def test_encode_patch_creates_vector(self, encoder):
        """Should encode patch as 10000-dim vector."""
        patch = CodePatch(
            file="utils.py",
            operation="add_line",
            location="42",
            content="x = 1",
        )

        vec = encoder.encode_patch(patch)

        assert vec.shape == (10000,)
        assert torch.isfinite(vec).all()

    def test_encode_issue_creates_vector(self, encoder):
        """Should encode issue text as vector."""
        issue_text = "Fix the null pointer exception in utils.py"

        vec = encoder.encode_issue(issue_text)

        assert vec.shape == (10000,)
        assert torch.isfinite(vec).all()

    def test_different_patches_different_vectors(self, encoder):
        """Different patches should produce different vectors."""
        patch1 = CodePatch(
            file="a.py",
            operation="add_line",
            location="1",
            content="x = 1",
        )
        patch2 = CodePatch(
            file="b.py",
            operation="delete_line",
            location="2",
            content="y = 2",
        )

        vec1 = encoder.encode_patch(patch1)
        vec2 = encoder.encode_patch(patch2)

        # Should not be identical
        assert not torch.allclose(vec1, vec2)


class TestCodeEncoderVocabulary:
    """Tests for vocabulary management."""

    def test_get_operation_vocabulary(self, encoder):
        """Should return operation vocabulary."""
        names, vectors = encoder.get_operation_vocabulary()

        assert len(names) > 0
        assert len(vectors) == len(names)
        assert "add_line" in names
        assert vectors.shape[1] == 10000

    def test_register_file(self, encoder):
        """Should register files in vocabulary."""
        encoder.register_file("test.py")

        names, vectors = encoder.get_file_vocabulary()

        assert "test.py" in names

    def test_file_vectors_consistent(self, encoder):
        """Same file should produce same vector."""
        encoder.register_file("test.py")

        vec1 = encoder._get_or_create_file_vector("test.py")
        vec2 = encoder._get_or_create_file_vector("test.py")

        assert torch.allclose(vec1, vec2)


class TestCodeEncoderLocationInference:
    """Tests for location type inference."""

    def test_infer_line_number(self, encoder):
        """Should infer line number location."""
        assert encoder._infer_location_type("42") == "line_number"
        assert encoder._infer_location_type("100") == "line_number"

    def test_infer_function_name(self, encoder):
        """Should infer function name location."""
        assert encoder._infer_location_type("def foo()") == "function_name"
        assert encoder._infer_location_type("process()") == "function_name"

    def test_infer_class_name(self, encoder):
        """Should infer class name location."""
        assert encoder._infer_location_type("class MyClass") == "class_name"

    def test_infer_module_level(self, encoder):
        """Should infer module level location."""
        assert encoder._infer_location_type("module") == "module_level"
        assert encoder._infer_location_type("top") == "module_level"
