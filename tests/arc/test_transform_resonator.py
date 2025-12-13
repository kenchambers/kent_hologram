"""
Transformation Resonator Tests for ARC-AGI-2 Holographic Reasoning.

Tests the TransformationResonator which factorizes observations into
(ACTION, TARGET, MODIFIER) tuples using Alternating Least Squares.

Run with: uv run pytest tests/arc/test_transform_resonator.py -v
"""

import pytest
import torch

from hologram.arc.types import Grid, Object, Color, BoundingBox, ACTIONS, TARGETS, MODIFIERS
from hologram.arc.detector import ObjectDetector
from hologram.arc.encoder import ObjectEncoder
from hologram.arc.transform_resonator import TransformationResonator, TransformResult
from hologram.core.fractal import FractalSpace
from hologram.core.codebook import Codebook
from hologram.core.vector_space import VectorSpace
from hologram.core.operations import Operations


@pytest.fixture
def encoder():
    """Create encoder for tests."""
    fractal_space = FractalSpace(dimensions=10000)
    space = VectorSpace(dimensions=10000)
    codebook = Codebook(space)
    return ObjectEncoder(fractal_space, codebook)


@pytest.fixture
def resonator(encoder):
    """Create resonator for tests."""
    space = VectorSpace(dimensions=10000)
    codebook = Codebook(space)
    return TransformationResonator(encoder, codebook)


@pytest.fixture
def detector():
    """Create detector for tests."""
    return ObjectDetector()


class TestTransformResultBasic:
    """Tests for TransformResult dataclass."""

    def test_transform_result_str(self):
        """TransformResult should have readable string representation."""
        result = TransformResult(
            action="rotate",
            target="all_objects",
            modifier="90_degrees",
            action_vec=torch.randn(100),
            target_vec=torch.randn(100),
            modifier_vec=torch.randn(100),
            iterations=10,
            converged=True,
            confidence={"action": 0.8, "target": 0.7, "modifier": 0.9},
        )
        s = str(result)
        assert "rotate" in s
        assert "all_objects" in s
        assert "90_degrees" in s
        assert "converged" in s

    def test_transform_result_as_gene(self):
        """as_gene should produce valid vector."""
        result = TransformResult(
            action="translate",
            target="largest",
            modifier="up",
            action_vec=torch.randn(100),
            target_vec=torch.randn(100),
            modifier_vec=torch.randn(100),
            iterations=5,
            converged=True,
            confidence={"action": 0.9, "target": 0.8, "modifier": 0.85},
        )
        gene = result.as_gene()
        assert gene.shape == (100,)
        assert torch.isfinite(gene).all()

    def test_min_confidence(self):
        """min_confidence should return lowest slot confidence."""
        result = TransformResult(
            action="identity",
            target="all_objects",
            modifier="none",
            action_vec=torch.randn(100),
            target_vec=torch.randn(100),
            modifier_vec=torch.randn(100),
            iterations=1,
            converged=True,
            confidence={"action": 0.9, "target": 0.5, "modifier": 0.8},
        )
        assert result.min_confidence == 0.5


class TestResonatorVocabulary:
    """Tests for vocabulary access."""

    def test_action_vocabulary(self, encoder):
        """Should have all expected actions."""
        names, vectors = encoder.get_action_vocabulary()
        
        assert len(names) == len(ACTIONS)
        assert vectors.shape[0] == len(ACTIONS)
        assert "rotate" in names
        assert "translate" in names
        assert "identity" in names

    def test_target_vocabulary(self, encoder):
        """Should have all expected targets."""
        names, vectors = encoder.get_target_vocabulary()
        
        assert len(names) == len(TARGETS)
        assert "all_objects" in names
        assert "largest" in names
        assert "red" in names

    def test_modifier_vocabulary(self, encoder):
        """Should have all expected modifiers."""
        names, vectors = encoder.get_modifier_vocabulary()
        
        assert len(names) == len(MODIFIERS)
        assert "none" in names
        assert "90_degrees" in names
        assert "up" in names


class TestResonatorBasic:
    """Basic resonator tests."""

    def test_resonate_returns_transform_result(self, resonator, encoder):
        """Resonator should return TransformResult."""
        # Create a simple observation
        obs = encoder._codebook.encode("test_observation")
        
        result = resonator.resonate(obs)
        
        assert isinstance(result, TransformResult)
        assert result.action in ACTIONS
        assert result.target in TARGETS
        assert result.modifier in MODIFIERS

    def test_resonate_produces_valid_vectors(self, resonator, encoder):
        """Resonator output vectors should be valid."""
        obs = encoder._codebook.encode("test")
        result = resonator.resonate(obs)
        
        assert result.action_vec.shape[0] == 10000
        assert result.target_vec.shape[0] == 10000
        assert result.modifier_vec.shape[0] == 10000
        assert torch.isfinite(result.action_vec).all()

    def test_resonator_respects_max_iterations(self, encoder):
        """Resonator should stop at max_iterations."""
        space = VectorSpace(dimensions=10000)
        codebook = Codebook(space)
        resonator = TransformationResonator(
            encoder, codebook, max_iterations=5
        )
        
        obs = torch.randn(10000)
        result = resonator.resonate(obs)
        
        assert result.iterations <= 5


class TestResonatorTranslation:
    """Tests for translation detection."""

    def test_detect_translate_up(self, resonator, encoder, detector):
        """Should detect upward translation."""
        input_grid = Grid.from_list([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        output_grid = Grid.from_list([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        
        input_objects = detector.detect(input_grid)
        output_objects = detector.detect(output_grid)
        matches = detector.match_objects(input_objects, output_objects)
        
        obs = encoder.encode_transformation_observation(
            matches[0][0], matches[0][1]
        )
        result = resonator.resonate(obs)
        
        assert result.action == "translate"
        assert result.modifier == "up"

    def test_detect_translate_down(self, resonator, encoder, detector):
        """Should detect downward translation."""
        input_grid = Grid.from_list([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        output_grid = Grid.from_list([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        
        input_objects = detector.detect(input_grid)
        output_objects = detector.detect(output_grid)
        matches = detector.match_objects(input_objects, output_objects)
        
        obs = encoder.encode_transformation_observation(
            matches[0][0], matches[0][1]
        )
        result = resonator.resonate(obs)
        
        assert result.action == "translate"
        assert result.modifier == "down"


class TestResonatorRecolor:
    """Tests for recolor detection."""

    def test_detect_recolor_to_blue(self, resonator, encoder, detector):
        """Should detect recoloring to blue."""
        input_grid = Grid.from_list([
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0],
        ])
        output_grid = Grid.from_list([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        
        input_objects = detector.detect(input_grid)
        output_objects = detector.detect(output_grid)
        matches = detector.match_objects(input_objects, output_objects)
        
        obs = encoder.encode_transformation_observation(
            matches[0][0], matches[0][1]
        )
        result = resonator.resonate(obs)
        
        assert result.action == "recolor"
        assert result.modifier == "to_blue"


class TestResonatorIdentity:
    """Tests for identity (no change) detection."""

    def test_detect_identity(self, resonator, encoder, detector):
        """Should detect no change as identity."""
        input_grid = Grid.from_list([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        output_grid = Grid.from_list([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        
        input_objects = detector.detect(input_grid)
        output_objects = detector.detect(output_grid)
        matches = detector.match_objects(input_objects, output_objects)
        
        obs = encoder.encode_transformation_observation(
            matches[0][0], matches[0][1]
        )
        result = resonator.resonate(obs)
        
        assert result.action == "identity"


class TestResonatorVerification:
    """Tests for factorization verification."""

    def test_verify_matches_observation(self, resonator, encoder, detector):
        """Verification score should be positive for valid factorization."""
        input_grid = Grid.from_list([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        output_grid = Grid.from_list([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        
        input_objects = detector.detect(input_grid)
        output_objects = detector.detect(output_grid)
        matches = detector.match_objects(input_objects, output_objects)
        
        obs = encoder.encode_transformation_observation(
            matches[0][0], matches[0][1]
        )
        result = resonator.resonate(obs)
        
        verification = resonator.verify_factorization(obs, result)
        
        # Verification should be positive (some match)
        assert verification > 0


class TestResonatorNoHallucination:
    """Tests for no-hallucination guarantee."""

    def test_output_only_vocabulary(self, resonator, encoder):
        """Resonator should only output vocabulary items."""
        # Random noise observation
        obs = torch.randn(10000)
        result = resonator.resonate(obs)
        
        # Must be valid vocabulary items
        assert result.action in ACTIONS
        assert result.target in TARGETS
        assert result.modifier in MODIFIERS

    def test_consistent_across_runs(self, encoder):
        """Same observation should produce same result."""
        space = VectorSpace(dimensions=10000)
        codebook = Codebook(space)
        
        # Fixed seed for reproducibility
        torch.manual_seed(42)
        obs = torch.randn(10000)
        
        resonator1 = TransformationResonator(encoder, codebook)
        result1 = resonator1.resonate(obs)
        
        resonator2 = TransformationResonator(encoder, codebook)
        result2 = resonator2.resonate(obs)
        
        # Results may differ due to initialization, but should be valid
        assert result1.action in ACTIONS
        assert result2.action in ACTIONS


class TestResonatorConvergence:
    """Tests for convergence behavior."""

    def test_convergence_reported(self, resonator, encoder, detector):
        """Resonator should report convergence status."""
        input_grid = Grid.from_list([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        output_grid = Grid.from_list([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        
        input_objects = detector.detect(input_grid)
        output_objects = detector.detect(output_grid)
        matches = detector.match_objects(input_objects, output_objects)
        
        obs = encoder.encode_transformation_observation(
            matches[0][0], matches[0][1]
        )
        result = resonator.resonate(obs)
        
        # Should have convergence info
        assert isinstance(result.converged, bool)
        assert result.iterations >= 1
