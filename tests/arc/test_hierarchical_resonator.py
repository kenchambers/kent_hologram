"""
Tests for RelationalEncoder and HierarchicalSalienceResonator.

These tests verify:
1. Salient relation extraction
2. Adjacency detection
3. Confidence gating
4. Relation augmentation
"""

import pytest
import torch
import numpy as np

from hologram.arc.types import Grid, Object, BoundingBox, Color
from hologram.arc.relational_encoder import RelationalEncoder, SalientRelation
from hologram.arc.hierarchical_resonator import HierarchicalSalienceResonator, HierarchicalResult
from hologram.arc.encoder import ObjectEncoder
from hologram.core.codebook import Codebook
from hologram.core.vector_space import VectorSpace
from hologram.core.fractal import FractalSpace


class TestRelationalEncoder:
    """Tests for RelationalEncoder class."""

    @pytest.fixture
    def encoder(self):
        """Create relational encoder."""
        space = VectorSpace(dimensions=1000)
        codebook = Codebook(space)
        return RelationalEncoder(codebook, space)

    def test_extract_no_relations_single_object(self, encoder):
        """Single object should have no relations."""
        obj = Object(
            color=Color.RED,
            pixels=frozenset([(0, 0), (0, 1), (1, 0), (1, 1)]),
            bbox=BoundingBox(0, 0, 1, 1),
            mask=((1, 1), (1, 1)),
        )

        relations = encoder.extract_salient_relations([obj])
        assert len(relations) == 0

    def test_adjacency_detection_horizontal(self, encoder):
        """Test adjacency detection for horizontally adjacent objects."""
        obj_a = Object(
            color=Color.RED,
            pixels=frozenset([(0, 0), (0, 1)]),
            bbox=BoundingBox(0, 0, 0, 1),
            mask=((1, 1),),
        )
        obj_b = Object(
            color=Color.BLUE,
            pixels=frozenset([(0, 2), (0, 3)]),
            bbox=BoundingBox(0, 2, 0, 3),
            mask=((1, 1),),
        )

        # Objects share column 1-2 edge
        assert encoder._are_adjacent(obj_a, obj_b)

    def test_adjacency_detection_diagonal(self, encoder):
        """Test adjacency detection for diagonally adjacent objects."""
        obj_a = Object(
            color=Color.RED,
            pixels=frozenset([(0, 0)]),
            bbox=BoundingBox(0, 0, 0, 0),
            mask=((1,),),
        )
        obj_b = Object(
            color=Color.BLUE,
            pixels=frozenset([(1, 1)]),
            bbox=BoundingBox(1, 1, 1, 1),
            mask=((1,),),
        )

        # Diagonal adjacency should be detected
        assert encoder._are_adjacent(obj_a, obj_b)

    def test_non_adjacent_objects(self, encoder):
        """Test that non-adjacent objects are correctly identified."""
        obj_a = Object(
            color=Color.RED,
            pixels=frozenset([(0, 0)]),
            bbox=BoundingBox(0, 0, 0, 0),
            mask=((1,),),
        )
        obj_b = Object(
            color=Color.BLUE,
            pixels=frozenset([(5, 5)]),
            bbox=BoundingBox(5, 5, 5, 5),
            mask=((1,),),
        )

        assert not encoder._are_adjacent(obj_a, obj_b)

    def test_same_color_relation(self, encoder):
        """Test same_color_as relation detection."""
        obj_a = Object(
            color=Color.RED,
            pixels=frozenset([(0, 0), (0, 1)]),
            bbox=BoundingBox(0, 0, 0, 1),
            mask=((1, 1),),
        )
        obj_b = Object(
            color=Color.RED,
            pixels=frozenset([(0, 2), (0, 3)]),  # Adjacent
            bbox=BoundingBox(0, 2, 0, 3),
            mask=((1, 1),),
        )

        relations = encoder.extract_salient_relations([obj_a, obj_b])

        # Should have adjacent_to and same_color_as
        relation_types = [r.relation for r in relations]
        assert "adjacent_to" in relation_types
        assert "same_color_as" in relation_types

    def test_same_shape_relation(self, encoder):
        """Test same_shape_as relation detection."""
        # Two objects with same mask (same shape)
        obj_a = Object(
            color=Color.RED,
            pixels=frozenset([(0, 0), (0, 1)]),
            bbox=BoundingBox(0, 0, 0, 1),
            mask=((1, 1),),
        )
        obj_b = Object(
            color=Color.BLUE,
            pixels=frozenset([(0, 2), (0, 3)]),  # Adjacent, same shape
            bbox=BoundingBox(0, 2, 0, 3),
            mask=((1, 1),),  # Same mask
        )

        relations = encoder.extract_salient_relations([obj_a, obj_b])

        relation_types = [r.relation for r in relations]
        assert "same_shape_as" in relation_types

    def test_containment_relation(self, encoder):
        """Test inside_of relation detection."""
        # Large outer object
        outer = Object(
            color=Color.RED,
            pixels=frozenset([(r, c) for r in range(5) for c in range(5)]),
            bbox=BoundingBox(0, 0, 4, 4),
            mask=tuple(tuple(1 for _ in range(5)) for _ in range(5)),
        )
        # Small inner object (fully contained)
        inner = Object(
            color=Color.BLUE,
            pixels=frozenset([(2, 2)]),
            bbox=BoundingBox(2, 2, 2, 2),
            mask=((1,),),
        )

        relations = encoder.extract_salient_relations([inner, outer])

        relation_types = [r.relation for r in relations]
        assert "inside_of" in relation_types

    def test_max_relations_cap(self, encoder):
        """Test that relations are capped at MAX_RELATIONS."""
        # Create many objects to generate many relations
        objects = []
        for i in range(10):
            obj = Object(
                color=Color.RED,
                pixels=frozenset([(0, i * 2), (0, i * 2 + 1)]),
                bbox=BoundingBox(0, i * 2, 0, i * 2 + 1),
                mask=((1, 1),),
            )
            objects.append(obj)

        relations = encoder.extract_salient_relations(objects)

        # Should be capped at max_relations
        assert len(relations) <= encoder._max_relations

    def test_encode_relation_context(self, encoder):
        """Test encoding relations into context vector."""
        obj_a = Object(
            color=Color.RED,
            pixels=frozenset([(0, 0), (0, 1)]),
            bbox=BoundingBox(0, 0, 0, 1),
            mask=((1, 1),),
        )
        obj_b = Object(
            color=Color.BLUE,
            pixels=frozenset([(0, 2), (0, 3)]),
            bbox=BoundingBox(0, 2, 0, 3),
            mask=((1, 1),),
        )

        context = encoder.encode_salient_relations([obj_a, obj_b])

        # Should produce non-zero vector
        assert context.shape[0] > 0
        assert torch.norm(context) > 0


class TestHierarchicalSalienceResonator:
    """Tests for HierarchicalSalienceResonator class."""

    @pytest.fixture
    def resonator(self):
        """Create hierarchical resonator."""
        space = VectorSpace(dimensions=1000)
        fractal = FractalSpace(dimensions=1000)
        codebook = Codebook(space)
        encoder = ObjectEncoder(fractal, codebook)
        return HierarchicalSalienceResonator(encoder, codebook)

    def test_fast_path_high_confidence(self, resonator):
        """Test that high confidence skips relations."""
        # Create a simple observation that should resonate with high confidence
        role_action = resonator._codebook.encode("__ROLE_ACTION__")
        role_target = resonator._codebook.encode("__ROLE_TARGET__")
        role_modifier = resonator._codebook.encode("__ROLE_MODIFIER__")

        action_vec = resonator._codebook.encode("action_identity")
        target_vec = resonator._codebook.encode("target_all_objects")
        modifier_vec = resonator._codebook.encode("modifier_none")

        from hologram.core.operations import Operations

        observation = Operations.bundle(
            Operations.bind(action_vec, role_action),
            Operations.bind(target_vec, role_target),
            Operations.bind(modifier_vec, role_modifier),
        )

        # Empty object list - relations shouldn't matter for fast path
        result = resonator.resonate_with_relations(observation, [])

        assert isinstance(result, HierarchicalResult)
        assert result.result is not None

    def test_compute_gate_weight(self, resonator):
        """Test gate weight computation."""
        # High confidence -> low alpha
        alpha_high = resonator._compute_gate_weight(0.8)
        assert alpha_high == 0.0

        # Low confidence -> high alpha
        alpha_low = resonator._compute_gate_weight(0.3)
        assert alpha_low == 1.0

        # Medium confidence -> interpolated
        alpha_mid = resonator._compute_gate_weight(0.6)
        assert 0.0 < alpha_mid < 1.0

    def test_standard_resonate_delegates(self, resonator):
        """Test that standard resonate() delegates to base resonator."""
        from hologram.core.operations import Operations

        role_action = resonator._codebook.encode("__ROLE_ACTION__")
        action_vec = resonator._codebook.encode("action_identity")

        observation = Operations.bind(action_vec, role_action)

        result = resonator.resonate(observation)

        # Should return TransformResult
        assert result is not None
        assert hasattr(result, "action")

    def test_hierarchical_result_metadata(self, resonator):
        """Test HierarchicalResult contains proper metadata."""
        from hologram.core.operations import Operations

        role_action = resonator._codebook.encode("__ROLE_ACTION__")
        role_target = resonator._codebook.encode("__ROLE_TARGET__")
        role_modifier = resonator._codebook.encode("__ROLE_MODIFIER__")

        observation = Operations.bundle(
            Operations.bind(resonator._codebook.encode("action_identity"), role_action),
            Operations.bind(resonator._codebook.encode("target_all_objects"), role_target),
            Operations.bind(resonator._codebook.encode("modifier_none"), role_modifier),
        )

        result = resonator.resonate_with_relations(observation, [])

        assert hasattr(result, "used_relations")
        assert hasattr(result, "relation_count")
        assert hasattr(result, "alpha")
        assert hasattr(result, "confidence_improvement")


class TestIntegration:
    """Integration tests for relational encoding."""

    def test_full_pipeline_with_objects(self):
        """Test full pipeline: objects -> relations -> resonation."""
        space = VectorSpace(dimensions=1000)
        fractal = FractalSpace(dimensions=1000)
        codebook = Codebook(space)
        encoder = ObjectEncoder(fractal, codebook)
        resonator = HierarchicalSalienceResonator(encoder, codebook)

        # Create test objects
        obj_a = Object(
            color=Color.RED,
            pixels=frozenset([(0, 0), (0, 1)]),
            bbox=BoundingBox(0, 0, 0, 1),
            mask=((1, 1),),
        )
        obj_b = Object(
            color=Color.RED,
            pixels=frozenset([(0, 2), (0, 3)]),
            bbox=BoundingBox(0, 2, 0, 3),
            mask=((1, 1),),
        )

        # Create observation
        obs = encoder.encode_transformation_observation(obj_a, obj_b)

        # Resonate with relations
        result = resonator.resonate_with_relations(obs, [obj_a, obj_b])

        assert result.result is not None
        assert result.result.action is not None
