"""
Tests for ConstraintAccumulator - Phase 1.1 implementation.

Verifies that:
1. Transformations are tracked correctly
2. Partial successes are ranked by score
3. Penalty vectors suppress failed transformations
4. Bias vectors promote variations of partial successes
"""

import pytest
import torch

from hologram.arc.constraint_accumulator import (
    ConstraintAccumulator,
    TransformSignature,
)
from hologram.arc.transform_resonator import TransformResult


@pytest.fixture
def dummy_transform():
    """Create a dummy transformation for testing."""
    dummy_vec = torch.zeros(10000)
    return TransformResult(
        action="rotate",
        target="all_objects",
        modifier="90_degrees",
        action_vec=dummy_vec,
        target_vec=dummy_vec,
        modifier_vec=dummy_vec,
        iterations=1,
        converged=True,
        confidence={"action": 0.8, "target": 0.9, "modifier": 0.7},
    )


@pytest.fixture
def vocabulary_vectors():
    """Create small vocabulary vectors for testing."""
    dim = 100
    actions = ["rotate", "translate", "recolor"]
    targets = ["all_objects", "largest", "smallest"]
    modifiers = ["90_degrees", "up", "to_red"]

    action_vecs = torch.randn(len(actions), dim)
    target_vecs = torch.randn(len(targets), dim)
    modifier_vecs = torch.randn(len(modifiers), dim)

    return {
        "action_vecs": action_vecs,
        "target_vecs": target_vecs,
        "modifier_vecs": modifier_vecs,
        "action_names": actions,
        "target_names": targets,
        "modifier_names": modifiers,
    }


class TestTransformSignature:
    """Test the TransformSignature dataclass."""

    def test_from_result(self, dummy_transform):
        """Test creating signature from TransformResult."""
        sig = TransformSignature.from_result(dummy_transform)
        assert sig.action == "rotate"
        assert sig.target == "all_objects"
        assert sig.modifier == "90_degrees"

    def test_hashable(self, dummy_transform):
        """Test that signatures are hashable (for use in sets)."""
        sig1 = TransformSignature.from_result(dummy_transform)
        sig2 = TransformSignature.from_result(dummy_transform)

        # Should be equal and hash to same value
        assert sig1 == sig2
        assert hash(sig1) == hash(sig2)

        # Should work in a set
        sig_set = {sig1, sig2}
        assert len(sig_set) == 1

    def test_string_representation(self, dummy_transform):
        """Test string representation."""
        sig = TransformSignature.from_result(dummy_transform)
        assert str(sig) == "rotate(all_objects, 90_degrees)"


class TestConstraintAccumulator:
    """Test the ConstraintAccumulator class."""

    def test_initialization(self):
        """Test accumulator initializes empty."""
        acc = ConstraintAccumulator()
        assert len(acc) == 0
        assert acc.get_best_partial() is None

    def test_record_attempt_tracks_signature(self, dummy_transform):
        """Test that recording an attempt tracks the signature."""
        acc = ConstraintAccumulator()
        acc.record_attempt(dummy_transform, partial_score=0.5)

        assert len(acc) == 1
        assert acc.has_tried(dummy_transform)

    def test_record_attempt_ignores_low_scores(self, dummy_transform):
        """Test that attempts with low scores don't become partial successes."""
        acc = ConstraintAccumulator(failure_threshold=0.2)
        acc.record_attempt(dummy_transform, partial_score=0.1)

        # Should be tracked as tried
        assert acc.has_tried(dummy_transform)

        # But not as partial success
        assert acc.get_best_partial() is None

    def test_partial_successes_sorted_by_score(self, vocabulary_vectors):
        """Test that partial successes are sorted best-first."""
        acc = ConstraintAccumulator()

        # Create three different transforms with different scores
        dummy_vec = torch.zeros(100)

        transforms_and_scores = [
            (TransformResult(
                action="rotate", target="all_objects", modifier="90_degrees",
                action_vec=dummy_vec, target_vec=dummy_vec, modifier_vec=dummy_vec,
                iterations=1, converged=True,
                confidence={"action": 0.8, "target": 0.8, "modifier": 0.8}
            ), 0.3),
            (TransformResult(
                action="translate", target="largest", modifier="up",
                action_vec=dummy_vec, target_vec=dummy_vec, modifier_vec=dummy_vec,
                iterations=1, converged=True,
                confidence={"action": 0.7, "target": 0.7, "modifier": 0.7}
            ), 0.8),  # Best score
            (TransformResult(
                action="recolor", target="smallest", modifier="to_red",
                action_vec=dummy_vec, target_vec=dummy_vec, modifier_vec=dummy_vec,
                iterations=1, converged=True,
                confidence={"action": 0.6, "target": 0.6, "modifier": 0.6}
            ), 0.5),
        ]

        for transform, score in transforms_and_scores:
            acc.record_attempt(transform, score)

        # Best partial should be the one with score 0.8
        best_sig, best_score = acc.get_best_partial()
        assert best_score == 0.8
        assert best_sig.action == "translate"

    def test_penalty_vector_has_correct_shape(self, dummy_transform, vocabulary_vectors):
        """Test that penalty vector has correct dimensions."""
        acc = ConstraintAccumulator()
        acc.record_attempt(dummy_transform, partial_score=0.1)  # Failure

        penalty = acc.get_penalty_vector(
            vocabulary_vectors["action_vecs"],
            vocabulary_vectors["target_vecs"],
            vocabulary_vectors["modifier_vecs"],
            vocabulary_vectors["action_names"],
            vocabulary_vectors["target_names"],
            vocabulary_vectors["modifier_names"],
        )

        # Should match vocabulary dimension
        assert penalty.shape == (100,)

    def test_penalty_vector_zero_when_no_failures(self, vocabulary_vectors):
        """Test that penalty is zero when nothing has been tried."""
        acc = ConstraintAccumulator()

        penalty = acc.get_penalty_vector(
            vocabulary_vectors["action_vecs"],
            vocabulary_vectors["target_vecs"],
            vocabulary_vectors["modifier_vecs"],
            vocabulary_vectors["action_names"],
            vocabulary_vectors["target_names"],
            vocabulary_vectors["modifier_names"],
        )

        assert torch.allclose(penalty, torch.zeros(100))

    def test_penalty_strength_scales_penalty(self, dummy_transform, vocabulary_vectors):
        """Test that penalty_strength scales the penalty vector."""
        acc = ConstraintAccumulator()
        acc.record_attempt(dummy_transform, partial_score=0.05)

        penalty_weak = acc.get_penalty_vector(
            vocabulary_vectors["action_vecs"],
            vocabulary_vectors["target_vecs"],
            vocabulary_vectors["modifier_vecs"],
            vocabulary_vectors["action_names"],
            vocabulary_vectors["target_names"],
            vocabulary_vectors["modifier_names"],
            penalty_strength=0.1,
        )

        penalty_strong = acc.get_penalty_vector(
            vocabulary_vectors["action_vecs"],
            vocabulary_vectors["target_vecs"],
            vocabulary_vectors["modifier_vecs"],
            vocabulary_vectors["action_names"],
            vocabulary_vectors["target_names"],
            vocabulary_vectors["modifier_names"],
            penalty_strength=0.5,
        )

        # Strong penalty should be larger
        assert penalty_strong.norm() > penalty_weak.norm()

    def test_bias_vector_none_when_no_partial_successes(self, vocabulary_vectors):
        """Test that bias is None when there are no partial successes."""
        acc = ConstraintAccumulator()

        bias = acc.suggest_search_bias(
            vocabulary_vectors["action_vecs"],
            vocabulary_vectors["target_vecs"],
            vocabulary_vectors["modifier_vecs"],
            vocabulary_vectors["action_names"],
            vocabulary_vectors["target_names"],
            vocabulary_vectors["modifier_names"],
        )

        assert bias is None

    def test_bias_vector_provided_for_partial_success(self, dummy_transform, vocabulary_vectors):
        """Test that bias is provided when there's a partial success."""
        acc = ConstraintAccumulator()
        acc.record_attempt(dummy_transform, partial_score=0.6)  # Partial success

        bias = acc.suggest_search_bias(
            vocabulary_vectors["action_vecs"],
            vocabulary_vectors["target_vecs"],
            vocabulary_vectors["modifier_vecs"],
            vocabulary_vectors["action_names"],
            vocabulary_vectors["target_names"],
            vocabulary_vectors["modifier_names"],
        )

        assert bias is not None
        assert bias.shape == (100,)
        assert not torch.allclose(bias, torch.zeros(100))

    def test_reset_clears_accumulator(self, dummy_transform):
        """Test that reset clears all tracked data."""
        acc = ConstraintAccumulator()
        acc.record_attempt(dummy_transform, partial_score=0.5)

        assert len(acc) == 1
        assert acc.get_best_partial() is not None

        acc.reset()

        assert len(acc) == 0
        assert acc.get_best_partial() is None
        assert not acc.has_tried(dummy_transform)

    def test_string_representation(self, dummy_transform):
        """Test string representation for debugging."""
        acc = ConstraintAccumulator()

        # Empty accumulator
        assert "tried=0" in str(acc)
        assert "best_partial=none" in str(acc)

        # With partial success
        acc.record_attempt(dummy_transform, partial_score=0.75)
        result_str = str(acc)
        assert "tried=1" in result_str
        assert "rotate(all_objects, 90_degrees)" in result_str
        assert "75" in result_str  # Score percentage


class TestConstraintAccumulatorIntegration:
    """Integration tests with realistic scenarios."""

    def test_multiple_failures_then_partial_success(self, vocabulary_vectors):
        """Test realistic scenario: several failures then a partial match."""
        acc = ConstraintAccumulator()
        dummy_vec = torch.zeros(100)

        # Fail with several different transforms
        for action, target, modifier, score in [
            ("rotate", "all_objects", "90_degrees", 0.0),
            ("translate", "largest", "up", 0.05),
            ("recolor", "smallest", "to_red", 0.02),
        ]:
            transform = TransformResult(
                action=action, target=target, modifier=modifier,
                action_vec=dummy_vec, target_vec=dummy_vec, modifier_vec=dummy_vec,
                iterations=1, converged=True,
                confidence={"action": 0.5, "target": 0.5, "modifier": 0.5}
            )
            acc.record_attempt(transform, score)

        # Then get a partial success
        good_transform = TransformResult(
            action="rotate", target="largest", modifier="90_degrees",
            action_vec=dummy_vec, target_vec=dummy_vec, modifier_vec=dummy_vec,
            iterations=1, converged=True,
            confidence={"action": 0.8, "target": 0.8, "modifier": 0.8}
        )
        acc.record_attempt(good_transform, partial_score=0.6)

        # Verify state
        assert len(acc) == 4
        best_sig, best_score = acc.get_best_partial()
        assert best_score == 0.6
        assert best_sig.action == "rotate"
        assert best_sig.target == "largest"

        # Penalty should include the failures
        penalty = acc.get_penalty_vector(
            vocabulary_vectors["action_vecs"],
            vocabulary_vectors["target_vecs"],
            vocabulary_vectors["modifier_vecs"],
            vocabulary_vectors["action_names"],
            vocabulary_vectors["target_names"],
            vocabulary_vectors["modifier_names"],
        )
        assert penalty.norm() > 0  # Non-zero penalty

        # Bias should favor the partial success
        bias = acc.suggest_search_bias(
            vocabulary_vectors["action_vecs"],
            vocabulary_vectors["target_vecs"],
            vocabulary_vectors["modifier_vecs"],
            vocabulary_vectors["action_names"],
            vocabulary_vectors["target_names"],
            vocabulary_vectors["modifier_names"],
        )
        assert bias is not None
        assert bias.norm() > 0
