"""Tests for CadenceJazz - cadence-based response composition."""

import pytest
import torch

from hologram.core.codebook import Codebook
from hologram.core.vector_space import VectorSpace
from hologram.generation.jazz import CadenceJazz, ComposedResponse
from hologram.generation.cadence_extractor import CadencePattern


@pytest.fixture
def space():
    """Create a vector space for testing."""
    return VectorSpace(dimensions=1000)


@pytest.fixture
def codebook(space):
    """Create a codebook for testing."""
    return Codebook(space)


@pytest.fixture
def cadence_jazz(codebook):
    """Create a CadenceJazz for testing."""
    return CadenceJazz(codebook)


@pytest.fixture
def capital_pattern(codebook):
    """Create a cadence pattern for capital facts."""
    return CadencePattern(
        template="__SLOT_ENTITY__ is the capital of __SLOT_ENTITY__",
        structure_vector=codebook.encode("capital_pattern"),
        slot_positions=[(0, "ENTITY", "Paris"), (27, "ENTITY", "France")],
        original_text="Paris is the capital of France",
    )


@pytest.fixture
def single_slot_pattern(codebook):
    """Create a pattern with single slot."""
    return CadencePattern(
        template="The answer is __SLOT_ENTITY__",
        structure_vector=codebook.encode("answer_pattern"),
        slot_positions=[(14, "ENTITY", "Paris")],
        original_text="The answer is Paris",
    )


class TestCadenceComposition:
    """Tests for cadence-based composition."""

    def test_compose_with_cadence_basic(self, cadence_jazz, capital_pattern, codebook):
        """Test basic composition with cadence pattern."""
        facts = [
            ("Paris", codebook.encode("Paris")),
            ("France", codebook.encode("France")),
        ]

        result = cadence_jazz.compose_with_cadence(facts, capital_pattern)

        assert isinstance(result, ComposedResponse)
        assert "Paris" in result.text
        assert "France" in result.text
        assert result.confidence > 0

    def test_compose_fills_slots_in_order(self, cadence_jazz, capital_pattern, codebook):
        """Test that slots are filled in correct order."""
        facts = [
            ("Berlin", codebook.encode("Berlin")),
            ("Germany", codebook.encode("Germany")),
        ]

        result = cadence_jazz.compose_with_cadence(facts, capital_pattern)

        # Should be "Berlin is the capital of Germany"
        assert result.text == "Berlin is the capital of Germany"

    def test_compose_with_single_fact(self, cadence_jazz, single_slot_pattern, codebook):
        """Test composition with single fact."""
        facts = [("Paris", codebook.encode("Paris"))]

        result = cadence_jazz.compose_with_cadence(facts, single_slot_pattern)

        assert result.text == "The answer is Paris"
        assert result.confidence > 0

    def test_compose_returns_vector(self, cadence_jazz, capital_pattern, codebook):
        """Test that composition returns a vector."""
        facts = [
            ("Paris", codebook.encode("Paris")),
            ("France", codebook.encode("France")),
        ]

        result = cadence_jazz.compose_with_cadence(facts, capital_pattern)

        assert isinstance(result.vector, torch.Tensor)
        assert torch.norm(result.vector).item() > 0


class TestCompositionConfidence:
    """Tests for composition confidence calculation."""

    def test_confidence_higher_with_all_slots_filled(self, cadence_jazz, capital_pattern, codebook):
        """Test that confidence is higher when all slots are filled."""
        # All slots filled
        full_facts = [
            ("Paris", codebook.encode("Paris")),
            ("France", codebook.encode("France")),
        ]
        full_result = cadence_jazz.compose_with_cadence(full_facts, capital_pattern)

        # Only one slot filled
        partial_facts = [("Paris", codebook.encode("Paris"))]
        partial_result = cadence_jazz.compose_with_cadence(partial_facts, capital_pattern)

        assert full_result.confidence > partial_result.confidence

    def test_confidence_with_empty_facts(self, cadence_jazz, capital_pattern):
        """Test confidence with no facts."""
        facts = []

        result = cadence_jazz.compose_with_cadence(facts, capital_pattern)

        assert result.confidence == 0.0

    def test_confidence_bounded(self, cadence_jazz, capital_pattern, codebook):
        """Test that confidence is bounded between 0 and 1."""
        facts = [
            ("Paris", codebook.encode("Paris")),
            ("France", codebook.encode("France")),
        ]

        result = cadence_jazz.compose_with_cadence(facts, capital_pattern)

        assert 0 <= result.confidence <= 1


class TestSlotFilling:
    """Tests for slot filling behavior."""

    def test_extra_facts_ignored(self, cadence_jazz, single_slot_pattern, codebook):
        """Test that extra facts beyond slots are ignored."""
        facts = [
            ("Paris", codebook.encode("Paris")),
            ("France", codebook.encode("France")),
            ("Europe", codebook.encode("Europe")),
        ]

        result = cadence_jazz.compose_with_cadence(facts, single_slot_pattern)

        # Only first fact should be used
        assert result.text == "The answer is Paris"
        assert "France" not in result.text
        assert "Europe" not in result.text

    def test_unfilled_slots_remain(self, cadence_jazz, capital_pattern, codebook):
        """Test that unfilled slots remain as markers."""
        facts = [("Paris", codebook.encode("Paris"))]

        result = cadence_jazz.compose_with_cadence(facts, capital_pattern)

        # Second slot should still be a marker
        assert "__SLOT_ENTITY__" in result.text

    def test_no_slots_pattern(self, cadence_jazz, codebook):
        """Test composition with pattern that has no slots."""
        pattern = CadencePattern(
            template="Hello, how are you?",
            structure_vector=codebook.encode("greeting_pattern"),
            slot_positions=[],
            original_text="Hello, how are you?",
        )
        facts = [("Paris", codebook.encode("Paris"))]

        result = cadence_jazz.compose_with_cadence(facts, pattern)

        # Text should be unchanged
        assert result.text == "Hello, how are you?"
        # Confidence should be reasonable (match_ratio=1.0, structure_quality varies)
        assert result.confidence > 0.5


class TestVectorComposition:
    """Tests for vector binding in composition."""

    def test_composed_vector_binds_all_facts(self, cadence_jazz, capital_pattern, codebook):
        """Test that composed vector binds all fact vectors."""
        paris_vec = codebook.encode("Paris")
        france_vec = codebook.encode("France")
        facts = [("Paris", paris_vec), ("France", france_vec)]

        result = cadence_jazz.compose_with_cadence(facts, capital_pattern)

        # Composed vector should be different from any single fact
        assert not torch.allclose(result.vector, paris_vec)
        assert not torch.allclose(result.vector, france_vec)

    def test_composed_vector_starts_with_structure(self, cadence_jazz, single_slot_pattern, codebook):
        """Test that composed vector includes structure vector."""
        facts = [("Paris", codebook.encode("Paris"))]

        result = cadence_jazz.compose_with_cadence(facts, single_slot_pattern)

        # Result should be different from structure alone (bound with fact)
        assert not torch.allclose(
            result.vector, single_slot_pattern.structure_vector
        )
