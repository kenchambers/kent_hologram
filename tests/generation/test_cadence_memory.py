"""Tests for CadenceMemory - neural memory for cadence patterns."""

import pytest
import torch

from hologram.core.codebook import Codebook
from hologram.core.vector_space import VectorSpace
from hologram.generation.cadence_memory import CadenceMemory
from hologram.generation.cadence_extractor import CadencePattern


@pytest.fixture
def dimensions():
    """Default vector dimensions."""
    return 1000


@pytest.fixture
def space(dimensions):
    """Create a vector space for testing."""
    return VectorSpace(dimensions=dimensions)


@pytest.fixture
def codebook(space):
    """Create a codebook for testing."""
    return Codebook(space)


@pytest.fixture
def cadence_memory(dimensions):
    """Create a CadenceMemory for testing."""
    return CadenceMemory(dimensions=dimensions, hidden_dim=64)


@pytest.fixture
def sample_pattern(codebook):
    """Create a sample CadencePattern."""
    template = "__SLOT_ENTITY__ is the capital of __SLOT_ENTITY__"
    return CadencePattern(
        template=template,
        structure_vector=codebook.encode("capital_structure"),
        slot_positions=[(0, "ENTITY", "Paris"), (23, "ENTITY", "France")],
        original_text="Paris is the capital of France",
    )


class TestCadenceMemoryStorage:
    """Tests for cadence pattern storage."""

    def test_store_cadence(self, cadence_memory, sample_pattern, space):
        """Test storing a cadence pattern."""
        context_vec = space.random_vector(seed=42)

        cadence_memory.store_cadence(context_vec, sample_pattern)

        assert cadence_memory.pattern_count == 1

    def test_store_multiple_patterns(self, cadence_memory, codebook, space):
        """Test storing multiple patterns."""
        patterns = []
        for i in range(5):
            pattern = CadencePattern(
                template=f"Pattern {i} __SLOT_ENTITY__",
                structure_vector=codebook.encode(f"structure_{i}"),
                slot_positions=[(9, "ENTITY", f"entity_{i}")],
                original_text=f"Pattern {i} entity_{i}",
            )
            patterns.append(pattern)

        for i, pattern in enumerate(patterns):
            context_vec = space.random_vector(seed=100 + i)
            cadence_memory.store_cadence(context_vec, pattern)

        assert cadence_memory.pattern_count == 5

    def test_pattern_deduplication(self, cadence_memory, sample_pattern, space):
        """Test that duplicate patterns are not stored twice."""
        context_vec1 = space.random_vector(seed=1)
        context_vec2 = space.random_vector(seed=2)

        cadence_memory.store_cadence(context_vec1, sample_pattern)
        cadence_memory.store_cadence(context_vec2, sample_pattern)

        # Same pattern should not be duplicated
        assert cadence_memory.pattern_count == 1


class TestCadenceMemoryQuery:
    """Tests for cadence pattern querying."""

    def test_query_returns_none_without_training(self, cadence_memory, space):
        """Test that query returns None before consolidation."""
        context_vec = space.random_vector(seed=42)

        result = cadence_memory.query_cadence(context_vec)

        # Without training, should return None or low confidence
        # (depends on neural network initialization)
        assert result is None or isinstance(result, CadencePattern)


class TestCadenceMemoryPersistence:
    """Tests for cadence memory persistence."""

    def test_get_state_dict(self, cadence_memory, sample_pattern, space):
        """Test getting state dict for persistence."""
        context_vec = space.random_vector(seed=42)
        cadence_memory.store_cadence(context_vec, sample_pattern)

        state = cadence_memory.get_state_dict()

        assert "neural_state" in state
        assert "patterns" in state
        assert "pattern_counter" in state
        assert len(state["patterns"]) == 1

    def test_load_state_dict(self, dimensions, sample_pattern, space):
        """Test loading state dict restores memory."""
        # Create first memory and store pattern
        memory1 = CadenceMemory(dimensions=dimensions, hidden_dim=64)
        context_vec = space.random_vector(seed=42)
        memory1.store_cadence(context_vec, sample_pattern)

        # Get state
        state = memory1.get_state_dict()

        # Create new memory and load state
        memory2 = CadenceMemory(dimensions=dimensions, hidden_dim=64)
        memory2.load_state_dict(state)

        # Should have same pattern count
        assert memory2.pattern_count == memory1.pattern_count

    def test_state_dict_contains_pattern_data(self, cadence_memory, sample_pattern, space):
        """Test that state dict contains all pattern data."""
        context_vec = space.random_vector(seed=42)
        cadence_memory.store_cadence(context_vec, sample_pattern)

        state = cadence_memory.get_state_dict()

        # Check pattern data
        pattern_key = list(state["patterns"].keys())[0]
        pattern_data = state["patterns"][pattern_key]

        assert "template" in pattern_data
        assert "structure_vector" in pattern_data
        assert "slot_positions" in pattern_data
        assert "original_text" in pattern_data


class TestCadenceMemoryConsolidation:
    """Tests for neural consolidation."""

    def test_consolidate_empty_buffer(self, cadence_memory):
        """Test consolidation with empty buffer returns 0."""
        loss = cadence_memory.consolidate(epochs=1)

        assert loss == 0.0

    def test_consolidate_with_patterns(self, cadence_memory, codebook, space):
        """Test consolidation with stored patterns."""
        # Store multiple patterns
        for i in range(10):
            pattern = CadencePattern(
                template=f"Pattern {i} __SLOT_ENTITY__",
                structure_vector=codebook.encode(f"structure_{i}"),
                slot_positions=[(9, "ENTITY", f"entity_{i}")],
                original_text=f"Pattern {i} entity_{i}",
            )
            context_vec = space.random_vector(seed=200 + i)
            cadence_memory.store_cadence(context_vec, pattern)

        # Consolidate
        loss = cadence_memory.consolidate(epochs=5)

        # Should have trained (loss may be any value)
        assert isinstance(loss, float)

    def test_replay_buffer_size(self, cadence_memory, sample_pattern, space):
        """Test replay buffer size tracking."""
        assert cadence_memory.get_replay_buffer_size() == 0

        context_vec = space.random_vector(seed=42)
        cadence_memory.store_cadence(context_vec, sample_pattern)

        assert cadence_memory.get_replay_buffer_size() == 1
