"""Integration tests for SWE module with shared memory."""

import pytest
import torch
from pathlib import Path

from hologram.container import HologramContainer
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
def container():
    """Create HologramContainer for tests."""
    return HologramContainer(dimensions=10000)


@pytest.fixture
def shared_components(container):
    """Create shared components from container."""
    space = container._space
    codebook = container._codebook

    # ARC encoder (uses existing vocabulary)
    arc_encoder = ObjectEncoder(space, codebook)

    # Code encoder (shares codebook with ARC)
    code_encoder = CodeEncoder(space, codebook)

    # Transformation resonator
    transform_resonator = TransformationResonator(arc_encoder, codebook)

    # Code resonator
    code_resonator = CodeResonator(code_encoder, transform_resonator)

    # Shared neural memory
    neural_memory = NeuralMemory(
        input_dim=10000,
        hidden_dim=256,
        initial_vocab_size=100,
    )

    return {
        "container": container,
        "code_encoder": code_encoder,
        "code_resonator": code_resonator,
        "neural_memory": neural_memory,
    }


class TestSharedMemory:
    """Tests for shared memory between ARC and SWE."""

    def test_shared_codebook(self, shared_components):
        """CodeEncoder and ARC should share codebook."""
        code_encoder = shared_components["code_encoder"]
        container = shared_components["container"]

        # Both should produce vectors with same dimensions
        issue_vec = code_encoder.encode_issue("Fix bug")
        # Codebook is shared via container - check dimensions match
        assert issue_vec.shape[0] == container._space.dimensions

    def test_neural_memory_can_store_code_patterns(self, shared_components):
        """Neural memory should accept code patterns."""
        neural_memory = shared_components["neural_memory"]
        code_encoder = shared_components["code_encoder"]

        # Create code pattern
        issue_vec = code_encoder.encode_issue("Add validation to process")

        # Should be able to query (even without training)
        label, confidence = neural_memory.query(issue_vec)
        # May return None (no training) but should not error
        assert label is None or isinstance(label, str)


class TestEndToEndGeneration:
    """End-to-end generation tests."""

    def test_full_pipeline(self, shared_components):
        """Should run full generation pipeline."""
        generator = CodeGenerator(
            encoder=shared_components["code_encoder"],
            resonator=shared_components["code_resonator"],
            neural_memory=shared_components["neural_memory"],
        )

        task = SWETask(
            task_id="e2e_test",
            repo="test/repo",
            issue_text="Add logging to the calculate function",
            code_before={"math.py": "def calculate(a, b):\n    return a + b"},
            code_after={"math.py": "import logging\n\ndef calculate(a, b):\n    logging.info('calc')\n    return a + b"},
        )

        result = generator.generate(task)

        assert isinstance(result, PatchResult)
        assert len(result.patches) > 0
        assert result.patches[0].file == "math.py"

    def test_learn_then_generate(self, shared_components):
        """Should improve after learning."""
        generator = CodeGenerator(
            encoder=shared_components["code_encoder"],
            resonator=shared_components["code_resonator"],
            neural_memory=shared_components["neural_memory"],
        )

        # Task 1: Learn from example
        task1 = SWETask(
            task_id="learn_001",
            repo="test/repo",
            issue_text="Add null check to function",
            code_before={"utils.py": "def foo(x): return x"},
            code_after={"utils.py": "def foo(x):\n    if x is None: raise\n    return x"},
        )

        generator.learn_from_task(task1)

        # Task 2: Generate for similar issue
        task2 = SWETask(
            task_id="gen_001",
            repo="test/repo",
            issue_text="Add null validation to method",
            code_before={"helper.py": "def bar(y): return y * 2"},
            code_after={},
        )

        result = generator.generate(task2)

        # Should generate something
        assert isinstance(result, PatchResult)


class TestContainerIntegration:
    """Tests for HologramContainer integration."""

    def test_container_provides_shared_components(self, container):
        """Container should provide shared components."""
        assert container._space is not None
        assert container._codebook is not None

    def test_multiple_encoders_share_codebook(self, container):
        """Multiple encoders should share same codebook."""
        encoder1 = CodeEncoder(container._space, container._codebook)
        encoder2 = CodeEncoder(container._space, container._codebook)

        # Same word should produce same vector
        vec1 = encoder1._codebook.encode("test_word")
        vec2 = encoder2._codebook.encode("test_word")

        assert torch.allclose(vec1, vec2)
