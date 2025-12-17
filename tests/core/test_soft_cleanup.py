"""Tests for soft cleanup functionality."""

import pytest
import torch

from hologram.core.resonator import Resonator
from hologram.core.codebook import Codebook
from hologram.core.vector_space import VectorSpace


class TestSoftCleanup:
    """Tests for soft cleanup mechanism."""

    @pytest.fixture
    def resonator(self):
        space = VectorSpace(dimensions=1000)
        codebook = Codebook(space)
        return Resonator(codebook)

    def test_soft_cleanup_returns_unit_vector(self, resonator):
        """Soft cleanup should return normalized vector."""
        vocab = torch.randn(10, 1000)
        proposal = torch.randn(1000)

        result = resonator._cleanup_soft(proposal, vocab, temperature=0.5)

        assert torch.abs(torch.norm(result) - 1.0) < 1e-5

    def test_low_temperature_approaches_hard(self, resonator):
        """Very low temperature should behave like argmax."""
        vocab = torch.randn(10, 1000)
        vocab = vocab / torch.norm(vocab, dim=1, keepdim=True)
        proposal = vocab[3] + torch.randn(1000) * 0.1  # Close to item 3

        soft_result = resonator._cleanup_soft(proposal, vocab, temperature=0.01)

        # Should be very close to vocab[3]
        sim = torch.cosine_similarity(soft_result.unsqueeze(0), vocab[3].unsqueeze(0))
        assert sim > 0.95

    def test_high_temperature_is_smooth(self, resonator):
        """High temperature should spread weight across items."""
        vocab = torch.randn(10, 1000)
        vocab = vocab / torch.norm(vocab, dim=1, keepdim=True)
        proposal = torch.randn(1000)

        soft_result = resonator._cleanup_soft(proposal, vocab, temperature=10.0)

        # Result should not be close to any single item
        similarities = torch.cosine_similarity(
            soft_result.unsqueeze(0), vocab
        )
        assert similarities.max() < 0.5  # Spread out
