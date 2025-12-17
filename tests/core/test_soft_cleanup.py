"""Tests for soft cleanup functionality."""

import math
import pytest
import torch

from hologram.core.resonator import Resonator
from hologram.core.codebook import Codebook
from hologram.core.vector_space import VectorSpace
from hologram.core.similarity import Similarity


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


class TestCosineAnnealing:
    """Tests for cosine temperature annealing schedule."""

    def test_temperature_starts_high_ends_low(self):
        """Temperature should decay smoothly from 0.5 to 0.01."""
        max_iters = 100

        # Start temperature (iteration 0)
        progress_start = 0 / max(1, max_iters - 1)
        temp_start = 0.01 + 0.49 * (1 + math.cos(math.pi * progress_start)) / 2
        assert abs(temp_start - 0.5) < 0.01, f"Start temp should be ~0.5, got {temp_start}"

        # End temperature (last iteration)
        progress_end = (max_iters - 1) / max(1, max_iters - 1)
        temp_end = 0.01 + 0.49 * (1 + math.cos(math.pi * progress_end)) / 2
        assert abs(temp_end - 0.01) < 0.01, f"End temp should be ~0.01, got {temp_end}"

    def test_temperature_is_monotonically_decreasing(self):
        """Temperature should decrease monotonically (no discontinuities)."""
        max_iters = 100
        temps = []

        for i in range(max_iters):
            progress = i / max(1, max_iters - 1)
            temp = 0.01 + 0.49 * (1 + math.cos(math.pi * progress)) / 2
            temps.append(temp)

        # Check monotonic decrease
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1] + 1e-6, \
                f"Temperature should decrease: temps[{i}]={temps[i]} > temps[{i-1}]={temps[i-1]}"

    def test_no_hard_phase_transition(self):
        """There should be no abrupt jump in temperature at midpoint."""
        max_iters = 100
        midpoint = max_iters // 2

        # Get temperatures around midpoint
        progress_before = (midpoint - 1) / max(1, max_iters - 1)
        progress_at = midpoint / max(1, max_iters - 1)
        progress_after = (midpoint + 1) / max(1, max_iters - 1)

        temp_before = 0.01 + 0.49 * (1 + math.cos(math.pi * progress_before)) / 2
        temp_at = 0.01 + 0.49 * (1 + math.cos(math.pi * progress_at)) / 2
        temp_after = 0.01 + 0.49 * (1 + math.cos(math.pi * progress_after)) / 2

        # All transitions should be smooth (small delta)
        delta_1 = abs(temp_at - temp_before)
        delta_2 = abs(temp_after - temp_at)

        assert delta_1 < 0.02, f"Transition before midpoint too sharp: {delta_1}"
        assert delta_2 < 0.02, f"Transition after midpoint too sharp: {delta_2}"


class TestConfidenceFromSoftOutput:
    """Tests for confidence calculation from soft output vector."""

    @pytest.fixture
    def resonator(self):
        space = VectorSpace(dimensions=1000)
        codebook = Codebook(space)
        return Resonator(codebook, max_iterations=100)

    def test_confidence_reflects_soft_output_certainty(self, resonator):
        """Confidence should be computed from soft output, not raw proposal."""
        vocab = torch.randn(10, 1000)
        vocab = vocab / torch.norm(vocab, dim=1, keepdim=True)
        vocabulary = [f"word_{i}" for i in range(10)]

        # Create proposal close to vocab[3]
        proposal = vocab[3] + torch.randn(1000) * 0.1

        # Get soft output
        soft_vec = resonator._cleanup_soft(proposal, vocab, temperature=0.1)

        # Compute confidence the way _solve_for_slot does
        soft_sims = Similarity.cosine_batch(soft_vec, vocab)
        top2 = torch.topk(soft_sims, 2)
        soft_conf = float((top2.values[0] - top2.values[1]).item())

        # Confidence from soft output should be high (soft output close to vocab item)
        assert soft_conf > 0.5, f"Soft output confidence should be high, got {soft_conf}"

    def test_confidence_changes_with_temperature(self, resonator):
        """Lower temperature should yield higher confidence."""
        vocab = torch.randn(10, 1000)
        vocab = vocab / torch.norm(vocab, dim=1, keepdim=True)

        proposal = vocab[3] + torch.randn(1000) * 0.3

        # High temperature (uncertain)
        soft_high = resonator._cleanup_soft(proposal, vocab, temperature=1.0)
        sims_high = Similarity.cosine_batch(soft_high, vocab)
        top2_high = torch.topk(sims_high, 2)
        conf_high = float((top2_high.values[0] - top2_high.values[1]).item())

        # Low temperature (certain)
        soft_low = resonator._cleanup_soft(proposal, vocab, temperature=0.01)
        sims_low = Similarity.cosine_batch(soft_low, vocab)
        top2_low = torch.topk(sims_low, 2)
        conf_low = float((top2_low.values[0] - top2_low.values[1]).item())

        assert conf_low > conf_high, \
            f"Low temp should yield higher confidence: {conf_low} vs {conf_high}"


class TestZeroNormFallback:
    """Tests for zero-norm edge case handling."""

    @pytest.fixture
    def resonator(self):
        space = VectorSpace(dimensions=1000)
        codebook = Codebook(space)
        return Resonator(codebook)

    def test_fallback_returns_unit_vector(self, resonator):
        """Even with pathological input, should return unit vector."""
        # Create vocab that could cause issues
        vocab = torch.randn(5, 1000)
        vocab = vocab / torch.norm(vocab, dim=1, keepdim=True)

        # Orthogonal proposal (worst case)
        proposal = torch.randn(1000)

        result = resonator._cleanup_soft(proposal, vocab, temperature=0.5)

        # Should always return unit vector
        norm = torch.norm(result).item()
        assert abs(norm - 1.0) < 1e-5, f"Result should be unit vector, got norm={norm}"

    def test_fallback_returns_vocabulary_item(self, resonator):
        """Fallback should return a valid vocabulary vector."""
        vocab = torch.randn(5, 1000)
        vocab = vocab / torch.norm(vocab, dim=1, keepdim=True)
        proposal = torch.randn(1000)

        result = resonator._cleanup_soft(proposal, vocab, temperature=0.5)

        # Result should have non-trivial similarity to at least one vocab item
        similarities = torch.cosine_similarity(result.unsqueeze(0), vocab)
        max_sim = similarities.max().item()
        assert max_sim > 0.3, f"Result should match some vocab item, max_sim={max_sim}"
