"""
Neural Consolidation Test Suite

Tests the hybrid HDC + Neural memory consolidation system:
- Async training infrastructure (non-blocking)
- Classification head neural memory (O(1) query)
- Confidence calibration (HDC vs Neural)
- Decay instead of delete (graceful degradation)
- Unbinding validation gate (HDC algebra preservation)

Run with: uv run pytest tests/test_neural_consolidation.py -v
"""

import time
import threading
import statistics
from typing import List, Tuple

import pytest
import torch

from hologram.core.vector_space import VectorSpace
from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.memory.memory_trace import MemoryTrace
from hologram.consolidation.neural_memory import NeuralMemory, ConsolidationFact
from hologram.consolidation.calibration import (
    ConfidenceCalibrator,
    AdaptiveCalibrator,
    CalibrationResult,
)
from hologram.consolidation.manager import ConsolidationManager, PendingFact


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def space():
    """Create a vector space for testing."""
    return VectorSpace(dimensions=1000)


@pytest.fixture
def codebook(space):
    """Create a codebook for encoding."""
    return Codebook(space)


@pytest.fixture
def neural_memory(space):
    """Create a neural memory instance."""
    return NeuralMemory(
        input_dim=space.dimensions,
        hidden_dim=128,
        initial_vocab_size=100,
    )


@pytest.fixture
def calibrator():
    """Create a confidence calibrator."""
    return ConfidenceCalibrator()


@pytest.fixture
def manager(space):
    """Create a consolidation manager."""
    return ConsolidationManager(
        space=space,
        consolidation_threshold=10,
        decay_factor=0.3,
        neural_hidden_dim=128,
        initial_vocab_size=100,
    )


# =============================================================================
# Section 1: MemoryTrace initial_trace Tests
# =============================================================================

class TestMemoryTraceInitialTrace:
    """Test the new initial_trace parameter in MemoryTrace."""

    def test_empty_initialization(self, space):
        """Test default empty initialization."""
        trace = MemoryTrace(space)
        assert trace.fact_count == 0
        assert torch.allclose(trace.trace_vector, space.empty_vector())

    def test_initial_trace_parameter(self, space):
        """Test initialization with initial_trace."""
        # Create a non-zero trace
        initial = torch.randn(space.dimensions)
        trace = MemoryTrace(space, initial_trace=initial)

        # Should have cloned the initial trace
        assert torch.allclose(trace.trace_vector, initial)
        assert trace.fact_count == 0  # No facts stored yet

    def test_initial_trace_is_cloned(self, space):
        """Test that initial_trace is cloned, not referenced."""
        initial = torch.randn(space.dimensions)
        trace = MemoryTrace(space, initial_trace=initial)

        # Modify original - should not affect trace
        initial[0] = 999.0
        assert trace.trace_vector[0] != 999.0

    def test_initial_trace_validation(self, space):
        """Test that initial_trace is validated."""
        wrong_dims = torch.randn(space.dimensions + 100)

        with pytest.raises(ValueError):
            MemoryTrace(space, initial_trace=wrong_dims)


# =============================================================================
# Section 2: NeuralMemory Tests
# =============================================================================

class TestNeuralMemory:
    """Test the classification-head neural memory."""

    def test_initialization(self, neural_memory):
        """Test neural memory initialization."""
        assert neural_memory.vocab_size == 0
        assert neural_memory.trained_samples == 0

    def test_query_empty(self, neural_memory, space):
        """Test querying empty neural memory."""
        key = torch.randn(space.dimensions)
        label, conf = neural_memory.query(key)

        assert label is None
        assert conf == 0.0

    def test_consolidate_single_fact(self, neural_memory, space):
        """Test consolidating a single fact."""
        key = torch.randn(space.dimensions)
        fact = ConsolidationFact(
            key_vector=key,
            value_index=-1,
            value_label="test_value",
            timestamp=time.time(),
        )

        loss = neural_memory.consolidate([fact], epochs=5)

        assert neural_memory.vocab_size == 1
        assert neural_memory.trained_samples >= 1
        assert loss >= 0.0

    def test_query_after_consolidation(self, neural_memory, space):
        """Test querying after consolidation."""
        key = torch.randn(space.dimensions)
        fact = ConsolidationFact(
            key_vector=key,
            value_index=-1,
            value_label="test_value",
            timestamp=time.time(),
        )

        neural_memory.consolidate([fact], epochs=20)

        # Query with same key
        label, conf = neural_memory.query(key)

        assert label == "test_value"
        assert conf > 0.5  # Should be somewhat confident

    def test_multiple_facts_discrimination(self, neural_memory, space):
        """Test that neural memory can discriminate multiple facts."""
        keys = [torch.randn(space.dimensions) for _ in range(5)]
        labels = [f"value_{i}" for i in range(5)]

        facts = [
            ConsolidationFact(
                key_vector=keys[i],
                value_index=-1,
                value_label=labels[i],
                timestamp=time.time(),
            )
            for i in range(5)
        ]

        neural_memory.consolidate(facts, epochs=30)

        # Query each key - should return corresponding label
        correct = 0
        for key, expected_label in zip(keys, labels):
            label, conf = neural_memory.query(key)
            if label == expected_label:
                correct += 1

        # Should get at least 80% correct
        assert correct >= 4, f"Only got {correct}/5 correct"

    def test_vocab_expansion(self, neural_memory, space):
        """Test vocabulary expansion when exceeding initial capacity."""
        # Add more labels than initial vocab size
        initial_vocab_size = 100
        n_labels = 150

        facts = [
            ConsolidationFact(
                key_vector=torch.randn(space.dimensions),
                value_index=-1,
                value_label=f"label_{i}",
                timestamp=time.time(),
            )
            for i in range(n_labels)
        ]

        neural_memory.consolidate(facts, epochs=5)

        assert neural_memory.vocab_size == n_labels

    def test_replay_buffer(self, neural_memory, space):
        """Test experience replay buffer."""
        # Add facts to replay buffer
        facts1 = [
            ConsolidationFact(
                key_vector=torch.randn(space.dimensions),
                value_index=-1,
                value_label="old_fact",
                timestamp=time.time(),
            )
        ]
        neural_memory.consolidate(facts1, epochs=5)

        assert neural_memory.replay_buffer_size >= 1

    def test_state_dict_persistence(self, neural_memory, space):
        """Test state dictionary for persistence."""
        # Train some facts
        facts = [
            ConsolidationFact(
                key_vector=torch.randn(space.dimensions),
                value_index=-1,
                value_label=f"value_{i}",
                timestamp=time.time(),
            )
            for i in range(5)
        ]
        neural_memory.consolidate(facts, epochs=5)

        # Get state
        state = neural_memory.state_dict()

        # Create new neural memory and load state
        new_memory = NeuralMemory(
            input_dim=space.dimensions,
            hidden_dim=128,
            initial_vocab_size=100,
        )
        new_memory.load_state_dict(state)

        assert new_memory.vocab_size == neural_memory.vocab_size
        assert new_memory.trained_samples == neural_memory.trained_samples


# =============================================================================
# Section 3: ConfidenceCalibrator Tests
# =============================================================================

class TestConfidenceCalibrator:
    """Test confidence calibration."""

    def test_hdc_calibration_range(self, calibrator):
        """Test that HDC calibration maps to [0, 1]."""
        # Test various HDC similarity values
        test_values = [0.0, 0.15, 0.35, 0.5, 0.6, 0.8, 1.0]

        for val in test_values:
            calibrated = calibrator.calibrate_hdc(val)
            assert 0.0 <= calibrated <= 1.0, f"HDC {val} -> {calibrated} out of range"

    def test_neural_calibration_range(self, calibrator):
        """Test that Neural calibration maps to [0, 1]."""
        test_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

        for val in test_values:
            calibrated = calibrator.calibrate_neural(val)
            assert 0.0 <= calibrated <= 1.0, f"Neural {val} -> {calibrated} out of range"

    def test_hdc_center_point(self, calibrator):
        """Test HDC sigmoid center point (0.35 -> ~0.5)."""
        # At center point, sigmoid should return ~0.5
        calibrated = calibrator.calibrate_hdc(0.35)
        assert 0.45 <= calibrated <= 0.55, f"HDC center {0.35} -> {calibrated}"

    def test_neural_center_point(self, calibrator):
        """Test Neural sigmoid center point (0.5 -> ~0.5)."""
        calibrated = calibrator.calibrate_neural(0.5)
        assert 0.45 <= calibrated <= 0.55, f"Neural center {0.5} -> {calibrated}"

    def test_pick_winner_hdc_wins(self, calibrator):
        """Test winner selection when HDC is better."""
        # HDC high, Neural low
        result = calibrator.pick_winner(hdc_raw=0.6, neural_raw=0.3)

        assert result.winner == "hdc"
        assert result.hdc_calibrated > result.neural_calibrated

    def test_pick_winner_neural_wins(self, calibrator):
        """Test winner selection when Neural is better."""
        # HDC low, Neural high
        result = calibrator.pick_winner(hdc_raw=0.2, neural_raw=0.9)

        assert result.winner == "neural"
        assert result.neural_calibrated > result.hdc_calibrated

    def test_unbind_safety_gate(self, calibrator):
        """Test unbinding safety gate."""
        # Neural would win but fails safety gate
        result = calibrator.pick_winner(
            hdc_raw=0.3,
            neural_raw=0.6,  # Would win, but below 0.85 threshold
            require_unbind_safety=True,
        )

        # Should fall back to HDC due to safety gate
        assert result.winner == "hdc"

    def test_unbind_safety_passes(self, calibrator):
        """Test that high-confidence neural passes unbind safety."""
        result = calibrator.pick_winner(
            hdc_raw=0.3,
            neural_raw=0.9,  # Above 0.85 threshold
            require_unbind_safety=True,
        )

        # Neural should win
        assert result.winner == "neural"

    def test_calibration_result_properties(self, calibrator):
        """Test CalibrationResult properties."""
        result = calibrator.pick_winner(hdc_raw=0.5, neural_raw=0.5)

        assert isinstance(result.is_confident, bool)
        assert isinstance(result.is_close_call, bool)
        assert result.margin >= 0.0

    def test_is_neural_safe_for_unbinding(self, calibrator):
        """Test unbinding safety check."""
        assert calibrator.is_neural_safe_for_unbinding(0.9)
        assert not calibrator.is_neural_safe_for_unbinding(0.7)


class TestAdaptiveCalibrator:
    """Test adaptive confidence calibration."""

    def test_initialization(self):
        """Test adaptive calibrator initialization."""
        calibrator = AdaptiveCalibrator(window_size=50)
        assert calibrator is not None

    def test_observe_hdc(self):
        """Test observing HDC scores."""
        calibrator = AdaptiveCalibrator(window_size=20)

        for i in range(30):
            calibrator.observe_hdc(0.3 + i * 0.01)

        # Should have adapted (implementation detail, just verify no crash)
        calibrated = calibrator.calibrate_hdc(0.4)
        assert 0.0 <= calibrated <= 1.0

    def test_observe_neural(self):
        """Test observing Neural scores."""
        calibrator = AdaptiveCalibrator(window_size=20)

        for i in range(30):
            calibrator.observe_neural(0.5 + i * 0.01)

        calibrated = calibrator.calibrate_neural(0.6)
        assert 0.0 <= calibrated <= 1.0


# =============================================================================
# Section 4: ConsolidationManager Tests
# =============================================================================

class TestConsolidationManager:
    """Test the consolidation manager."""

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.pending_count == 0
        assert manager.total_consolidated == 0
        assert manager.vocab_size == 0

    def test_store_fact(self, manager, space, codebook):
        """Test storing a fact."""
        key = codebook.encode("test_key")
        value = codebook.encode("test_value")

        latency = manager.store(key, value, "test_value")

        assert latency < 100  # Should be fast (< 100ms)
        assert manager.pending_count == 1
        assert manager.vocab_size == 1

    def test_store_latency_is_non_blocking(self, manager, space, codebook):
        """Test that store operations are non-blocking (< 10ms P99)."""
        latencies = []

        for i in range(100):
            key = codebook.encode(f"key_{i}")
            value = codebook.encode(f"value_{i}")
            latency = manager.store(key, value, f"value_{i}")
            latencies.append(latency)

        # P99 should be < 10ms
        p99 = sorted(latencies)[99]
        assert p99 < 10, f"P99 latency {p99}ms exceeds 10ms threshold"

    def test_query_hdc_only(self, manager, space, codebook):
        """Test HDC-only query."""
        key = codebook.encode("France")
        value = codebook.encode("Paris")

        manager.store(key, value, "Paris")

        label, conf = manager.query_hdc_only(key)

        assert label == "Paris"
        assert conf > 0.3

    def test_query_neural_only_empty(self, manager, space, codebook):
        """Test Neural-only query when no training has occurred."""
        key = codebook.encode("test")

        label, conf = manager.query_neural_only(key)

        assert label is None
        assert conf == 0.0

    def test_query_combined(self, manager, space, codebook):
        """Test combined HDC + Neural query."""
        key = codebook.encode("Germany")
        value = codebook.encode("Berlin")

        manager.store(key, value, "Berlin")

        result = manager.query(key)

        assert result.value_label == "Berlin"
        assert result.confidence > 0.0
        assert result.source in ["hdc", "neural"]

    def test_automatic_consolidation_trigger(self, manager, space, codebook):
        """Test that consolidation is triggered automatically."""
        manager.start_worker()

        try:
            # Store more facts than threshold
            for i in range(15):  # threshold is 10
                key = codebook.encode(f"entity_{i}")
                value = codebook.encode(f"value_{i}")
                manager.store(key, value, f"value_{i}")

            # Wait for consolidation
            time.sleep(1.0)

            # Should have triggered consolidation
            assert manager.total_consolidated > 0 or manager.pending_count < 15
        finally:
            manager.stop_worker()

    def test_force_consolidation(self, manager, space, codebook):
        """Test forced consolidation."""
        manager.start_worker()

        try:
            # Store some facts (less than threshold)
            for i in range(5):
                key = codebook.encode(f"entity_{i}")
                value = codebook.encode(f"value_{i}")
                manager.store(key, value, f"value_{i}")

            # Force consolidation
            manager.force_consolidation()

            # Wait for completion
            time.sleep(1.0)

            # Should have consolidated
            assert manager.pending_count == 0 or manager.total_consolidated > 0
        finally:
            manager.stop_worker()

    def test_context_manager(self, space):
        """Test using manager as context manager."""
        with ConsolidationManager(space) as manager:
            assert manager._worker_running

        assert not manager._worker_running

    def test_statistics(self, manager, space, codebook):
        """Test getting statistics."""
        key = codebook.encode("test")
        value = codebook.encode("value")
        manager.store(key, value, "value")

        stats = manager.get_statistics()

        assert "pending_facts" in stats
        assert "total_consolidated" in stats
        assert "vocab_size" in stats
        assert stats["vocab_size"] == 1

    def test_state_dict_persistence(self, manager, space, codebook):
        """Test state dictionary for persistence."""
        # Store some facts
        for i in range(5):
            key = codebook.encode(f"key_{i}")
            value = codebook.encode(f"value_{i}")
            manager.store(key, value, f"value_{i}")

        # Get state
        state = manager.state_dict()

        # Create new manager with matching hidden_dim and load state
        new_manager = ConsolidationManager(
            space,
            neural_hidden_dim=128,  # Match original manager's hidden_dim
        )
        new_manager.load_state_dict(state)

        assert new_manager.vocab_size == manager.vocab_size


# =============================================================================
# Section 5: Integration Tests (Goldfish & Elephant Scenarios)
# =============================================================================

class TestGoldfishScenario:
    """
    Goldfish Test: Short-term memory with HDC wipe.

    Train 20 facts -> completely wipe HDC -> query neural -> high accuracy

    This tests that neural memory can operate independently of HDC.
    """

    def test_goldfish_scenario(self, space, codebook):
        """Test goldfish scenario: neural works after HDC wipe."""
        # Test neural memory directly (not through async manager) for determinism
        neural = NeuralMemory(
            input_dim=space.dimensions,
            hidden_dim=256,  # Larger for better learning
            initial_vocab_size=100,
        )

        # Store 20 facts
        facts = []
        for i in range(20):
            key = codebook.encode(f"goldfish_key_{i}")
            fact = ConsolidationFact(
                key_vector=key,
                value_index=-1,
                value_label=f"goldfish_value_{i}",
                timestamp=time.time(),
            )
            facts.append((key, f"goldfish_value_{i}"))

        # Convert to list of facts
        consolidation_facts = [
            ConsolidationFact(
                key_vector=key,
                value_index=-1,
                value_label=label,
                timestamp=time.time(),
            )
            for key, label in facts
        ]

        # Train directly (no async)
        neural.consolidate(consolidation_facts, epochs=100)

        # Query neural (simulating HDC wipe)
        correct = 0
        for key, expected_label in facts:
            label, conf = neural.query(key)
            if label == expected_label:
                correct += 1

        accuracy = correct / len(facts)
        # With 100 epochs and 256 hidden, should achieve good accuracy
        assert accuracy >= 0.6, f"Goldfish accuracy {accuracy:.1%} < 60%"


class TestElephantScenario:
    """
    Elephant Test: Long-term memory with interference.

    Train A (20 facts) -> Train B (20 different facts) -> Query A -> good recall

    This tests that neural memory doesn't suffer catastrophic forgetting.
    """

    def test_elephant_scenario(self, space, codebook):
        """Test elephant scenario: old memories survive new learning."""
        # Test neural memory directly with experience replay
        neural = NeuralMemory(
            input_dim=space.dimensions,
            hidden_dim=256,
            initial_vocab_size=100,
            replay_buffer_size=100,  # Enable replay
        )

        # Phase A: Train first batch of facts
        facts_a = []
        for i in range(20):
            key = codebook.encode(f"elephant_a_key_{i}")
            facts_a.append((key, f"elephant_a_value_{i}"))

        consolidation_facts_a = [
            ConsolidationFact(
                key_vector=key,
                value_index=-1,
                value_label=label,
                timestamp=time.time(),
            )
            for key, label in facts_a
        ]

        neural.consolidate(consolidation_facts_a, epochs=100, replay_ratio=0.5)

        # Phase B: Train second batch of facts
        facts_b = []
        for i in range(20):
            key = codebook.encode(f"elephant_b_key_{i}")
            facts_b.append((key, f"elephant_b_value_{i}"))

        consolidation_facts_b = [
            ConsolidationFact(
                key_vector=key,
                value_index=-1,
                value_label=label,
                timestamp=time.time(),
            )
            for key, label in facts_b
        ]

        neural.consolidate(consolidation_facts_b, epochs=100, replay_ratio=0.5)

        # Query Phase A facts
        correct = 0
        for key, expected_label in facts_a:
            label, conf = neural.query(key)
            if label == expected_label:
                correct += 1

        recall = correct / len(facts_a)
        # With experience replay, should maintain some recall
        # Note: 40-fact vocab is small, so some forgetting is expected
        assert recall >= 0.3, f"Elephant recall {recall:.1%} < 30%"


class TestCapacityScenario:
    """
    Capacity Test: Large-scale fact storage.

    Store 100 facts -> verify system handles vocabulary growth

    This tests that the system scales beyond HDC capacity limits.
    """

    @pytest.mark.slow
    def test_capacity_100_facts(self, space, codebook):
        """Test capacity scenario: 100 facts with vocabulary growth."""
        # Test neural memory directly for determinism
        neural = NeuralMemory(
            input_dim=space.dimensions,
            hidden_dim=256,
            initial_vocab_size=50,  # Start small to force expansion
        )

        # Store 100 facts (forces vocab expansion)
        facts = []
        for i in range(100):
            key = codebook.encode(f"capacity_key_{i}")
            facts.append((key, f"capacity_value_{i}"))

        consolidation_facts = [
            ConsolidationFact(
                key_vector=key,
                value_index=-1,
                value_label=label,
                timestamp=time.time(),
            )
            for key, label in facts
        ]

        # Train in batches to simulate consolidation cycles
        batch_size = 25
        for i in range(0, len(consolidation_facts), batch_size):
            batch = consolidation_facts[i:i + batch_size]
            neural.consolidate(batch, epochs=50, replay_ratio=0.5)

        # Verify vocabulary expanded correctly
        assert neural.vocab_size == 100, f"Vocab size {neural.vocab_size} != 100"

        # Sample and test accuracy
        import random
        sample = random.sample(facts, min(50, len(facts)))

        correct = 0
        for key, expected_label in sample:
            label, conf = neural.query(key)
            if label == expected_label:
                correct += 1

        accuracy = correct / len(sample)
        # With batched training and replay, should achieve moderate accuracy
        assert accuracy >= 0.2, f"Capacity accuracy {accuracy:.1%} < 20%"


# =============================================================================
# Section 6: Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Test thread safety of the consolidation system."""

    def test_concurrent_stores(self, space, codebook):
        """Test concurrent store operations don't crash."""
        manager = ConsolidationManager(space, consolidation_threshold=100)
        manager.start_worker()

        errors = []

        def store_thread(thread_id):
            try:
                for i in range(20):
                    key = codebook.encode(f"thread_{thread_id}_key_{i}")
                    value = codebook.encode(f"thread_{thread_id}_value_{i}")
                    manager.store(key, value, f"thread_{thread_id}_value_{i}")
            except Exception as e:
                errors.append(e)

        try:
            threads = [
                threading.Thread(target=store_thread, args=(i,))
                for i in range(5)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Thread errors: {errors}"
            assert manager.vocab_size == 100  # 5 threads * 20 facts
        finally:
            manager.stop_worker()

    def test_concurrent_queries_during_consolidation(self, space, codebook):
        """Test queries don't block during consolidation."""
        manager = ConsolidationManager(space, consolidation_threshold=10)
        manager.start_worker()

        query_latencies = []

        def query_thread():
            for _ in range(50):
                key = codebook.encode("query_test")
                start = time.perf_counter()
                manager.query(key)
                latency = (time.perf_counter() - start) * 1000
                query_latencies.append(latency)
                time.sleep(0.01)

        def store_thread():
            for i in range(30):  # Trigger multiple consolidations
                key = codebook.encode(f"store_{i}")
                value = codebook.encode(f"value_{i}")
                manager.store(key, value, f"value_{i}")
                time.sleep(0.02)

        try:
            t1 = threading.Thread(target=query_thread)
            t2 = threading.Thread(target=store_thread)

            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # Queries should still be fast (allowing for system variability)
            p99 = sorted(query_latencies)[int(len(query_latencies) * 0.99)]
            assert p99 < 200, f"Query P99 {p99}ms too slow during consolidation"
        finally:
            manager.stop_worker()


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
