"""
Integration tests for self-improvement system.

Tests the integration of CircuitObserver with ConstraintAccumulator
and MetacognitiveLoop.
"""

import pytest
from unittest.mock import Mock, MagicMock

from hologram.introspection import CircuitObserver, SelfImprovementManager


class MockTransformResult:
    """Mock TransformResult for testing."""
    def __init__(self, action: str, target: str, modifier: str, min_confidence: float = 0.8):
        self.action = action
        self.target = target
        self.modifier = modifier
        self.min_confidence = min_confidence


class TestConstraintAccumulatorIntegration:
    """Tests for ConstraintAccumulator + CircuitObserver integration."""

    def test_observer_receives_observations(self):
        """Test that ConstraintAccumulator reports to CircuitObserver."""
        from hologram.arc.constraint_accumulator import ConstraintAccumulator

        accumulator = ConstraintAccumulator()
        observer = CircuitObserver(min_observations=0.5)

        # Attach observer
        accumulator.set_circuit_observer(observer)

        # Create mock transform result
        result = MockTransformResult(
            action="rotate",
            target="grid",
            modifier="90deg",
            min_confidence=0.8
        )

        # Record attempt
        accumulator.record_attempt(result, partial_score=0.6)

        # Verify observation was recorded
        assert len(observer) == 3  # action, target, modifier
        assert observer.get_observation_count("rotate") > 0
        assert observer.get_observation_count("grid") > 0
        assert observer.get_observation_count("90deg") > 0

    def test_success_threshold(self):
        """Test that success is based on partial_score > 0.5."""
        from hologram.arc.constraint_accumulator import ConstraintAccumulator

        accumulator = ConstraintAccumulator()
        observer = CircuitObserver(min_observations=0.5)
        accumulator.set_circuit_observer(observer)

        # High partial score = success
        result1 = MockTransformResult("action1", "target1", "mod1")
        accumulator.record_attempt(result1, partial_score=0.7)

        # Low partial score = failure
        result2 = MockTransformResult("action2", "target2", "mod2")
        accumulator.record_attempt(result2, partial_score=0.3)

        # Check success rates
        assert observer.get_success_rate("action1") == 1.0  # Success
        assert observer.get_success_rate("action2") == 0.0  # Failure

    def test_get_transformation_prior(self):
        """Test ConstraintAccumulator queries priors correctly."""
        from hologram.arc.constraint_accumulator import ConstraintAccumulator

        accumulator = ConstraintAccumulator()
        observer = CircuitObserver(min_observations=1)
        accumulator.set_circuit_observer(observer)

        # Add successful observations for a transformation
        for _ in range(5):
            result = MockTransformResult("rotate", "largest", "90deg")
            accumulator.record_attempt(result, partial_score=0.9)

        # Get prior for the same transformation
        query_result = MockTransformResult("rotate", "largest", "90deg")
        prior = accumulator.get_transformation_prior(query_result)

        assert prior > 0.7  # Should have high prior due to success history

    def test_no_observer_graceful(self):
        """Test ConstraintAccumulator works without observer."""
        from hologram.arc.constraint_accumulator import ConstraintAccumulator

        accumulator = ConstraintAccumulator()
        # Don't attach observer

        result = MockTransformResult("rotate", "grid", "90deg")

        # Should not crash
        accumulator.record_attempt(result, partial_score=0.6)

        # Get prior should return neutral
        prior = accumulator.get_transformation_prior(result)
        assert prior == 0.5


class TestMetacognitiveLoopIntegration:
    """Tests for MetacognitiveLoop + CircuitObserver integration."""

    def test_observer_receives_query_observations(self):
        """Test MetacognitiveLoop reports to CircuitObserver."""
        from hologram.cognition.metacognition import MetacognitiveLoop
        from hologram.core.codebook import Codebook
        from hologram.core.vector_space import VectorSpace

        space = VectorSpace(dimensions=512)
        codebook = Codebook(space)
        loop = MetacognitiveLoop(codebook)
        observer = CircuitObserver(min_observations=0.5)

        # Attach observer
        loop.set_circuit_observer(observer)

        # Create mock query function
        def mock_query(text, **kwargs):
            return ("answer", 0.9)

        # Execute query
        result, confidence = loop.execute_query(
            query_func=mock_query,
            query_text="What is the capital of France?"
        )

        # Verify observations were recorded
        assert len(observer) > 0
        # Check that some query words were tracked
        stats = observer.get_stats_summary()
        assert stats["total_observations"] >= 1

    def test_extract_query_items(self):
        """Test query text item extraction."""
        from hologram.cognition.metacognition import MetacognitiveLoop
        from hologram.core.codebook import Codebook
        from hologram.core.vector_space import VectorSpace

        space = VectorSpace(dimensions=512)
        codebook = Codebook(space)
        loop = MetacognitiveLoop(codebook)

        items = loop._extract_query_items("What is the capital of France?")

        # Should extract words longer than 2 chars, lowercase
        assert "what" in items
        assert "capital" in items
        assert "france" in items
        # Short words filtered out
        assert "is" not in items
        assert "of" not in items

    def test_success_based_on_retry_threshold(self):
        """Test that success is based on retry_threshold."""
        from hologram.cognition.metacognition import MetacognitiveLoop
        from hologram.core.codebook import Codebook
        from hologram.core.vector_space import VectorSpace

        space = VectorSpace(dimensions=512)
        codebook = Codebook(space)
        loop = MetacognitiveLoop(codebook, retry_threshold=0.5)
        observer = CircuitObserver(min_observations=0.5)
        loop.set_circuit_observer(observer)

        # High confidence = success
        def high_conf_query(text, **kwargs):
            return ("answer", 0.8)

        loop.execute_query(query_func=high_conf_query, query_text="test query one")

        # Check that observations were marked as successful
        # (confidence 0.8 >= retry_threshold 0.5)
        for item in observer._activation_counts:
            if observer.get_observation_count(item) > 0:
                rate = observer.get_success_rate(item)
                assert rate == 1.0  # Should be 100% success

    def test_no_observer_graceful(self):
        """Test MetacognitiveLoop works without observer."""
        from hologram.cognition.metacognition import MetacognitiveLoop
        from hologram.core.codebook import Codebook
        from hologram.core.vector_space import VectorSpace

        space = VectorSpace(dimensions=512)
        codebook = Codebook(space)
        loop = MetacognitiveLoop(codebook)
        # Don't attach observer

        def mock_query(text, **kwargs):
            return ("answer", 0.9)

        # Should not crash
        result, confidence = loop.execute_query(
            query_func=mock_query,
            query_text="test query"
        )

        assert result == "answer"
        assert confidence == 0.9


class TestSelfImprovementManager:
    """Tests for SelfImprovementManager."""

    def test_init_without_persistence(self):
        """Test manager initialization without persistence."""
        manager = SelfImprovementManager()

        assert manager.observer is not None
        assert manager.analyzer is not None

    def test_observer_attachment(self):
        """Test that observer can be attached to components."""
        from hologram.arc.constraint_accumulator import ConstraintAccumulator

        manager = SelfImprovementManager()
        accumulator = ConstraintAccumulator()

        # Attach manager's observer
        accumulator.set_circuit_observer(manager.observer)

        # Record some attempts
        result = MockTransformResult("rotate", "grid", "90deg")
        accumulator.record_attempt(result, partial_score=0.8)

        # Check manager's observer received the data
        stats = manager.get_statistics()
        assert stats["unique_items"] > 0

    def test_get_statistics(self):
        """Test statistics retrieval."""
        manager = SelfImprovementManager()

        # Add some observations
        manager.observer.observe(
            items=["item_a"],
            success=True,
            confidence=0.9
        )

        stats = manager.get_statistics()

        assert "total_observations" in stats
        assert "unique_items" in stats
        assert "items_to_prune" in stats
        assert "items_to_reinforce" in stats

    def test_improvement_report(self):
        """Test improvement report generation."""
        manager = SelfImprovementManager()

        # Add observations for report content
        for _ in range(10):
            manager.observer.observe(
                items=["good_pattern"],
                success=True,
                confidence=0.9
            )
            manager.observer.observe(
                items=["bad_pattern"],
                success=False,
                confidence=0.9
            )

        report = manager.get_improvement_report()

        assert "Circuit Introspection Report" in report
        assert "Observations Summary" in report

    def test_reset(self):
        """Test manager reset."""
        manager = SelfImprovementManager()

        manager.observer.observe(items=["item"], success=True, confidence=0.9)
        assert len(manager.observer) > 0

        manager.reset()
        assert len(manager.observer) == 0

    def test_persistence_roundtrip(self, tmp_path):
        """Test save and load functionality."""
        persist_path = tmp_path / "learned_patterns.json"

        # Create manager and add observations
        manager1 = SelfImprovementManager(persist_path=str(persist_path))
        manager1.observer.observe(items=["persistent_item"], success=True, confidence=0.9)
        manager1._observation_count = 1  # Manually track since observe doesn't auto-increment
        manager1.save()

        # Create new manager and load
        manager2 = SelfImprovementManager(persist_path=str(persist_path))

        # Should have loaded the previous observations
        # Check that the item exists in activation counts
        assert "persistent_item" in manager2.observer._activation_counts
