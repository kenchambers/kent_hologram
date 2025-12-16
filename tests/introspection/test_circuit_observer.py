"""
Tests for CircuitObserver - the core observation layer.
"""

import pytest
import time

from hologram.introspection import CircuitObserver, ActivationRecord


class TestCircuitObserver:
    """Tests for CircuitObserver functionality."""

    def test_init(self):
        """Test CircuitObserver initialization."""
        observer = CircuitObserver()
        assert len(observer) == 0
        assert observer.get_stats_summary()["total_observations"] == 0

    def test_observe_single_item(self):
        """Test observing a single item."""
        observer = CircuitObserver(min_observations=0.5)  # Lower threshold

        observer.observe(
            items=["rotate"],
            success=True,
            confidence=0.9,
            context="test"
        )

        assert len(observer) == 1
        # With 0.9 confidence, weighted count is 0.9, which meets min_observations=0.5
        assert observer.get_success_rate("rotate") == 1.0  # 100% success

    def test_observe_multiple_items(self):
        """Test observing multiple items together."""
        observer = CircuitObserver(min_observations=0.5)

        observer.observe(
            items=["rotate", "grid", "90deg"],
            success=True,
            confidence=0.9,
            context="arc"
        )

        assert len(observer) == 3
        # All items should have 100% success rate
        assert observer.get_success_rate("rotate") == 1.0
        assert observer.get_success_rate("grid") == 1.0
        assert observer.get_success_rate("90deg") == 1.0

    def test_success_rate_calculation(self):
        """Test success rate calculation with mixed outcomes."""
        observer = CircuitObserver(min_observations=1)

        # 2 successes, 1 failure for "rotate"
        observer.observe(items=["rotate"], success=True, confidence=1.0)
        observer.observe(items=["rotate"], success=True, confidence=1.0)
        observer.observe(items=["rotate"], success=False, confidence=1.0)

        # Success rate should be 2/3 ≈ 0.67
        rate = observer.get_success_rate("rotate")
        assert 0.65 < rate < 0.68

    def test_unknown_item_returns_neutral(self):
        """Test that unknown items return neutral prior."""
        observer = CircuitObserver()

        rate = observer.get_success_rate("unknown_item")
        assert rate == 0.5  # Neutral prior

    def test_suggest_pruning(self):
        """Test pruning suggestions for failing items."""
        observer = CircuitObserver(min_observations=2)

        # Create an item that consistently fails
        for _ in range(5):
            observer.observe(items=["bad_item"], success=False, confidence=1.0)

        # Create an item that consistently succeeds
        for _ in range(5):
            observer.observe(items=["good_item"], success=True, confidence=1.0)

        pruning = observer.suggest_pruning(threshold=0.3)

        # bad_item should be suggested for pruning
        pruning_items = [item for item, rate in pruning]
        assert "bad_item" in pruning_items
        assert "good_item" not in pruning_items

    def test_suggest_reinforcement(self):
        """Test reinforcement suggestions for successful items."""
        observer = CircuitObserver(min_observations=2)

        # Create an item that consistently succeeds
        for _ in range(5):
            observer.observe(items=["star_performer"], success=True, confidence=1.0)

        # Create an item that consistently fails
        for _ in range(5):
            observer.observe(items=["poor_performer"], success=False, confidence=1.0)

        reinforcement = observer.suggest_reinforcement(threshold=0.8)

        # star_performer should be suggested for reinforcement
        reinf_items = [item for item, rate in reinforcement]
        assert "star_performer" in reinf_items
        assert "poor_performer" not in reinf_items

    def test_co_activations(self):
        """Test co-activation tracking."""
        observer = CircuitObserver(min_observations=1)

        # Items that succeed together
        for _ in range(5):
            observer.observe(
                items=["action_a", "target_b"],
                success=True,
                confidence=1.0
            )

        co_acts = observer.get_co_activations("action_a", min_count=1)

        # target_b should be co-activated with action_a
        co_act_items = [item for item, count in co_acts]
        assert "target_b" in co_act_items

    def test_get_prior(self):
        """Test combined prior calculation."""
        observer = CircuitObserver(min_observations=1)

        # Create items with known success rates
        for _ in range(10):
            observer.observe(items=["high_success"], success=True, confidence=1.0)
        for _ in range(10):
            observer.observe(items=["low_success"], success=False, confidence=1.0)

        # Prior for high_success should be high
        high_prior = observer.get_prior(["high_success"])
        assert high_prior > 0.7

        # Prior for low_success should be low or neutral (0.0 success rate)
        low_prior = observer.get_prior(["low_success"])
        assert low_prior <= 0.5  # Could be 0 or neutral due to geometric mean

    def test_context_specific_rates(self):
        """Test context-specific success rates."""
        observer = CircuitObserver(min_observations=1)

        # Same item, different contexts
        observer.observe(items=["item"], success=True, confidence=1.0, context="good_ctx")
        observer.observe(items=["item"], success=True, confidence=1.0, context="good_ctx")
        observer.observe(items=["item"], success=False, confidence=1.0, context="bad_ctx")
        observer.observe(items=["item"], success=False, confidence=1.0, context="bad_ctx")

        # Context-specific rates should differ
        good_rate = observer.get_success_rate("item", context="good_ctx")
        bad_rate = observer.get_success_rate("item", context="bad_ctx")

        assert good_rate > 0.9
        assert bad_rate < 0.1

    def test_decay_old_observations(self):
        """Test recency decay."""
        observer = CircuitObserver(min_observations=1, recency_weight=0.5)

        # Add some observations
        observer.observe(items=["item"], success=True, confidence=1.0)
        initial_count = observer._activation_counts["item"]

        # Apply decay
        observer.decay_old_observations()
        decayed_count = observer._activation_counts["item"]

        # Count should be halved (recency_weight=0.5)
        assert decayed_count == initial_count * 0.5

    def test_state_dict_persistence(self):
        """Test state serialization and deserialization."""
        observer1 = CircuitObserver(min_observations=0.5)

        # Add observations
        observer1.observe(items=["item_a"], success=True, confidence=0.9)
        observer1.observe(items=["item_b"], success=False, confidence=0.8)

        # Serialize
        state = observer1.state_dict()

        # Create new observer and load state
        observer2 = CircuitObserver(min_observations=0.5)
        observer2.load_state_dict(state)

        # Verify state was restored
        assert observer2.get_success_rate("item_a") == observer1.get_success_rate("item_a")
        assert observer2.get_success_rate("item_b") == observer1.get_success_rate("item_b")
        assert len(observer2) == len(observer1)

    def test_reset(self):
        """Test reset clears all observations."""
        observer = CircuitObserver(min_observations=1)

        observer.observe(items=["item"], success=True, confidence=1.0)
        assert len(observer) > 0

        observer.reset()
        assert len(observer) == 0

    def test_max_history_limit(self):
        """Test that history respects max_history limit."""
        max_hist = 10
        observer = CircuitObserver(max_history=max_hist)

        # Add more observations than max_history
        for i in range(20):
            observer.observe(items=[f"item_{i}"], success=True, confidence=1.0)

        # History should be capped
        assert len(observer._history) <= max_hist

    def test_thread_safety(self):
        """Test that concurrent observations don't crash."""
        import threading

        observer = CircuitObserver()
        errors = []

        def observe_loop():
            try:
                for i in range(100):
                    observer.observe(
                        items=[f"item_{i % 10}"],
                        success=i % 2 == 0,
                        confidence=0.5
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=observe_loop) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
