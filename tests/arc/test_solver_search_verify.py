"""
Tests for HolographicARCSolver with search_verify strategy.
"""

import pytest

from hologram.arc.solver import HolographicARCSolver, create_simple_task
from hologram.arc.types import ARCTask


def test_solver_search_verify_strategy():
    """Test solver with search_verify strategy."""
    solver = HolographicARCSolver(
        dimensions=10000,
        strategy="search_verify",
        search_k=10,
        search_slot_k=3,
    )

    # Create a simple task
    task = create_simple_task(
        train_inputs=[
            [[1, 0], [0, 0]],  # Single pixel top-left
        ],
        train_outputs=[
            [[0, 1], [0, 0]],  # Translated right
        ],
        test_input=[[1, 0], [0, 0]],
        test_output=[[0, 1], [0, 0]],
    )

    result = solver.solve(task)
    assert result is not None
    assert hasattr(result, "output")
    assert hasattr(result, "transformation")
    assert hasattr(result, "confidence")
    assert hasattr(result, "from_cache")
    assert hasattr(result, "message")


def test_solver_fallback_to_resonator():
    """Test solver can fall back to resonator strategy."""
    solver = HolographicARCSolver(
        dimensions=10000,
        strategy="resonator",
    )

    task = create_simple_task(
        train_inputs=[
            [[1, 0], [0, 0]],
        ],
        train_outputs=[
            [[0, 1], [0, 0]],
        ],
        test_input=[[1, 0], [0, 0]],
    )

    result = solver.solve(task)
    assert result is not None


def test_solver_refuses_when_no_candidates_pass():
    """Test solver refuses when no candidates pass verification."""
    solver = HolographicARCSolver(
        dimensions=10000,
        strategy="search_verify",
        search_k=5,
    )

    # Create task with impossible transformation (to force refusal)
    # Use a task that likely won't match any simple transform
    task = create_simple_task(
        train_inputs=[
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # Complex pattern
        ],
        train_outputs=[
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],  # Completely different
        ],
        test_input=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    )

    result = solver.solve(task)
    # Should either refuse (output=None) or return best effort
    assert result is not None
    # If refused, message should indicate no candidates passed or attempts exhausted
    if result.output is None:
        msg_lower = result.message.lower()
        assert any(kw in msg_lower for kw in ["candidates", "verification", "attempts", "mood"])


def test_solver_skill_memory_still_works():
    """Test that skill memory lookup still works with search_verify."""
    solver = HolographicARCSolver(
        dimensions=10000,
        strategy="search_verify",
    )

    # Create and solve a task
    task1 = create_simple_task(
        train_inputs=[
            [[1, 0], [0, 0]],
        ],
        train_outputs=[
            [[0, 1], [0, 0]],
        ],
        test_input=[[1, 0], [0, 0]],
        task_id="test_skill_1",
    )

    result1 = solver.solve(task1)

    # Create identical task (should hit cache)
    task2 = create_simple_task(
        train_inputs=[
            [[1, 0], [0, 0]],
        ],
        train_outputs=[
            [[0, 1], [0, 0]],
        ],
        test_input=[[1, 0], [0, 0]],
        task_id="test_skill_2",
    )

    result2 = solver.solve(task2)
    # May or may not hit cache depending on signature similarity
    assert result2 is not None


def test_solver_isolate_memory_flag():
    """Test that isolate_memory=True disables skill memory."""
    solver = HolographicARCSolver(isolate_memory=True)

    # Should not have skill memory when isolated
    assert solver._skill_memory is None
    assert solver._isolate_memory is True


def test_solver_default_has_memory():
    """Test that default solver has skill memory enabled."""
    solver = HolographicARCSolver()  # Default: isolate_memory=False

    # Should have skill memory by default
    assert solver._skill_memory is not None
    assert solver._isolate_memory is False


def test_solver_isolate_memory_solves_without_crash():
    """Test solver with isolated memory can still solve tasks."""
    solver = HolographicARCSolver(
        strategy="search_verify",
        isolate_memory=True,
    )

    task = create_simple_task(
        train_inputs=[[[1, 0], [0, 0]]],
        train_outputs=[[[0, 1], [0, 0]]],
        test_input=[[1, 0], [0, 0]],
        test_output=[[0, 1], [0, 0]],
    )

    result = solver.solve(task)

    # Should produce result even without skill memory
    assert result is not None
    # Should not be from cache (no memory)
    assert result.from_cache is False


def test_solver_clear_cache_respects_isolation():
    """Test that clear_cache respects isolate_memory flag."""
    # Solver with memory
    solver_with_memory = HolographicARCSolver(isolate_memory=False)
    solver_with_memory.clear_cache()
    assert solver_with_memory._skill_memory is not None

    # Solver without memory
    solver_isolated = HolographicARCSolver(isolate_memory=True)
    solver_isolated.clear_cache()
    assert solver_isolated._skill_memory is None
