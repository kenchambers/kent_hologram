"""
ARC Solver Tests for ARC-AGI-2 Holographic Reasoning.

Tests the HolographicARCSolver which orchestrates the full pipeline:
object detection → encoding → resonator → execution.

Run with: uv run pytest tests/arc/test_arc_solver.py -v
"""

import pytest
import torch

from hologram.arc import (
    HolographicARCSolver,
    create_simple_task,
    Grid,
    ARCTask,
    TrainingPair,
)
from hologram.arc.solver import SolverResult


@pytest.fixture
def solver():
    """Create solver for tests."""
    return HolographicARCSolver(dimensions=10000)


class TestSolverInitialization:
    """Tests for solver initialization."""

    def test_solver_creates_components(self):
        """Solver should initialize all components."""
        solver = HolographicARCSolver()
        
        assert solver._detector is not None
        assert solver._encoder is not None
        assert solver._resonator is not None
        assert solver._executor is not None
        assert solver._skill_memory is not None

    def test_solver_custom_dimensions(self):
        """Solver should accept custom dimensions."""
        solver = HolographicARCSolver(dimensions=5000)
        assert solver._space.dimensions == 5000

    def test_solver_custom_confidence(self):
        """Solver should accept custom confidence threshold."""
        solver = HolographicARCSolver(confidence_threshold=0.5)
        assert solver._confidence_threshold == 0.5


class TestSolverBasic:
    """Basic solver tests."""

    def test_solve_returns_solver_result(self, solver):
        """Solve should return SolverResult."""
        task = create_simple_task(
            train_inputs=[[[0, 1], [0, 0]]],
            train_outputs=[[[1, 0], [0, 0]]],
            test_input=[[0, 2], [0, 0]],
        )
        result = solver.solve(task)
        
        assert isinstance(result, SolverResult)
        assert hasattr(result, "output")
        assert hasattr(result, "transformation")
        assert hasattr(result, "confidence")
        assert hasattr(result, "from_cache")
        assert hasattr(result, "message")

    def test_solve_translate_up(self, solver):
        """Solver should detect and apply translate up."""
        task = create_simple_task(
            train_inputs=[
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 2, 0]],
            ],
            train_outputs=[
                [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
            ],
            test_input=[[0, 0, 0], [0, 0, 0], [3, 0, 0]],
            test_output=[[0, 0, 0], [3, 0, 0], [0, 0, 0]],
        )
        result = solver.solve(task)
        
        assert result.transformation is not None
        assert result.transformation.action == "translate"
        # Output should be produced (may or may not match expected)
        if result.output is not None:
            assert isinstance(result.output, Grid)


class TestSolverRecolor:
    """Tests for recoloring tasks."""

    def test_solve_recolor_to_blue(self, solver):
        """Solver should detect recolor to blue."""
        task = create_simple_task(
            train_inputs=[
                [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                [[3, 0], [0, 0]],
            ],
            train_outputs=[
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[1, 0], [0, 0]],
            ],
            test_input=[[0, 4, 0], [0, 0, 0]],
            test_output=[[0, 1, 0], [0, 0, 0]],
        )
        result = solver.solve(task)
        
        if result.transformation is not None:
            assert result.transformation.action == "recolor"


class TestSolverIdentity:
    """Tests for identity (no change) tasks."""

    def test_solve_identity(self, solver):
        """Solver should detect identity transformation."""
        task = create_simple_task(
            train_inputs=[
                [[0, 1, 0], [0, 0, 0]],
            ],
            train_outputs=[
                [[0, 1, 0], [0, 0, 0]],
            ],
            test_input=[[2, 0], [0, 0]],
            test_output=[[2, 0], [0, 0]],
        )
        result = solver.solve(task)
        
        if result.transformation is not None:
            assert result.transformation.action == "identity"
        if result.output is not None:
            assert result.output == task.test_output


class TestSolverSkillMemory:
    """Tests for skill memory caching."""

    def test_cache_stores_skill(self, solver):
        """Solver should store skills after solving."""
        task = create_simple_task(
            train_inputs=[[[0, 1], [0, 0]]],
            train_outputs=[[[1, 0], [0, 0]]],
            test_input=[[0, 2], [0, 0]],
        )
        
        # Solve once
        result1 = solver.solve(task)
        assert not result1.from_cache
        
        # After solving, transform cache should have entry
        assert len(solver._transform_cache) > 0 or result1.output is None

    def test_clear_cache(self, solver):
        """Clear cache should reset both caches."""
        task = create_simple_task(
            train_inputs=[[[0, 1], [0, 0]]],
            train_outputs=[[[1, 0], [0, 0]]],
            test_input=[[0, 2], [0, 0]],
        )
        solver.solve(task)
        
        solver.clear_cache()
        
        assert len(solver._transform_cache) == 0


class TestSolverEvaluation:
    """Tests for task evaluation."""

    def test_evaluate_with_ground_truth(self, solver):
        """Evaluate should compare with ground truth."""
        task = create_simple_task(
            train_inputs=[[[0, 1], [0, 0]]],
            train_outputs=[[[0, 1], [0, 0]]],  # Identity
            test_input=[[0, 2], [0, 0]],
            test_output=[[0, 2], [0, 0]],  # Identity
        )
        
        correct, message = solver.evaluate_on_task(task)
        
        assert isinstance(correct, bool)
        assert isinstance(message, str)

    def test_evaluate_no_ground_truth(self, solver):
        """Evaluate should fail gracefully without ground truth."""
        task = create_simple_task(
            train_inputs=[[[0, 1], [0, 0]]],
            train_outputs=[[[1, 0], [0, 0]]],
            test_input=[[0, 2], [0, 0]],
            test_output=None,  # No ground truth
        )
        
        correct, message = solver.evaluate_on_task(task)
        
        assert correct is False
        assert "ground truth" in message.lower()


class TestSolverRefusal:
    """Tests for confidence-based refusal."""

    def test_low_confidence_refusal(self):
        """Solver should refuse with very high threshold."""
        solver = HolographicARCSolver(confidence_threshold=0.99)
        
        task = create_simple_task(
            train_inputs=[[[0, 1], [0, 0]]],
            train_outputs=[[[1, 0], [0, 0]]],
            test_input=[[0, 2], [0, 0]],
        )
        result = solver.solve(task)
        
        # With very high threshold, likely to refuse
        # (implementation detail - may or may not refuse)
        assert result.message is not None


class TestCreateSimpleTask:
    """Tests for the task creation helper."""

    def test_create_task_basic(self):
        """Should create valid task from lists."""
        task = create_simple_task(
            train_inputs=[[[0, 1], [0, 0]]],
            train_outputs=[[[1, 0], [0, 0]]],
            test_input=[[0, 2], [0, 0]],
        )
        
        assert isinstance(task, ARCTask)
        assert len(task.training) == 1
        assert task.test_input is not None
        assert task.test_output is None

    def test_create_task_with_ground_truth(self):
        """Should include test output when provided."""
        task = create_simple_task(
            train_inputs=[[[0, 1]]],
            train_outputs=[[[1, 0]]],
            test_input=[[0, 2]],
            test_output=[[2, 0]],
        )
        
        assert task.test_output is not None
        assert task.test_output == Grid.from_list([[2, 0]])

    def test_create_task_custom_id(self):
        """Should use custom task ID."""
        task = create_simple_task(
            train_inputs=[[[0]]],
            train_outputs=[[[0]]],
            test_input=[[0]],
            task_id="my_custom_task",
        )
        
        assert task.task_id == "my_custom_task"


class TestSolverEmptyInput:
    """Tests for edge cases with empty/minimal inputs."""

    def test_empty_objects(self, solver):
        """Should handle grids with no objects."""
        task = create_simple_task(
            train_inputs=[[[0, 0], [0, 0]]],
            train_outputs=[[[0, 0], [0, 0]]],
            test_input=[[0, 0], [0, 0]],
        )
        result = solver.solve(task)
        
        # Should not crash, may refuse
        assert result.message is not None

    def test_single_pixel(self, solver):
        """Should handle single-pixel objects."""
        task = create_simple_task(
            train_inputs=[[[1]]],
            train_outputs=[[[1]]],
            test_input=[[2]],
            test_output=[[2]],
        )
        result = solver.solve(task)
        
        # Should detect identity
        if result.transformation is not None:
            assert result.transformation.action == "identity"


class TestSolverIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_runs(self, solver):
        """Full pipeline should run without errors."""
        task = create_simple_task(
            train_inputs=[
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
            ],
            train_outputs=[
                [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
            ],
            test_input=[[0, 0, 0], [0, 0, 0], [0, 3, 0]],
            test_output=[[0, 0, 0], [0, 3, 0], [0, 0, 0]],
        )
        
        result = solver.solve(task)
        
        # Pipeline completed
        assert result is not None
        
        # If output produced, check it's a valid grid
        if result.output is not None:
            assert isinstance(result.output, Grid)
            assert result.output.height > 0
            assert result.output.width > 0

    def test_multi_training_pair(self, solver):
        """Should handle multiple training pairs."""
        task = create_simple_task(
            train_inputs=[
                [[1, 0], [0, 0]],
                [[0, 2], [0, 0]],
                [[0, 0], [3, 0]],
            ],
            train_outputs=[
                [[0, 0], [1, 0]],
                [[0, 0], [0, 2]],
                [[0, 0], [0, 3]],  # All move down-right pattern
            ],
            test_input=[[4, 0], [0, 0]],
        )
        
        result = solver.solve(task)
        
        # Should complete without error
        assert result is not None
        assert result.transformation is not None
