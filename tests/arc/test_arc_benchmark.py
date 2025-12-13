"""
ARC-AGI-2 Benchmark using existing HolographicARCSolver.

Run with: uv run pytest tests/arc/test_arc_benchmark.py -v --tb=no
"""

import json
from pathlib import Path
from typing import List

import pytest

from hologram.arc.solver import HolographicARCSolver
from hologram.arc.types import ARCTask


DATA_PATH = Path(__file__).parent.parent.parent / "data" / "ARC-AGI-2" / "data"


def load_tasks(subset: str, limit: int = None) -> List[ARCTask]:
    """Load ARC tasks from JSON files."""
    task_dir = DATA_PATH / subset
    if not task_dir.exists():
        pytest.skip(f"Dataset not found at {task_dir}")

    tasks = []
    for i, f in enumerate(sorted(task_dir.glob("*.json"))):
        if limit and i >= limit:
            break
        with open(f) as fp:
            tasks.append(ARCTask.from_dict(json.load(fp), task_id=f.stem))
    return tasks


# Shared solver instance for efficiency (training only)
_solver = None

def get_solver(subset: str = "training", fresh: bool = False):
    """
    Get solver instance.

    For evaluation subset, always returns fresh instance to avoid
    cross-task leakage from skill memory/caching.

    Args:
        subset: Dataset subset ("training" or "evaluation")
        fresh: Force fresh instance even for training

    Returns:
        HolographicARCSolver instance
    """
    global _solver
    
    # Always use fresh solver for evaluation (honest benchmarking)
    if subset == "evaluation" or fresh:
        return HolographicARCSolver(dimensions=10000, strategy="search_verify")
    
    # Reuse solver for training (efficiency)
    if _solver is None:
        _solver = HolographicARCSolver(dimensions=10000, strategy="search_verify")
    return _solver


class TestARCBenchmark:
    """Benchmark tests for ARC-AGI-2 dataset."""

    @pytest.mark.parametrize(
        "task",
        load_tasks("training", limit=50),
        ids=lambda t: t.task_id
    )
    def test_training_task(self, task: ARCTask):
        """Evaluate solver on a single training task."""
        solver = get_solver("training")
        correct, msg = solver.evaluate_on_task(task)

        # Record result (visible in pytest output with -v)
        status = "✓ CORRECT" if correct else "✗ WRONG"
        print(f"\n{task.task_id}: {status} - {msg}")

        # Don't fail - we're benchmarking, not testing correctness
        # The parametrize will show pass/fail count in summary

    @pytest.mark.parametrize(
        "task",
        load_tasks("evaluation", limit=50),
        ids=lambda t: t.task_id
    )
    def test_evaluation_task(self, task: ARCTask):
        """Evaluate solver on evaluation tasks (fresh solver per task)."""
        # Always use fresh solver for evaluation to avoid cross-task leakage
        solver = get_solver("evaluation", fresh=True)
        correct, msg = solver.evaluate_on_task(task)

        # Record result (visible in pytest output with -v)
        status = "✓ CORRECT" if correct else "✗ WRONG"
        print(f"\n{task.task_id}: {status} - {msg}")

        # Don't fail - we're benchmarking, not testing correctness


def run_full_benchmark(subset: str = "training", limit: int = 100):
    """
    Run benchmark and print summary stats.

    Usage: python -c "from tests.arc.test_arc_benchmark import run_full_benchmark; run_full_benchmark()"
    """
    tasks = load_tasks(subset, limit=limit)
    
    # Use fresh solver for evaluation, shared for training
    use_fresh = subset == "evaluation"

    correct = 0
    refused = 0
    incorrect = 0

    print(f"Running benchmark on {len(tasks)} {subset} tasks...")
    print(f"Strategy: search_verify (fresh solver per task: {use_fresh})")
    print("-" * 60)

    for i, task in enumerate(tasks):
        # Get fresh solver for evaluation, reuse for training
        solver = get_solver(subset, fresh=use_fresh)
        is_correct, msg = solver.evaluate_on_task(task)

        if is_correct:
            correct += 1
            status = "✓"
        elif "Refused" in msg or "No valid" in msg or "below threshold" in msg:
            refused += 1
            status = "○"
        else:
            incorrect += 1
            status = "✗"

        print(f"[{i+1:3d}/{len(tasks)}] {status} {task.task_id}: {msg[:50]}")

    print("-" * 60)
    print(f"Results ({subset}, n={len(tasks)}):")
    print(f"  Correct:   {correct:3d} ({correct/len(tasks)*100:.1f}%)")
    print(f"  Incorrect: {incorrect:3d} ({incorrect/len(tasks)*100:.1f}%)")
    print(f"  Refused:   {refused:3d} ({refused/len(tasks)*100:.1f}%)")
    print(f"  Accuracy (attempted): {correct/(correct+incorrect)*100:.1f}%" if correct+incorrect > 0 else "  N/A")


if __name__ == "__main__":
    run_full_benchmark(limit=50)
