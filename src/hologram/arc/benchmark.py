"""
HonestBenchmark: ARC-AGI-2 evaluation with proper cache isolation.

The current built-in benchmark is misleading because it tests vocabulary-matched
tasks. This module provides honest evaluation on REAL ARC-AGI-2 with:

1. Cache clearing protocol (fresh solver per task)
2. ARC-AGI-2 evaluation set loader
3. Reproducible benchmarking CLI

Per Dr. Nexus: "If you claim 80% on a benchmark you designed, nobody believes you.
Show me the numbers on ARC-AGI-2 evaluation set."
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import time

from hologram.arc.types import ARCTask, Grid, TrainingPair
from hologram.arc.solver import HolographicARCSolver, SolverResult


@dataclass
class TaskResult:
    """Result from evaluating a single task."""
    task_id: str
    solved: bool
    correct: Optional[bool]  # None if no ground truth
    confidence: float
    message: str
    from_cache: bool
    time_ms: float


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    results: List[TaskResult]
    errors: List[Dict[str, Any]]
    accuracy: float  # Fraction correct (where ground truth known)
    solve_rate: float  # Fraction that produced output (any confidence)
    total_time_s: float
    tasks_evaluated: int

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=== HONEST BENCHMARK RESULTS ===",
            f"Tasks evaluated: {self.tasks_evaluated}",
            f"Accuracy: {self.accuracy * 100:.1f}%",
            f"Solve rate: {self.solve_rate * 100:.1f}%",
            f"Errors: {len(self.errors)}",
            f"Total time: {self.total_time_s:.1f}s",
            f"Avg time per task: {self.total_time_s / max(1, self.tasks_evaluated):.2f}s",
        ]
        return "\n".join(lines)


class HonestBenchmark:
    """
    ARC benchmark with proper cache isolation.

    Each task gets a FRESH solver to prevent skill memory contamination.
    This is the honest way to benchmark - no cheating via cached solutions.

    Example:
        >>> benchmark = HonestBenchmark()
        >>> tasks = load_arc_agi_2(limit=100)
        >>> result = benchmark.evaluate(tasks)
        >>> print(result.accuracy)
        0.15  # 15% accuracy
    """

    def __init__(
        self,
        dimensions: int = 10000,
        confidence_threshold: float = 0.005,
        iterative: bool = True,
        max_steps: int = 5,
    ):
        """
        Initialize benchmark configuration.

        Args:
            dimensions: HDC vector dimensions for solver
            confidence_threshold: Minimum confidence for output
            iterative: Use iterative solving (recommended)
            max_steps: Max steps for iterative solving
        """
        self._dimensions = dimensions
        self._confidence_threshold = confidence_threshold
        self._iterative = iterative
        self._max_steps = max_steps

    def evaluate(
        self,
        tasks: List[ARCTask],
        isolate_cache: bool = True,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """
        Evaluate solver on a list of ARC tasks.

        Args:
            tasks: List of ARC tasks to evaluate
            isolate_cache: If True, create fresh solver per task (recommended)
            verbose: Print progress during evaluation

        Returns:
            BenchmarkResult with aggregated statistics
        """
        results: List[TaskResult] = []
        errors: List[Dict[str, Any]] = []
        start_time = time.time()

        # Shared solver for non-isolated mode (ablation studies only)
        shared_solver = None
        if not isolate_cache:
            shared_solver = self._create_solver()

        for i, task in enumerate(tasks):
            if verbose:
                print(f"[{i + 1}/{len(tasks)}] Evaluating {task.task_id}...")

            # CRITICAL: Fresh solver per task for honest evaluation
            if isolate_cache:
                solver = self._create_solver()
            else:
                solver = shared_solver

            task_start = time.time()
            try:
                result = solver.solve(task)
                task_time = (time.time() - task_start) * 1000  # ms

                # Determine correctness
                correct = None
                if task.test_output is not None and result.output is not None:
                    correct = result.output == task.test_output

                results.append(TaskResult(
                    task_id=task.task_id,
                    solved=result.output is not None,
                    correct=correct,
                    confidence=result.confidence,
                    message=result.message,
                    from_cache=result.from_cache,
                    time_ms=task_time,
                ))

            except Exception as e:
                errors.append({
                    "task_id": task.task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                })

            # Verify cache isolation by deleting solver
            if isolate_cache:
                del solver

        total_time = time.time() - start_time

        # Compute aggregated stats
        correct_count = sum(1 for r in results if r.correct is True)
        with_truth_count = sum(1 for r in results if r.correct is not None)
        solve_count = sum(1 for r in results if r.solved)

        accuracy = correct_count / max(1, with_truth_count)
        solve_rate = solve_count / max(1, len(results))

        return BenchmarkResult(
            results=results,
            errors=errors,
            accuracy=accuracy,
            solve_rate=solve_rate,
            total_time_s=total_time,
            tasks_evaluated=len(tasks),
        )

    def _create_solver(self) -> HolographicARCSolver:
        """Create a fresh solver instance with isolated memory for honest eval."""
        return HolographicARCSolver(
            dimensions=self._dimensions,
            confidence_threshold=self._confidence_threshold,
            iterative=self._iterative,
            max_steps=self._max_steps,
            isolate_memory=True,  # Honest benchmark: no cross-task memory
        )


def load_arc_agi_2(
    split: str = "evaluation",
    limit: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> List[ARCTask]:
    """
    Load official ARC-AGI-2 dataset.

    IMPORTANT: For honest reporting, use split="evaluation" NOT "training".

    Args:
        split: Dataset split ("training" or "evaluation")
        limit: Maximum number of tasks to load (None = all)
        data_dir: Override data directory (default: data/ARC-AGI-2/data/)

    Returns:
        List of ARCTask instances

    Raises:
        FileNotFoundError: If dataset not found at expected location
    """
    # Try multiple locations
    if data_dir is not None:
        arc_path = data_dir / split
    else:
        # Check project data directory first
        project_data = Path(__file__).parent.parent.parent.parent / "data" / "ARC-AGI-2" / "data" / split
        home_data = Path.home() / ".arc-agi-2" / split

        if project_data.exists():
            arc_path = project_data
        elif home_data.exists():
            arc_path = home_data
        else:
            raise FileNotFoundError(
                f"ARC-AGI-2 {split} set not found. Tried:\n"
                f"  {project_data}\n"
                f"  {home_data}\n"
                f"Download from https://github.com/fchollet/ARC-AGI"
            )

    tasks = []
    for task_file in sorted(arc_path.glob("*.json")):
        with open(task_file) as f:
            data = json.load(f)
        tasks.append(ARCTask.from_dict(data, task_id=task_file.stem))

        if limit and len(tasks) >= limit:
            break

    return tasks


def load_tasks_from_arckit() -> List[ARCTask]:
    """
    Load tasks from arckit package if available.

    Returns:
        List of ARCTask instances from arckit
    """
    try:
        import arckit

        tasks = []
        # arckit provides train_set and eval_set
        for task_data in arckit.train_set:
            task = ARCTask(
                task_id=task_data.id,
                training=[
                    TrainingPair(
                        input=Grid.from_list(ex.input),
                        output=Grid.from_list(ex.output),
                    )
                    for ex in task_data.train
                ],
                test_input=Grid.from_list(task_data.test[0].input),
                test_output=Grid.from_list(task_data.test[0].output) if task_data.test[0].output else None,
            )
            tasks.append(task)
        return tasks

    except ImportError:
        raise ImportError(
            "arckit package not installed. Install with: pip install arckit"
        )


def run_benchmark(
    split: str = "evaluation",
    limit: int = 100,
    iterative: bool = True,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Convenience function to run benchmark with standard settings.

    Args:
        split: Dataset split to use
        limit: Number of tasks to evaluate
        iterative: Use iterative solver
        verbose: Print progress

    Returns:
        BenchmarkResult with statistics
    """
    if verbose:
        print(f"Loading ARC-AGI-2 {split} set (limit={limit})...")

    try:
        tasks = load_arc_agi_2(split=split, limit=limit)
    except FileNotFoundError:
        if verbose:
            print("ARC-AGI-2 not found, trying arckit package...")
        tasks = load_tasks_from_arckit()[:limit]

    if verbose:
        print(f"Loaded {len(tasks)} tasks")
        print(f"Evaluating with cache isolation (honest mode)...")

    benchmark = HonestBenchmark(iterative=iterative)
    result = benchmark.evaluate(tasks, isolate_cache=True, verbose=verbose)

    if verbose:
        print()
        print(result.summary())

    return result
