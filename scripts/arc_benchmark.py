#!/usr/bin/env python3
"""
ARC-AGI-2 Holographic Reasoning Benchmark.

Evaluates the HolographicARCSolver on ARC tasks and generates
a performance report.

Usage:
    # Run on built-in test tasks
    uv run python scripts/arc_benchmark.py
    
    # Run on ARC dataset (requires arckit)
    uv run python scripts/arc_benchmark.py --dataset training
    
    # Run with custom JSON tasks
    uv run python scripts/arc_benchmark.py --tasks path/to/tasks/

Requirements:
    - arckit (optional): pip install arckit
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, "src")

from hologram.arc import (
    HolographicARCSolver,
    create_simple_task,
    Grid,
    ARCTask,
)


@dataclass
class BenchmarkResult:
    """Result for a single task."""
    task_id: str
    correct: bool
    refused: bool
    confidence: float
    action: Optional[str]
    target: Optional[str]
    modifier: Optional[str]
    from_cache: bool
    time_ms: float
    message: str


@dataclass
class BenchmarkReport:
    """Summary report for benchmark run."""
    total_tasks: int
    solved_correct: int
    solved_incorrect: int
    refused: int
    accuracy: float
    accuracy_with_refusal: float  # Correct / (Correct + Incorrect)
    avg_confidence: float
    avg_time_ms: float
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("ARC-AGI-2 HOLOGRAPHIC REASONING BENCHMARK RESULTS")
        print("=" * 60)
        print(f"\nTotal Tasks:        {self.total_tasks}")
        print(f"Solved Correct:     {self.solved_correct}")
        print(f"Solved Incorrect:   {self.solved_incorrect}")
        print(f"Refused:            {self.refused}")
        print(f"\nAccuracy (all):     {self.accuracy:.1%}")
        print(f"Accuracy (solved):  {self.accuracy_with_refusal:.1%}")
        print(f"Avg Confidence:     {self.avg_confidence:.3f}")
        print(f"Avg Time (ms):      {self.avg_time_ms:.1f}")
        
        # Action distribution
        actions = {}
        for r in self.results:
            if r.action:
                actions[r.action] = actions.get(r.action, 0) + 1
        
        print("\nAction Distribution:")
        for action, count in sorted(actions.items(), key=lambda x: -x[1]):
            print(f"  {action:15} {count:3} ({count/self.total_tasks:.0%})")
        
        print("\n" + "=" * 60)


def load_builtin_tasks() -> List[ARCTask]:
    """Load built-in test tasks for benchmarking."""
    tasks = []
    
    # Task 1: Translate Up
    tasks.append(create_simple_task(
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
        task_id="translate_up",
    ))
    
    # Task 2: Translate Down
    tasks.append(create_simple_task(
        train_inputs=[
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        train_outputs=[
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [2, 0, 0], [0, 0, 0]],
        ],
        test_input=[[0, 0, 3], [0, 0, 0], [0, 0, 0]],
        test_output=[[0, 0, 0], [0, 0, 3], [0, 0, 0]],
        task_id="translate_down",
    ))
    
    # Task 3: Translate Left
    tasks.append(create_simple_task(
        train_inputs=[
            [[0, 0, 1], [0, 0, 0]],
            [[0, 2, 0], [0, 0, 0]],
        ],
        train_outputs=[
            [[0, 1, 0], [0, 0, 0]],
            [[2, 0, 0], [0, 0, 0]],
        ],
        test_input=[[0, 0, 0], [0, 0, 3]],
        test_output=[[0, 0, 0], [0, 3, 0]],
        task_id="translate_left",
    ))
    
    # Task 4: Translate Right
    tasks.append(create_simple_task(
        train_inputs=[
            [[1, 0, 0], [0, 0, 0]],
            [[0, 2, 0], [0, 0, 0]],
        ],
        train_outputs=[
            [[0, 1, 0], [0, 0, 0]],
            [[0, 0, 2], [0, 0, 0]],
        ],
        test_input=[[3, 0, 0], [0, 0, 0]],
        test_output=[[0, 3, 0], [0, 0, 0]],
        task_id="translate_right",
    ))
    
    # Task 5: Recolor to Blue
    tasks.append(create_simple_task(
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
        task_id="recolor_to_blue",
    ))
    
    # Task 6: Recolor to Red
    tasks.append(create_simple_task(
        train_inputs=[
            [[1, 0], [0, 0]],
            [[0, 3], [0, 0]],
        ],
        train_outputs=[
            [[2, 0], [0, 0]],
            [[0, 2], [0, 0]],
        ],
        test_input=[[0, 0], [4, 0]],
        test_output=[[0, 0], [2, 0]],
        task_id="recolor_to_red",
    ))
    
    # Task 7: Identity
    tasks.append(create_simple_task(
        train_inputs=[
            [[0, 1, 0], [0, 0, 0]],
            [[2, 0], [0, 0]],
        ],
        train_outputs=[
            [[0, 1, 0], [0, 0, 0]],
            [[2, 0], [0, 0]],
        ],
        test_input=[[0, 0, 3], [0, 0, 0]],
        test_output=[[0, 0, 3], [0, 0, 0]],
        task_id="identity",
    ))
    
    # Task 8: Delete (all objects become background)
    tasks.append(create_simple_task(
        train_inputs=[
            [[1, 0], [0, 2]],
            [[3, 3], [0, 0]],
        ],
        train_outputs=[
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        test_input=[[0, 4], [5, 0]],
        test_output=[[0, 0], [0, 0]],
        task_id="delete_all",
    ))
    
    # Task 9: Flip Horizontal
    tasks.append(create_simple_task(
        train_inputs=[
            [[1, 1, 0], [0, 0, 0]],
        ],
        train_outputs=[
            [[0, 1, 1], [0, 0, 0]],
        ],
        test_input=[[2, 2, 0], [0, 0, 0]],
        test_output=[[0, 2, 2], [0, 0, 0]],
        task_id="flip_horizontal",
    ))
    
    # Task 10: Complex multi-object (harder)
    tasks.append(create_simple_task(
        train_inputs=[
            [[1, 0, 2], [0, 0, 0], [3, 0, 4]],
        ],
        train_outputs=[
            [[0, 0, 0], [1, 0, 2], [0, 0, 0]],  # All move down? unclear
        ],
        test_input=[[5, 0, 6], [0, 0, 0], [0, 0, 0]],
        test_output=[[0, 0, 0], [5, 0, 6], [0, 0, 0]],
        task_id="multi_object_translate",
    ))
    
    return tasks


def load_json_tasks(path: Path) -> List[ARCTask]:
    """Load tasks from JSON files."""
    tasks = []
    
    if path.is_file():
        files = [path]
    else:
        files = list(path.glob("*.json"))
    
    for file in files:
        try:
            with open(file) as f:
                data = json.load(f)
            task = ARCTask.from_dict(data, task_id=file.stem)
            tasks.append(task)
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")
    
    return tasks


def load_arckit_tasks(dataset: str = "training") -> List[ARCTask]:
    """Load tasks from arckit (if installed)."""
    try:
        import arckit
    except ImportError:
        print("arckit not installed. Install with: pip install arckit")
        return []
    
    tasks = []
    arc_dataset = arckit.load_data(dataset)
    
    for task_id, task_data in arc_dataset.items():
        try:
            task = ARCTask.from_dict(task_data, task_id=task_id)
            tasks.append(task)
        except Exception as e:
            print(f"Warning: Failed to load {task_id}: {e}")
    
    return tasks


def run_benchmark(
    tasks: List[ARCTask],
    solver: Optional[HolographicARCSolver] = None,
    verbose: bool = True,
) -> BenchmarkReport:
    """
    Run benchmark on a list of tasks.
    
    Args:
        tasks: List of ARC tasks
        solver: Optional solver (creates new one if None)
        verbose: Print progress
        
    Returns:
        BenchmarkReport with results
    """
    if solver is None:
        solver = HolographicARCSolver()
    
    results = []
    
    for i, task in enumerate(tasks):
        if verbose:
            print(f"[{i+1}/{len(tasks)}] {task.task_id}...", end=" ", flush=True)
        
        start = time.time()
        
        try:
            result = solver.solve(task)
            elapsed_ms = (time.time() - start) * 1000
            
            # Check correctness
            if result.output is None:
                correct = False
                refused = True
            elif task.test_output is None:
                correct = False  # No ground truth
                refused = False
            else:
                correct = result.output == task.test_output
                refused = False
            
            bench_result = BenchmarkResult(
                task_id=task.task_id,
                correct=correct,
                refused=refused,
                confidence=result.confidence,
                action=result.transformation.action if result.transformation else None,
                target=result.transformation.target if result.transformation else None,
                modifier=result.transformation.modifier if result.transformation else None,
                from_cache=result.from_cache,
                time_ms=elapsed_ms,
                message=result.message,
            )
            
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            bench_result = BenchmarkResult(
                task_id=task.task_id,
                correct=False,
                refused=True,
                confidence=0.0,
                action=None,
                target=None,
                modifier=None,
                from_cache=False,
                time_ms=elapsed_ms,
                message=f"Error: {e}",
            )
        
        results.append(bench_result)
        
        if verbose:
            status = "✓" if bench_result.correct else ("⊘" if bench_result.refused else "✗")
            print(f"{status} ({bench_result.time_ms:.0f}ms)")
    
    # Compute summary statistics
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    incorrect = sum(1 for r in results if not r.correct and not r.refused)
    refused = sum(1 for r in results if r.refused)
    
    accuracy = correct / total if total > 0 else 0.0
    solved = correct + incorrect
    accuracy_solved = correct / solved if solved > 0 else 0.0
    
    avg_conf = sum(r.confidence for r in results) / total if total > 0 else 0.0
    avg_time = sum(r.time_ms for r in results) / total if total > 0 else 0.0
    
    return BenchmarkReport(
        total_tasks=total,
        solved_correct=correct,
        solved_incorrect=incorrect,
        refused=refused,
        accuracy=accuracy,
        accuracy_with_refusal=accuracy_solved,
        avg_confidence=avg_conf,
        avg_time_ms=avg_time,
        results=results,
    )


def main():
    parser = argparse.ArgumentParser(
        description="ARC-AGI-2 Holographic Reasoning Benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="ARC dataset to use: 'training' or 'evaluation' (requires arckit)",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="Path to JSON task file or directory",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=10000,
        help="HDC vector dimensions (default: 10000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        help="Confidence threshold (default: 0.005)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to run",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Output results to JSON file",
    )
    
    args = parser.parse_args()
    
    # Load tasks
    if args.dataset:
        print(f"Loading ARC {args.dataset} dataset...")
        tasks = load_arckit_tasks(args.dataset)
    elif args.tasks:
        print(f"Loading tasks from {args.tasks}...")
        tasks = load_json_tasks(args.tasks)
    else:
        print("Loading built-in test tasks...")
        tasks = load_builtin_tasks()
    
    if not tasks:
        print("No tasks loaded!")
        return 1
    
    # Apply limit
    if args.limit:
        tasks = tasks[:args.limit]
    
    print(f"Loaded {len(tasks)} tasks")
    
    # Create solver
    solver = HolographicARCSolver(
        dimensions=args.dimensions,
        confidence_threshold=args.threshold,
    )
    
    # Run benchmark
    print("\nRunning benchmark...")
    report = run_benchmark(tasks, solver, verbose=not args.quiet)
    
    # Print summary
    report.print_summary()
    
    # Save JSON if requested
    if args.json:
        output = {
            "summary": {
                "total_tasks": report.total_tasks,
                "solved_correct": report.solved_correct,
                "solved_incorrect": report.solved_incorrect,
                "refused": report.refused,
                "accuracy": report.accuracy,
                "accuracy_with_refusal": report.accuracy_with_refusal,
                "avg_confidence": report.avg_confidence,
                "avg_time_ms": report.avg_time_ms,
            },
            "results": [
                {
                    "task_id": r.task_id,
                    "correct": r.correct,
                    "refused": r.refused,
                    "confidence": r.confidence,
                    "action": r.action,
                    "target": r.target,
                    "modifier": r.modifier,
                    "time_ms": r.time_ms,
                }
                for r in report.results
            ],
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.json}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
