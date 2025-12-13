#!/usr/bin/env python3
"""
Honest ARC-AGI-2 benchmark.

Evaluates the holographic ARC solver on real ARC-AGI-2 tasks with
proper cache isolation for honest reporting.

Usage:
    python scripts/arc_benchmark_honest.py --split evaluation --limit 100
    python scripts/arc_benchmark_honest.py --split training --limit 50 --no-iterative
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hologram.arc.benchmark import run_benchmark, BenchmarkResult


def main():
    parser = argparse.ArgumentParser(
        description="Honest ARC-AGI-2 benchmark with cache isolation"
    )
    parser.add_argument(
        "--split",
        default="evaluation",
        choices=["training", "evaluation"],
        help="Dataset split to evaluate (default: evaluation)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of tasks to evaluate (default: 100)",
    )
    parser.add_argument(
        "--no-iterative",
        action="store_true",
        help="Disable iterative solving (use single-shot only)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save detailed results to JSON file",
    )

    args = parser.parse_args()

    result = run_benchmark(
        split=args.split,
        limit=args.limit,
        iterative=not args.no_iterative,
        verbose=not args.quiet,
    )

    # Save detailed results if requested
    if args.output:
        import json
        output_data = {
            "summary": {
                "accuracy": result.accuracy,
                "solve_rate": result.solve_rate,
                "tasks_evaluated": result.tasks_evaluated,
                "total_time_s": result.total_time_s,
                "errors_count": len(result.errors),
            },
            "results": [
                {
                    "task_id": r.task_id,
                    "solved": r.solved,
                    "correct": r.correct,
                    "confidence": r.confidence,
                    "message": r.message,
                    "from_cache": r.from_cache,
                    "time_ms": r.time_ms,
                }
                for r in result.results
            ],
            "errors": result.errors,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")

    # Return exit code based on success
    return 0 if result.accuracy > 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
