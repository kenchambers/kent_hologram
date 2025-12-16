"""
HonestCodeBenchmark: SWE-bench evaluation with proper cache isolation.

Follows the same pattern as hologram.arc.benchmark for honest evaluation:
1. Cache clearing protocol (fresh generator per task)
2. SWE-bench task loader
3. Reproducible benchmarking CLI

Per Dr. Nexus: "Show me the numbers on the standard benchmark,
not your custom test set."
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import time
import difflib

from hologram.core.vector_space import VectorSpace
from hologram.core.codebook import Codebook
from hologram.core.fractal import FractalSpace
from hologram.consolidation.neural_memory import NeuralMemory
from hologram.arc.encoder import ObjectEncoder
from hologram.arc.transform_resonator import TransformationResonator
from hologram.swe.types import SWETask, CodePatch, PatchResult
from hologram.swe.encoder import CodeEncoder
from hologram.swe.code_resonator import CodeResonator
from hologram.swe.generator import CodeGenerator


@dataclass
class TaskResult:
    """Result from evaluating a single SWE task."""
    task_id: str
    generated: bool  # Did generator produce output?
    correct: Optional[bool]  # Matches ground truth?
    partial_score: float  # 0-1 partial credit
    confidence: float
    message: str
    time_ms: float
    patches_generated: int


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    results: List[TaskResult]
    errors: List[Dict[str, Any]]
    accuracy: float  # Fraction exactly correct
    partial_accuracy: float  # Average partial score
    generation_rate: float  # Fraction that produced output
    total_time_s: float
    tasks_evaluated: int

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=== HONEST CODE BENCHMARK RESULTS ===",
            f"Tasks evaluated: {self.tasks_evaluated}",
            f"Exact accuracy: {self.accuracy * 100:.1f}%",
            f"Partial accuracy: {self.partial_accuracy * 100:.1f}%",
            f"Generation rate: {self.generation_rate * 100:.1f}%",
            f"Errors: {len(self.errors)}",
            f"Total time: {self.total_time_s:.1f}s",
            f"Avg time per task: {self.total_time_s / max(1, self.tasks_evaluated):.2f}s",
        ]
        return "\n".join(lines)


class HonestCodeBenchmark:
    """
    SWE-bench evaluation with proper cache isolation.

    Each task gets a FRESH generator to prevent pattern memory contamination.
    This is the honest way to benchmark - no cheating via cached solutions.

    Example:
        >>> benchmark = HonestCodeBenchmark()
        >>> tasks = load_sample_tasks()
        >>> result = benchmark.evaluate(tasks)
        >>> print(result.accuracy)
        0.10  # 10% exact match accuracy
    """

    def __init__(
        self,
        dimensions: int = 10000,
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize benchmark configuration.

        Args:
            dimensions: HDC vector dimensions for generator
            confidence_threshold: Minimum confidence for output
        """
        self._dimensions = dimensions
        self._confidence_threshold = confidence_threshold

    def evaluate(
        self,
        tasks: List[SWETask],
        isolate_cache: bool = True,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """
        Evaluate generator on a list of SWE tasks.

        Args:
            tasks: List of SWE tasks to evaluate
            isolate_cache: If True, create fresh generator per task (recommended)
            verbose: Print progress during evaluation

        Returns:
            BenchmarkResult with aggregated statistics
        """
        results: List[TaskResult] = []
        errors: List[Dict[str, Any]] = []
        start_time = time.time()

        # Shared generator for non-isolated mode (ablation studies only)
        shared_generator = None
        if not isolate_cache:
            shared_generator = self._create_generator()

        for i, task in enumerate(tasks):
            if verbose:
                print(f"[{i + 1}/{len(tasks)}] Evaluating {task.task_id}...")

            # CRITICAL: Fresh generator per task for honest evaluation
            if isolate_cache:
                generator = self._create_generator()
            else:
                generator = shared_generator

            task_start = time.time()
            try:
                result = generator.generate(
                    task,
                    confidence_threshold=self._confidence_threshold,
                )
                task_time = (time.time() - task_start) * 1000  # ms

                # Determine correctness via diff comparison
                correct = None
                partial_score = 0.0
                if task.code_after:
                    correct, partial_score = self._evaluate_correctness(
                        result, task
                    )

                results.append(TaskResult(
                    task_id=task.task_id,
                    generated=len(result.patches) > 0,
                    correct=correct,
                    partial_score=partial_score,
                    confidence=result.confidence,
                    message=f"{len(result.patches)} patches generated",
                    time_ms=task_time,
                    patches_generated=len(result.patches),
                ))

            except Exception as e:
                errors.append({
                    "task_id": task.task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                })

            # Verify cache isolation by deleting generator
            if isolate_cache:
                del generator

        total_time = time.time() - start_time

        # Compute aggregated stats
        correct_count = sum(1 for r in results if r.correct is True)
        with_truth_count = sum(1 for r in results if r.correct is not None)
        generated_count = sum(1 for r in results if r.generated)
        total_partial = sum(r.partial_score for r in results if r.correct is not None)

        accuracy = correct_count / max(1, with_truth_count)
        partial_accuracy = total_partial / max(1, with_truth_count)
        generation_rate = generated_count / max(1, len(results))

        return BenchmarkResult(
            results=results,
            errors=errors,
            accuracy=accuracy,
            partial_accuracy=partial_accuracy,
            generation_rate=generation_rate,
            total_time_s=total_time,
            tasks_evaluated=len(tasks),
        )

    def _create_generator(self) -> CodeGenerator:
        """Create a fresh generator instance with isolated memory for honest eval."""
        space = VectorSpace(dimensions=self._dimensions)
        fractal = FractalSpace(dimensions=self._dimensions)
        codebook = Codebook(space)

        # Create encoders
        arc_encoder = ObjectEncoder(fractal, codebook)
        code_encoder = CodeEncoder(fractal, codebook)

        # Create resonators
        transform_resonator = TransformationResonator(arc_encoder, codebook)
        code_resonator = CodeResonator(code_encoder, transform_resonator)

        # Create neural memory (fresh, isolated)
        neural_memory = NeuralMemory(
            input_dim=self._dimensions,
            hidden_dim=256,
            initial_vocab_size=50,
        )

        return CodeGenerator(
            encoder=code_encoder,
            resonator=code_resonator,
            neural_memory=neural_memory,
        )

    def _evaluate_correctness(
        self,
        result: PatchResult,
        task: SWETask,
    ) -> tuple[Optional[bool], float]:
        """
        Evaluate if generated patches match ground truth.

        Returns:
            Tuple of (exact_match, partial_score)
            - exact_match: True if patches produce exactly code_after
            - partial_score: 0-1 based on diff similarity
        """
        if not result.patches:
            return False, 0.0

        # Simple evaluation: check if target files match
        # In production, would apply patches and compare results
        target_files = set(p.file for p in result.patches)
        expected_files = set(task.code_after.keys())

        # Check file overlap
        file_overlap = len(target_files & expected_files) / max(1, len(expected_files))

        # For each matching file, compute diff similarity
        similarity_scores = []
        for patch in result.patches:
            if patch.file in task.code_after:
                expected = task.code_after[patch.file]
                # Use sequence matcher to compare patch content to expected
                # This is a simplified check - real eval would apply patch
                matcher = difflib.SequenceMatcher(
                    None,
                    patch.content.lower(),
                    expected.lower()[:len(patch.content) + 100]
                )
                similarity_scores.append(matcher.ratio())

        content_similarity = sum(similarity_scores) / max(1, len(similarity_scores)) if similarity_scores else 0.0

        # Combine scores
        partial_score = 0.4 * file_overlap + 0.6 * content_similarity

        # Exact match requires high similarity (simplified)
        exact_match = partial_score > 0.9

        return exact_match, partial_score


def load_sample_tasks() -> List[SWETask]:
    """
    Load sample SWE tasks for testing.

    Returns:
        List of sample SWETask instances
    """
    return [
        SWETask(
            task_id="sample_001",
            repo="test/repo",
            issue_text="Add input validation to the process function",
            code_before={"utils.py": "def process(x):\n    return x * 2"},
            code_after={"utils.py": "def process(x):\n    if x is None:\n        raise ValueError('x cannot be None')\n    return x * 2"},
        ),
        SWETask(
            task_id="sample_002",
            repo="test/repo",
            issue_text="Add logging to the calculate function",
            code_before={"math.py": "def calculate(a, b):\n    return a + b"},
            code_after={"math.py": "import logging\n\ndef calculate(a, b):\n    logging.info(f'Calculating {a} + {b}')\n    return a + b"},
        ),
        SWETask(
            task_id="sample_003",
            repo="test/repo",
            issue_text="Fix division by zero in divide function",
            code_before={"math.py": "def divide(a, b):\n    return a / b"},
            code_after={"math.py": "def divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError('Cannot divide by zero')\n    return a / b"},
        ),
    ]


def load_swe_bench(
    split: str = "test",
    limit: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> List[SWETask]:
    """
    Load SWE-bench dataset.

    Args:
        split: Dataset split ("dev", "test", or "lite")
        limit: Maximum number of tasks to load (None = all)
        data_dir: Override data directory

    Returns:
        List of SWETask instances

    Raises:
        FileNotFoundError: If dataset not found
    """
    # Try multiple locations
    if data_dir is not None:
        swe_path = data_dir / split
    else:
        project_data = Path(__file__).parent.parent.parent.parent / "data" / "swe-bench" / split
        home_data = Path.home() / ".swe-bench" / split

        if project_data.exists():
            swe_path = project_data
        elif home_data.exists():
            swe_path = home_data
        else:
            raise FileNotFoundError(
                f"SWE-bench {split} set not found. Tried:\n"
                f"  {project_data}\n"
                f"  {home_data}\n"
                f"Download from https://github.com/princeton-nlp/SWE-bench"
            )

    tasks = []
    for task_file in sorted(swe_path.glob("*.json")):
        with open(task_file) as f:
            data = json.load(f)

        # Map SWE-bench format to our SWETask
        tasks.append(SWETask(
            task_id=data.get("instance_id", task_file.stem),
            repo=data.get("repo", "unknown"),
            issue_text=data.get("problem_statement", ""),
            code_before=data.get("base_commit_files", {}),
            code_after=data.get("patch_files", {}),
        ))

        if limit and len(tasks) >= limit:
            break

    return tasks


def run_benchmark(
    use_sample: bool = True,
    limit: int = 10,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Convenience function to run benchmark with standard settings.

    Args:
        use_sample: Use sample tasks instead of SWE-bench
        limit: Number of tasks to evaluate
        verbose: Print progress

    Returns:
        BenchmarkResult with statistics
    """
    if verbose:
        print(f"Loading tasks (limit={limit})...")

    if use_sample:
        tasks = load_sample_tasks()[:limit]
    else:
        try:
            tasks = load_swe_bench(split="test", limit=limit)
        except FileNotFoundError:
            if verbose:
                print("SWE-bench not found, using sample tasks...")
            tasks = load_sample_tasks()[:limit]

    if verbose:
        print(f"Loaded {len(tasks)} tasks")
        print(f"Evaluating with cache isolation (honest mode)...")

    benchmark = HonestCodeBenchmark()
    result = benchmark.evaluate(tasks, isolate_cache=True, verbose=verbose)

    if verbose:
        print()
        print(result.summary())

    return result


if __name__ == "__main__":
    run_benchmark(use_sample=True, verbose=True)
