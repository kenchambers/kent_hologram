#!/usr/bin/env python3
"""
ARC Trainer: Train Hologram's shared memory on ARC transformation patterns.

Unlike crew_trainer.py (LLM-based conversation), this extracts transformation
patterns directly from ground-truth training pairs, storing them in the SAME
shared memory system. ARC patterns and vocabulary facts reinforce each other.

Usage:
    # Train on easy tasks first (curriculum learning)
    python scripts/arc_trainer.py --curriculum easy --max-rounds 5

    # Full curriculum training with validation
    python scripts/arc_trainer.py --max-rounds 20 --validate-every 5

    # Use same memory as crew_trainer (default)
    python scripts/arc_trainer.py --persist-dir ./data/crew_training_facts
"""

import argparse
import json
import os
import signal
import sys
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Generator

import torch
import numpy as np

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hologram.arc.benchmark import load_arc_agi_2, HonestBenchmark, BenchmarkResult
from hologram.arc.solver import HolographicARCSolver, SolverResult
from hologram.arc.types import ARCTask, Grid, TrainingPair, ACTIONS, TARGETS, MODIFIERS
from hologram.arc.encoder import ObjectEncoder
from hologram.arc.detector import ObjectDetector
from hologram.arc.search_verifier import SearchVerifier, VerificationStats
from hologram.arc.transform_resonator import TransformationResonator, TransformResult
from hologram.arc.executor import TransformationExecutor
from hologram.container import HologramContainer
from hologram.core.operations import Operations
from hologram.introspection import SelfImprovementManager


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExtractedPattern:
    """A transformation pattern extracted from training pairs."""
    task_id: str
    action: str
    target: str
    modifier: str
    signature_vector: torch.Tensor
    confidence: float
    verified: bool  # True if pattern reproduces all training outputs


@dataclass
class DifficultyMetrics:
    """Metrics for sorting tasks by difficulty."""
    task_id: str
    grid_size: int  # max(height, width) of any grid
    object_count: float  # average objects per training pair
    color_count: int  # unique colors used
    training_pairs: int  # number of training examples
    grid_change: bool  # does grid size change between input/output?

    @property
    def difficulty_score(self) -> float:
        """Compute numeric difficulty score (lower = easier)."""
        score = 0.0
        score += self.grid_size * 0.5  # Larger grids are harder
        score += self.object_count * 2.0  # More objects are harder
        score += self.color_count * 0.3  # More colors add complexity
        score += 5.0 if self.grid_change else 0.0  # Grid changes are hard
        score -= self.training_pairs * 0.5  # More examples help
        return score


@dataclass
class TrainingStats:
    """Statistics for tracking training progress."""
    rounds_completed: int = 0
    tasks_attempted: int = 0
    tasks_solved: int = 0
    patterns_stored: int = 0
    patterns_already_known: int = 0
    extraction_failures: int = 0
    validation_accuracy: float = 0.0

    def __str__(self) -> str:
        solve_rate = (
            self.tasks_solved / max(1, self.tasks_attempted) * 100
        )
        return (
            f"Rounds: {self.rounds_completed}, "
            f"Tasks: {self.tasks_attempted}, "
            f"Solved: {self.tasks_solved} ({solve_rate:.1f}%), "
            f"Patterns stored: {self.patterns_stored}, "
            f"Already known: {self.patterns_already_known}, "
            f"Extraction failures: {self.extraction_failures}"
        )


# =============================================================================
# Curriculum Management
# =============================================================================

class ARCCurriculum:
    """
    Manage curriculum of ARC tasks sorted by difficulty.

    Difficulty tiers:
    - EASY: Small grids (<=5), few objects (<=3), no grid size changes
    - MEDIUM: Medium grids (<=10), moderate objects (<=6)
    - HARD: Large grids, many objects, complex transformations
    """

    DIFFICULTY_TIERS = ["easy", "medium", "hard"]

    def __init__(self, tasks: List[ARCTask]):
        """
        Initialize curriculum with task analysis.

        Args:
            tasks: List of ARC tasks to organize
        """
        self._tasks = tasks
        self._metrics: Dict[str, DifficultyMetrics] = {}
        self._by_tier: Dict[str, List[ARCTask]] = {
            "easy": [], "medium": [], "hard": []
        }
        self._detector = ObjectDetector()
        self._analyze_and_sort()

    def _compute_difficulty(self, task: ARCTask) -> DifficultyMetrics:
        """Compute difficulty metrics for a task."""
        max_grid_size = 0
        total_objects = 0
        all_colors = set()
        grid_change = False

        for pair in task.training:
            # Grid sizes
            max_grid_size = max(
                max_grid_size,
                pair.input.height, pair.input.width,
                pair.output.height, pair.output.width
            )

            # Check for grid size change
            if pair.input.shape != pair.output.shape:
                grid_change = True

            # Object counts (input only - more stable)
            try:
                objects = self._detector.detect(pair.input)
                total_objects += len(objects)
            except Exception:
                total_objects += 1  # Assume at least 1 object

            # Unique colors
            all_colors.update(pair.input.data.flatten().tolist())
            all_colors.update(pair.output.data.flatten().tolist())

        avg_objects = total_objects / max(1, len(task.training))

        return DifficultyMetrics(
            task_id=task.task_id,
            grid_size=max_grid_size,
            object_count=avg_objects,
            color_count=len(all_colors),
            training_pairs=len(task.training),
            grid_change=grid_change,
        )

    def _classify_tier(self, metrics: DifficultyMetrics) -> str:
        """Classify task into difficulty tier."""
        # Easy: small grids, few objects, no grid changes
        if (metrics.grid_size <= 5 and
            metrics.object_count <= 3 and
            not metrics.grid_change):
            return "easy"

        # Medium: moderate complexity
        if (metrics.grid_size <= 10 and
            metrics.object_count <= 6):
            return "medium"

        # Hard: everything else
        return "hard"

    def _analyze_and_sort(self):
        """Analyze all tasks and sort into difficulty tiers."""
        for task in self._tasks:
            metrics = self._compute_difficulty(task)
            self._metrics[task.task_id] = metrics
            tier = self._classify_tier(metrics)
            self._by_tier[tier].append(task)

        # Sort within each tier by difficulty score
        for tier in self.DIFFICULTY_TIERS:
            self._by_tier[tier].sort(
                key=lambda t: self._metrics[t.task_id].difficulty_score
            )

    def get_tier(self, tier: str) -> List[ARCTask]:
        """Get all tasks in a difficulty tier."""
        return self._by_tier.get(tier, [])

    def get_tier_count(self, tier: str) -> int:
        """Get count of tasks in a tier."""
        return len(self._by_tier.get(tier, []))

    def iterate_by_difficulty(self) -> Generator[ARCTask, None, None]:
        """Iterate through tasks from easiest to hardest."""
        for tier in self.DIFFICULTY_TIERS:
            for task in self._by_tier[tier]:
                yield task

    def summary(self) -> str:
        """Return summary of curriculum distribution."""
        lines = ["ARC Curriculum Distribution:"]
        for tier in self.DIFFICULTY_TIERS:
            count = self.get_tier_count(tier)
            lines.append(f"  {tier.upper()}: {count} tasks")
        return "\n".join(lines)


# =============================================================================
# Transformation Extraction
# =============================================================================

class TransformationExtractor:
    """
    Extract transformation patterns from ARC training pairs.

    Uses search+verify to find the correct (ACTION, TARGET, MODIFIER)
    that transforms all training inputs to their outputs.
    """

    def __init__(
        self,
        encoder: ObjectEncoder,
        detector: ObjectDetector,
        dimensions: int = 10000,
    ):
        """
        Initialize extractor.

        Args:
            encoder: ObjectEncoder for creating vectors
            detector: ObjectDetector for finding objects
            dimensions: HDC vector dimensions
        """
        self._encoder = encoder
        self._detector = detector
        self._executor = TransformationExecutor()
        self._verifier = SearchVerifier(
            executor=self._executor,
            detector=detector
        )

    def extract(self, task: ARCTask) -> Optional[ExtractedPattern]:
        """
        Extract the transformation pattern from a task's training pairs.

        Returns the pattern that correctly transforms all training examples.
        """
        if not task.training:
            return None

        # Build candidate transformations to test
        candidates = self._generate_candidates()

        # Verify against all training pairs
        stats = self._verifier.verify_candidates_with_stats(
            candidates, task.training
        )

        if stats.verified_transform is None:
            return None

        # Compute task signature for memory key
        signature = self._compute_task_signature(task)

        return ExtractedPattern(
            task_id=task.task_id,
            action=stats.verified_transform.action,
            target=stats.verified_transform.target,
            modifier=stats.verified_transform.modifier,
            signature_vector=signature,
            confidence=1.0,  # Verified = 100% confidence
            verified=True,
        )

    def _generate_candidates(self) -> List[TransformResult]:
        """Generate all possible transformation candidates."""
        candidates = []

        # Create dummy vectors (we only need action/target/modifier strings for verification)
        dummy_vec = torch.zeros(self._encoder._fractal_space.dimensions)

        # Generate all combinations (ACTION x TARGET x MODIFIER)
        # This is expensive but exhaustive
        for action in ACTIONS:
            for target in TARGETS:
                for modifier in MODIFIERS:
                    candidates.append(TransformResult(
                        action=action,
                        target=target,
                        modifier=modifier,
                        action_vec=dummy_vec,
                        target_vec=dummy_vec,
                        modifier_vec=dummy_vec,
                        iterations=0,
                        converged=True,
                        confidence={
                            "action": 1.0,
                            "target": 1.0,
                            "modifier": 1.0,
                        },
                    ))

        return candidates

    def _compute_task_signature(self, task: ARCTask) -> torch.Tensor:
        """
        Compute a signature vector for the task.

        This enables similar tasks to share learned transformations.
        """
        # Hash the structural properties of the task
        sig_parts = []

        for pair in task.training:
            # Grid dimensions
            sig_parts.append(f"in:{pair.input.height}x{pair.input.width}")
            sig_parts.append(f"out:{pair.output.height}x{pair.output.width}")

            # Object counts
            try:
                in_objs = self._detector.detect(pair.input)
                out_objs = self._detector.detect(pair.output)
                sig_parts.append(f"objs:{len(in_objs)}->{len(out_objs)}")
            except Exception:
                pass

        # Create deterministic hash
        sig_str = "|".join(sig_parts)
        sig_hash = hashlib.md5(sig_str.encode()).hexdigest()

        # Convert to tensor (use hash as seed for reproducibility)
        seed = int(sig_hash[:8], 16) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(self._encoder._fractal_space.dimensions, generator=gen)


# =============================================================================
# Main Trainer
# =============================================================================

class ARCTrainer:
    """
    Main ARC training orchestrator.

    Trains Hologram's shared memory by:
    1. Loading ARC tasks sorted by difficulty
    2. Attempting to solve each task
    3. If wrong, extracting correct pattern from training pairs
    4. Storing transformation pattern in shared ConsolidationManager
    """

    def __init__(
        self,
        persist_dir: str = "./data/crew_training_facts",
        dimensions: int = 10000,
        consolidation_threshold: int = 20,
        log_dir: Path = Path("./arc_training_logs"),
        enable_self_improvement: bool = True,
    ):
        """
        Initialize trainer with shared memory.

        Args:
            persist_dir: Shared persistence directory (same as crew_trainer)
            dimensions: HDC vector dimensions
            consolidation_threshold: Facts before neural consolidation
            log_dir: Directory for training logs
            enable_self_improvement: Enable circuit observer for self-improvement (default: True)
        """
        self.persist_dir = persist_dir
        self.dimensions = dimensions
        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"arc_training_{timestamp}.log"

        self._log("=" * 60)
        self._log("ARC Trainer Initialization")
        self._log("=" * 60)

        # Initialize container for shared components
        self._log("Creating HologramContainer...")
        self.container = HologramContainer(dimensions=dimensions)

        # Create ConsolidationManager (shared with crew_trainer)
        self._log("Creating ConsolidationManager...")
        self._consolidation_manager = self.container.create_consolidation_manager(
            threshold=consolidation_threshold
        )

        # Load existing memory if available
        self._load_existing_memory()

        # ARC-specific components
        self._log("Creating ARC components...")
        self._encoder = ObjectEncoder(
            self.container._space,  # FractalSpace
            self.container._codebook
        )
        self._detector = ObjectDetector()
        self._extractor = TransformationExtractor(
            self._encoder, self._detector, dimensions
        )

        # Create self-improvement manager for cross-task learning
        self._self_improvement = None
        if enable_self_improvement:
            self_improvement_path = str(Path(persist_dir) / "arc_learned_patterns.json")
            self._self_improvement = SelfImprovementManager(persist_path=self_improvement_path)
            self._log(f"Self-improvement enabled: {self_improvement_path}")

        # Create solver with shared memory (NOT isolated)
        self._log("Creating HolographicARCSolver (shared memory mode)...")
        self._solver = HolographicARCSolver(
            dimensions=dimensions,
            isolate_memory=False,  # Use shared memory!
            enable_self_improvement=enable_self_improvement,
            self_improvement_path=str(Path(persist_dir) / "arc_learned_patterns.json") if enable_self_improvement else None,
        )

        # Statistics
        self._stats = TrainingStats()
        self.running = True

        self._log(f"Initialization complete. Persist dir: {persist_dir}")

    def _log(self, message: str):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        with open(self.log_file, "a") as f:
            f.write(log_line + "\n")

    def _load_existing_memory(self):
        """Load existing neural memory from shared persistence."""
        neural_path = Path(self.persist_dir) / "neural_memory.pt"
        if neural_path.exists():
            try:
                state = torch.load(neural_path, weights_only=False)
                self._consolidation_manager.load_state_dict(state)
                vocab_size = len(self._consolidation_manager._value_vocab)
                self._log(f"Loaded existing memory: {vocab_size} patterns")
            except Exception as e:
                self._log(f"Warning: Could not load existing memory: {e}")

    def _store_pattern(self, pattern: ExtractedPattern):
        """
        Store transformation pattern in shared memory.

        Uses ConsolidationManager.store() to add to both HDC working memory
        and queue for neural consolidation.
        """
        # Create label for the transformation
        label = f"{pattern.action}_{pattern.target}_{pattern.modifier}"

        # Encode the value (transformation label)
        value_vector = self.container._codebook.encode(label)

        # Store in ConsolidationManager
        try:
            latency = self._consolidation_manager.store(
                key=pattern.signature_vector,
                value=value_vector,
                value_label=label,
            )
            self._stats.patterns_stored += 1
            self._log(f"  Stored pattern: {label} ({latency:.1f}ms)")
        except Exception as e:
            self._log(f"  Failed to store pattern: {e}")

    def train_on_task(self, task: ARCTask) -> bool:
        """
        Train on a single task.

        Returns True if task was solved (pattern already known or learned).
        """
        self._stats.tasks_attempted += 1

        # Step 1: Try to solve with current knowledge
        try:
            result = self._solver.solve(task)
        except Exception as e:
            self._log(f"  Solver error: {e}")
            result = SolverResult(
                output=None, transformation=None,
                confidence=0.0, from_cache=False, message=str(e)
            )

        # Check if already solved correctly
        if result.output is not None and task.test_output is not None:
            if result.output == task.test_output:
                self._stats.tasks_solved += 1
                self._stats.patterns_already_known += 1
                self._log(f"  Already known (from cache: {result.from_cache})")
                return True

        # Step 2: Extract correct pattern from training pairs
        pattern = self._extractor.extract(task)

        if pattern is None or not pattern.verified:
            self._stats.extraction_failures += 1
            self._log(f"  Could not extract verified pattern")
            return False

        # Step 3: Store pattern in shared memory
        self._store_pattern(pattern)
        self._stats.tasks_solved += 1

        return True

    def train_round(
        self,
        tasks: List[ARCTask],
        tier_name: str = "all"
    ) -> Tuple[int, int]:
        """
        Train on a list of tasks.

        Returns (success_count, total_count).
        """
        success_count = 0
        self._log(f"\nTraining on {tier_name} tier ({len(tasks)} tasks)")

        for i, task in enumerate(tasks):
            if not self.running:
                break

            self._log(f"[{i+1}/{len(tasks)}] Task {task.task_id}")

            if self.train_on_task(task):
                success_count += 1

        return success_count, len(tasks)

    def validate(self, validation_tasks: List[ARCTask]) -> float:
        """
        Run validation benchmark on held-out tasks.

        Returns accuracy (0.0 to 1.0).
        """
        self._log(f"\n--- Validation ({len(validation_tasks)} tasks) ---")

        benchmark = HonestBenchmark(
            dimensions=self.dimensions,
            iterative=True,
        )

        # Use shared memory for validation (not isolated)
        result = benchmark.evaluate(
            validation_tasks,
            isolate_cache=False,  # Use shared memory
            verbose=False
        )

        self._stats.validation_accuracy = result.accuracy
        self._log(f"Validation accuracy: {result.accuracy:.1%}")
        self._log(f"Solve rate: {result.solve_rate:.1%}")

        return result.accuracy

    def save_memory(self, force_consolidation: bool = False):
        """Save neural memory to persistence directory."""
        try:
            # Force consolidation if requested
            if force_consolidation:
                self._consolidation_manager.force_consolidation()

            # Save state
            state = self._consolidation_manager.state_dict()
            neural_path = Path(self.persist_dir) / "neural_memory.pt"
            torch.save(state, neural_path)

            self._log(f"Saved memory to {neural_path}")
        except Exception as e:
            self._log(f"Error saving memory: {e}")

    def run_continuous(
        self,
        curriculum: ARCCurriculum,
        max_rounds: int,
        validate_every: int,
        validation_tasks: List[ARCTask],
    ):
        """
        Run continuous training loop.

        Args:
            curriculum: ARCCurriculum with sorted tasks
            max_rounds: Maximum training rounds
            validate_every: Validate every N rounds
            validation_tasks: Tasks for validation
        """
        self._log("\n" + "=" * 60)
        self._log("Starting Continuous Training")
        self._log("=" * 60)
        self._log(curriculum.summary())

        # Start consolidation worker
        self._consolidation_manager.start_worker()

        try:
            for round_num in range(max_rounds):
                if not self.running:
                    break

                self._stats.rounds_completed = round_num + 1
                self._log(f"\n{'='*60}")
                self._log(f"Round {round_num + 1}/{max_rounds}")
                self._log(f"{'='*60}")

                # Train on each difficulty tier
                for tier in ARCCurriculum.DIFFICULTY_TIERS:
                    if not self.running:
                        break

                    tasks = curriculum.get_tier(tier)
                    if tasks:
                        success, total = self.train_round(tasks, tier)
                        self._log(f"  {tier.upper()}: {success}/{total} solved")

                # Periodic validation
                if (round_num + 1) % validate_every == 0:
                    self.validate(validation_tasks)

                # Save checkpoint
                self.save_memory()

                # Status report
                self._log(f"\nRound {round_num + 1} complete: {self._stats}")

        except KeyboardInterrupt:
            self._log("\nTraining interrupted by user")
            self.running = False

        finally:
            # Final save
            self._log("\nFinal save...")
            self.save_memory(force_consolidation=True)
            self._consolidation_manager.stop_worker()

            # Save self-improvement learned patterns
            if self._self_improvement:
                self._self_improvement.save()
                self._log("Saved self-improvement patterns")

            # Final report
            self._log("\n" + "=" * 60)
            self._log("Final Training Report")
            self._log("=" * 60)
            self._log(str(self._stats))
            if validation_tasks:
                self.validate(validation_tasks)

            # Self-improvement report
            if self._self_improvement:
                self._log("\n" + "=" * 60)
                self._log("Self-Improvement Learning Report")
                self._log("=" * 60)
                self._log(self._solver.get_improvement_report())


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ARC Trainer - Train Hologram on ARC transformation patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on easy tasks first
  python scripts/arc_trainer.py --curriculum easy --max-rounds 5

  # Full curriculum training
  python scripts/arc_trainer.py --max-rounds 20 --validate-every 5

  # Use same memory as crew_trainer (default)
  python scripts/arc_trainer.py --persist-dir ./data/crew_training_facts
        """
    )

    parser.add_argument(
        "--persist-dir",
        default="./data/crew_training_facts",
        help="Shared persistence directory (default: ./data/crew_training_facts)"
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Number of training rounds (default: 10)"
    )

    parser.add_argument(
        "--curriculum",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Difficulty tier to train on (default: all)"
    )

    parser.add_argument(
        "--validate-every",
        type=int,
        default=5,
        help="Run validation every N rounds (default: 5)"
    )

    parser.add_argument(
        "--validation-limit",
        type=int,
        default=50,
        help="Number of validation tasks to use (default: 50)"
    )

    parser.add_argument(
        "--train-split",
        default="training",
        help="Dataset split to train on (default: training)"
    )

    parser.add_argument(
        "--consolidation-threshold",
        type=int,
        default=20,
        help="Facts before neural consolidation triggers (default: 20)"
    )

    parser.add_argument(
        "--log-dir",
        default="./arc_training_logs",
        help="Directory for training logs (default: ./arc_training_logs)"
    )

    args = parser.parse_args()

    # Load datasets
    print("=" * 60)
    print("ARC Trainer - Loading Dataset")
    print("=" * 60)

    try:
        print(f"Loading ARC-AGI-2 {args.train_split} set...")
        train_tasks = load_arc_agi_2(split=args.train_split)
        print(f"Loaded {len(train_tasks)} training tasks")

        print(f"Loading validation tasks (limit={args.validation_limit})...")
        eval_tasks = load_arc_agi_2(split="evaluation", limit=args.validation_limit)
        print(f"Loaded {len(eval_tasks)} validation tasks")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download ARC-AGI-2 dataset:")
        print("  git clone https://github.com/fchollet/ARC-AGI data/ARC-AGI-2")
        sys.exit(1)

    # Build curriculum
    print("\nBuilding curriculum...")
    curriculum = ARCCurriculum(train_tasks)
    print(curriculum.summary())

    # Filter by difficulty if requested
    if args.curriculum != "all":
        # Create filtered curriculum with only specified tier
        tier_tasks = curriculum.get_tier(args.curriculum)
        print(f"\nFiltered to {args.curriculum} tier: {len(tier_tasks)} tasks")
        curriculum = ARCCurriculum(tier_tasks)

    # Create trainer
    print("\nInitializing trainer...")
    trainer = ARCTrainer(
        persist_dir=args.persist_dir,
        consolidation_threshold=args.consolidation_threshold,
        log_dir=Path(args.log_dir),
    )

    # Run training
    trainer.run_continuous(
        curriculum=curriculum,
        max_rounds=args.max_rounds,
        validate_every=args.validate_every,
        validation_tasks=eval_tasks,
    )

    print(f"\nTraining log saved to: {trainer.log_file}")
    print(f"Memory persisted to: {args.persist_dir}")


if __name__ == "__main__":
    main()
