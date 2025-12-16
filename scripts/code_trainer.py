#!/usr/bin/env python3
"""
Code Trainer: Train Hologram's shared memory on SWE-bench patterns.

Uses the SAME shared memory as arc_trainer.py - code patterns and ARC
patterns reinforce each other in the unified holographic memory.

Usage:
    # Train on sample tasks
    python scripts/code_trainer.py --max-rounds 5

    # Use same memory as arc_trainer (default)
    python scripts/code_trainer.py --persist-dir ./data/crew_training_facts

    # Generate patches for a task
    python scripts/code_trainer.py --mode generate --issue "Fix null pointer"
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hologram.container import HologramContainer
from hologram.consolidation.neural_memory import NeuralMemory, ConsolidationFact
from hologram.arc.transform_resonator import TransformationResonator
from hologram.arc.encoder import ObjectEncoder
from hologram.introspection import SelfImprovementManager
from hologram.swe import (
    SWETask,
    CodePatch,
    PatchResult,
    CodeEncoder,
    CodeResonator,
    CodeGenerator,
)


# =============================================================================
# Sample Tasks (for testing without SWE-bench dataset)
# =============================================================================

SAMPLE_TASKS = [
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


# =============================================================================
# Training Statistics
# =============================================================================

@dataclass
class TrainingStats:
    """Statistics for tracking training progress."""
    rounds_completed: int = 0
    tasks_attempted: int = 0
    tasks_learned: int = 0
    patterns_stored: int = 0
    generation_attempts: int = 0
    generation_successes: int = 0

    def __str__(self) -> str:
        learn_rate = self.tasks_learned / max(1, self.tasks_attempted) * 100
        gen_rate = self.generation_successes / max(1, self.generation_attempts) * 100
        return (
            f"Rounds: {self.rounds_completed}, "
            f"Tasks: {self.tasks_attempted}, "
            f"Learned: {self.tasks_learned} ({learn_rate:.1f}%), "
            f"Patterns: {self.patterns_stored}, "
            f"Gen success: {self.generation_successes}/{self.generation_attempts} ({gen_rate:.1f}%)"
        )


# =============================================================================
# Main Trainer
# =============================================================================

class CodeTrainer:
    """
    Train Hologram on code patterns using shared memory.

    Uses the SAME ConsolidationManager as ARCTrainer, enabling
    cross-domain learning between ARC patterns and code patterns.

    Args:
        persist_dir: Shared persistence directory
        dimensions: HDC vector dimensions
        consolidation_threshold: Facts before neural consolidation
        enable_self_improvement: Enable circuit observer
    """

    def __init__(
        self,
        persist_dir: str = "./data/crew_training_facts",
        dimensions: int = 10000,
        consolidation_threshold: int = 20,
        log_dir: Path = Path("./code_training_logs"),
        enable_self_improvement: bool = True,
    ):
        self.persist_dir = persist_dir
        self.dimensions = dimensions
        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"code_training_{timestamp}.log"

        self._log("=" * 60)
        self._log("Code Trainer Initialization")
        self._log("=" * 60)

        # Initialize container (shared components)
        self._log("Creating HologramContainer...")
        self.container = HologramContainer(dimensions=dimensions)

        # Create ConsolidationManager (SHARED with arc_trainer)
        self._log("Creating ConsolidationManager (shared)...")
        self._consolidation_manager = self.container.create_consolidation_manager(
            threshold=consolidation_threshold
        )

        # Load existing memory if available
        self._load_existing_memory()

        # Create code-specific components
        self._log("Creating code components...")
        self._code_encoder = CodeEncoder(
            self.container._space,  # FractalSpace
            self.container._codebook,
        )

        # Create ARC encoder for resonator (reuses existing vocabulary)
        self._arc_encoder = ObjectEncoder(
            self.container._space,
            self.container._codebook,
        )

        # Create transformation resonator
        self._resonator = TransformationResonator(
            self._arc_encoder,
            self.container._codebook,
        )

        # Create code resonator (wraps transformation resonator)
        self._code_resonator = CodeResonator(
            self._code_encoder,
            self._resonator,
        )

        # Get neural memory FROM consolidation manager (shared instance)
        self._neural_memory = self._consolidation_manager._neural_memory

        # Create self-improvement manager
        self._self_improvement = None
        if enable_self_improvement:
            self_improvement_path = str(Path(persist_dir) / "code_learned_patterns.json")
            self._self_improvement = SelfImprovementManager(persist_path=self_improvement_path)
            self._log(f"Self-improvement enabled: {self_improvement_path}")

        # Create code generator using container factory
        self._generator = self.container.create_code_generator(
            fact_store=None,  # Will create if needed
            neural_memory=self._neural_memory,
            enable_dependency_graph=True,
            circuit_observer=self._self_improvement.observer if self._self_improvement else None,
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
                vocab_size = self._consolidation_manager.vocab_size
                self._log(f"Loaded existing memory: {vocab_size} patterns")
            except Exception as e:
                self._log(f"Warning: Could not load existing memory: {e}")

    def train_on_task(self, task: SWETask) -> bool:
        """
        Learn from a single SWE task.

        Args:
            task: SWE task with ground truth

        Returns:
            True if pattern was learned
        """
        self._stats.tasks_attempted += 1
        self._log(f"  Training on task: {task.task_id}")

        # Try to learn from ground truth
        success = self._generator.learn_from_task(task)

        if success:
            self._stats.tasks_learned += 1
            self._stats.patterns_stored += 1
            self._log(f"    Learned pattern from {task.task_id}")
        else:
            self._log(f"    Could not extract pattern from {task.task_id}")

        return success

    def generate_for_task(self, task: SWETask) -> PatchResult:
        """
        Generate patches for a task.

        Args:
            task: SWE task

        Returns:
            PatchResult with generated patches
        """
        self._stats.generation_attempts += 1

        result = self._generator.generate(task)

        if result.verification_passed:
            self._stats.generation_successes += 1
            self._log(f"  Generated verified patch: {result.patches[0] if result.patches else 'none'}")
        else:
            self._log(f"  Generation failed verification (confidence={result.confidence:.2f})")

        return result

    def train_round(self, tasks: List[SWETask]) -> int:
        """
        Train on a list of tasks.

        Args:
            tasks: List of SWE tasks

        Returns:
            Number of successfully learned tasks
        """
        success_count = 0

        for task in tasks:
            if not self.running:
                break

            if self.train_on_task(task):
                success_count += 1

        return success_count

    def run_continuous(
        self,
        tasks: List[SWETask],
        max_rounds: int = 10,
        validate_every: int = 5,
    ):
        """
        Run continuous training loop.

        Args:
            tasks: Tasks to train on
            max_rounds: Maximum training rounds
            validate_every: Validate every N rounds
        """
        self._log("\n" + "=" * 60)
        self._log("Starting Continuous Training")
        self._log("=" * 60)
        self._log(f"Tasks: {len(tasks)}, Max rounds: {max_rounds}")

        try:
            for round_num in range(max_rounds):
                if not self.running:
                    break

                self._stats.rounds_completed = round_num + 1
                self._log(f"\n{'='*60}")
                self._log(f"Round {round_num + 1}/{max_rounds}")
                self._log(f"{'='*60}")

                # Train on all tasks
                success = self.train_round(tasks)
                self._log(f"  Learned {success}/{len(tasks)} tasks")

                # Periodic validation
                if (round_num + 1) % validate_every == 0:
                    self._validate(tasks[:3])  # Validate on first 3 tasks

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
            self.save_memory()

            if self._self_improvement:
                self._self_improvement.save()
                self._log("Saved self-improvement patterns")

            # Final report
            self._log("\n" + "=" * 60)
            self._log("Final Training Report")
            self._log("=" * 60)
            self._log(str(self._stats))

    def _validate(self, tasks: List[SWETask]):
        """Run validation on tasks."""
        self._log("\n--- Validation ---")

        for task in tasks:
            result = self.generate_for_task(task)
            status = "PASS" if result.verification_passed else "FAIL"
            self._log(f"  {task.task_id}: {status} (conf={result.confidence:.2f})")

    def save_memory(self):
        """Save neural memory to persistence directory."""
        try:
            neural_path = Path(self.persist_dir) / "neural_memory.pt"
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

            # Save consolidation manager state (includes neural memory)
            state = self._consolidation_manager.state_dict()
            torch.save(state, neural_path)

            self._log(f"Saved memory to {neural_path}")
        except Exception as e:
            self._log(f"Error saving memory: {e}")


# =============================================================================
# Training Data Loading
# =============================================================================

def load_training_data(data_path: str) -> List[SWETask]:
    """
    Load training data from JSON file (generated by swe_data_extractor.py).

    Args:
        data_path: Path to training data JSON file

    Returns:
        List of SWETask instances
    """
    import json

    with open(data_path) as f:
        data = json.load(f)

    tasks = []

    # Load from patterns section
    for pattern in data.get("patterns", []):
        task = SWETask(
            task_id=f"pattern_{pattern['pattern_id']}",
            repo="synthetic/patterns",
            issue_text=pattern.get("example_issue", ""),
            code_before={"example.py": "# Placeholder code"},
            code_after={"example.py": pattern.get("example_fix", "")},
        )
        tasks.append(task)

    # Load from examples section
    for example in data.get("examples", []):
        task = SWETask(
            task_id=example.get("task_id", "unknown"),
            repo=example.get("repo", "unknown"),
            issue_text=example.get("issue_text", ""),
            code_before={example.get("file_path", "file.py"): example.get("code_before", "")},
            code_after={example.get("file_path", "file.py"): example.get("code_after", "")},
        )
        tasks.append(task)

    return tasks


def web_teach_code_patterns(
    topics: List[str],
    max_results_per_topic: int = 3,
) -> List[SWETask]:
    """
    Search web for code patterns and create training tasks.

    Like crew_trainer's WebTeacher but for code patterns.

    Args:
        topics: List of topics to search (e.g., ["null check python", "input validation"])
        max_results_per_topic: Web results per topic

    Returns:
        List of SWETask instances created from web patterns
    """
    # Try to import web search
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            print("Web search unavailable. Install with: pip install ddgs")
            return []

    # Try to import LLM for extraction
    try:
        from langchain_anthropic import ChatAnthropic
        import os

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("ANTHROPIC_API_KEY not set. Using simple extraction.")
            llm = None
        else:
            llm = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                api_key=api_key,
                temperature=0.3,
            )
    except ImportError:
        print("LangChain not available. Using simple extraction.")
        llm = None

    tasks = []

    for topic in topics:
        print(f"  Searching: {topic}")

        try:
            ddgs = DDGS()
            search_query = f"{topic} code fix example python"
            results = list(ddgs.text(search_query, max_results=max_results_per_topic))
        except Exception as e:
            print(f"    Search failed: {e}")
            continue

        print(f"    Found {len(results)} results")

        for i, result in enumerate(results):
            title = result.get("title", "")[:40]
            body = result.get("body", "")

            # Extract pattern from result
            if llm:
                # Use LLM for extraction
                pattern = extract_pattern_with_llm(llm, topic, title, body)
            else:
                # Simple extraction
                pattern = extract_pattern_simple(topic, title, body)

            if pattern:
                task = SWETask(
                    task_id=f"web_{topic.replace(' ', '_')}_{i}",
                    repo="web/search",
                    issue_text=pattern["issue"],
                    code_before={"example.py": pattern.get("before", "# Original code")},
                    code_after={"example.py": pattern.get("after", "# Fixed code")},
                )
                tasks.append(task)
                print(f"      [{i+1}] Extracted: {pattern['issue'][:50]}...")

    return tasks


def extract_pattern_with_llm(llm, topic: str, title: str, body: str) -> dict:
    """Extract code pattern using LLM."""
    import json

    prompt = f"""Extract a code fix pattern from this web search result about "{topic}".

Title: {title}
Content: {body[:1500]}

Return JSON:
{{
  "issue": "Brief description of the issue being fixed",
  "before": "Code before the fix (1-3 lines)",
  "after": "Code after the fix (1-5 lines)"
}}

If no clear code pattern is found, return {{"issue": "", "before": "", "after": ""}}

JSON:"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)
        if data.get("issue"):
            return data
    except Exception as e:
        print(f"      LLM extraction failed: {e}")

    return None


def extract_pattern_simple(topic: str, title: str, body: str) -> dict:
    """Extract code pattern using simple heuristics."""
    import re

    # Look for code snippets in the body
    code_pattern = re.search(r'```python\s*(.*?)\s*```', body, re.DOTALL)
    if not code_pattern:
        code_pattern = re.search(r'def \w+\([^)]*\):[^\n]*\n\s+[^\n]+', body)

    if code_pattern:
        code = code_pattern.group(1) if '```' in body else code_pattern.group(0)
        return {
            "issue": f"Fix {topic} - {title[:50]}",
            "before": "# Original code",
            "after": code[:200],
        }

    # Fallback: create from topic
    return {
        "issue": f"Fix: {topic}",
        "before": "# Code with issue",
        "after": f"# Fixed: {topic}",
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Code Trainer - Train Hologram on SWE-bench patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on sample tasks
  python scripts/code_trainer.py --max-rounds 5

  # Use same memory as arc_trainer
  python scripts/code_trainer.py --persist-dir ./data/crew_training_facts

  # Generate patches for an issue
  python scripts/code_trainer.py --mode generate --issue "Fix null pointer"

  # Load training data from file (generated by swe_data_extractor.py)
  python scripts/code_trainer.py --data data/swe_training.json --max-rounds 10

  # Web search for code patterns before training
  python scripts/code_trainer.py --web-teach "null check python" "input validation" --max-rounds 5
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
        default=5,
        help="Number of training rounds (default: 5)"
    )

    parser.add_argument(
        "--validate-every",
        type=int,
        default=2,
        help="Run validation every N rounds (default: 2)"
    )

    parser.add_argument(
        "--mode",
        choices=["train", "generate"],
        default="train",
        help="Mode: train or generate (default: train)"
    )

    parser.add_argument(
        "--issue",
        type=str,
        default=None,
        help="Issue text for generate mode"
    )

    parser.add_argument(
        "--log-dir",
        default="./code_training_logs",
        help="Directory for training logs (default: ./code_training_logs)"
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Load training data from JSON file (generated by swe_data_extractor.py)"
    )

    parser.add_argument(
        "--web-teach",
        nargs="+",
        metavar="TOPIC",
        help="Search web for code patterns before training (e.g., --web-teach 'null check' 'validation')"
    )

    parser.add_argument(
        "--web-results",
        type=int,
        default=3,
        help="Number of web search results per topic (default: 3)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Code Trainer - SWE-bench Pattern Learning")
    print("=" * 60)

    # Create trainer
    trainer = CodeTrainer(
        persist_dir=args.persist_dir,
        log_dir=Path(args.log_dir),
    )

    # Load training data if specified
    tasks = SAMPLE_TASKS

    if args.data:
        print(f"\nLoading training data from {args.data}...")
        tasks = load_training_data(args.data)
        print(f"Loaded {len(tasks)} tasks from file")

    # Web teach mode - search for patterns before training
    if args.web_teach:
        print(f"\nWeb teaching mode: searching for patterns...")
        web_tasks = web_teach_code_patterns(
            topics=args.web_teach,
            max_results_per_topic=args.web_results,
        )
        tasks = web_tasks + tasks
        print(f"Added {len(web_tasks)} web-sourced tasks")

    if args.mode == "train":
        # Train on tasks
        print(f"\nUsing {len(tasks)} tasks for training")
        trainer.run_continuous(
            tasks=tasks,
            max_rounds=args.max_rounds,
            validate_every=args.validate_every,
        )
    else:
        # Generate mode
        if args.issue is None:
            print("Error: --issue is required for generate mode")
            sys.exit(1)

        task = SWETask(
            task_id="user_task",
            repo="user/repo",
            issue_text=args.issue,
            code_before={"main.py": "# Your code here"},
            code_after={},
        )

        result = trainer.generate_for_task(task)
        print(f"\nGenerated patches:")
        for patch in result.patches:
            print(f"  {patch}")
        print(f"\nConfidence: {result.confidence:.2f}")
        print(f"Verification: {'PASSED' if result.verification_passed else 'FAILED'}")

    print(f"\nTraining log saved to: {trainer.log_file}")
    print(f"Memory persisted to: {args.persist_dir}")


if __name__ == "__main__":
    main()
