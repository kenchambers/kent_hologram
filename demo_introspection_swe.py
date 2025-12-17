#!/usr/bin/env python3
"""
Quick Demo: Introspection & SWE Features

Run this to see Kent's self-improvement and code generation in action.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hologram import HologramContainer
from hologram.introspection import SelfImprovementManager
from hologram.swe import SWETask


def demo_introspection():
    """Demo: Self-Improvement System"""
    print("\n" + "="*70)
    print("DEMO 1: INTROSPECTION - Self-Improvement System")
    print("="*70)

    print("\n📊 Creating SelfImprovementManager...")
    manager = SelfImprovementManager(
        persist_path="./data/demo_patterns.json",
        auto_save_interval=0  # Manual save for demo
    )

    print(f"✓ Manager initialized")
    print(f"  Observer: {manager.observer}")
    print(f"  Analyzer: {manager.analyzer}")

    print("\n📝 Simulating observations...")
    # Simulate some successes and failures
    observations = [
        (["rotate_90", "LARGEST", "clockwise"], True, 0.92),
        (["rotate_90", "LARGEST", "clockwise"], True, 0.88),
        (["flip", "ALL", "horizontal"], True, 0.85),
        (["tile", "SMALLEST", "vertical"], False, 0.15),
        (["tile", "SMALLEST", "vertical"], False, 0.12),
        (["rotate_90", "LARGEST", "clockwise"], True, 0.91),
    ]

    for items, success, confidence in observations:
        manager.observer.observe(items, success, confidence, context="demo")
        print(f"  {'✓' if success else '✗'} {items} (conf: {confidence:.2f})")

    print("\n📈 Getting improvement report...")
    report = manager.get_improvement_report()
    print(report)

    print("\n💾 Getting statistics...")
    stats = manager.get_statistics()
    print(f"  Total observations: {stats['total_observations']}")
    print(f"  Unique items: {stats['unique_items']}")
    print(f"  Items to reinforce: {stats['items_to_reinforce']}")
    print(f"  Items to prune: {stats['items_to_prune']}")

    if stats['top_performers']:
        print(f"\n  🏆 Top performer: {stats['top_performers'][0]}")

    print("\n💾 Saving patterns...")
    manager.save()
    print(f"✓ Saved to: {manager._persist_path}")

    print("\n✅ Introspection demo complete!")
    print("   The system learned from observations and can now:")
    print("   - Reinforce successful patterns")
    print("   - Prune failing patterns")
    print("   - Persist knowledge across sessions")


def demo_swe():
    """Demo: Software Engineering Code Generation"""
    print("\n" + "="*70)
    print("DEMO 2: SWE - Code Generation")
    print("="*70)

    print("\n🏗️  Initializing code generator...")
    container = HologramContainer(dimensions=10000)

    # Create self-improvement manager for code generator
    from hologram.introspection import SelfImprovementManager
    swe_manager = SelfImprovementManager(
        persist_path="./data/demo_swe_patterns.json"
    )

    # Create code generator with circuit observer
    generator = container.create_code_generator(
        fact_store=None,
        circuit_observer=swe_manager.observer
    )

    print("✓ Code generator ready")

    print("\n📝 Creating sample coding task...")
    task = SWETask(
        task_id="demo_validation",
        repo="demo/project",
        issue_text="Add input validation to the process() function to check if data is an integer",
        code_before={
            "main.py": """def process(data):
    result = data * 2
    return result
"""
        },
        code_after={
            "main.py": """def process(data):
    if not isinstance(data, int):
        raise ValueError("data must be an integer")
    result = data * 2
    return result
"""
        }
    )

    print(f"  Task: {task.issue_text}")
    print(f"  Files: {list(task.code_before.keys())}")

    print("\n🤖 Generating patches...")
    result = generator.generate(task, max_patches=5, confidence_threshold=0.3)

    print(f"\n✓ Generated {len(result.patches)} patch(es)")
    print(f"  Confidence: {result.confidence:.2f}")

    if result.patches:
        print("\n📋 Patches:")
        for i, patch in enumerate(result.patches, 1):
            print(f"\n  Patch {i}:")
            print(f"    File: {patch.file}")
            print(f"    Operation: {patch.operation}")
            print(f"    Location: {patch.location}")
            print(f"    Content: {patch.content[:80]}...")

    print("\n✅ SWE demo complete!")
    print("   The code generator:")
    print("   - Parsed natural language issue")
    print("   - Encoded code context")
    print("   - Generated appropriate patches")
    print("   - Verified against HDC encoding")


def demo_arc_with_introspection():
    """Demo: ARC Solver with Self-Improvement"""
    print("\n" + "="*70)
    print("DEMO 3: ARC Solver + Introspection")
    print("="*70)

    from hologram.arc.solver import HolographicARCSolver
    from hologram.arc.types import ARCTask, Grid, TrainingPair

    print("\n🧠 Creating ARC solver with self-improvement...")
    solver = HolographicARCSolver(
        dimensions=10000,
        iterative=True,
        enable_self_improvement=True,
        self_improvement_path="./data/demo_arc_patterns.json"
    )

    print("✓ Solver ready with introspection enabled")

    print("\n🎯 Creating simple ARC task (rotation)...")
    # Simple rotation task
    task = ARCTask(
        task_id="demo_rotate",
        training=[
            TrainingPair(
                input=Grid.from_list([[1, 0], [0, 0]]),
                output=Grid.from_list([[0, 1], [0, 0]])
            )
        ],
        test_input=Grid.from_list([[0, 1], [0, 0]]),
        test_output=Grid.from_list([[0, 0], [1, 0]])
    )

    print("  Task: Rotate pattern")
    print("  Training pairs: 1")

    print("\n🔍 Solving with self-improvement tracking...")
    result = solver.solve(task)

    print(f"\n✓ Result:")
    print(f"  Solved: {result.from_cache or (result.output is not None)}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Message: {result.message}")

    print("\n📈 Checking what was learned...")
    stats = solver.get_improvement_stats()

    if stats:
        print(f"  Total observations: {stats.get('total_observations', 0)}")
        print(f"  Unique items: {stats.get('unique_items', 0)}")

        if stats.get('top_performers'):
            print(f"\n  🏆 Top performers:")
            for perf in stats['top_performers'][:3]:
                print(f"    - {perf}")

    print("\n✅ ARC + Introspection demo complete!")
    print("   The solver:")
    print("   - Attempted to solve the task")
    print("   - Tracked which transformations worked")
    print("   - Saved patterns for future use")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("KENT HOLOGRAM - Introspection & SWE Demo")
    print("="*70)
    print("\nThis demo showcases:")
    print("1. Self-Improvement System (Introspection)")
    print("2. Code Generation (SWE)")
    print("3. ARC Solver with Learning")

    try:
        # Demo 1: Introspection
        demo_introspection()

        # Demo 2: SWE
        demo_swe()

        # Demo 3: ARC with introspection
        demo_arc_with_introspection()

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "="*70)
    print("ALL DEMOS COMPLETE")
    print("="*70)
    print("\n✨ Next Steps:")
    print("  1. Read INTROSPECTION_AND_SWE_GUIDE.md for full documentation")
    print("  2. Try interactive chat: uv run hologram")
    print("  3. Use /code command for code generation")
    print("  4. Run scripts/arc_trainer.py for real training")
    print("  5. Check data/ folder for learned patterns")
    print("\n" + "="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
