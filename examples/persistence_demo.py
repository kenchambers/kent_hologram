#!/usr/bin/env python3
"""
Persistence demonstration for Hologram.

Shows how to:
1. Create and populate a FactStore
2. Save to disk
3. Load from disk
4. Verify state is preserved
"""

from pathlib import Path

from hologram.container import HologramContainer
from hologram.persistence.state_manager import StateManager


def main():
    print("=" * 60)
    print("Persistence Demo: Save and Load Holographic Memory")
    print("=" * 60)
    print()

    # Setup save path
    save_path = Path("./data/demo_save")

    # ======================================================================
    # PHASE 1: Create and populate memory
    # ======================================================================
    print("Phase 1: Creating holographic memory...")
    container = HologramContainer(dimensions=10000)
    fact_store = container.create_fact_store()

    # Add facts about countries and capitals
    facts = [
        ("France", "capital", "Paris", "Wikipedia"),
        ("Germany", "capital", "Berlin", "Wikipedia"),
        ("Japan", "capital", "Tokyo", "Wikipedia"),
        ("Brazil", "capital", "Brasilia", "Wikipedia"),
        ("Australia", "capital", "Canberra", "Wikipedia"),
    ]

    print("Storing facts...")
    for subject, predicate, obj, source in facts:
        fact_store.add_fact(subject, predicate, obj, source=source)
        print(f"  ✓ {subject} --{predicate}--> {obj}")

    print(f"\n{fact_store}")
    print()

    # Test query before save
    print("Testing query before save...")
    answer, conf = fact_store.query("France", "capital")
    print(f"  Query: France --capital--> {answer} (confidence: {conf:.3f})")
    print()

    # ======================================================================
    # PHASE 2: Save to disk
    # ======================================================================
    print("Phase 2: Saving to disk...")
    manager = StateManager()
    manager.save(
        fact_store,
        save_path,
        description="Demo of country capitals"
    )
    print(f"  ✓ Saved to: {save_path}")
    print()

    # Show what was saved
    print("Files created:")
    for item in sorted(save_path.rglob("*")):
        if item.is_file():
            size = item.stat().st_size
            rel_path = item.relative_to(save_path)
            print(f"  - {rel_path} ({size:,} bytes)")
    print()

    # ======================================================================
    # PHASE 3: Clear memory (simulate new session)
    # ======================================================================
    print("Phase 3: Clearing memory (simulating new session)...")
    del fact_store
    del container
    print("  ✓ Memory cleared")
    print()

    # ======================================================================
    # PHASE 4: Load from disk
    # ======================================================================
    print("Phase 4: Loading from disk...")
    restored_fs = manager.load(save_path, validate_checksum=True)
    print(f"  ✓ Loaded: {restored_fs}")
    print()

    # ======================================================================
    # PHASE 5: Verify state preserved
    # ======================================================================
    print("Phase 5: Verifying state preserved...")

    # Test all facts are retrievable
    test_queries = [
        ("France", "capital", "Paris"),
        ("Germany", "capital", "Berlin"),
        ("Japan", "capital", "Tokyo"),
    ]

    all_correct = True
    for subject, predicate, expected in test_queries:
        answer, conf = restored_fs.query(subject, predicate)
        status = "✓" if answer == expected else "✗"
        print(f"  {status} {subject} --{predicate}--> {answer} (expected: {expected})")
        if answer != expected:
            all_correct = False

    print()
    if all_correct:
        print("✅ SUCCESS: All facts preserved across save/load cycle!")
    else:
        print("❌ FAILURE: Some facts were corrupted!")

    print()

    # ======================================================================
    # PHASE 6: List available saves
    # ======================================================================
    print("Phase 6: Listing available saves...")
    saves = manager.list_saves(Path("./data"))
    for i, save in enumerate(saves, 1):
        print(f"\n  Save #{i}:")
        print(f"    Path: {save['path']}")
        print(f"    Description: {save.get('description', 'N/A')}")
        print(f"    Saved at: {save['saved_at']}")
        print(f"    Facts: {save['stats']['fact_count']}")
        print(f"    Saturation: {save['stats']['saturation']:.2%}")

    print()
    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
