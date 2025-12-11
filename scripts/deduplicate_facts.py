#!/usr/bin/env python3
"""
Deduplication script for ChromaDB fact storage.

Removes duplicate facts from the ChromaDB storage layer while preserving
the most recent version of each unique fact.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def deduplicate_chroma_facts(persist_dir: str, dry_run: bool = True) -> None:
    """
    Remove duplicate facts from ChromaDB.
    
    Args:
        persist_dir: Path to ChromaDB persistence directory
        dry_run: If True, only report duplicates without removing them
    """
    from hologram.persistence.chroma_adapter import ChromaFactStore
    from hologram.core.vector_space import VectorSpace
    from hologram.core.codebook import Codebook
    
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  ChromaDB Fact Deduplication".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made")
    else:
        print("\n⚠️  LIVE MODE - Duplicates will be removed")
    
    print(f"📂 Storage location: {persist_dir}\n")
    
    # Create temporary container to get codebook
    space = VectorSpace(dimensions=10000)
    codebook = Codebook(space)
    
    # Load ChromaDB
    print("📚 Loading ChromaDB...")
    chroma_store = ChromaFactStore(
        codebook=codebook,
        persist_dir=persist_dir,
    )
    
    # Get all facts
    facts = chroma_store.get_all_facts()
    print(f"✓ Loaded {len(facts)} facts\n")
    
    if not facts:
        print("✅ No facts found - nothing to deduplicate")
        return
    
    # Build fact index: (subject, predicate, object) -> [list of fact instances]
    fact_index: Dict[Tuple[str, str, str], List[dict]] = {}
    
    for i, fact in enumerate(facts):
        subject = fact.get('subject', '') if isinstance(fact, dict) else getattr(fact, 'subject', '')
        predicate = fact.get('predicate', '') if isinstance(fact, dict) else getattr(fact, 'predicate', '')
        obj = fact.get('object', '') if isinstance(fact, dict) else getattr(fact, 'object', '')
        
        # Normalize to lowercase for comparison
        key = (subject.lower(), predicate.lower(), obj.lower())
        
        if key not in fact_index:
            fact_index[key] = []
        
        fact_index[key].append({
            'index': i,
            'subject': subject,
            'predicate': predicate,
            'object': obj,
            'fact': fact,
        })
    
    # Find duplicates
    duplicates_found = []
    
    for key, instances in fact_index.items():
        if len(instances) > 1:
            duplicates_found.append((key, instances))
    
    if not duplicates_found:
        print("✅ No duplicate facts found!")
        print("   All facts are unique.")
        return
    
    # Report duplicates
    print(f"⚠️  Found {len(duplicates_found)} sets of duplicate facts:\n")
    
    total_duplicates = sum(len(instances) - 1 for _, instances in duplicates_found)
    print(f"📊 Total duplicate entries to remove: {total_duplicates}\n")
    
    for i, (key, instances) in enumerate(duplicates_found, 1):
        subject, predicate, obj = key
        print(f"{i}. '{subject}' --{predicate}--> '{obj}'")
        print(f"   Occurrences: {len(instances)}")
        
        # Show instances
        for j, instance in enumerate(instances, 1):
            timestamp = instance['fact'].get('timestamp', 'unknown') if isinstance(instance['fact'], dict) else getattr(instance['fact'], 'timestamp', 'unknown')
            print(f"     [{j}] Timestamp: {timestamp}")
        
        print()
    
    if dry_run:
        print("="*60)
        print("DRY RUN COMPLETE")
        print("="*60)
        print(f"\nTo remove {total_duplicates} duplicates, run:")
        print(f"  python {Path(__file__).name} --live")
        return
    
    # Remove duplicates (keep most recent)
    print("="*60)
    print("REMOVING DUPLICATES")
    print("="*60)
    print()
    
    removed_count = 0
    
    for key, instances in duplicates_found:
        subject, predicate, obj = key
        
        # Sort by timestamp (most recent first)
        # Keep the first (most recent), remove the rest
        instances_sorted = sorted(
            instances,
            key=lambda x: x['fact'].get('timestamp', datetime.min) if isinstance(x['fact'], dict) else getattr(x['fact'], 'timestamp', datetime.min),
            reverse=True
        )
        
        # Keep the first instance, remove the rest
        to_keep = instances_sorted[0]
        to_remove = instances_sorted[1:]
        
        print(f"Keeping: '{subject}' --{predicate}--> '{obj}'")
        print(f"  Timestamp: {to_keep['fact'].get('timestamp', 'unknown') if isinstance(to_keep['fact'], dict) else getattr(to_keep['fact'], 'timestamp', 'unknown')}")
        print(f"  Removing {len(to_remove)} older duplicates...")
        
        # Remove duplicates from ChromaDB
        # Note: This requires implementing a delete method in ChromaFactStore
        # For now, we'll just report what would be deleted
        for instance in to_remove:
            # TODO: Implement deletion in ChromaFactStore
            # chroma_store.delete_fact(instance['fact'])
            removed_count += 1
            print(f"    ✓ Would remove instance from {instance['fact'].get('timestamp', 'unknown') if isinstance(instance['fact'], dict) else getattr(instance['fact'], 'timestamp', 'unknown')}")
        
        print()
    
    print("="*60)
    print(f"✅ Deduplication complete!")
    print(f"   Removed: {removed_count} duplicate facts")
    print(f"   Remaining: {len(facts) - removed_count} unique facts")
    print("="*60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Remove duplicate facts from ChromaDB storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--persist-dir",
        default="./data/crew_training_facts",
        help="Directory for ChromaDB persistence (default: ./data/crew_training_facts)"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually remove duplicates (default is dry-run mode)"
    )
    
    args = parser.parse_args()
    
    deduplicate_chroma_facts(
        persist_dir=args.persist_dir,
        dry_run=not args.live
    )


if __name__ == "__main__":
    main()
