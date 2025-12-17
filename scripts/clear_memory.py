#!/usr/bin/env python3
"""
Clear all neural memory, vectors, and learned patterns.

This script completely resets Hologram's memory by deleting:
- Neural network weights (neural_memory.pt)
- Cadence patterns (cadence_memory.pt)
- ChromaDB vector database (chroma.sqlite3)
- Self-improvement patterns (*.json)
- Training logs

Use this to start fresh training from scratch.
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime


def clear_memory(
    persist_dir: str = "./data/crew_training_facts",
    backup: bool = True,
    clear_logs: bool = False
) -> None:
    """
    Clear all neural memory and learned patterns.

    Args:
        persist_dir: Path to persistence directory
        backup: If True, create timestamped backup before deleting
        clear_logs: If True, also clear training logs
    """
    persist_path = Path(persist_dir)

    if not persist_path.exists():
        print(f"Directory does not exist: {persist_path}")
        print("Nothing to clear!")
        return

    print("=" * 60)
    print("CLEAR HOLOGRAM MEMORY")
    print("=" * 60)
    print(f"Target directory: {persist_path}")

    # List what will be deleted
    files_to_delete = list(persist_path.glob("*"))
    if not files_to_delete:
        print("\nDirectory is already empty!")
        return

    print(f"\nFiles to delete ({len(files_to_delete)}):")
    total_size = 0
    for file in files_to_delete:
        if file.is_file():
            size = file.stat().st_size
            total_size += size
            size_mb = size / (1024 * 1024)
            print(f"  - {file.name:<30} {size_mb:>8.2f} MB")

    print(f"\nTotal size: {total_size / (1024 * 1024):.2f} MB")

    # Confirmation
    print("\n⚠️  WARNING: This will delete all learned memory and patterns!")
    print("Hologram will start completely fresh, like a new installation.")
    response = input("\nAre you sure you want to continue? (yes/N): ")

    if response.lower() != 'yes':
        print("Aborted.")
        return

    # Create backup if requested
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = persist_path.parent / f"{persist_path.name}_backup_{timestamp}"

        print(f"\nCreating backup to: {backup_dir}")
        shutil.copytree(persist_path, backup_dir)
        print("✓ Backup created")

    # Delete all files
    print(f"\nDeleting files...")
    for file in files_to_delete:
        if file.is_file():
            file.unlink()
            print(f"  ✓ Deleted {file.name}")
        elif file.is_dir():
            shutil.rmtree(file)
            print(f"  ✓ Deleted directory {file.name}")

    print("✓ All memory files deleted")

    # Optionally clear training logs
    if clear_logs:
        log_dirs = [
            "./arc_training_logs",
            "./code_training_logs",
            "./conversation_logs"
        ]

        print("\nClearing training logs...")
        for log_dir in log_dirs:
            log_path = Path(log_dir)
            if log_path.exists():
                shutil.rmtree(log_path)
                print(f"  ✓ Deleted {log_dir}")

    print("\n" + "=" * 60)
    print("MEMORY CLEARED SUCCESSFULLY!")
    print("=" * 60)
    print("\nHologram's memory has been reset.")
    print("Next training session will start fresh.")

    if backup:
        print(f"\nBackup saved to: {backup_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Clear all neural memory, vectors, and learned patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clear with backup (safe)
  python scripts/clear_memory.py

  # Clear specific directory
  python scripts/clear_memory.py --persist-dir ./data/my_facts

  # Clear without backup (faster, irreversible)
  python scripts/clear_memory.py --no-backup

  # Clear memory AND training logs
  python scripts/clear_memory.py --clear-logs
        """
    )

    parser.add_argument(
        "--persist-dir",
        default="./data/crew_training_facts",
        help="Path to persistence directory (default: ./data/crew_training_facts)"
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create a backup before deleting (DANGEROUS)"
    )

    parser.add_argument(
        "--clear-logs",
        action="store_true",
        help="Also delete training logs (arc_training_logs, code_training_logs, conversation_logs)"
    )

    args = parser.parse_args()

    clear_memory(
        persist_dir=args.persist_dir,
        backup=not args.no_backup,
        clear_logs=args.clear_logs
    )


if __name__ == "__main__":
    main()
