#!/usr/bin/env python3
"""
Recovery script for corrupted ChromaDB databases.

This script can be used to manually recover from ChromaDB corruption
by backing up and recreating the database.
"""

import argparse
import shutil
from pathlib import Path


def recover_chromadb(persist_dir: str, backup: bool = True) -> None:
    """
    Recover from corrupted ChromaDB database.

    Args:
        persist_dir: Path to ChromaDB persistence directory
        backup: If True, create a backup before deleting
    """
    persist_path = Path(persist_dir)

    if not persist_path.exists():
        print(f"Directory does not exist: {persist_path}")
        return

    print(f"Recovering ChromaDB database at: {persist_path}")

    if backup:
        backup_dir = persist_path.parent / f"{persist_path.name}_backup"
        if backup_dir.exists():
            print(f"Backup directory already exists: {backup_dir}")
            response = input("Delete existing backup and create new one? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
            shutil.rmtree(backup_dir)

        print(f"Creating backup to: {backup_dir}")
        shutil.copytree(persist_path, backup_dir)
        print("✓ Backup created")

    # Delete corrupted database
    print(f"Deleting corrupted database...")
    shutil.rmtree(persist_path)
    print("✓ Corrupted database deleted")

    # Recreate directory
    persist_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Recreated directory: {persist_path}")

    print("\nRecovery complete! You can now run 'uv run hologram' again.")
    print("Note: All facts stored in the database have been lost.")


def main():
    parser = argparse.ArgumentParser(
        description="Recover from corrupted ChromaDB database"
    )
    parser.add_argument(
        "persist_dir",
        nargs="?",
        default="./data/hologram_facts",
        help="Path to ChromaDB persistence directory (default: ./data/hologram_facts)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create a backup before deleting",
    )

    args = parser.parse_args()

    recover_chromadb(args.persist_dir, backup=not args.no_backup)


if __name__ == "__main__":
    main()



