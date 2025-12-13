#!/usr/bin/env python3
"""
Semantic code search using EmbeddixDB

Usage: python search_code.py "your search query"

Returns absolute paths for compatibility with Claude Code Read tool.
"""

import sys
import os
from pathlib import Path

# Add embeddix/src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'embeddix', 'src'))

from novita_integration import TradingDocumentManager


def search_code(query: str, limit: int = 5):
    """Search the indexed codebase"""

    # Get project root for absolute paths
    project_root = Path(__file__).parent.absolute()

    # Initialize manager
    manager = TradingDocumentManager(
        collection_name="kent_hologram_code",
        embeddix_port=8082
    )

    print(f"\nSearching for: '{query}'\n")
    print("=" * 80)

    results = manager.search(query, limit=limit)

    if not results:
        print("No results found.")
        print("\nMake sure:")
        print("  1. EmbeddixDB server is running: ./embeddix/scripts/status.sh")
        print("  2. Codebase is indexed: PYTHONPATH=./embeddix/src python embeddix/src/index_codebase.py")
        return

    for i, result in enumerate(results, 1):
        score = result.get('score', 0)
        metadata = result.get('metadata', {})

        file_path = metadata.get('file_path', 'Unknown')
        file_type = metadata.get('file_type', 'Unknown')
        block_type = metadata.get('block_type', '')
        name = metadata.get('name', '')

        # Convert to absolute path (Claude Code needs absolute paths)
        abs_path = project_root / file_path

        print(f"\n[Result {i}] Score: {score:.4f}")
        print(f"File: {abs_path}")
        if block_type and name:
            print(f"Type: {block_type} '{name}'")

        # Show preview
        content = metadata.get('content', '')
        if content:
            preview = content[:200].replace('\n', ' ')
            if len(content) > 200:
                preview += "..."
            print(f"Preview: {preview}")

        print("-" * 80)

    print("\n" + "=" * 80)
    print(f"Found {len(results)} results")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_code.py 'your search query'")
        print("\nExamples:")
        print("  python search_code.py 'neural network'")
        print("  python search_code.py 'memory consolidation'")
        print("  python search_code.py 'vector embedding'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    search_code(query)
