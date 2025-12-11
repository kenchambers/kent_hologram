#!/usr/bin/env python3
"""Entry point for running chat interface as a module.

Usage: 
    python -m hologram.chat
    python -m hologram.chat --persist-dir ./data/crew_training_facts
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Hologram conversational chatbot interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default directory
  python -m hologram.chat
  
  # Query training facts
  python -m hologram.chat --persist-dir ./data/crew_training_facts
  
  # Use custom directory
  python -m hologram.chat --persist-dir ./data/my_facts
  
  # In-memory only (no persistence)
  python -m hologram.chat --no-persist
        """
    )
    
    parser.add_argument(
        "--persist-dir",
        default="./data/hologram_facts",
        help="Directory for ChromaDB fact persistence (default: ./data/hologram_facts)"
    )
    
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Disable persistence (in-memory only)"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid circular import warning
    from hologram.chat.interface import ChatInterface
    interface = ChatInterface(
        persist_dir=args.persist_dir,
        persistent=not args.no_persist,
    )
    interface.start()


if __name__ == "__main__":
    main()
