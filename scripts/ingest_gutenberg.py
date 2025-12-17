#!/usr/bin/env python3
"""
Project Gutenberg Dataset Ingestion Script

NOTE: First run may take 30-60 seconds to initialize the trainer.

Ingests books from the Hugging Face Project Gutenberg dataset into the
Hologram knowledge base using crew_trainer's document teaching mode.

Features:
- Resume capability: Tracks progress in a JSON checkpoint file
- Graceful shutdown: Press Ctrl+C to save progress and exit cleanly
- Configurable batch size and language filter
- Progress reporting with ETA

Dataset: https://huggingface.co/datasets/manu/project_gutenberg
- 75,570 books total
- 61,300 English books
- Full text content with Project Gutenberg headers/footers

Usage:
    # Start fresh ingestion (English books)
    python scripts/ingest_gutenberg.py

    # Resume from checkpoint
    python scripts/ingest_gutenberg.py --resume

    # Ingest French books
    python scripts/ingest_gutenberg.py --language fr

    # Limit number of books
    python scripts/ingest_gutenberg.py --max-books 100

    # Custom chunk size
    python scripts/ingest_gutenberg.py --chunk-size 1000

Author: Claude Code (Generated)
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Set
import re

# Force unbuffered output for real-time progress
os.environ['PYTHONUNBUFFERED'] = '1'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# Checkpoint file for resume capability
DEFAULT_CHECKPOINT_FILE = "./data/gutenberg_checkpoint.json"
DEFAULT_PERSIST_DIR = "./data/crew_training_facts"


class GutenbergIngester:
    """
    Ingests Project Gutenberg books into Hologram's knowledge base.

    Supports resume capability via checkpoint files.
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        checkpoint_file: str = DEFAULT_CHECKPOINT_FILE,
        language: str = "en",
        chunk_size: int = 1000,
        max_books: Optional[int] = None,
        skip_existing: bool = True,
    ):
        """
        Initialize the ingester.

        Args:
            persist_dir: Directory for Hologram fact persistence
            checkpoint_file: Path to checkpoint JSON file
            language: Language split to use (en, fr, de, zh, etc.)
            chunk_size: Characters per chunk for document processing
            max_books: Maximum number of books to process (None for all)
            skip_existing: Skip books already in checkpoint
        """
        self.persist_dir = persist_dir
        self.checkpoint_file = checkpoint_file
        self.language = language
        self.chunk_size = chunk_size
        self.max_books = max_books
        self.skip_existing = skip_existing

        # State
        self.running = True
        self.checkpoint: Dict[str, Any] = self._load_checkpoint()
        self.processed_ids: Set[str] = set(self.checkpoint.get("processed_ids", []))

        # Statistics
        self.session_start = datetime.now()
        self.books_processed_this_session = 0
        self.facts_added_this_session = 0
        self.errors_this_session = 0

        # Components (lazy loaded)
        self._trainer = None
        self._web_teacher = None
        self._dataset = None

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint from file if it exists."""
        checkpoint_path = Path(self.checkpoint_file)
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                    print(f"✓ Loaded checkpoint: {len(data.get('processed_ids', []))} books already processed")
                    return data
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠ Warning: Could not load checkpoint: {e}")
        return {
            "processed_ids": [],
            "language": self.language,
            "total_facts": 0,
            "total_books": 0,
            "last_updated": None,
            "errors": [],
        }

    def _save_checkpoint(self):
        """Save current progress to checkpoint file."""
        self.checkpoint["processed_ids"] = list(self.processed_ids)
        self.checkpoint["language"] = self.language
        self.checkpoint["last_updated"] = datetime.now().isoformat()

        checkpoint_path = Path(self.checkpoint_file)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)

        print(f"💾 Checkpoint saved: {len(self.processed_ids)} books processed")

    def _setup_trainer(self):
        """Lazy-load the CrewTrainer and WebTeacher."""
        if self._trainer is not None:
            return

        print("Initializing Hologram trainer...")
        print("  (This may take 30-60 seconds on first run)")
        sys.stdout.flush()  # Force output

        # Import here to avoid circular imports and speed up startup
        from scripts.crew_trainer import CrewTrainer, WebTeacher

        self._trainer = CrewTrainer(
            persist_dir=self.persist_dir,
            max_rounds=0,  # We don't need conversation rounds
        )

        self._web_teacher = WebTeacher(llm_provider="anthropic")

        print("✓ Trainer initialized")
        sys.stdout.flush()

    def _setup_dataset(self):
        """Lazy-load the Hugging Face dataset."""
        if self._dataset is not None:
            return

        print(f"Loading Project Gutenberg dataset (language: {self.language})...")

        try:
            from datasets import load_dataset
        except ImportError:
            print("ERROR: datasets library not installed.")
            print("Install with: pip install datasets")
            sys.exit(1)

        # Use streaming to avoid downloading entire dataset
        self._dataset = load_dataset(
            "manu/project_gutenberg",
            split=self.language,
            streaming=True,
        )

        print("✓ Dataset loaded (streaming mode)")
        sys.stdout.flush()

    def _clean_gutenberg_text(self, text: str) -> str:
        """
        Clean Project Gutenberg text by removing headers/footers.

        The dataset includes standard PG headers and footers marked with ***.
        """
        # Find content between *** START and *** END markers
        start_markers = [
            "*** START OF THIS PROJECT GUTENBERG",
            "*** START OF THE PROJECT GUTENBERG",
            "***START OF THIS PROJECT GUTENBERG",
            "*END*THE SMALL PRINT",
        ]
        end_markers = [
            "*** END OF THIS PROJECT GUTENBERG",
            "*** END OF THE PROJECT GUTENBERG",
            "***END OF THIS PROJECT GUTENBERG",
            "End of Project Gutenberg",
            "End of the Project Gutenberg",
        ]

        # Find start position
        start_pos = 0
        for marker in start_markers:
            pos = text.find(marker)
            if pos != -1:
                # Find the end of this line
                newline_pos = text.find('\n', pos)
                if newline_pos != -1:
                    start_pos = newline_pos + 1
                break

        # Find end position
        end_pos = len(text)
        for marker in end_markers:
            pos = text.find(marker)
            if pos != -1 and pos > start_pos:
                end_pos = pos
                break

        # Extract content
        content = text[start_pos:end_pos].strip()

        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)

        return content

    def _extract_title(self, text: str, book_id: str) -> str:
        """Extract book title from text or use ID as fallback."""
        # Try to find title in header
        title_patterns = [
            r"Title:\s*(.+?)(?:\n|$)",
            r"^(.+?)\n\n",  # First line before double newline
        ]

        for pattern in title_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                # Clean up title
                title = re.sub(r'\s+', ' ', title)
                if len(title) > 5 and len(title) < 200:
                    return title

        return f"Gutenberg Book {book_id}"

    def process_book(self, book_id: str, text: str) -> int:
        """
        Process a single book and extract facts.

        Args:
            book_id: Gutenberg book ID
            text: Full book text

        Returns:
            Number of facts extracted
        """
        # Clean the text
        cleaned_text = self._clean_gutenberg_text(text)

        if len(cleaned_text) < 1000:
            print(f"  ⚠ Skipping {book_id}: too short after cleaning ({len(cleaned_text)} chars)")
            return 0

        # Extract title for topic label
        title = self._extract_title(text, book_id)

        print(f"  📖 Processing: {title[:60]}{'...' if len(title) > 60 else ''}")
        print(f"     Size: {len(cleaned_text):,} chars")

        # Use WebTeacher's document teaching
        try:
            facts_added = self._web_teacher.teach_document(
                text=cleaned_text,
                fact_store=self._trainer.chatbot._fact_store,
                codebook=self._trainer.container._codebook,
                topic=f"Gutenberg/{title}",
                chunk_size=self.chunk_size,
                overlap=100,
            )

            return facts_added

        except Exception as e:
            print(f"  ❌ Error processing book: {e}")
            self.checkpoint.setdefault("errors", []).append({
                "book_id": book_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            return 0

    def run(self):
        """Run the ingestion process."""
        # Setup signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print("\n\n⚠ Interrupt received, saving progress...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Initialize components
        self._setup_trainer()
        self._setup_dataset()

        print("\n" + "=" * 70)
        print("  PROJECT GUTENBERG INGESTION")
        print("=" * 70)
        print(f"  Language: {self.language}")
        print(f"  Chunk size: {self.chunk_size} chars")
        print(f"  Max books: {self.max_books or 'unlimited'}")
        print(f"  Already processed: {len(self.processed_ids)} books")
        print(f"  Checkpoint file: {self.checkpoint_file}")
        print("=" * 70)
        print("\nPress Ctrl+C to stop and save progress\n")

        # Process books
        book_count = 0

        try:
            for example in self._dataset:
                if not self.running:
                    break

                book_id = example.get("id", f"unknown_{book_count}")

                # Skip if already processed
                if book_id in self.processed_ids:
                    continue

                book_count += 1

                # Check max books limit
                if self.max_books and book_count > self.max_books:
                    print(f"\n✓ Reached max books limit ({self.max_books})")
                    break

                # Get book text
                text = example.get("text", "")
                if not text:
                    print(f"  ⚠ Skipping {book_id}: no text content")
                    continue

                # Process book
                print(f"\n[{book_count}] Book ID: {book_id}")

                try:
                    facts = self.process_book(book_id, text)

                    # Update statistics
                    self.processed_ids.add(book_id)
                    self.books_processed_this_session += 1
                    self.facts_added_this_session += facts
                    self.checkpoint["total_facts"] = self.checkpoint.get("total_facts", 0) + facts
                    self.checkpoint["total_books"] = len(self.processed_ids)

                    print(f"  ✓ Extracted {facts} facts (session total: {self.facts_added_this_session})")

                except Exception as e:
                    self.errors_this_session += 1
                    print(f"  ❌ Error: {e}")

                # Save checkpoint periodically (every 10 books)
                if book_count % 10 == 0:
                    self._save_checkpoint()
                    self._print_progress()

                # Brief pause to avoid rate limiting
                time.sleep(0.5)

        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Always save on exit
            self._save_checkpoint()
            self._print_final_summary()

            # Save trainer state
            if self._trainer:
                self._trainer.chatbot.save_memory(self.persist_dir, force_consolidation=True)
                self._trainer.chatbot.end_session()

    def _print_progress(self):
        """Print progress update."""
        elapsed = datetime.now() - self.session_start
        rate = self.books_processed_this_session / max(elapsed.total_seconds() / 60, 0.1)

        print(f"\n--- Progress Update ---")
        print(f"  Books this session: {self.books_processed_this_session}")
        print(f"  Facts this session: {self.facts_added_this_session}")
        print(f"  Total books (all sessions): {len(self.processed_ids)}")
        print(f"  Rate: {rate:.1f} books/min")
        print(f"  Elapsed: {elapsed}")
        print(f"-----------------------\n")

    def _print_final_summary(self):
        """Print final summary."""
        elapsed = datetime.now() - self.session_start

        print("\n" + "=" * 70)
        print("  INGESTION COMPLETE")
        print("=" * 70)
        print(f"  Session duration: {elapsed}")
        print(f"  Books processed (this session): {self.books_processed_this_session}")
        print(f"  Facts extracted (this session): {self.facts_added_this_session}")
        print(f"  Errors (this session): {self.errors_this_session}")
        print(f"  Total books (all sessions): {len(self.processed_ids)}")
        print(f"  Total facts (all sessions): {self.checkpoint.get('total_facts', 0)}")
        print(f"  Checkpoint saved to: {self.checkpoint_file}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Project Gutenberg books into Hologram knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start ingestion (English books, default)
  python scripts/ingest_gutenberg.py

  # Resume from where you left off
  python scripts/ingest_gutenberg.py --resume

  # Ingest French books
  python scripts/ingest_gutenberg.py --language fr

  # Process only 100 books
  python scripts/ingest_gutenberg.py --max-books 100

  # Use larger chunks for faster processing
  python scripts/ingest_gutenberg.py --chunk-size 2000

  # Clear checkpoint and start fresh
  python scripts/ingest_gutenberg.py --fresh

Available languages: en (61k), fr (5.5k), de (3.1k), pt (1.1k),
                     nl (1.4k), es (1.2k), it (1k), zh (437),
                     sv (388), pl (34), ru (6)
        """
    )

    parser.add_argument(
        "--language", "-l",
        default="en",
        help="Language split to use (default: en)"
    )

    parser.add_argument(
        "--max-books", "-n",
        type=int,
        default=None,
        help="Maximum number of books to process (default: all)"
    )

    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=1000,
        help="Characters per chunk for document processing (default: 1000)"
    )

    parser.add_argument(
        "--checkpoint-file",
        default=DEFAULT_CHECKPOINT_FILE,
        help=f"Path to checkpoint file (default: {DEFAULT_CHECKPOINT_FILE})"
    )

    parser.add_argument(
        "--persist-dir",
        default=DEFAULT_PERSIST_DIR,
        help=f"Directory for fact persistence (default: {DEFAULT_PERSIST_DIR})"
    )

    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from existing checkpoint (same as default behavior, kept for clarity)"
    )

    parser.add_argument(
        "--fresh", "-f",
        action="store_true",
        help="Start fresh, ignoring existing checkpoint"
    )

    args = parser.parse_args()

    # Handle fresh start
    if args.fresh:
        checkpoint_path = Path(args.checkpoint_file)
        if checkpoint_path.exists():
            print(f"Removing existing checkpoint: {checkpoint_path}")
            checkpoint_path.unlink()

    # Create and run ingester
    ingester = GutenbergIngester(
        persist_dir=args.persist_dir,
        checkpoint_file=args.checkpoint_file,
        language=args.language,
        chunk_size=args.chunk_size,
        max_books=args.max_books,
    )

    ingester.run()


if __name__ == "__main__":
    main()
