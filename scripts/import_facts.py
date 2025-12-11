#!/usr/bin/env python3
"""
Bulk import facts into the Hologram trainer from JSON or CSV.

Expected formats:
- JSON: [{"subject": "...", "predicate": "...", "object": "..."}, ...]
- CSV:  subject,predicate,object

Notes:
- Uses CrewTrainer.bulk_import_facts() so facts persist to the same store
  used during conversational training.
- Requires GEMINI_API_KEY and ANTHROPIC_API_KEY (CrewTrainer dependency).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

from crew_trainer import CrewTrainer


def _load_json(path: Path) -> List[Tuple[str, str, str]]:
    data = json.loads(path.read_text())
    facts = []
    for entry in data:
        subject = entry.get("subject")
        predicate = entry.get("predicate")
        obj = entry.get("object")
        if subject and predicate and obj:
            facts.append((subject, predicate, obj))
    return facts


def _load_csv(path: Path) -> List[Tuple[str, str, str]]:
    facts = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row.get("subject")
            predicate = row.get("predicate")
            obj = row.get("object")
            if subject and predicate and obj:
                facts.append((subject, predicate, obj))
    return facts


def _load_file(path: Path) -> List[Tuple[str, str, str]]:
    if path.suffix.lower() == ".json":
        return _load_json(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}. Use .json or .csv.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk import facts into Hologram FactStore",
    )
    parser.add_argument("--file", required=True, type=Path, help="Path to JSON or CSV file")
    parser.add_argument(
        "--persist-dir",
        default="./data/crew_training_facts",
        help="Persistence directory (same as CrewTrainer)",
    )
    parser.add_argument(
        "--source",
        default="bulk",
        help="Source label to attach to imported facts",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=1.0,
        help="Confidence / learning rate for imported facts (0-1)",
    )
    args = parser.parse_args()

    facts = _load_file(args.file)
    if not facts:
        raise SystemExit(f"No facts loaded from {args.file}")

    trainer = CrewTrainer(persist_dir=args.persist_dir)
    added = trainer.bulk_import_facts(facts, source=args.source, confidence=args.confidence)

    print(f"Imported {added}/{len(facts)} facts into {args.persist_dir}")


if __name__ == "__main__":
    main()
