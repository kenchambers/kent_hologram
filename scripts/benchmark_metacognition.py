#!/usr/bin/env python3
"""
Benchmark the current metacognition retry loop.

The goal is to measure whether retries improve confidence/accuracy.
This reflects the current implementation (no context modulation yet).
"""

from __future__ import annotations

import argparse
import statistics
from typing import Dict, List, Tuple

from hologram.cognition.metacognition import MetacognitiveLoop
from hologram.core.codebook import Codebook
from hologram.core.vector_space import VectorSpace
from hologram.memory.fact_store import FactStore


def _seed_facts(store: FactStore) -> None:
    data = [
        ("France", "capital", "Paris"),
        ("Germany", "capital", "Berlin"),
        ("Japan", "capital", "Tokyo"),
        ("Canada", "capital", "Ottawa"),
        ("Kenya", "capital", "Nairobi"),
    ]
    for s, p, o in data:
        store.add_fact(s, p, o, source="benchmark")


def _build_fact_store(dimensions: int) -> FactStore:
    space = VectorSpace(dimensions=dimensions)
    codebook = Codebook(space)
    store = FactStore(space, codebook)
    _seed_facts(store)
    return store


def _queries() -> List[Tuple[str, str, str]]:
    """
    Returns (label, subject, predicate) tuples.
    Includes noisy inputs to see if retries help.
    """
    return [
        ("exact_france", "France", "capital"),
        ("exact_germany", "Germany", "capital"),
        ("noisy_france", "frnce", "capital"),  # missing vowel
        ("noisy_kenya", "Ken ya", "capital"),  # space noise
        ("wrong_predicate", "France", "capitol"),  # spelling difference
    ]


def _run_plain(store: FactStore) -> List[Dict[str, object]]:
    rows = []
    for label, subject, predicate in _queries():
        answer, conf = store.query(subject, predicate)
        rows.append(
            {
                "mode": "plain",
                "label": label,
                "answer": answer,
                "confidence": conf,
                "correct": answer.lower() in {"paris", "berlin", "tokyo", "ottawa", "nairobi"},
            }
        )
    return rows


def _run_metacog(store: FactStore, codebook: Codebook) -> List[Dict[str, object]]:
    loop = MetacognitiveLoop(codebook)

    def query_func(text: str) -> Tuple[str, float]:
        subject, predicate = text.split("|")
        return store.query(subject, predicate)

    rows = []
    for label, subject, predicate in _queries():
        query_text = f"{subject}|{predicate}"
        answer, conf = loop.execute_query(query_func, query_text)
        rows.append(
            {
                "mode": "metacog",
                "label": label,
                "answer": answer,
                "confidence": conf,
                "correct": answer.lower() in {"paris", "berlin", "tokyo", "ottawa", "nairobi"},
                "retries": len(loop.state.confidence_history),
            }
        )
        loop.state.reset()
    return rows


def _summarize(rows: List[Dict[str, object]]) -> Dict[str, float]:
    confidences = [r["confidence"] for r in rows]
    accuracy = sum(1 for r in rows if r["correct"]) / len(rows)
    return {
        "accuracy": accuracy,
        "conf_mean": statistics.mean(confidences),
        "conf_min": min(confidences),
    }


def _print(rows: List[Dict[str, object]]) -> None:
    for r in rows:
        print(
            f"{r['mode']:8} | {r['label']:16} | ans={r['answer'] or '""'} | "
            f"conf={r['confidence']:.3f} | correct={r['correct']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark metacognition retry effectiveness",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=10000,
        help="Vector dimensions (default: 10000)",
    )
    args = parser.parse_args()

    store = _build_fact_store(args.dimensions)
    codebook = Codebook(VectorSpace(dimensions=args.dimensions))

    plain_rows = _run_plain(store)
    metacog_rows = _run_metacog(store, codebook)

    print("=== Plain ===")
    _print(plain_rows)
    print("\n=== Metacognition (current wiring) ===")
    _print(metacog_rows)

    print("\nSummary:")
    plain_summary = _summarize(plain_rows)
    meta_summary = _summarize(metacog_rows)
    print(f"Plain: {plain_summary}")
    print(f"Metacog: {meta_summary}")


if __name__ == "__main__":
    main()
