#!/usr/bin/env python3
"""
Compare HDC FactStore vs FAISS adapter for scale.

Metrics per store:
- Query latency (P50/P95)
- Similarity/confidence statistics

Defaults: 10,000 facts, 200 random queries.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import faiss  # noqa: F401
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "FAISS is required for this benchmark. Install faiss-cpu and retry."
    ) from exc

from hologram.core.codebook import Codebook
from hologram.core.operations import Operations
from hologram.core.vector_space import VectorSpace
from hologram.memory.fact_store import FactStore
from hologram.persistence.faiss_adapter import FaissAdapter


def _make_fact(i: int) -> Tuple[str, str, str]:
    return (f"entity_{i}", "related_to", f"value_{i}")


# ----------------------------
# HDC baseline (FactStore)
# ----------------------------
def _build_hdc(num_facts: int, dimensions: int) -> FactStore:
    space = VectorSpace(dimensions=dimensions)
    codebook = Codebook(space)
    store = FactStore(space, codebook)
    for i in range(num_facts):
        s, p, o = _make_fact(i)
        store.add_fact(s, p, o, source="benchmark")
    return store


def _query_hdc(store: FactStore, subject: str, predicate: str) -> Tuple[str, float]:
    return store.query(subject, predicate)


# ----------------------------
# FAISS adapter
# ----------------------------
def _build_faiss(num_facts: int, dimensions: int, persist_dir: Path) -> Tuple[FaissAdapter, Codebook]:
    space = VectorSpace(dimensions=dimensions)
    codebook = Codebook(space)
    adapter = FaissAdapter(dimensions=dimensions, persist_path=str(persist_dir))

    for i in range(num_facts):
        subj, pred, obj = _make_fact(i)
        key = Operations.bind(codebook.encode(subj.lower()), codebook.encode(pred.lower()))
        adapter.store(key, {"subject": subj, "predicate": pred, "object": obj})

    return adapter, codebook


def _query_faiss(adapter: FaissAdapter, codebook: Codebook, subject: str, predicate: str) -> Tuple[str, float]:
    key = Operations.bind(codebook.encode(subject.lower()), codebook.encode(predicate.lower()))
    results = adapter.query(key, k=1)
    if not results:
        return "", 0.0
    _, score, meta = results[0]
    return meta.get("object", ""), float(score)


# ----------------------------
# Benchmark runner
# ----------------------------
def _sample_queries(num_queries: int, max_index: int) -> List[int]:
    return [random.randint(0, max_index - 1) for _ in range(num_queries)]


def _run_hdc(num_facts: int, dimensions: int, num_queries: int) -> Dict[str, float]:
    store = _build_hdc(num_facts, dimensions)
    query_ids = _sample_queries(num_queries, num_facts)

    latencies: List[float] = []
    confidences: List[float] = []
    for idx in query_ids:
        subj, pred, _ = _make_fact(idx)
        start = time.perf_counter()
        _, conf = _query_hdc(store, subj, pred)
        latencies.append(time.perf_counter() - start)
        confidences.append(conf)

    return {
        "mode": "HDC",
        "facts": num_facts,
        "latency_p50_ms": statistics.median(latencies) * 1000,
        "latency_p95_ms": statistics.quantiles(latencies, n=20)[-1] * 1000,
        "conf_mean": statistics.mean(confidences),
        "conf_min": min(confidences),
    }


def _run_faiss(num_facts: int, dimensions: int, num_queries: int, persist_dir: Path) -> Dict[str, float]:
    adapter, codebook = _build_faiss(num_facts, dimensions, persist_dir)
    query_ids = _sample_queries(num_queries, num_facts)

    latencies: List[float] = []
    scores: List[float] = []
    for idx in query_ids:
        subj, pred, _ = _make_fact(idx)
        start = time.perf_counter()
        _, score = _query_faiss(adapter, codebook, subj, pred)
        latencies.append(time.perf_counter() - start)
        scores.append(score)

    return {
        "mode": "FAISS",
        "facts": num_facts,
        "latency_p50_ms": statistics.median(latencies) * 1000,
        "latency_p95_ms": statistics.quantiles(latencies, n=20)[-1] * 1000,
        "conf_mean": statistics.mean(scores),
        "conf_min": min(scores),
    }


def _print(rows: List[Dict[str, float]]) -> None:
    header = f"{'mode':>5} | {'facts':>6} | {'p50 (ms)':>8} | {'p95 (ms)':>8} | {'conf(mean)':>10} | {'conf(min)':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['mode']:>5} | "
            f"{int(r['facts']):6d} | "
            f"{r['latency_p50_ms']:8.2f} | "
            f"{r['latency_p95_ms']:8.2f} | "
            f"{r['conf_mean']:10.3f} | "
            f"{r['conf_min']:9.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FAISS vs HDC FactStore for large-scale retrieval",
    )
    parser.add_argument(
        "--facts",
        type=int,
        default=10000,
        help="Number of facts to load (default: 10000)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=10000,
        help="Vector dimensions (default: 10000)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=200,
        help="Number of random queries to run (default: 200)",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("/tmp/faiss_benchmark"),
        help="Directory for FAISS metadata (default: /tmp/faiss_benchmark)",
    )
    args = parser.parse_args()

    random.seed(42)

    # Ensure persist dir exists (no files written unless save() is called)
    args.persist_dir.mkdir(parents=True, exist_ok=True)

    hdc_row = _run_hdc(args.facts, args.dimensions, args.queries)
    faiss_row = _run_faiss(args.facts, args.dimensions, args.queries, args.persist_dir)

    _print([hdc_row, faiss_row])


if __name__ == "__main__":
    main()
