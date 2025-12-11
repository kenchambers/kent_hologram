#!/usr/bin/env python3
"""
Capacity saturation benchmark for Hologram FactStore.

Runs incremental fact loads (10 → 1000) and measures:
- Query latency (P50/P95)
- Retrieval confidence (mean/min)
- Approximate process RSS

This script is intentionally dependency-light and CPU-only.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from typing import Dict, List, Tuple

try:
    import resource  # POSIX only; used for RSS snapshot

    def _rss_mb() -> float:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS/Linux ru_maxrss is kilobytes
        return usage.ru_maxrss / 1024.0

except ImportError:  # pragma: no cover - non-POSIX

    def _rss_mb() -> float:
        return 0.0

from hologram.core.codebook import Codebook
from hologram.core.vector_space import VectorSpace
from hologram.memory.fact_store import FactStore


def _make_fact(i: int) -> Tuple[str, str, str]:
    """Generate a deterministic synthetic fact."""
    return (f"entity_{i}", "related_to", f"value_{i}")


def _build_store(num_facts: int, dimensions: int) -> FactStore:
    """Create and populate a FactStore with synthetic facts."""
    space = VectorSpace(dimensions=dimensions)
    codebook = Codebook(space)
    store = FactStore(space, codebook)

    for i in range(num_facts):
        s, p, o = _make_fact(i)
        store.add_fact(s, p, o, source="benchmark")

    return store


def _sample_queries(num_queries: int, max_index: int) -> List[int]:
    return [random.randint(0, max_index - 1) for _ in range(num_queries)]


def _run_trial(num_facts: int, dimensions: int, num_queries: int) -> Dict[str, float]:
    store = _build_store(num_facts, dimensions)
    query_ids = _sample_queries(num_queries, num_facts)

    latencies: List[float] = []
    confidences: List[float] = []

    for i in query_ids:
        subject, predicate, _ = _make_fact(i)
        start = time.perf_counter()
        _, confidence = store.query(subject, predicate)
        latencies.append(time.perf_counter() - start)
        confidences.append(confidence)

    return {
        "facts": num_facts,
        "latency_p50_ms": statistics.median(latencies) * 1000,
        "latency_p95_ms": statistics.quantiles(latencies, n=20)[-1] * 1000,
        "conf_mean": statistics.mean(confidences),
        "conf_min": min(confidences),
        "rss_mb": _rss_mb(),
    }


def run_benchmark(
    fact_counts: List[int], dimensions: int, num_queries: int
) -> List[Dict[str, float]]:
    results = []
    for n in fact_counts:
        results.append(_run_trial(n, dimensions, num_queries))
    return results


def _print_results(results: List[Dict[str, float]]) -> None:
    header = (
        f"{'facts':>6} | {'p50 (ms)':>8} | {'p95 (ms)':>8} | "
        f"{'conf(mean)':>10} | {'conf(min)':>9} | {'rss(MB)':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{row['facts']:6d} | "
            f"{row['latency_p50_ms']:8.2f} | "
            f"{row['latency_p95_ms']:8.2f} | "
            f"{row['conf_mean']:10.3f} | "
            f"{row['conf_min']:9.3f} | "
            f"{row['rss_mb']:7.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FactStore capacity saturation",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=10000,
        help="Vector dimensions (default: 10000)",
    )
    parser.add_argument(
        "--facts",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 200, 500, 1000],
        help="Fact counts to benchmark",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=50,
        help="Queries per checkpoint",
    )
    args = parser.parse_args()

    random.seed(42)
    results = run_benchmark(args.facts, args.dimensions, args.queries)
    _print_results(results)


if __name__ == "__main__":
    main()
