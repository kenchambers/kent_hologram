#!/usr/bin/env python3
"""
Task benchmark harness with synthetic domain-sized datasets.

Loads 800+ facts (geography, science, code) and reports:
- Accuracy (% correct on known facts)
- Confidence mean/min
- Latency P50/P95
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict, List, Tuple

from hologram.core.codebook import Codebook
from hologram.core.vector_space import VectorSpace
from hologram.memory.fact_store import FactStore


def _generate_geography(n: int = 500) -> List[Tuple[str, str, str]]:
    return [(f"Country{i}", "capital", f"City{i}") for i in range(n)]


def _generate_science(n: int = 200) -> List[Tuple[str, str, str]]:
    return [(f"Element{i}", "atomic_number", str(i)) for i in range(n)]


def _generate_code(n: int = 100) -> List[Tuple[str, str, str]]:
    return [(f"func_{i}", "returns", f"value_{i}") for i in range(n)]


def _load_facts(store: FactStore, facts: List[Tuple[str, str, str]]) -> None:
    for s, p, o in facts:
        store.add_fact(s, p, o, source="benchmark")


def _evaluate(store: FactStore, facts: List[Tuple[str, str, str]]) -> Dict[str, float]:
    latencies: List[float] = []
    confidences: List[float] = []
    correct = 0

    for s, p, o in facts:
        start = time.perf_counter()
        answer, conf = store.query(s, p)
        latencies.append(time.perf_counter() - start)
        confidences.append(conf)
        if answer.lower() == o.lower():
            correct += 1

    return {
        "count": len(facts),
        "accuracy": correct / len(facts),
        "latency_p50_ms": statistics.median(latencies) * 1000,
        "latency_p95_ms": statistics.quantiles(latencies, n=20)[-1] * 1000,
        "conf_mean": statistics.mean(confidences),
        "conf_min": min(confidences),
    }


def _print_summary(results: Dict[str, Dict[str, float]]) -> None:
    header = f"{'domain':>10} | {'facts':>5} | {'acc':>5} | {'p50 (ms)':>8} | {'p95 (ms)':>8} | {'conf_mean':>9} | {'conf_min':>8}"
    print(header)
    print("-" * len(header))
    for domain, stats in results.items():
        print(
            f"{domain:>10} | "
            f"{int(stats['count']):5d} | "
            f"{stats['accuracy']*100:5.1f}% | "
            f"{stats['latency_p50_ms']:8.2f} | "
            f"{stats['latency_p95_ms']:8.2f} | "
            f"{stats['conf_mean']:9.3f} | "
            f"{stats['conf_min']:8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark factual QA across synthetic domains",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=10000,
        help="Vector dimensions (default: 10000)",
    )
    args = parser.parse_args()

    space = VectorSpace(dimensions=args.dimensions)
    codebook = Codebook(space)
    store = FactStore(space, codebook)

    geo = _generate_geography()
    sci = _generate_science()
    code = _generate_code()

    _load_facts(store, geo + sci + code)

    results = {
        "geography": _evaluate(store, geo),
        "science": _evaluate(store, sci),
        "code": _evaluate(store, code),
        "total": _evaluate(store, geo + sci + code),
    }

    _print_summary(results)


if __name__ == "__main__":
    main()
