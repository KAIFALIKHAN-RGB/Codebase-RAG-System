import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import statistics
import time

from src.retrieval.retriever import search


QUERIES = [
    "Which class represents a CLI command in Click?",
    "Where is the execution context for a Click command represented?",
    "How does Click prompt the user for interactive input?",
    "Where is Click's usage error handling defined?",
    "Where are command-line arguments parsed?",
]

REPOSITORY = "heldout_click"
THRESHOLD = 35
RUNS = 10


def benchmark(k):
    latencies = []

    # Warm-up
    for query in QUERIES:
        search(
            query,
            repository=REPOSITORY,
            k=k,
            threshold=THRESHOLD,
        )

    for _ in range(RUNS):
        for query in QUERIES:
            start = time.perf_counter()

            search(
                query,
                repository=REPOSITORY,
                k=k,
                threshold=THRESHOLD,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

    latencies.sort()

    p50 = statistics.median(latencies)

    p95_index = int(len(latencies) * 0.95) - 1
    p95 = latencies[max(0, p95_index)]

    print(f"\nk={k}")
    print(f"Runs       : {len(latencies)}")
    print(f"Average    : {statistics.mean(latencies):.2f} ms")
    print(f"P50        : {p50:.2f} ms")
    print(f"P95        : {p95:.2f} ms")
    print(f"Min        : {min(latencies):.2f} ms")
    print(f"Max        : {max(latencies):.2f} ms")


if __name__ == "__main__":
    print("=" * 60)
    print("RETRIEVAL LATENCY BENCHMARK")
    print("=" * 60)

    for k in [3, 5, 10]:
        benchmark(k)