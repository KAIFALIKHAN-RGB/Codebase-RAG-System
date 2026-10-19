import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.retrieval.retriever as retriever_module
from src.retrieval.graph_aware_retriever import graph_aware_search
from src.relationships.import_graph import build_import_graph
from src.utils.repository_paths import get_repository_path
from tests.retrieval_quality_benchmark import (
    K,
    REPOSITORY,
    THRESHOLD,
    is_correct,
    reciprocal_rank,
    test_cases,
)


RUNS = 5

# These cases remain separate from the original vector benchmark. Their
# expected chunks are related through imports and are not used to tune it.
RELATIONSHIP_CASES = [
    {
        "query": "How does load_dotenv parse stream data?",
        "expected_chunks": [
            {
                "file": "src/dotenv/parser.py",
                "symbol": "parse_stream",
            },
        ],
    },
    {
        "query": "Which module does cli.py import for dotenv operations?",
        "expected_chunks": [
            {
                "file": "src/dotenv/main.py",
                "symbol": "DotEnv",
            },
        ],
    },
]


def chunk_key(result):
    metadata = result.get("metadata", {})
    return (
        metadata.get("repository"),
        str(metadata.get("file_path", "")).replace("\\", "/"),
        metadata.get("start_line"),
        metadata.get("end_line"),
    )


def evaluate(results, test):
    top_k = results[:K]
    return (
        any(is_correct(result, test) for result in top_k),
        reciprocal_rank(top_k, test),
    )


def context_recall(results, test):
    expected = test.get("expected_chunks")
    if not expected:
        return None

    found = sum(
        any(
            result_matches_expected(result, item)
            for result in results[:K]
        )
        for item in expected
    )
    return found / len(expected)


def result_matches_expected(result, expected):
    metadata = result.get("metadata", {})
    file_path = metadata.get("file_path")
    symbol = metadata.get("symbol") or metadata.get("name")

    return (
        file_path is not None
        and symbol == expected["symbol"]
        and str(file_path).replace("\\", "/").lower().endswith(
            expected["file"].replace("\\", "/").lower()
        )
    )


def run_vector(test):
    return retriever_module.search(
        query=test["query"],
        repository=REPOSITORY,
        k=K,
        threshold=THRESHOLD,
    )


def run_graph(test, graph):
    return graph_aware_search(
        retriever=retriever_module,
        graph=graph,
        query=test["query"],
        repository=REPOSITORY,
        k=K,
        threshold=THRESHOLD,
    )


def measure_pair(test, graph):
    timings = {"vector": [], "graph": []}
    outputs = {}

    for run in range(RUNS):
        modes = ("vector", "graph") if run % 2 == 0 else ("graph", "vector")

        for mode in modes:
            start = time.perf_counter()
            outputs[mode] = (
                run_vector(test)
                if mode == "vector"
                else run_graph(test, graph)
            )
            timings[mode].append((time.perf_counter() - start) * 1000)

    return outputs, timings


def warm_up(cases, graph):
    for test in cases:
        run_vector(test)
        run_graph(test, graph)


def report_suite(label, cases, graph):
    metrics = {
        "vector": {"hits": 0, "rr": [], "latency": [], "recall": []},
        "graph": {"hits": 0, "rr": [], "latency": [], "recall": []},
    }
    added_chunks = []
    score_limitations = 0

    for index, test in enumerate(cases, start=1):
        outputs, timings = measure_pair(test, graph)
        vector_results = outputs["vector"]["results"][:K]
        graph_results = outputs["graph"]["results"][:K]

        vector_keys = {chunk_key(result) for result in vector_results}
        graph_keys = {chunk_key(result) for result in graph_results}
        added_chunks.append(len(graph_keys - vector_keys))

        if any("ranking_score" not in result for result in graph_results):
            score_limitations += 1

        for mode in ("vector", "graph"):
            hit, rr = evaluate(outputs[mode]["results"], test)
            metrics[mode]["hits"] += int(hit)
            metrics[mode]["rr"].append(rr)
            metrics[mode]["latency"].extend(timings[mode])
            recall = context_recall(outputs[mode]["results"], test)
            if recall is not None:
                metrics[mode]["recall"].append(recall)

        print(
            f"{index:02d}. {test['query']}\n"
            f"    Vector Hit={evaluate(outputs['vector']['results'], test)[0]} "
            f"RR={evaluate(outputs['vector']['results'], test)[1]:.3f}\n"
            f"    Graph  Hit={evaluate(outputs['graph']['results'], test)[0]} "
            f"RR={evaluate(outputs['graph']['results'], test)[1]:.3f}\n"
            f"    Added unique top-{K} chunks: {added_chunks[-1]}"
        )

    total = len(cases)
    print(f"\n=== {label} ===")
    print(f"Queries: {total} | K={K} | threshold={THRESHOLD} | runs={RUNS}")

    for mode, values in metrics.items():
        print(f"\n{mode.upper()}")
        print(f"Hit@{K}: {values['hits']}/{total}")
        print(f"MRR: {statistics.mean(values['rr']):.3f}")
        print(f"Mean paired latency: {statistics.mean(values['latency']):.2f} ms")
        print(f"P50 paired latency: {statistics.median(values['latency']):.2f} ms")
        if values["recall"]:
            print(
                "Mean context recall: "
                f"{statistics.mean(values['recall']):.3f}"
            )

    print(
        "Mean added unique top-k chunks: "
        f"{statistics.mean(added_chunks):.2f}"
    )
    print(
        "Queries with added unique top-k chunks: "
        f"{sum(value > 0 for value in added_chunks)}/{total}"
    )
    if score_limitations:
        print(
            "Ranking-score limitation: "
            f"{score_limitations} graph queries lacked complete ranking_score data; "
            "stable retrieval order was used for those results."
        )


def main():
    repository_path = get_repository_path(REPOSITORY)
    if not repository_path:
        raise RuntimeError(
            f"No repository path registered for {REPOSITORY!r}. "
            "Register the repository path before running this benchmark."
        )

    start = time.perf_counter()
    graph = build_import_graph(repository_path)
    graph_build_ms = (time.perf_counter() - start) * 1000

    all_cases = test_cases + RELATIONSHIP_CASES
    warm_up(all_cases, graph)

    print(f"Graph build time: {graph_build_ms:.2f} ms")
    report_suite("ORIGINAL VECTOR GROUND TRUTH", test_cases, graph)
    report_suite("RELATIONSHIP-AWARE CASES", RELATIONSHIP_CASES, graph)
    report_suite("ALL CASES", all_cases, graph)
    print(
        "Amortized graph build per query (one build): "
        f"{graph_build_ms / len(all_cases):.2f} ms"
    )


if __name__ == "__main__":
    main()
