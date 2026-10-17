import statistics
import time

from tests.retrieval_quality_benchmark import (
    K,
    REPOSITORY,
    THRESHOLD,
    test_cases,
    is_correct,
    reciprocal_rank,
)
from src.retrieval.retriever import search
from src.retrieval.graph_aware_retriever import graph_aware_search
from src.relationships.import_graph import build_import_graph
from src.utils.repository_paths import get_repository_path


def evaluate(results, test):
    return (
        any(is_correct(result, test) for result in results),
        reciprocal_rank(results, test),
    )


def main():
    repository_path = get_repository_path(REPOSITORY)

    if not repository_path:
        if REPOSITORY == "python-dotenv":
            repository_path = r"C:\Users\kaifa\Desktop\python-dotenv"
        else:
            raise RuntimeError(
                f"No indexed repository path registered for {REPOSITORY!r}"
            )

    start = time.perf_counter()
    graph = build_import_graph(repository_path)
    graph_build_ms = (time.perf_counter() - start) * 1000

    modes = {
        "vector": {"hits": 0, "rr": [], "latency": []},
        "graph": {"hits": 0, "rr": [], "latency": []},
    }

    additional_chunks_total = 0
    queries_with_new_chunks = 0

    for index, test in enumerate(test_cases, start=1):
        # Warm-up and measurement use the same query, k and threshold.
        start = time.perf_counter()
        vector_output = search(
            query=test["query"],
            repository=REPOSITORY,
            k=K,
            threshold=THRESHOLD,
        )
        vector_wall_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        graph_output = graph_aware_search(
            retriever=__import__(
                "src.retrieval.retriever", fromlist=["search"]
            ),
            graph=graph,
            query=test["query"],
            repository=REPOSITORY,
            k=K,
            threshold=THRESHOLD,
        )
        vector_results = vector_output["results"]
        graph_results = graph_output["results"]

        vector_keys = {
            (
                result.get("metadata", {}).get("repository"),
                str(result.get("metadata", {}).get("file_path", "")).replace("\\", "/"),
                result.get("metadata", {}).get("start_line"),
                result.get("metadata", {}).get("end_line"),
            )
            for result in vector_results
        }

        graph_keys = {
            (
                result.get("metadata", {}).get("repository"),
                str(result.get("metadata", {}).get("file_path", "")).replace("\\", "/"),
                result.get("metadata", {}).get("start_line"),
                result.get("metadata", {}).get("end_line"),
            )
            for result in graph_results
        }

        new_chunk_count = len(graph_keys - vector_keys)
        additional_chunks_total += new_chunk_count

        if new_chunk_count > 0:
            queries_with_new_chunks += 1

        print(f"Additional unique chunks: {new_chunk_count}")
        graph_wall_ms = (time.perf_counter() - start) * 1000

        for name, output, wall_ms in (
            ("vector", vector_output, vector_wall_ms),
            ("graph", graph_output, graph_wall_ms),
        ):
            results = output["results"]
            hit, rr = evaluate(results, test)
            modes[name]["hits"] += int(hit)
            modes[name]["rr"].append(rr)
            modes[name]["latency"].append(wall_ms)

        print(
            f"{index:02d}. {test['query']}\n"
            f"    Vector Hit={evaluate(vector_output['results'], test)[0]} "
            f"RR={evaluate(vector_output['results'], test)[1]:.3f}\n"
            f"    Graph  Hit={evaluate(graph_output['results'], test)[0]} "
            f"RR={evaluate(graph_output['results'], test)[1]:.3f}"
        )

    total = len(test_cases)
    print("\n=== COMPARISON ===")
    print(f"Queries: {total} | K={K} | threshold={THRESHOLD}")
    print(f"Graph build time: {graph_build_ms:.2f} ms")

    for name, metrics in modes.items():
        print(f"\n{name.upper()}")
        print(
            f"Hit@{K}: {metrics['hits']}/{total} "
            f"({metrics['hits'] / total * 100:.2f}%)"
        )
        print(f"MRR: {statistics.mean(metrics['rr']):.3f}")
        print(f"Mean end-to-end retrieval time: "
              f"{statistics.mean(metrics['latency']):.2f} ms")


if __name__ == "__main__":
    main()