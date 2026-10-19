import math
from typing import Any

from src.relationships.graph_expander import expand_related_files


def _normalize_path(path: str) -> str:
    return str(path).replace("\\", "/")


def _get_file_path(chunk: dict[str, Any]) -> str | None:
    return chunk.get("metadata", {}).get("file_path")


def _chunk_key(chunk: dict[str, Any]) -> tuple:
    metadata = chunk.get("metadata", {})

    return (
        metadata.get("repository"),
        _normalize_path(metadata.get("file_path", "")),
        metadata.get("start_line"),
        metadata.get("end_line"),
    )


def _has_repository(chunk: dict[str, Any], repository: str | None) -> bool:
    return (
        repository is None
        or chunk.get("metadata", {}).get("repository") == repository
    )


def _rank_combined_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use the retriever's score contract when all candidates provide it."""
    if not all(
        isinstance(chunk.get("ranking_score"), (int, float))
        and not isinstance(chunk.get("ranking_score"), bool)
        and math.isfinite(chunk["ranking_score"])
        for chunk in chunks
    ):
        return chunks

    return sorted(
        chunks,
        key=lambda chunk: (
            -chunk["ranking_score"],
            chunk.get("metadata", {}).get("repository") or "",
            _normalize_path(chunk.get("metadata", {}).get("file_path", "")),
            chunk.get("metadata", {}).get("start_line", 0) or 0,
            chunk.get("metadata", {}).get("end_line", 0) or 0,
        ),
    )


def _related_query_paths(
    seed_path_map: dict[str, str],
    related_files: list[str],
) -> set[str]:
    paths = set()

    for node, seed_path in seed_path_map.items():
        prefix = seed_path[:-len(node)] if seed_path.endswith(node) else ""

        for path in related_files:
            normalized_path = prefix + _normalize_path(path)
            paths.add(normalized_path)
            paths.add(normalized_path.replace("/", "\\"))

    return paths


def graph_aware_search(
    retriever,
    graph,
    query: str,
    repository: str | None = None,
    k: int = 3,
    threshold: float = 35.0,
    max_hops: int = 1,
    max_related_files: int = 5,
) -> dict[str, Any]:
    """
    Perform vector retrieval and supplement it with chunks from
    files related through the import graph.

    Vector retrieval remains the primary ranking signal.
    Graph expansion only adds additional context.
    """

    if not query or not query.strip() or k <= 0:
        return {
            "results": [],
            "retrieval_time_ms": 0.0,
        }

    # Primary vector retrieval
    seed_response = retriever.search(
        query=query,
        repository=repository,
        k=k,
        threshold=threshold,
    )

    seed_chunks = [
        chunk
        for chunk in seed_response.get("results", [])
        if _has_repository(chunk, repository)
    ]

    if not seed_chunks:
        return {
            "results": [],
            "retrieval_time_ms": seed_response.get("retrieval_time_ms", 0.0),
        }

    # Match ChromaDB paths to repository-relative graph nodes.
    graph_nodes = {
        _normalize_path(node)
        for node in getattr(graph, "nodes", set())
    }

    seed_path_map = {}

    for chunk in seed_chunks:
        file_path = _get_file_path(chunk)
        if not file_path:
            continue

        normalized_path = _normalize_path(file_path)

        for node in sorted(graph_nodes):
            if normalized_path == node or normalized_path.endswith("/" + node):
                seed_path_map[node] = normalized_path
                break

    seed_files = set(seed_path_map)

    if not seed_files:
        return {
            "results": seed_chunks[:k],
            "retrieval_time_ms": seed_response.get("retrieval_time_ms", 0.0),
        }

    related_files = expand_related_files(
        graph=graph,
        seed_files=seed_files,
        max_hops=max_hops,
        max_nodes=max_related_files,
        include_dependents=True,
    )

    if not related_files:
        return {
            "results": seed_chunks[:k],
            "retrieval_time_ms": seed_response.get("retrieval_time_ms", 0.0),
        }

    related_query_paths = _related_query_paths(seed_path_map, related_files)

    if not related_query_paths:
        return {
            "results": seed_chunks[:k],
            "retrieval_time_ms": seed_response.get("retrieval_time_ms", 0.0),
        }

    # Retrieve query-relevant chunks ONLY from graph-related files
    related_response = retriever.search(
        query=query,
        repository=repository,
        k=max(k * 2, 6),
        threshold=threshold,
        file_paths=related_query_paths,
    )

    related_chunks = [
        chunk
        for chunk in related_response.get("results", [])
        if _has_repository(chunk, repository)
    ]

    combined = []
    seen = set()

    for chunk in seed_chunks + related_chunks:
        key = _chunk_key(chunk)

        if key in seen:
            continue

        seen.add(key)
        combined.append(chunk)

    combined = _rank_combined_chunks(combined)

    return {
        "results": combined[:k],
        "retrieval_time_ms": (
            seed_response.get("retrieval_time_ms", 0.0)
            + related_response.get("retrieval_time_ms", 0.0)
        ),
    }