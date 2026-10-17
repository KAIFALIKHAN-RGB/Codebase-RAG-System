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

    seed_chunks = seed_response.get("results", [])

    if not seed_chunks:
        return seed_response

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

        for node in graph_nodes:
            if normalized_path == node or normalized_path.endswith("/" + node):
                seed_path_map[node] = normalized_path
                break

    seed_files = set(seed_path_map)

    if not seed_files:
        return seed_response

    related_files = expand_related_files(
        graph=graph,
        seed_files=seed_files,
        max_hops=max_hops,
        max_nodes=max_related_files,
        include_dependents=True,
    )

    if not related_files:
        return seed_response

    # Preserve the exact path prefix used by ChromaDB metadata.
    prefixes = {
        seed_path[:-len(node)]
        for node, seed_path in seed_path_map.items()
        if node and seed_path.endswith(node)
    }

    related_query_paths = {
        prefix.replace("/", "\\") + _normalize_path(path).replace("/", "\\")
        for prefix in prefixes
        for path in related_files
    }

    if not related_query_paths:
        return seed_response

    # Retrieve query-relevant chunks ONLY from graph-related files
    related_response = retriever.search(
        query=query,
        repository=repository,
        k=max(k * 2, 6),
        threshold=threshold,
        file_paths=related_query_paths,
    )

    related_chunks = related_response.get("results", [])

    # Merge while preserving primary vector results first
    combined = []
    seen = set()

    for chunk in seed_chunks + related_chunks:
        key = _chunk_key(chunk)

        if key in seen:
            continue

        seen.add(key)
        combined.append(chunk)

    return {
        "results": combined,
        "retrieval_time_ms": (
            seed_response.get("retrieval_time_ms", 0.0)
            + related_response.get("retrieval_time_ms", 0.0)
        ),
    }