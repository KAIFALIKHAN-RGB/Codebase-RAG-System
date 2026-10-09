from collections import deque
from pathlib import PurePosixPath


def _normalize_path(path):
    """Return a normalized repository-relative POSIX path."""
    return PurePosixPath(str(path).replace("\\", "/")).as_posix()


def expand_related_files(
    graph,
    seed_files,
    max_hops=1,
    max_nodes=10,
    include_dependents=True,
):
    """
    Expand seed files through the repository import graph.

    Returns repository-local files related to the seed files.
    Seed files themselves are not included in the result.
    """

    if max_hops < 0:
        raise ValueError("max_hops must be >= 0")

    if max_nodes < 0:
        raise ValueError("max_nodes must be >= 0")

    seeds = {
        _normalize_path(file_path)
        for file_path in seed_files
        if file_path is not None
    }

    if not seeds or max_hops == 0 or max_nodes == 0:
        return []

    # Normalize graph nodes and edges once.
    nodes = {
        _normalize_path(node)
        for node in getattr(graph, "nodes", set())
    }

    edges = {
        _normalize_path(source): {
            _normalize_path(target)
            for target in targets
        }
        for source, targets in getattr(graph, "edges", {}).items()
    }

    # Build reverse edges so we can find files that import the seed.
    reverse_edges = {}

    for source, targets in edges.items():
        for target in targets:
            reverse_edges.setdefault(target, set()).add(source)

    queue = deque()
    visited = set(seeds)
    related = []

    for seed in sorted(seeds):
        if seed in nodes:
            queue.append((seed, 0))

    while queue and len(related) < max_nodes:
        current, depth = queue.popleft()

        if depth >= max_hops:
            continue

        neighbors = set(edges.get(current, set()))

        if include_dependents:
            neighbors.update(reverse_edges.get(current, set()))

        for neighbor in sorted(neighbors):
            if neighbor in visited:
                continue

            visited.add(neighbor)

            if neighbor not in nodes:
                continue

            related.append(neighbor)

            if len(related) >= max_nodes:
                break

            queue.append((neighbor, depth + 1))

    return related