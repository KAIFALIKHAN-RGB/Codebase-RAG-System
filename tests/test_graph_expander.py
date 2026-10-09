from src.relationships.graph_expander import expand_related_files


class FakeGraph:
    def __init__(self, nodes, edges):
        self.nodes = set(nodes)
        self.edges = edges


def test_expands_imported_files():
    graph = FakeGraph(
        {"a.py", "b.py"},
        {"a.py": {"b.py"}},
    )

    result = expand_related_files(graph, {"a.py"})

    assert result == ["b.py"]


def test_expands_dependents():
    graph = FakeGraph(
        {"a.py", "b.py"},
        {"b.py": {"a.py"}},
    )

    result = expand_related_files(
        graph,
        {"a.py"},
        include_dependents=True,
    )

    assert result == ["b.py"]


def test_does_not_expand_when_max_hops_is_zero():
    graph = FakeGraph(
        {"a.py", "b.py"},
        {"a.py": {"b.py"}},
    )

    result = expand_related_files(
        graph,
        {"a.py"},
        max_hops=0,
    )

    assert result == []


def test_supports_multiple_hops():
    graph = FakeGraph(
        {"a.py", "b.py", "c.py"},
        {
            "a.py": {"b.py"},
            "b.py": {"c.py"},
        },
    )

    result = expand_related_files(
        graph,
        {"a.py"},
        max_hops=2,
    )

    assert result == ["b.py", "c.py"]


def test_handles_circular_imports():
    graph = FakeGraph(
        {"a.py", "b.py"},
        {
            "a.py": {"b.py"},
            "b.py": {"a.py"},
        },
    )

    result = expand_related_files(
        graph,
        {"a.py"},
        max_hops=10,
    )

    assert result == ["b.py"]


def test_respects_max_nodes():
    graph = FakeGraph(
        {"a.py", "b.py", "c.py", "d.py"},
        {
            "a.py": {"b.py", "c.py", "d.py"},
        },
    )

    result = expand_related_files(
        graph,
        {"a.py"},
        max_nodes=2,
    )

    assert len(result) == 2
    assert result == ["b.py", "c.py"]


def test_ignores_nonexistent_seed_files():
    graph = FakeGraph(
        {"a.py", "b.py"},
        {"a.py": {"b.py"}},
    )

    result = expand_related_files(
        graph,
        {"missing.py"},
    )

    assert result == []


def test_rejects_negative_limits():
    graph = FakeGraph(
        {"a.py", "b.py"},
        {"a.py": {"b.py"}},
    )

    try:
        expand_related_files(graph, {"a.py"}, max_hops=-1)
        assert False
    except ValueError:
        pass