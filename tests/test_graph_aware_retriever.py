from src.retrieval.graph_aware_retriever import graph_aware_search
from src.relationships.import_graph import ImportGraph


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(
        self,
        query,
        repository=None,
        k=3,
        threshold=35.0,
        file_paths=None,
    ):
        chunks = self.chunks

        if file_paths is not None:
            normalized_paths = {
                str(path).replace("\\", "/")
                for path in file_paths
            }

            chunks = [
                chunk
                for chunk in chunks
                if str(
                    chunk["metadata"].get("file_path", "")
                ).replace("\\", "/") in normalized_paths
            ]

        return {
            "results": chunks,
            "retrieval_time_ms": 1.0,
        }


def chunk(
    file_path,
    start=1,
    end=5,
    repository="repo",
):
    return {
        "metadata": {
            "repository": repository,
            "file_path": file_path,
            "start_line": start,
            "end_line": end,
        }
    }


def build_graph():
    graph = ImportGraph()

    graph.add_edge(
        "src/main.py",
        "src/utils.py",
    )

    graph.add_edge(
        "src/helper.py",
        "src/main.py",
    )

    return graph


def test_seed_chunks_are_preserved():
    seed = chunk("src/main.py")

    retriever = FakeRetriever([seed])

    result = graph_aware_search(
        retriever,
        build_graph(),
        "main logic",
    )

    assert result["results"][0] == seed


def test_imported_file_is_added():
    seed = chunk("src/main.py")
    related = chunk("src/utils.py")

    retriever = FakeRetriever(
        [seed, related]
    )

    result = graph_aware_search(
        retriever,
        build_graph(),
        "utility logic",
    )

    files = {
        item["metadata"]["file_path"]
        for item in result["results"]
    }

    assert "src/main.py" in files
    assert "src/utils.py" in files


def test_dependent_file_is_added():
    seed = chunk("src/main.py")
    dependent = chunk("src/helper.py")

    retriever = FakeRetriever(
        [seed, dependent]
    )

    result = graph_aware_search(
        retriever,
        build_graph(),
        "helper logic",
    )

    files = {
        item["metadata"]["file_path"]
        for item in result["results"]
    }

    assert "src/helper.py" in files


def test_duplicate_chunks_are_removed():
    seed = chunk("src/main.py")

    retriever = FakeRetriever([seed])

    result = graph_aware_search(
        retriever,
        build_graph(),
        "main",
    )

    assert len(result["results"]) == 1


def test_no_related_files_returns_vector_results():
    seed = chunk("src/main.py")

    retriever = FakeRetriever([seed])

    graph = ImportGraph()
    graph.add_node("src/main.py")

    result = graph_aware_search(
        retriever,
        graph,
        "main",
    )

    assert result["results"] == [seed]


def test_empty_query_returns_empty():
    retriever = FakeRetriever([])

    result = graph_aware_search(
        retriever,
        build_graph(),
        "   ",
    )

    assert result == {
        "results": [],
        "retrieval_time_ms": 0.0,
    }


def test_non_positive_k_returns_empty():
    seed = chunk("src/main.py")

    retriever = FakeRetriever([seed])

    result_zero = graph_aware_search(
        retriever,
        build_graph(),
        "main",
        k=0,
    )

    result_negative = graph_aware_search(
        retriever,
        build_graph(),
        "main",
        k=-1,
    )

    expected = {
        "results": [],
        "retrieval_time_ms": 0.0,
    }

    assert result_zero == expected
    assert result_negative == expected


def test_repository_metadata_is_preserved():
    seed = chunk(
        "src/main.py",
        repository="repo-a",
    )

    related = chunk(
        "src/utils.py",
        repository="repo-a",
    )

    retriever = FakeRetriever(
        [seed, related]
    )
    result = graph_aware_search(
        retriever,
        build_graph(),
        "utils",
        repository="repo-a",
    )

    assert all(
        item["metadata"]["repository"] == "repo-a"
        for item in result["results"]
    )