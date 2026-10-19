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

        if file_paths is None:
            chunks = chunks[:k]

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
    ranking_score=None,
):
    result = {
        "metadata": {
            "repository": repository,
            "file_path": file_path,
            "start_line": start,
            "end_line": end,
        }
    }
    if ranking_score is not None:
        result["ranking_score"] = ranking_score
    return result


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


def test_results_are_limited_to_k():
    seed = chunk("src/main.py", ranking_score=1)
    imported = chunk("src/utils.py", ranking_score=3)
    dependent = chunk("src/helper.py", ranking_score=2)

    retriever = FakeRetriever([seed, imported, dependent])

    result = graph_aware_search(
        retriever,
        build_graph(),
        "related logic",
        k=2,
    )

    assert len(result["results"]) == 2
    assert [
        item["metadata"]["file_path"]
        for item in result["results"]
    ] == ["src/utils.py", "src/helper.py"]


def test_combined_candidates_are_ranked_by_existing_score():
    seed = chunk("src/main.py", ranking_score=10)
    imported = chunk("src/utils.py", ranking_score=20)

    result = graph_aware_search(
        FakeRetriever([seed, imported]),
        build_graph(),
        "utility logic",
        k=2,
    )

    assert [
        item["metadata"]["file_path"]
        for item in result["results"]
    ] == ["src/utils.py", "src/main.py"]


def test_missing_scores_preserve_stable_order_without_fabricating_scores():
    seed = chunk("src/main.py")
    imported = chunk("src/utils.py")

    result = graph_aware_search(
        FakeRetriever([seed, imported]),
        build_graph(),
        "utility logic",
        k=2,
    )

    assert result["results"] == [seed, imported]


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


def test_cross_repository_chunks_are_excluded():
    seed = chunk("src/main.py", repository="repo-a", ranking_score=10)
    related = chunk("src/utils.py", repository="repo-b", ranking_score=100)

    result = graph_aware_search(
        FakeRetriever([seed, related]),
        build_graph(),
        "utility logic",
        repository="repo-a",
    )

    assert result["results"] == [seed]


def test_windows_path_prefix_is_preserved_for_related_files():
    seed = chunk(
        r"..\python-dotenv\src\dotenv\main.py",
        repository="python-dotenv",
    )
    related = chunk(
        r"..\python-dotenv\src\dotenv\cli.py",
        repository="python-dotenv",
    )

    retriever = FakeRetriever([seed, related])

    result = graph_aware_search(
        retriever,
        build_graph(),
        "command line functionality",
        repository="python-dotenv",
    )

    assert len(result["results"]) == 2
    assert any(
        item["metadata"]["file_path"]
        == r"..\python-dotenv\src\dotenv\cli.py"
        for item in result["results"]
    )


def test_posix_path_prefix_is_preserved_for_related_files():
    seed = chunk(
        "../python-dotenv/src/dotenv/main.py",
        repository="python-dotenv",
    )
    related = chunk(
        "../python-dotenv/src/dotenv/cli.py",
        repository="python-dotenv",
    )

    result = graph_aware_search(
        FakeRetriever([seed, related]),
        build_graph(),
        "command line functionality",
        repository="python-dotenv",
    )

    assert len(result["results"]) == 2
    assert any(
        item["metadata"]["file_path"]
        == "../python-dotenv/src/dotenv/cli.py"
        for item in result["results"]
    )