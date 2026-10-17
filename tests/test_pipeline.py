from unittest.mock import MagicMock, patch

from src.rag.pipeline import run_rag_pipeline


def test_pipeline_forwards_threshold_without_scaling():
    mock_retrieval_output = {
        "results": [
            {
                "code": "def add(a, b):\n    return a + b",
                "metadata": {
                    "file_path": "sample.py",
                    "symbol_name": "add",
                    "start_line": 1,
                    "end_line": 2,
                },
                "distance": 0.1,
                "similarity": 90.0,
            }
        ],
        "retrieval_time_ms": 5.0,
    }

    with patch(
        "src.rag.pipeline.search",
        return_value=mock_retrieval_output,
    ) as mock_search, patch(
        "src.rag.pipeline.build_context",
        return_value="CODE CONTEXT",
    ), patch(
        "src.rag.pipeline.generate_answer",
        return_value="The add function returns the sum.",
    ):

        result = run_rag_pipeline(
            "What does add do?",
            repository="sample_repo",
            k=3,
            threshold=10,
        )

    mock_search.assert_called_once_with(
        query="What does add do?",
        repository="sample_repo",
        k=3,
        threshold=10,
    )

    assert result["answer"] == "The add function returns the sum."
    assert result["sources"] == mock_retrieval_output["results"]
    assert result["retrieval_time_ms"] == 5.0

def test_pipeline_stops_when_no_relevant_chunks_found():
    mock_retrieval_output = {
        "results": [],
        "retrieval_time_ms": 4.0,
    }

    with patch(
        "src.rag.pipeline.search",
        return_value=mock_retrieval_output,
    ) as mock_search, patch(
        "src.rag.pipeline.build_context"
    ) as mock_build_context, patch(
        "src.rag.pipeline.generate_answer"
    ) as mock_generate_answer:

        result = run_rag_pipeline(
            "What does nonexistent_function do?",
            repository="sample_repo",
            k=3,
            threshold=10,
        )

    assert result["answer"] == "No relevant code chunks found."
    assert result["sources"] == []
    assert result["retrieval_time_ms"] == 4.0

    mock_search.assert_called_once_with(
        query="What does nonexistent_function do?",
        repository="sample_repo",
        k=3,
        threshold=10,
    )

    mock_build_context.assert_not_called()
    mock_generate_answer.assert_not_called()

def test_pipeline_runs_full_retrieval_context_and_generation_flow():
    mock_retrieval_output = {
        "results": [
            {
                "code": "def add(a, b):\n    return a + b",
                "metadata": {
                    "file_path": "sample.py",
                    "symbol_name": "add",
                    "start_line": 1,
                    "end_line": 2,
                },
                "distance": 0.1,
                "similarity": 90.0,
            }
        ],
        "retrieval_time_ms": 6.5,
    }

    with patch(
        "src.rag.pipeline.search",
        return_value=mock_retrieval_output,
    ) as mock_search, patch(
        "src.rag.pipeline.build_context",
        return_value="FILE: sample.py\nCODE:\ndef add(a, b):\n    return a + b",
    ) as mock_build_context, patch(
        "src.rag.pipeline.generate_answer",
        return_value="The add function returns the sum of two numbers.",
    ) as mock_generate_answer:

        result = run_rag_pipeline(
            "What does add do?",
            repository="sample_repo",
            k=3,
            threshold=10,
        )

    mock_search.assert_called_once_with(
        query="What does add do?",
        repository="sample_repo",
        k=3,
        threshold=10,
    )

    mock_build_context.assert_called_once_with(
        mock_retrieval_output["results"]
    )

    mock_generate_answer.assert_called_once_with(
        "What does add do?",
        "FILE: sample.py\nCODE:\ndef add(a, b):\n    return a + b",
    )

    assert result["answer"] == "The add function returns the sum of two numbers."
    assert result["sources"] == mock_retrieval_output["results"]
    assert result["retrieval_time_ms"] == 6.5

def test_pipeline_uses_graph_aware_retrieval_when_repository_path_exists():
    mock_output = {
        "results": [
            {
                "code": "def add(a, b): return a + b",
                "metadata": {
                    "file_path": "sample.py",
                    "symbol_name": "add",
                    "start_line": 1,
                    "end_line": 1,
                },
            }
        ],
        "retrieval_time_ms": 8.0,
    }

    with (
        patch("src.rag.pipeline.get_repository_path", return_value="/fake/sample_repo"),
        patch("src.rag.pipeline.build_import_graph", return_value=MagicMock()) as mock_graph,
        patch("src.rag.pipeline.graph_aware_search", return_value=mock_output) as mock_graph_search,
        patch("src.rag.pipeline.search") as mock_search,
        patch("src.rag.pipeline.build_context", return_value="CODE CONTEXT"),
        patch("src.rag.pipeline.generate_answer", return_value="Answer"),
    ):
        result = run_rag_pipeline("What does add do?", repository="sample_repo")

    mock_graph.assert_called_once_with("/fake/sample_repo")
    mock_graph_search.assert_called_once()
    mock_search.assert_not_called()
    assert result["answer"] == "Answer"
    assert result["retrieval_time_ms"] == 8.0


def test_pipeline_uses_vector_search_when_repository_path_missing():
    mock_output = {"results": [], "retrieval_time_ms": 4.0}

    with (
        patch("src.rag.pipeline.get_repository_path", return_value=None),
        patch("src.rag.pipeline.search", return_value=mock_output) as mock_search,
        patch("src.rag.pipeline.build_import_graph") as mock_graph,
        patch("src.rag.pipeline.graph_aware_search") as mock_graph_search,
    ):
        result = run_rag_pipeline("Unknown function?", repository="sample_repo")

    mock_search.assert_called_once()
    mock_graph.assert_not_called()
    mock_graph_search.assert_not_called()
    assert result["sources"] == []
    assert result["retrieval_time_ms"] == 4.0


def test_pipeline_falls_back_to_vector_search_when_graph_build_fails():
    mock_output = {"results": [], "retrieval_time_ms": 4.0}

    with (
        patch("src.rag.pipeline.get_repository_path", return_value="/fake/sample_repo"),
        patch(
            "src.rag.pipeline.build_import_graph",
            side_effect=OSError("Graph unavailable"),
        ),
        patch("src.rag.pipeline.graph_aware_search") as mock_graph_search,
        patch("src.rag.pipeline.search", return_value=mock_output) as mock_search,
    ):
        result = run_rag_pipeline("Unknown function?", repository="sample_repo")

    mock_graph_search.assert_not_called()
    mock_search.assert_called_once()
    assert result["sources"] == []
    assert result["retrieval_time_ms"] == 4.0