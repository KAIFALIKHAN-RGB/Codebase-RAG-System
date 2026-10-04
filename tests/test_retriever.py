import src.retrieval.retriever as retriever


def test_source_files_are_ranked_above_test_files():
    retriever.get_embedding = lambda query: [0.1, 0.2, 0.3]

    retriever.collection.query = lambda **kwargs: {
        "documents": [[
            "test chunk",
            "source chunk"
        ]],
        "metadatas": [[
            {
                "repository": "python-dotenv",
                "file_path": "tests/test_cli.py",
                "name": "test_run"
            },
            {
                "repository": "python-dotenv",
                "file_path": "src/dotenv/cli.py",
                "name": "run_command"
            }
        ]],
        "distances": [[
            0.60,
            0.62
        ]]
    }

    result = retriever.search(
        "Where is the command execution logic implemented?",
        repository="python-dotenv",
        k=2,
        threshold=30
    )

    results = result["results"]

    assert results[0]["metadata"]["file_path"] == "src/dotenv/cli.py"
    assert results[0]["metadata"]["name"] == "run_command"

    assert results[0]["similarity"] == 38.0
    assert results[0]["ranking_score"] == 41.0

    assert results[1]["similarity"] == 40.0
    assert results[1]["ranking_score"] == 37.0

def test_repository_filter_isolated_with_source_test_ranking():
        retriever.get_embedding = lambda query: [0.1, 0.2, 0.3]

        def fake_query(**kwargs):
            repository = kwargs["where"]["repository"]

            if repository == "repo-a":
                return {
                    "documents": [[
                        "repo-a test chunk",
                        "repo-a source chunk",
                    ]],
                    "metadatas": [[
                        {
                            "repository": "repo-a",
                            "file_path": "tests/test_cli.py",
                            "name": "test_run",
                        },
                        {
                            "repository": "repo-a",
                            "file_path": "src/dotenv/cli.py",
                            "name": "run_command",
                        },
                    ]],
                    "distances": [[0.60, 0.62]],
                }

            return {
                "documents": [[
                    "repo-b test chunk",
                    "repo-b source chunk",
                ]],
                "metadatas": [[
                    {
                        "repository": "repo-b",
                        "file_path": "tests/test_cli.py",
                        "name": "test_run",
                    },
                    {
                        "repository": "repo-b",
                        "file_path": "src/dotenv/cli.py",
                        "name": "run_command",
                    },
                ]],
                "distances": [[0.60, 0.62]],
            }

        retriever.collection.query = fake_query

        for repository in ["repo-a", "repo-b"]:
            result = retriever.search(
                "Where is the command execution logic implemented?",
                repository=repository,
                k=2,
                threshold=30,
            )

            results = result["results"]

            assert len(results) == 2

            assert all(
                item["metadata"]["repository"] == repository
                for item in results
            )

            assert results[0]["metadata"]["file_path"] == "src/dotenv/cli.py"
            assert results[0]["metadata"]["name"] == "run_command"

            assert results[0]["ranking_score"] > results[1]["ranking_score"]

import pytest


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k(k):
    with pytest.raises(ValueError):
        retriever.search("find user model", k=k)

def test_search_handles_k_larger_than_available_chunks(monkeypatch):
    def fake_query(**kwargs):
        return {
            "documents": [["def example(): pass"]],
            "metadatas": [[{
                "repository": "repo_a",
                "file_path": "src/example.py",
                "name": "example",
                "start_line": 1,
                "end_line": 1,
            }]],
            "distances": [[0.1]],
        }

    monkeypatch.setattr(retriever.collection, "query", fake_query)
    monkeypatch.setattr(
        "src.retrieval.retriever.get_embedding",
        lambda query: [0.1, 0.2],
    )

    result = retriever.search("example", k=100000, threshold=0)

    assert len(result["results"]) == 1

def test_search_handles_missing_metadata_fields(monkeypatch):
    def fake_query(**kwargs):
        return {
            "documents": [["def example(): pass"]],
            "metadatas": [[{"repository": "repo_a"}]],
            "distances": [[0.1]],
        }

    monkeypatch.setattr(retriever.collection, "query", fake_query)
    monkeypatch.setattr(
        "src.retrieval.retriever.get_embedding",
        lambda query: [0.1, 0.2],
    )

    result = retriever.search("example", threshold=0)

    assert len(result["results"]) == 1
    assert result["results"][0]["metadata"] == {"repository": "repo_a"}

def test_search_uses_deterministic_tiebreaker(monkeypatch):
    chunks = [
        {
            "code": "def from_b(): pass",
            "metadata": {
                "repository": "repo_b",
                "file_path": "src/example.py",
                "name": "from_b",
                "start_line": 1,
                "end_line": 2,
            },
            "distance": 0.5,
        },
        {
            "code": "def from_a(): pass",
            "metadata": {
                "repository": "repo_a",
                "file_path": "src/example.py",
                "name": "from_a",
                "start_line": 1,
                "end_line": 2,
            },
            "distance": 0.5,
        },
    ]

    monkeypatch.setattr(
        "src.retrieval.retriever.get_embedding",
        lambda query: [0.1, 0.2],
    )

    def fake_query(**kwargs):
        ordered = chunks if fake_query.reverse else list(reversed(chunks))
        return {
            "documents": [[item["code"] for item in ordered]],
            "metadatas": [[item["metadata"] for item in ordered]],
            "distances": [[item["distance"] for item in ordered]],
        }

    fake_query.reverse = False
    monkeypatch.setattr(retriever.collection, "query", fake_query)
    first = retriever.search("example", k=2, threshold=0)["results"]

    fake_query.reverse = True
    second = retriever.search("example", k=2, threshold=0)["results"]

    assert [r["metadata"]["repository"] for r in first] == ["repo_a", "repo_b"]
    assert [r["metadata"]["repository"] for r in second] == ["repo_a", "repo_b"]

@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_search_rejects_empty_query_before_embedding(
    monkeypatch, query
):
    def fail_if_called(_query):
        pytest.fail("Embedding must not be called for an empty query")

    monkeypatch.setattr(
        "src.retrieval.retriever.get_embedding",
        fail_if_called,
    )

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.search(query)

def test_search_handles_none_metadata(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.retriever.get_embedding",
        lambda query: [0.1, 0.2],
    )

    def fake_query(**kwargs):
        return {
            "documents": [["example code"]],
            "metadatas": [[None]],
            "distances": [[0.1]],
        }

    monkeypatch.setattr(retriever.collection, "query", fake_query)

    result = retriever.search("example", threshold=0)

    assert result["results"] == []