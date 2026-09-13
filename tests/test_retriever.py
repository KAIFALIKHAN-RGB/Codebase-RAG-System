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