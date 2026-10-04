from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Codebase RAG API is running."
    }


def test_query_rejects_empty_query():
    response = client.post(
        "/query",
        json={"query": "", "repository": "sample_repo"},
    )

    assert response.status_code == 422


def test_query_rejects_whitespace_only_query():
    response = client.post(
        "/query",
        json={"query": "   ", "repository": "sample_repo"},
    )

    assert response.status_code == 422


def test_query_rejects_missing_repository():
    response = client.post(
        "/query",
        json={"query": "What does the code do?"},
    )

    assert response.status_code == 422


def test_query_returns_404_for_unknown_repository():
    with patch(
        "src.api.app.load_repositories",
        return_value=[],
    ):
        response = client.post(
            "/query",
            json={
                "query": "What does the code do?",
                "repository": "unknown_repo",
            },
        )

    assert response.status_code == 404


def test_query_returns_pipeline_result():
    expected = {
        "answer": "The function adds two numbers.",
        "sources": [],
    }

    with (
        patch(
            "src.api.app.load_repositories",
            return_value=["sample_repo"],
        ),
        patch(
            "src.api.app.run_rag_pipeline",
            return_value=expected,
        ) as mock_pipeline,
    ):
        response = client.post(
            "/query",
            json={
                "query": "What does add do?",
                "repository": "sample_repo",
            },
        )

    assert response.status_code == 200
    assert response.json() == expected
    mock_pipeline.assert_called_once_with(
        "What does add do?",
        "sample_repo",
    )


def test_query_returns_500_when_pipeline_fails():
    with (
        patch(
            "src.api.app.load_repositories",
            return_value=["sample_repo"],
        ),
        patch(
            "src.api.app.run_rag_pipeline",
            side_effect=RuntimeError("Pipeline failure"),
        ),
    ):
        response = client.post(
            "/query",
            json={
                "query": "What does add do?",
                "repository": "sample_repo",
            },
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Pipeline failure"