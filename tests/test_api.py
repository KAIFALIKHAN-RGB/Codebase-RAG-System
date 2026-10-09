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

def test_health_returns_healthy_when_chromadb_is_available():
    mock_collection = patch(
        "src.api.app.get_collection"
    )

    with mock_collection as mock_get_collection:
        mock_get_collection.return_value.count.return_value = 10

        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "chromadb": "connected",
    }


def test_health_returns_503_when_chromadb_fails():
    with patch(
        "src.api.app.get_collection",
        side_effect=RuntimeError("ChromaDB unavailable"),
    ):
        response = client.get("/health")

    assert response.status_code == 503
    assert "ChromaDB unavailable" in response.json()["detail"]


def test_index_rejects_invalid_repository_path():
    with patch("src.api.app.os.path.exists", return_value=False):
        response = client.post(
            "/index",
            json={"repo_path": "C:/does/not/exist"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid repository path."


def test_index_rejects_missing_repo_path():
    response = client.post("/index", json={})

    assert response.status_code == 422


def test_index_starts_background_task_for_valid_repository(tmp_path):
    from unittest.mock import patch

    repo_path = tmp_path / "sample_repo"
    repo_path.mkdir()

    with (
        patch("src.api.app.os.path.exists", return_value=True),
        patch("src.api.app.os.path.isdir", return_value=True),
        patch("src.api.app.save_repository_path", return_value=None),
        patch("src.api.app.run_indexing_task") as mock_task,
        patch(
            "src.utils.repository_paths.REPO_PATH_FILE",
            str(tmp_path / "repository_paths.json"),
            
        ),
        patch(
            "src.utils.repository_paths.LOCK_FILE",
            str(tmp_path / "repository_paths.lock"),
            create=True,
        ),
    ):
        response = client.post(
            "/index",
            json={"repo_path": str(repo_path)},
        )

    assert response.status_code == 202
    assert response.json()["success"] is True
    assert response.json()["repositories"] == "sample_repo"
    mock_task.assert_called_once_with(str(repo_path), "sample_repo")


def test_index_status_returns_status():
    with patch(
        "src.api.app.get_index_status",
        return_value="completed",
    ):
        response = client.get("/index/status/sample_repo")

    assert response.status_code == 200
    assert response.json() == {
        "repository": "sample_repo",
        "status": "completed",
    }


def test_index_status_returns_404_for_unknown_repository():
    with patch(
        "src.api.app.get_index_status",
        return_value="not_found",
    ):
        response = client.get("/index/status/unknown_repo")

    assert response.status_code == 404


def test_repositories_returns_repository_list():
    with patch(
        "src.api.app.load_repositories",
        return_value=["repo_a", "repo_b"],
    ):
        response = client.get("/repositories")

    assert response.status_code == 200
    assert response.json() == {
        "repositories": ["repo_a", "repo_b"],
    }


def test_repositories_returns_empty_list():
    with patch(
        "src.api.app.load_repositories",
        return_value=[],
    ):
        response = client.get("/repositories")

    assert response.status_code == 200
    assert response.json() == {
        "repositories": [],
    }


def test_delete_returns_404_for_unknown_repository():
    with patch(
        "src.api.app.load_repositories",
        return_value=[],
    ):
        response = client.delete("/repositories/unknown_repo")

    assert response.status_code == 404
    assert response.json()["detail"] == "Repository not found."


def test_delete_repository_success():
    with (
        patch(
            "src.api.app.load_repositories",
            return_value=["sample_repo"],
        ),
        patch("src.api.app.get_repo_lock"),
        patch("src.api.app.delete_chunks_by_repository") as mock_chunks,
        patch("src.api.app.delete_repository_state") as mock_state,
        patch("src.api.app.delete_repository") as mock_repo,
    ):
        response = client.delete("/repositories/sample_repo")

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message": "Repository 'sample_repo' deleted successfully.",
    }
    mock_chunks.assert_called_once_with("sample_repo")
    mock_state.assert_called_once_with("sample_repo")
    mock_repo.assert_called_once_with("sample_repo")


def test_delete_repository_returns_500_on_failure():
    with (
        patch(
            "src.api.app.load_repositories",
            return_value=["sample_repo"],
        ),
        patch("src.api.app.get_repo_lock"),
        patch(
            "src.api.app.delete_chunks_by_repository",
            side_effect=RuntimeError("Delete failed"),
        ),
    ):
        response = client.delete("/repositories/sample_repo")

    assert response.status_code == 500
    assert "Failed to delete repository" in response.json()["detail"]