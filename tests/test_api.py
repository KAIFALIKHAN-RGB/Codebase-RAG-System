import asyncio
from threading import Event, Thread
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from src.api.app import get_repo_lock, health, run_indexing_task
from src.utils.index_status import set_index_status
from src.indexing import indexer
from src.storage.storage_lock import storage_access

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
    assert response.json()["detail"] == "Internal server error."
    assert "Pipeline failure" not in response.json()["detail"]
    

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
    assert response.json()["detail"] == "ChromaDB unavailable."


def test_health_count_waits_for_shared_storage_access():
    entered = Event()
    release = Event()
    counted = Event()
    collection = MagicMock()
    collection.count.side_effect = lambda: counted.set()

    async def call_health():
        with patch("src.api.app.get_collection", return_value=collection):
            await health()

    def hold_storage():
        with storage_access():
            entered.set()
            release.wait(timeout=2)

    holder = Thread(target=hold_storage)
    holder.start()
    assert entered.wait(timeout=2)

    health_thread = Thread(target=lambda: asyncio.run(call_health()))
    health_thread.start()
    assert not counted.wait(timeout=0.1)

    release.set()
    health_thread.join(timeout=2)
    holder.join(timeout=2)
    assert counted.is_set()


def test_index_rejects_invalid_repository_path():
    with patch("src.api.app.os.path.exists", return_value=False):
        response = client.post(
            "/index",
            json={"repo_path": "C:/does/not/exist"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid repository path."


def test_index_rejects_existing_file_path_without_scheduling_task():
    with (
        patch("src.api.app.os.path.exists", return_value=True),
        patch("src.api.app.os.path.isdir", return_value=False),
        patch("src.api.app.save_repository_path") as mock_save_path,
        patch("src.api.app.run_indexing_task") as mock_task,
    ):
        response = client.post(
            "/index",
            json={"repo_path": "C:/files/not_a_repository.txt"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid repository path."
    mock_save_path.assert_not_called()
    mock_task.assert_not_called()


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
        patch("src.api.app.save_repository_path", return_value=None) as mock_save_path,
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
    mock_save_path.assert_called_once_with("sample_repo", str(repo_path))


def test_index_rejects_duplicate_active_indexing(tmp_path):
    repo_path = tmp_path / "sample_repo"
    repo_path.mkdir()
    set_index_status("sample_repo", "indexing")

    try:
        with (
            patch("src.api.app.os.path.exists", return_value=True),
            patch("src.api.app.os.path.isdir", return_value=True),
            patch("src.api.app.save_repository_path") as mock_save_path,
            patch("src.api.app.run_indexing_task") as mock_task,
        ):
            response = client.post("/index", json={"repo_path": str(repo_path)})
    finally:
        from src.utils.index_status import clear_index_status

        clear_index_status("sample_repo")

    assert response.status_code == 409
    mock_save_path.assert_not_called()
    mock_task.assert_not_called()


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
        patch("src.api.app.delete_repository_path") as mock_path,
        patch("src.api.app.get_index_status", return_value="completed"),
        patch("src.api.app.clear_index_status") as mock_clear_status,
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
    mock_path.assert_called_once_with("sample_repo")
    mock_clear_status.assert_called_once_with("sample_repo")


def test_delete_rejects_queued_indexing_and_preserves_status():
    with (
        patch("src.api.app.load_repositories", return_value=["sample_repo"]),
        patch("src.api.app.get_index_status", return_value="queued"),
        patch("src.api.app.delete_chunks_by_repository") as mock_chunks,
    ):
        response = client.delete("/repositories/sample_repo")

    assert response.status_code == 409
    mock_chunks.assert_not_called()


def test_delete_rejects_newly_queued_repository_before_persistent_membership():
    with (
        patch("src.api.app.load_repositories", return_value=[]),
        patch("src.api.app.get_index_status", return_value="queued"),
        patch("src.api.app.delete_chunks_by_repository") as mock_chunks,
    ):
        response = client.delete("/repositories/sample_repo")

    assert response.status_code == 409
    mock_chunks.assert_not_called()


def test_run_indexing_task_skips_deleted_queued_work(tmp_path):
    from src.api.app import run_indexing_task

    with (
        patch("src.api.app.get_repo_lock"),
        patch("src.api.app.get_index_status", return_value="not_found"),
        patch("src.api.app.index_codebase") as mock_index,
    ):
        run_indexing_task(str(tmp_path), "sample_repo")

    mock_index.assert_not_called()


def test_run_indexing_task_does_not_restore_failed_after_deletion(
    tmp_path,
    monkeypatch,
):
    statuses = {"sample_repo": "queued"}
    lock = MagicMock()

    def get_status(repo_name):
        return statuses.get(repo_name, "not_found")

    def set_status(repo_name, status):
        statuses[repo_name] = status

    def delete_during_indexing(*args, **kwargs):
        statuses.pop("sample_repo", None)
        raise RuntimeError("index failed")

    monkeypatch.setattr("src.api.app.get_repo_lock", lambda name: lock)
    monkeypatch.setattr("src.api.app.get_index_status", get_status)
    monkeypatch.setattr("src.api.app.set_index_status", set_status)
    monkeypatch.setattr(
        "src.api.app.index_codebase",
        delete_during_indexing,
    )

    run_indexing_task(str(tmp_path), "sample_repo")

    assert statuses == {}
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
    assert response.json()["detail"] == "Failed to delete repository."

@pytest.mark.parametrize(
    "repo_name",
    ["", ".", "..", "../outside", "nested/repo", r"nested\repo"],
)
def test_get_repo_lock_rejects_invalid_repository_name(repo_name):
    with pytest.raises(ValueError, match="Invalid repository name"):
        get_repo_lock(repo_name)


def test_api_and_indexer_use_the_same_repository_lock_helper():
    assert get_repo_lock is indexer.get_repository_lock


def test_query_rejects_query_longer_than_max_length():
    response = client.post(
        "/query",
        json={
            "query": "a" * 4001,
            "repository": "sample_repo",
        },
    )

    assert response.status_code == 422

def test_index_returns_409_when_repository_name_conflicts(tmp_path):
    repo_path = tmp_path / "sample_repo"
    repo_path.mkdir()

    with (
        patch("src.api.app.os.path.exists", return_value=True),
        patch("src.api.app.os.path.isdir", return_value=True),
        patch(
            "src.api.app.save_repository_path",
            side_effect=ValueError(
                "Repository name already belongs to a different path."
            ),
        ),
        patch("src.api.app.run_indexing_task") as mock_task,
    ):
        response = client.post(
            "/index",
            json={"repo_path": str(repo_path)},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Repository name already belongs to a different path."
    )
    mock_task.assert_not_called()