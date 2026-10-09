from pathlib import Path

from src.utils.repository_paths import (
    delete_repository_path,
    get_repository_path,
    load_repository_paths,
    save_repository_path,
)


def test_missing_registry_returns_empty(tmp_path, monkeypatch):
    registry = tmp_path / "repository_paths.json"

    monkeypatch.setattr(
        "src.utils.repository_paths.REPO_PATH_FILE",
        str(registry),
    )

    assert load_repository_paths() == {}


def test_save_and_get_repository_path(tmp_path, monkeypatch):
    registry = tmp_path / "repository_paths.json"
    lock_file = tmp_path / "repository_paths.lock"

    monkeypatch.setattr(
        "src.utils.repository_paths.REPO_PATH_FILE",
        str(registry),
    )
    monkeypatch.setattr(
        "src.utils.repository_paths.LOCK_FILE",
        str(lock_file),
    )

    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    save_repository_path("test-repo", str(repo_path))

    assert get_repository_path("test-repo") == str(
        Path(repo_path).resolve()
    )


def test_save_updates_existing_repository(tmp_path, monkeypatch):
    registry = tmp_path / "repository_paths.json"
    lock_file = tmp_path / "repository_paths.lock"

    monkeypatch.setattr(
        "src.utils.repository_paths.REPO_PATH_FILE",
        str(registry),
    )
    monkeypatch.setattr(
        "src.utils.repository_paths.LOCK_FILE",
        str(lock_file),
    )

    first_repo = tmp_path / "repo1"
    second_repo = tmp_path / "repo2"

    first_repo.mkdir()
    second_repo.mkdir()

    save_repository_path("test-repo", str(first_repo))
    save_repository_path("test-repo", str(second_repo))

    assert get_repository_path("test-repo") == str(
        Path(second_repo).resolve()
    )


def test_invalid_repository_path_rejected(tmp_path, monkeypatch):
    registry = tmp_path / "repository_paths.json"
    lock_file = tmp_path / "repository_paths.lock"

    monkeypatch.setattr(
        "src.utils.repository_paths.REPO_PATH_FILE",
        str(registry),
    )
    monkeypatch.setattr(
        "src.utils.repository_paths.LOCK_FILE",
        str(lock_file),
    )

    missing_path = tmp_path / "does-not-exist"

    try:
        save_repository_path(
            "test-repo",
            str(missing_path),
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_delete_repository_path(tmp_path, monkeypatch):
    registry = tmp_path / "repository_paths.json"
    lock_file = tmp_path / "repository_paths.lock"

    monkeypatch.setattr(
        "src.utils.repository_paths.REPO_PATH_FILE",
        str(registry),
    )
    monkeypatch.setattr(
        "src.utils.repository_paths.LOCK_FILE",
        str(lock_file),
    )

    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    save_repository_path("test-repo", str(repo_path))

    assert delete_repository_path("test-repo") is True
    assert get_repository_path("test-repo") is None


def test_delete_unknown_repository_returns_false(tmp_path, monkeypatch):
    registry = tmp_path / "repository_paths.json"
    lock_file = tmp_path / "repository_paths.lock"

    monkeypatch.setattr(
        "src.utils.repository_paths.REPO_PATH_FILE",
        str(registry),
    )
    monkeypatch.setattr(
        "src.utils.repository_paths.LOCK_FILE",
        str(lock_file),
    )

    assert delete_repository_path("unknown") is False