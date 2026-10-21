import logging
import json
from unittest.mock import MagicMock
from threading import Event, Thread

from src.indexing import indexer
from src.storage.storage_lock import storage_access
from src.utils.storage import repository_chunks_path, safe_repository_name, save_chunks


def test_indexing_reports_summary_through_logging_without_stdout(
    monkeypatch,
    caplog,
    capsys,
    tmp_path,
):
    monkeypatch.setattr(indexer, "load_python_file", lambda path: [])
    monkeypatch.setattr(
        indexer,
        "load_schema_version",
        lambda: indexer.SCHEMA_VERSION,
    )
    monkeypatch.setattr(indexer, "load_repository_state", lambda name: {})
    monkeypatch.setattr(indexer, "save_chunks", lambda chunks, path: None)
    monkeypatch.setattr(
        indexer,
        "save_repository_state",
        lambda name, state: None,
    )
    monkeypatch.setattr(indexer, "save_repository", lambda name: None)

    with caplog.at_level(logging.INFO, logger="src.indexing.indexer"):
        indexer._run_indexing(str(tmp_path), "sample_repo")

    assert capsys.readouterr().out == ""
    assert any(
        "Indexing summary for sample_repo" in message
        for message in caplog.messages
    )


def test_repository_chunk_exports_are_atomic_and_isolated(tmp_path):
    barrier = Event()
    output_paths = {
        name: tmp_path / "chunks" / f"{name}.json"
        for name in ("repo-a", "repo-b")
    }

    def write_export(name):
        barrier.wait(timeout=2)
        save_chunks(
            [{"repository": name, "code": f"code for {name}"}],
            output_paths[name],
        )

    workers = [
        Thread(target=write_export, args=(name,))
        for name in output_paths
    ]
    for worker in workers:
        worker.start()
    barrier.set()
    for worker in workers:
        worker.join(timeout=2)

    for name, path in output_paths.items():
        assert path.exists()
        assert json.loads(path.read_text(encoding="utf-8")) == [
            {"repository": name, "code": f"code for {name}"}
        ]


def test_repository_chunk_export_names_do_not_collide_after_sanitization():
    first = repository_chunks_path(r"repo/name")
    second = repository_chunks_path(r"repo\name")

    assert first != second
    assert first.name == f"{safe_repository_name(r'repo/name')}.json"
    assert second.name == f"{safe_repository_name(r'repo\name')}.json"


def test_indexer_uses_shared_repository_lock(monkeypatch, tmp_path):
    lock = MagicMock()
    monkeypatch.setattr(indexer, "get_repository_lock", lambda name: lock)
    monkeypatch.setattr(indexer, "_run_indexing", lambda path, name: None)

    indexer._index_codebase(str(tmp_path))

    lock.__enter__.assert_called_once_with()
    lock.__exit__.assert_called_once()


def test_schema_reset_boundary_blocks_other_storage_access(monkeypatch, tmp_path):
    from src.storage import chroma_store

    previous_generation = chroma_store.get_storage_generation()
    reset_entered = Event()
    release_reset = Event()
    storage_operation_entered = Event()

    def reset():
        reset_entered.set()
        release_reset.wait(timeout=2)

    def storage_operation():
        with storage_access():
            storage_operation_entered.set()

    monkeypatch.setattr(indexer, "load_python_file", lambda path: [])
    monkeypatch.setattr(indexer, "load_schema_version", lambda: None)
    monkeypatch.setattr(indexer, "reset_database_unlocked", reset)
    monkeypatch.setattr(indexer, "save_schema_version", lambda: None)
    monkeypatch.setattr(indexer.os.path, "exists", lambda path: False)
    monkeypatch.setattr(indexer, "save_chunks", lambda chunks, path: None)
    monkeypatch.setattr(indexer, "save_repository_state", lambda name, state: None)
    monkeypatch.setattr(indexer, "save_repository", lambda name: None)

    indexing = Thread(
        target=indexer._run_indexing,
        args=(str(tmp_path), "repo-a"),
    )
    storage = Thread(target=storage_operation)
    indexing.start()
    assert reset_entered.wait(timeout=2)
    storage.start()
    assert not storage_operation_entered.wait(timeout=0.1)

    release_reset.set()
    indexing.join(timeout=2)
    storage.join(timeout=2)

    assert storage_operation_entered.is_set()
    monkeypatch.setattr(chroma_store, "_storage_generation", previous_generation)
