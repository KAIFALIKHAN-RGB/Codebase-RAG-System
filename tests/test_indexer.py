import logging

from src.indexing import indexer


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
