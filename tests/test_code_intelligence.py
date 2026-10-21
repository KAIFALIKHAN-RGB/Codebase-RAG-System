import os
from threading import Event, Thread
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agents.code_intelligence import (
    MAX_RESULTS,
    MAX_INSPECTED_FILES,
    MAX_SOURCE_LINES,
    CodeIntelligenceTools,
    ToolBudgetExceeded,
    ToolValidationError,
)
from src.storage.storage_lock import storage_access


def test_search_code_delegates_to_existing_retriever():
    retriever = MagicMock()
    retriever.search.return_value = {"results": [], "retrieval_time_ms": 1.0}
    tools = CodeIntelligenceTools("repo", retriever=retriever)

    result = tools.call(
        "search_code",
        {"query": "find parser", "k": 100, "threshold": 10},
    )

    assert result == {"results": [], "retrieval_time_ms": 1.0}
    retriever.search.assert_called_once_with(
        query="find parser",
        repository="repo",
        k=MAX_RESULTS,
        threshold=10,
    )


def test_lookup_symbol_filters_by_repository_and_metadata():
    collection = MagicMock()
    collection.get.return_value = {
        "documents": ["def load(): pass"],
        "metadatas": [{
            "name": "load",
            "type": "FunctionDef",
            "repository": "repo",
            "file_path": "src/app.py",
            "start_line": 1,
            "end_line": 1,
        }],
    }
    tools = CodeIntelligenceTools("repo", collection=collection)

    result = tools.call(
        "lookup_symbol",
        {"symbol": "load", "symbol_type": "FunctionDef"},
    )

    assert result["results"][0]["metadata"]["repository"] == "repo"
    collection.get.assert_called_once_with(
        where={
            "$and": [
                {"repository": "repo"},
                {"name": "load"},
                {"type": "FunctionDef"},
            ]
        },
        limit=MAX_RESULTS,
        include=["documents", "metadatas"],
    )


def test_lookup_symbol_waits_for_shared_storage_access():
    entered = Event()
    release = Event()
    looked_up = Event()
    collection = MagicMock()
    collection.get.side_effect = lambda **kwargs: (
        looked_up.set() or {"documents": [], "metadatas": []}
    )
    tools = CodeIntelligenceTools("repo", collection=collection)

    def hold_storage():
        with storage_access():
            entered.set()
            release.wait(timeout=2)

    holder = Thread(target=hold_storage)
    lookup = Thread(
        target=lambda: tools.call("lookup_symbol", {"symbol": "load"})
    )
    holder.start()
    assert entered.wait(timeout=2)
    lookup.start()
    assert not looked_up.wait(timeout=0.1)

    release.set()
    holder.join(timeout=2)
    lookup.join(timeout=2)
    assert looked_up.is_set()


def test_default_tools_refresh_collection_after_reset(monkeypatch):
    first = MagicMock()
    second = MagicMock()
    first.get.return_value = {"documents": [], "metadatas": []}
    second.get.return_value = {"documents": [], "metadatas": []}
    tools = CodeIntelligenceTools("repo", retriever=MagicMock())
    tools.collection = first
    tools._uses_default_collection = True
    generation = tools._collection_generation
    monkeypatch.setattr(
        "src.agents.code_intelligence.chroma_store.get_collection_unlocked",
        lambda: second,
    )
    monkeypatch.setattr(
        "src.agents.code_intelligence.chroma_store.get_storage_generation",
        lambda: generation + 1,
    )

    tools.call("lookup_symbol", {"symbol": "load"})

    assert tools.collection is second
    second.get.assert_called_once()


def test_read_source_enforces_repository_containment_and_limits(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    source = repository / "src.py"
    source.write_text("one\ntwo\nthree\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("secret\n", encoding="utf-8")

    tools = CodeIntelligenceTools(
        "repo",
        repository_path_resolver=lambda name: str(repository),
        max_tool_calls=10,
    )

    result = tools.call(
        "read_source",
        {"file_path": "src.py", "start_line": 2, "end_line": 3},
    )
    assert result["code"] == "two\nthree\n"
    assert result["file_path"] == "src.py"

    for file_path in (
        str(outside),
        "../outside.py",
        r"..\outside.py",
        r"C:\outside.py",
        r"\\server\share\outside.py",
    ):
        with pytest.raises(ToolValidationError):
            tools.call("read_source", {"file_path": file_path})

    with pytest.raises(ToolValidationError, match="200 lines"):
        tools.call(
            "read_source",
            {
                "file_path": "src.py",
                "start_line": 1,
                "end_line": MAX_SOURCE_LINES + 1,
            },
        )

    with pytest.raises(ToolValidationError, match="characters"):
        tools.call(
            "read_source",
            {"file_path": "src.py", "max_characters": 3},
        )


def test_read_source_rejects_symlink_resolving_outside_repository(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("outside\n", encoding="utf-8")
    link = repository / "outside-link.py"

    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")

    tools = CodeIntelligenceTools(
        "repo",
        repository_path_resolver=lambda name: str(repository),
    )

    with pytest.raises(ToolValidationError, match="inside"):
        tools.call("read_source", {"file_path": "outside-link.py"})


def test_repository_isolation_is_enforced_for_search_lookup_and_read(tmp_path):
    repository_a = tmp_path / "repo-a"
    repository_b = tmp_path / "repo-b"
    repository_a.mkdir()
    repository_b.mkdir()
    (repository_a / "shared.py").write_text("repo a\n", encoding="utf-8")
    (repository_b / "shared.py").write_text("repo b\n", encoding="utf-8")

    retriever = MagicMock()
    retriever.search.return_value = {"results": []}
    collection = MagicMock()
    collection.get.return_value = {"documents": [], "metadatas": []}
    tools = CodeIntelligenceTools(
        "repo-a",
        retriever=retriever,
        collection=collection,
        repository_path_resolver=lambda name: {
            "repo-a": str(repository_a),
            "repo-b": str(repository_b),
        }.get(name),
        max_tool_calls=4,
    )

    tools.call("search_code", {"query": "shared"})
    tools.call("lookup_symbol", {"symbol": "shared"})
    with pytest.raises(ToolValidationError):
        tools.call("read_source", {"file_path": "../repo-b/shared.py"})

    retriever.search.assert_called_once_with(
        query="shared",
        repository="repo-a",
        k=MAX_RESULTS,
        threshold=35.0,
    )
    assert collection.get.call_args.kwargs["where"] == {
        "$and": [{"repository": "repo-a"}, {"name": "shared"}]
    }


def test_lookup_symbol_is_exact_and_result_limited():
    collection = MagicMock()

    def get_symbols(*, where, limit, include):
        assert where == {
            "$and": [{"repository": "repo"}, {"name": "load"}]
        }
        assert include == ["documents", "metadatas"]
        symbols = [
            {
                "document": f"def load_{index}(): pass",
                "metadata": {
                    "name": "load",
                    "repository": "repo",
                    "type": "FunctionDef",
                },
            }
            for index in range(5)
        ]
        symbols.extend(
            [
                {
                    "document": "def loader(): pass",
                    "metadata": {"name": "loader", "repository": "repo"},
                },
                {
                    "document": "def load(): pass",
                    "metadata": {"name": "load", "repository": "other-repo"},
                },
            ]
        )
        selected = symbols[:limit]
        return {
            "documents": [item["document"] for item in selected],
            "metadatas": [item["metadata"] for item in selected],
        }

    collection.get.side_effect = get_symbols
    tools = CodeIntelligenceTools("repo", collection=collection)

    result = tools.call("lookup_symbol", {"symbol": "load"})

    assert len(result["results"]) == MAX_RESULTS
    assert all(item["metadata"]["name"] == "load" for item in result["results"])
    assert all(
        item["metadata"]["repository"] == "repo" for item in result["results"]
    )


def test_read_source_rejects_directories_and_empty_ranges(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "folder").mkdir()
    (repository / "source.py").write_text("line\n", encoding="utf-8")
    tools = CodeIntelligenceTools(
        "repo",
        repository_path_resolver=lambda name: str(repository),
    )

    with pytest.raises(ToolValidationError, match="regular file"):
        tools.call("read_source", {"file_path": "folder"})

    with pytest.raises(ToolValidationError, match="empty"):
        tools.call(
            "read_source",
            {"file_path": "source.py", "start_line": 2},
        )


def test_related_files_resolves_internal_imports_and_excludes_external_files(
    tmp_path,
):
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "main.py").write_text(
        "import package.helper\nimport os\n",
        encoding="utf-8",
    )
    package = repository / "package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    tools = CodeIntelligenceTools(
        "repo",
        repository_path_resolver=lambda name: str(repository),
    )

    result = tools.call("related_files", {"file_path": "main.py"})

    assert result == {
        "source_file": "main.py",
        "related_files": ["package/helper.py"],
    }


def test_related_files_rejects_path_attacks_and_missing_roots(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "main.py").write_text("import os\n", encoding="utf-8")
    tools = CodeIntelligenceTools(
        "repo",
        repository_path_resolver=lambda name: str(repository),
        max_tool_calls=5,
    )

    for file_path in (
        "../main.py",
        r"..\main.py",
        r"C:\main.py",
        r"\\server\share\main.py",
    ):
        with pytest.raises(ToolValidationError):
            tools.call("related_files", {"file_path": file_path})

    missing_tools = CodeIntelligenceTools(
        "missing",
        repository_path_resolver=lambda name: None,
    )
    with pytest.raises(ToolValidationError, match="No registered path"):
        missing_tools.call("related_files", {"file_path": "main.py"})


def test_inspect_repository_is_sorted_bounded_and_ignores_directories(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    for relative_path in ("z.py", "a.py", "nested/m.py"):
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n", encoding="utf-8")
    (repository / ".venv").mkdir()
    (repository / ".venv" / "hidden.py").write_text("", encoding="utf-8")
    (repository / "tests").mkdir()
    (repository / "tests" / "test_hidden.py").write_text("", encoding="utf-8")
    (repository / "__pycache__").mkdir()
    (repository / "__pycache__" / "hidden.py").write_text("", encoding="utf-8")
    tools = CodeIntelligenceTools(
        "repo",
        repository_path_resolver=lambda name: str(repository),
    )

    result = tools.call("inspect_repository", {"limit": 2})

    assert result == {"files": ["a.py", "nested/m.py"]}


def test_inspect_repository_rejects_invalid_limits_and_non_directory_root(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    tools = CodeIntelligenceTools(
        "repo",
        repository_path_resolver=lambda name: str(repository),
    )

    with pytest.raises(ToolValidationError):
        tools.call("inspect_repository", {"limit": 0})
    with pytest.raises(ToolValidationError):
        tools.call("inspect_repository", {"unexpected": True})

    file_root = tmp_path / "file.py"
    file_root.write_text("", encoding="utf-8")
    invalid_tools = CodeIntelligenceTools(
        "repo",
        repository_path_resolver=lambda name: str(file_root),
    )
    with pytest.raises(ToolValidationError, match="not a directory"):
        invalid_tools.call("inspect_repository", {})


def test_tool_names_arguments_and_budget_are_validated():
    tools = CodeIntelligenceTools(
        "repo",
        retriever=MagicMock(),
        max_tool_calls=1,
    )

    with pytest.raises(ToolValidationError, match="Unknown tool"):
        tools.call("unknown", {})
    with pytest.raises(ToolValidationError, match="arguments"):
        tools.call("search_code", [])

    tools.retriever.search.return_value = {"results": []}
    tools.call("search_code", {"query": "anything"})

    with pytest.raises(ToolBudgetExceeded):
        tools.call("search_code", {"query": "again"})


def test_tool_budget_is_consumed_when_recognized_tool_raises():
    retriever = MagicMock()
    retriever.search.side_effect = RuntimeError("retriever unavailable")
    tools = CodeIntelligenceTools("repo", retriever=retriever, max_tool_calls=1)

    with pytest.raises(RuntimeError, match="unavailable"):
        tools.call("search_code", {"query": "anything"})

    with pytest.raises(ToolBudgetExceeded):
        tools.call("search_code", {"query": "again"})
