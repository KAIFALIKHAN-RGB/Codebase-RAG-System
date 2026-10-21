import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

import src.retrieval.retriever as retriever_module
from src.ingestion.loader import load_python_file
from src.relationships.import_graph import build_import_graph
from src.utils.repository_paths import get_repository_path
from src.storage import chroma_store
from src.storage.storage_lock import storage_access


MAX_TOOL_CALLS = 3
MAX_RESULTS = 5
MAX_INSPECTED_FILES = 20
MAX_SOURCE_LINES = 200
MAX_SOURCE_CHARACTERS = 12_000


class ToolValidationError(ValueError):
    """Raised when a tool name or argument is not allowed."""


class ToolBudgetExceeded(RuntimeError):
    """Raised when the per-request tool-call limit is reached."""


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolValidationError(f"{name} must be a non-empty string.")
    return value.strip()


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ToolValidationError(f"{name} must be a positive integer.")
    return value


def _is_absolute_path(value: str) -> bool:
    return (
        Path(value).is_absolute()
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    )


def _relative_source_path(repository_root: Path, file_path: str) -> Path:
    if _is_absolute_path(file_path):
        raise ToolValidationError("file_path must be repository-relative.")

    candidate = (repository_root / Path(file_path.replace("\\", os.sep))).resolve()

    try:
        candidate.relative_to(repository_root)
    except ValueError as error:
        raise ToolValidationError(
            "file_path must remain inside the selected repository."
        ) from error

    return candidate


def _chunk_result(document: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": document,
        "metadata": metadata,
    }


class CodeIntelligenceTools:
    """Small, bounded tool registry for repository code inspection."""

    def __init__(
        self,
        repository: str,
        *,
        retriever: Any = retriever_module,
        collection: Any = None,
        repository_path_resolver: Callable[[str], str | None] = (
            get_repository_path
        ),
        max_tool_calls: int = MAX_TOOL_CALLS,
    ):
        self.repository = _require_text(repository, "repository")
        self.retriever = retriever
        self.collection = (
            collection if collection is not None else retriever_module.collection
        )
        self._uses_default_collection = collection is None
        self._collection_generation = chroma_store.get_storage_generation()
        self.repository_path_resolver = repository_path_resolver

        if (
            not isinstance(max_tool_calls, int)
            or isinstance(max_tool_calls, bool)
            or max_tool_calls <= 0
        ):
            raise ValueError("max_tool_calls must be a positive integer.")

        self.max_tool_calls = max_tool_calls
        self.tool_calls = 0
        self._registry = {
            "search_code": self._search_code,
            "lookup_symbol": self._lookup_symbol,
            "read_source": self._read_source,
            "related_files": self._related_files,
            "inspect_repository": self._inspect_repository,
        }

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._registry)

    def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self._registry:
            raise ToolValidationError(f"Unknown tool: {tool_name}")

        if not isinstance(arguments, dict):
            raise ToolValidationError("Tool arguments must be an object.")

        if self.tool_calls >= self.max_tool_calls:
            raise ToolBudgetExceeded(
                f"Maximum tool calls ({self.max_tool_calls}) exceeded."
            )

        self.tool_calls += 1
        return self._registry[tool_name](arguments)

    def _search_code(self, arguments: dict[str, Any]) -> dict[str, Any]:
        allowed = {"query", "k", "threshold"}
        unknown = set(arguments) - allowed
        if unknown:
            raise ToolValidationError(
                f"Unsupported search_code arguments: {sorted(unknown)}"
            )

        query = _require_text(arguments.get("query"), "query")
        k = arguments.get("k", MAX_RESULTS)
        threshold = arguments.get("threshold", 35.0)

        _require_positive_int(k, "k")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ToolValidationError("threshold must be numeric.")

        return self.retriever.search(
            query=query,
            repository=self.repository,
            k=min(k, MAX_RESULTS),
            threshold=threshold,
        )

    def _lookup_symbol(self, arguments: dict[str, Any]) -> dict[str, Any]:
        allowed = {"symbol", "symbol_type"}
        unknown = set(arguments) - allowed
        if unknown:
            raise ToolValidationError(
                f"Unsupported lookup_symbol arguments: {sorted(unknown)}"
            )

        symbol = _require_text(arguments.get("symbol"), "symbol")
        symbol_type = arguments.get("symbol_type")
        if symbol_type is not None:
            symbol_type = _require_text(symbol_type, "symbol_type")

        filters = [
            {"repository": self.repository},
            {"name": symbol},
        ]
        if symbol_type is not None:
            filters.append({"type": symbol_type})

        with storage_access():
            if self._uses_default_collection and (
                self.collection is None
                or self._collection_generation
                != chroma_store.get_storage_generation()
            ):
                self.collection = chroma_store.get_collection_unlocked()
                self._collection_generation = (
                    chroma_store.get_storage_generation()
                )

            result = self.collection.get(
                where={"$and": filters},
                limit=MAX_RESULTS,
                include=["documents", "metadatas"],
            )
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        return {
            "results": [
                _chunk_result(document, metadata)
                for document, metadata in zip(documents, metadatas)
                if isinstance(document, str) and isinstance(metadata, dict)
            ]
        }

    def _read_source(self, arguments: dict[str, Any]) -> dict[str, Any]:
        allowed = {"file_path", "start_line", "end_line", "max_characters"}
        unknown = set(arguments) - allowed
        if unknown:
            raise ToolValidationError(
                f"Unsupported read_source arguments: {sorted(unknown)}"
            )

        file_path = _require_text(arguments.get("file_path"), "file_path")
        start_line = arguments.get("start_line", 1)
        end_line = arguments.get("end_line")
        max_characters = arguments.get(
            "max_characters",
            MAX_SOURCE_CHARACTERS,
        )

        _require_positive_int(start_line, "start_line")
        if end_line is not None:
            _require_positive_int(end_line, "end_line")
            if end_line < start_line:
                raise ToolValidationError(
                    "end_line must be greater than or equal to start_line."
                )
        if (
            not isinstance(max_characters, int)
            or isinstance(max_characters, bool)
            or max_characters <= 0
            or max_characters > MAX_SOURCE_CHARACTERS
        ):
            raise ToolValidationError(
                f"max_characters must be between 1 and {MAX_SOURCE_CHARACTERS}."
            )

        repository_path = self.repository_path_resolver(self.repository)
        if not repository_path:
            raise ToolValidationError(
                f"No registered path for repository {self.repository!r}."
            )

        repository_root = Path(repository_path).resolve()
        if not repository_root.is_dir():
            raise ToolValidationError("Registered repository path is not a directory.")

        source_path = _relative_source_path(repository_root, file_path)
        if not source_path.is_file():
            raise ToolValidationError("file_path must identify a regular file.")

        requested_end = (
            end_line
            if end_line is not None
            else start_line + MAX_SOURCE_LINES - 1
        )
        if requested_end - start_line + 1 > MAX_SOURCE_LINES:
            raise ToolValidationError(
                f"Source ranges are limited to {MAX_SOURCE_LINES} lines."
            )

        selected_lines = []
        with source_path.open("r", encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                if line_number < start_line:
                    continue
                if line_number > requested_end:
                    break
                selected_lines.append(line)

        if not selected_lines:
            raise ToolValidationError("Requested source range is empty.")

        code = "".join(selected_lines)
        if len(code) > max_characters:
            raise ToolValidationError(
                f"Source output is limited to {max_characters} characters."
            )

        return {
            "repository": self.repository,
            "file_path": source_path.relative_to(repository_root).as_posix(),
            "start_line": start_line,
            "end_line": start_line + len(selected_lines) - 1,
            "code": code,
        }

    def _repository_root(self) -> Path:
        repository_path = self.repository_path_resolver(self.repository)
        if not repository_path:
            raise ToolValidationError(
                f"No registered path for repository {self.repository!r}."
            )

        repository_root = Path(repository_path).resolve()
        if not repository_root.is_dir():
            raise ToolValidationError("Registered repository path is not a directory.")
        return repository_root

    def _related_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if set(arguments) != {"file_path"}:
            raise ToolValidationError(
                "related_files requires only file_path."
            )

        file_path = _require_text(arguments["file_path"], "file_path")
        repository_root = self._repository_root()
        source_path = _relative_source_path(repository_root, file_path)
        if not source_path.is_file():
            raise ToolValidationError("file_path must identify a regular file.")

        source_relative = source_path.relative_to(repository_root).as_posix()
        try:
            graph = build_import_graph(repository_root)
        except (OSError, UnicodeError) as error:
            raise ToolValidationError(
                "The repository import graph could not be read."
            ) from error

        related = []
        for related_path in sorted(graph.edges.get(source_relative, set())):
            try:
                candidate = _relative_source_path(repository_root, related_path)
            except ToolValidationError:
                continue
            if candidate.is_file():
                related.append(candidate.relative_to(repository_root).as_posix())

        return {
            "source_file": source_relative,
            "related_files": related[:MAX_RESULTS],
        }

    def _inspect_repository(self, arguments: dict[str, Any]) -> dict[str, Any]:
        allowed = {"limit"}
        unknown = set(arguments) - allowed
        if unknown:
            raise ToolValidationError(
                f"Unsupported inspect_repository arguments: {sorted(unknown)}"
            )

        limit = arguments.get("limit", MAX_INSPECTED_FILES)
        _require_positive_int(limit, "limit")
        limit = min(limit, MAX_INSPECTED_FILES)
        repository_root = self._repository_root()

        try:
            files = load_python_file(repository_root)
        except OSError as error:
            raise ToolValidationError(
                "The repository files could not be inspected."
            ) from error

        normalized = set()
        for file_path in files:
            try:
                candidate = file_path.resolve()
                candidate.relative_to(repository_root)
            except (OSError, ValueError):
                continue
            if candidate.is_file() and candidate.suffix.lower() == ".py":
                normalized.add(candidate.relative_to(repository_root).as_posix())

        return {"files": sorted(normalized)[:limit]}
