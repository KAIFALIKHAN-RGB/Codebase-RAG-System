from pathlib import Path

from src.relationships.import_graph import build_import_graph


def write_file(root: Path, relative_path: str, content: str):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_resolves_absolute_internal_import(tmp_path):
    write_file(tmp_path, "main.py", "from utils.helpers import add\n")
    write_file(tmp_path, "utils/init.py", "")
    write_file(tmp_path, "utils/helpers.py", "def add(): pass\n")

    graph = build_import_graph(tmp_path)

    assert graph.edges["main.py"] == {"utils/helpers.py"}
    assert graph.unresolved == {}


def test_resolves_relative_import(tmp_path):
    write_file(tmp_path, "package/init.py", "")
    write_file(tmp_path, "package/main.py", "from .utils import helper\n")
    write_file(tmp_path, "package/utils.py", "def helper(): pass\n")

    graph = build_import_graph(tmp_path)

    assert graph.edges["package/main.py"] == {"package/utils.py"}


def test_handles_missing_module_without_crashing(tmp_path):
    write_file(tmp_path, "main.py", "from missing.module import value\n")

    graph = build_import_graph(tmp_path)

    assert graph.edges["main.py"] == set()
    assert graph.unresolved["main.py"] == {"missing.module"}


def test_supports_circular_imports(tmp_path):
    write_file(tmp_path, "a.py", "import b\n")
    write_file(tmp_path, "b.py", "import a\n")

    graph = build_import_graph(tmp_path)

    assert graph.edges["a.py"] == {"b.py"}
    assert graph.edges["b.py"] == {"a.py"}


def test_deduplicates_duplicate_imports(tmp_path):
    write_file(
        tmp_path,
        "main.py",
        "import utils\nimport utils\nfrom utils import helper\n",
    )
    write_file(tmp_path, "utils.py", "helper = 1\n")

    graph = build_import_graph(tmp_path)

    assert graph.edges["main.py"] == {"utils.py"}


def test_ignores_cross_repository_absolute_path(tmp_path):
    outside = tmp_path.parent / "outside_module.py"
    outside.write_text("value = 1\n", encoding="utf-8")

    write_file(
        tmp_path,
        "main.py",
        f"import {outside.stem}\n",
    )

    graph = build_import_graph(tmp_path)

    assert graph.edges["main.py"] == set()


def test_skips_malformed_python_files(tmp_path):
    write_file(tmp_path, "broken.py", "def broken(:\n")
    write_file(tmp_path, "main.py", "import broken\n")

    graph = build_import_graph(tmp_path)

    assert "broken.py" in graph.nodes
    assert graph.edges["main.py"] == set()


def test_does_not_include_external_packages_as_internal_edges(tmp_path):
    write_file(
        tmp_path,
        "main.py",
        "import os\nimport numpy\nfrom fastapi import FastAPI\n",
    )

    graph = build_import_graph(tmp_path)

    assert graph.edges["main.py"] == set()
    assert graph.unresolved["main.py"] == {
        "os",
        "numpy",
        "fastapi",
    }