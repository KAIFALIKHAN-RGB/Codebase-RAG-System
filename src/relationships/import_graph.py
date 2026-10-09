import ast
from pathlib import Path


class ImportGraph:
    """Repository-local Python import graph."""

    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.unresolved = {}

    def add_node(self, node):
        if not hasattr(self, "nodes"):
            self.nodes = set()
        if not hasattr(self, "edges"):
            self.edges = {}

        self.nodes.add(node)
        self.edges.setdefault(node, set())
        

    def add_edge(self, source, target):
        if not hasattr(self, "nodes"):
            self.nodes = set()
        if not hasattr(self, "edges"):
            self.edges = {}

        self.nodes.add(source)
        self.nodes.add(target)
        self.edges.setdefault(source, set())
        self.edges[source].add(target)


def build_import_graph(repository_path):
    """
    Build an import graph for Python files inside a repository.

    Only imports that resolve to Python files inside the same repository
    are represented as edges.
    """

    repository_path = Path(repository_path).resolve()

    graph = ImportGraph()

    ignored_dirs = {
        ".venv",
        "venv",
        "pycache",
        ".git",
        "tests",
    }

    python_files = [
        path
        for path in repository_path.rglob("*.py")
        if not any(part in ignored_dirs for part in path.parts)
    ]

    # Map module names to actual files.
    module_map = {}

    for file_path in python_files:
        try:
            code = file_path.read_text(encoding="utf-8")
            ast.parse(code, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            # Keep malformed files as graph nodes,
            # but never use them as import-resolution targets.
            continue

        relative = file_path.relative_to(repository_path)
        parts = list(relative.with_suffix("").parts)

        if parts[-1] == "init":
            module_parts = parts[:-1]
        else:
            module_parts = parts

        module_name = ".".join(module_parts)

        if module_name:
            module_map[module_name] = file_path

    # Add every valid Python file as a graph node.
    for file_path in python_files:
        source = _relative_path(file_path, repository_path)
        graph.add_node(source)

    # Resolve imports.
    for file_path in python_files:
        source = _relative_path(file_path, repository_path)

        try:
            code = file_path.read_text(encoding="utf-8")
            tree = ast.parse(code, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            # Malformed/unreadable files remain nodes but produce no edges.
            continue

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = _resolve_absolute_import(
                        alias.name,
                        module_map,
                        repository_path,
                    )

                    if target:
                        graph.add_edge(source, target)

                    else:
                        graph.unresolved.setdefault(source,set()).add(alias.name)

            elif isinstance(node, ast.ImportFrom):
                target = _resolve_from_import(
                    file_path=file_path,
                    module=node.module,
                    level=node.level,
                    module_map=module_map,
                    repository_path=repository_path,
                )

                if target:
                    graph.add_edge(source, target)
                else:
                    if node.module:
                        graph.unresolved.setdefault(source,set()).add(node.module)

    return graph


def _relative_path(file_path, repository_path):
    """Return a normalized repository-relative path."""

    return file_path.relative_to(repository_path).as_posix()


def _resolve_absolute_import(module_name, module_map, repository_path):
    """Resolve an absolute import to a repository-local Python file."""

    # Exact module match.
    if module_name in module_map:
        return _relative_path(module_map[module_name], repository_path)

    parts = module_name.split(".")

    # Example:
    # from package.submodule import Class
    # should resolve package.submodule.py or package/submodule/init.py.
    while parts:
        candidate = ".".join(parts)

        if candidate in module_map:
            return _relative_path(module_map[candidate], repository_path)

        parts.pop()

    return None


def _resolve_from_import(
    file_path,
    module,
    level,
    module_map,
    repository_path,
):
    """Resolve absolute and relative 'from ... import ...' statements."""

    if level == 0:
        if not module:
            return None

        return _resolve_absolute_import(
            module,
            module_map,
            repository_path,
        )
    # Determine the package containing the current file.
    relative = file_path.relative_to(repository_path)
    parts = list(relative.with_suffix("").parts)

    if parts[-1] == "__init__":
        package_parts = parts[:-1]
    else:
        package_parts = parts[:-1]

    # Relative import:
    # .foo      -> current package + foo
    # ..foo     -> parent package + foo
    if level > len(package_parts) + 1:
        return None

    base = package_parts[: len(package_parts) - level + 1]

    if module:
        base.extend(module.split("."))

    if not base:
        return None

    module_name = ".".join(base)

    return _resolve_absolute_import(
        module_name,
        module_map,
        repository_path,
    )