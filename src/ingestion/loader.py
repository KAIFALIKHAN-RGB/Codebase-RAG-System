from pathlib import Path

def load_python_file(folder_path):
    python_files = []
    ignored_dirs = {".venv", "venv", "__pycache__", ".git", "tests"}

    for file in Path(folder_path).rglob("*.py"):
        if any (part in ignored_dirs for part in file.parts):
            continue
        python_files.append(file)

    return python_files