from pathlib import Path

def load_python_file(folder_path):
    python_files = []

    for file in Path(folder_path).rglob("*.py"):
        python_files.append(file)

    return python_files