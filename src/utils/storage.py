import json
import os
import tempfile
from urllib.parse import quote
from pathlib import Path


def safe_repository_name(repository_name):
    name = str(repository_name)
    if not name:
        raise ValueError("Repository name must contain a safe filename.")
    return quote(name, safe="-_.")


def repository_chunks_path(repository_name):
    return Path("data") / "chunks" / (
        f"{safe_repository_name(repository_name)}.json"
    )


def save_chunks(chunks, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(chunks, file, indent=4, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()