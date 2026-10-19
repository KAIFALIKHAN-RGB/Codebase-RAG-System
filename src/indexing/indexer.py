from src.ingestion.loader import load_python_file
from src.ingestion.reader import read_python_file
from src.parser.ast_parser import parse_code
from src.chunking.chunker import extract_chunks
from src.utils.storage import save_chunks
from src.storage.chroma_store import (store_chunk, delete_chunks_by_file,reset_database)
from src.embeddings.embedder import get_embedding
from src.utils.file_hash import get_hash_file
from src.utils.index_state import (load_repository_state, save_repository_state)
from src.utils.index_status import set_index_status
from src.utils.schema_version import (
    SCHEMA_VERSION,
    load_schema_version,
    save_schema_version,
)
from src.utils.repositories import save_repository
from filelock import FileLock
import logging
import os


logger = logging.getLogger(__name__)


def _index_codebase(repo_path):
    repository_name = os.path.basename(os.path.normpath(repo_path))
    lock_dir = "data/repository_locks"
    os.makedirs(lock_dir, exist_ok=True)

    lock_file = os.path.join(lock_dir, f"{repository_name}.lock")

    with FileLock(lock_file):
        return _run_indexing(repo_path, repository_name)

def _run_indexing(repo_path, repository_name):
    
    files = load_python_file(repo_path)

    saved_version = load_schema_version()

    if saved_version != SCHEMA_VERSION:
        logger.warning(
            "Schema version changed from %s to %s; full re-index required.",
            saved_version,
            SCHEMA_VERSION,
        )

        reset_database()

        if os.path.exists("data/index_state.json"):
            os.remove("data/index_state.json")

        if os.path.exists("data/file_hashes.json"):
            os.remove("data/file_hashes.json")

        old_state = {}

        save_schema_version()


    if saved_version == SCHEMA_VERSION:
      old_state = load_repository_state(repository_name)
    new_state = {}

    all_chunks = []

    new_files_count = 0
    modified_files_count = 0
    unchanged_files_count = 0
    deleted_files_count = 0

    for file in files:
        file_key = str(file)
        current_hash = get_hash_file(file)

        new_state[file_key] = current_hash

        if old_state.get(file_key) == current_hash:
            unchanged_files_count += 1
            logger.debug("Skipping unchanged file: %s", file)
            continue

        if file_key in old_state:
            modified_files_count += 1
            logger.debug("File modified; removing old chunks: %s", file)
            delete_chunks_by_file(repository_name, file_key)
        else:
            new_files_count += 1

        logger.debug("Reading: %s", file)

        code = read_python_file(file)
        tree = parse_code(code)
        chunks = extract_chunks(
            tree,
            code,
            file,
            repository_name
            )
        all_chunks.extend(chunks)

    deleted_files = set(old_state.keys()) - set(new_state.keys())

    for deleted_file in deleted_files:
        deleted_files_count += 1
        logger.debug("File deleted; removing old chunks: %s", deleted_file)
        delete_chunks_by_file(repository_name, deleted_file)

    save_chunks(all_chunks, "data/chunks.json")

    for chunk in all_chunks:
        embedding = get_embedding(chunk["code"])
        store_chunk(chunk, embedding)

    logger.info("Stored %s chunks in ChromaDB successfully.", len(all_chunks))

    save_repository_state(repository_name, new_state)

    save_repository(repository_name)

    logger.info(
        "Indexing summary for %s: new=%s modified=%s unchanged=%s "
        "deleted=%s chunks=%s",
        repository_name,
        new_files_count,
        modified_files_count,
        unchanged_files_count,
        deleted_files_count,
        len(all_chunks),
    )

def index_codebase(repo_path):
        repository_name = os.path.basename(os.path.normpath(repo_path))

        set_index_status(repository_name, "indexing")

        try:
            _index_codebase(repo_path)
            set_index_status(repository_name, "completed")

        except Exception:
            set_index_status(repository_name, "failed")
            logger.exception("Indexing failed for repository %s.", repository_name)
            raise
    
    