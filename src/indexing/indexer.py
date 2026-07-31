from src.ingestion.loader import load_python_file
from src.ingestion.reader import read_python_file
from src.parser.ast_parser import parse_code
from src.chunking.chunker import extract_chunks
from src.utils.storage import save_chunks
from src.storage.chroma_store import (store_chunk, delete_chunks_by_file,reset_database)
from src.embeddings.embedder import get_embedding
from src.utils.file_hash import get_hash_file
from src.utils.index_state import load_index_state, save_index_state
from src.utils.schema_version import (
    SCHEMA_VERSION,
    load_schema_version,
    save_schema_version,
)
from src.utils.repositories import save_repository
import os


def index_codebase(repo_path):
    repository_name = os.path.basename(os.path.normpath(repo_path))
    files = load_python_file(repo_path)

    saved_version = load_schema_version()

    if saved_version != SCHEMA_VERSION:
        print("=" * 60)
        print("Schema version changed.")
        print(f"Old Version : {saved_version}")
        print(f"New Version : {SCHEMA_VERSION}")
        print("Full re-index required.")
        print("=" * 60)

        reset_database()

        if os.path.exists("data/index_state.json"):
            os.remove("data/index_state.json")

        if os.path.exists("data/file_hashes.json"):
            os.remove("data/file_hashes.json")

        old_state = {}

        save_schema_version()


    if saved_version == SCHEMA_VERSION:
      old_state = load_index_state()
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
            print(f"Skipping unchanged file : {file}")
            continue

        if file_key in old_state:
            modified_files_count += 1
            print(f"File modified, removing old chunks : {file}")
            delete_chunks_by_file(repository_name, file_key)
        else:
            new_files_count += 1

        print("=" * 50)
        print(f"Reading : {file}")
        print("=" * 50)

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
        print(f"File deleted, removing old chunks : {deleted_file}")
        delete_chunks_by_file(repository_name, deleted_file)

    save_chunks(all_chunks, "data/chunks.json")

    for chunk in all_chunks:
        embedding = get_embedding(chunk["code"])
        store_chunk(chunk, embedding)

    print(f"\nStored {len(all_chunks)} chunks in ChromaDB successfully.")

    save_index_state(new_state)

    save_repository(repository_name)

    print("\n" + "=" * 50)
    print("INDEXING SUMMARY")
    print("=" * 50)
    print(f"New files       : {new_files_count}")
    print(f"Modified files  : {modified_files_count}")
    print(f"Unchanged files : {unchanged_files_count}")
    print(f"Deleted files   : {deleted_files_count}")
    print(f"Chunks indexed  : {len(all_chunks)}")
    print("=" * 50)