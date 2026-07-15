from src.ingestion.loader import load_python_file
from src.ingestion.reader import read_python_file
from src.parser.ast_parser import parse_code
from src.chunking.chunker import extract_chunks
from src.utils.storage import save_chunks
from src.storage.chroma_store import store_chunk, delete_chunks_by_file
from src.embeddings.embedder import get_embedding
from src.utils.file_hash import get_hash_file
from src.utils.index_state import load_index_state, save_index_state


def index_codebase(repo_path):
    files = load_python_file(repo_path)

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
            delete_chunks_by_file(file_key)
        else:
            new_files_count += 1

        print("=" * 50)
        print(f"Reading : {file}")
        print("=" * 50)

        code = read_python_file(file)
        tree = parse_code(code)
        chunks = extract_chunks(tree, code, file)
        all_chunks.extend(chunks)

    deleted_files = set(old_state.keys()) - set(new_state.keys())

    for deleted_file in deleted_files:
        deleted_files_count += 1
        print(f"File deleted, removing old chunks : {deleted_file}")
        delete_chunks_by_file(deleted_file)

    save_chunks(all_chunks, "data/chunks.json")

    for chunk in all_chunks:
        embedding = get_embedding(chunk["code"])
        store_chunk(chunk, embedding)

    print(f"\nStored {len(all_chunks)} chunks in ChromaDB successfully.")

    save_index_state(new_state)

    print("\n" + "=" * 50)
    print("INDEXING SUMMARY")
    print("=" * 50)
    print(f"New files       : {new_files_count}")
    print(f"Modified files  : {modified_files_count}")
    print(f"Unchanged files : {unchanged_files_count}")
    print(f"Deleted files   : {deleted_files_count}")
    print(f"Chunks indexed  : {len(all_chunks)}")
    print("=" * 50)