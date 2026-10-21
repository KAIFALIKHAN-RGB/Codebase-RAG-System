import chromadb 
import shutil
import os
from src.storage.storage_lock import storage_access

client = None
collection = None
_storage_generation = 0
_reset_callbacks = []


def get_collection():
    with storage_access():
        return get_collection_unlocked()


def get_collection_unlocked():
    global client, collection

    if collection is None:
        client = chromadb.PersistentClient(path="data/chroma_db")

        collection = client.get_or_create_collection(
            name="code_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    return collection


def get_storage_generation():
    return _storage_generation


def register_reset_callback(callback):
    if not any(existing is callback for existing in _reset_callbacks):
        _reset_callbacks.append(callback)


def store_chunk(chunk, embedding):
    with storage_access():
        store_chunk_unlocked(chunk, embedding)


def store_chunk_unlocked(chunk, embedding):
    collection = get_collection_unlocked()
    collection.upsert(
        ids=[f"{chunk['file_path']}:{chunk['start_line']}:{chunk['end_line']}"],
        documents=[chunk["code"]],
        embeddings=[embedding],
        metadatas=[{
            "name": chunk["name"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "parameters": ",".join(chunk["parameters"]) if chunk["parameters"] else "",
            "type": chunk["type"],
            "repository": chunk["repository"],
            "file_path": chunk["file_path"],
            "parent_class": str(chunk["parent_class"]),
            "return_type": str(chunk["return_type"])
        }]
    )


def delete_chunks_by_file(repository, file_path):
    with storage_access():
        delete_chunks_by_file_unlocked(repository, file_path)


def delete_chunks_by_file_unlocked(repository, file_path):
        collection = get_collection_unlocked()
        collection.delete(
    where={
        "$and": [
            {"repository": repository},
            {"file_path": str(file_path)}
        ]
    }
)  

def delete_chunks_by_repository(repository):
    with storage_access():
        delete_chunks_by_repository_unlocked(repository)


def delete_chunks_by_repository_unlocked(repository):
    collection = get_collection_unlocked()
    collection.delete(
    where={
        "repository": repository
    }
)

def reset_database():
    with storage_access():
        reset_database_unlocked()


def reset_database_unlocked():
    global client, collection, _storage_generation

    for callback in tuple(_reset_callbacks):
        callback()
    client = None
    collection = None
    _storage_generation += 1

    if os.path.exists("data/chroma_db"):
        shutil.rmtree("data/chroma_db")