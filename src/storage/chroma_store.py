import chromadb 
import shutil
import os

client = None
collection = None


def get_collection():
    global client, collection

    if collection is None:
        client = chromadb.PersistentClient(path="data/chroma_db")

        collection = client.get_or_create_collection(
            name="code_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    return collection

def store_chunk(chunk, embedding):
    collection = get_collection()
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
        collection = get_collection()
        collection.delete(
    where={
        "$and": [
            {"repository": repository},
            {"file_path": str(file_path)}
        ]
    }
)  

def delete_chunks_by_repository(repository):
    collection = get_collection()
    collection.delete(
    where={
        "repository": repository
    }
)

def reset_database():
    global client, collection

    client = None
    collection = None

    if os.path.exists("data/chroma_db"):
        shutil.rmtree("data/chroma_db")