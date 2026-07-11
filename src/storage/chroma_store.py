import chromadb 

client = chromadb.PersistentClient(path="data/chroma_db")

collection = client.get_or_create_collection(name="code_chunks", metadata={"hnsw:space": "cosine"})

def store_chunk(chunk, embedding):
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
            "file_path": chunk["file_path"],
            "parent_class": str(chunk["parent_class"]),
            "return_type": str(chunk["return_type"])
        }]
    )