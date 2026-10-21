import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.retrieval.retriever as retriever


NEGATIVE_QUERIES = [
    "How does Click implement CUDA kernel scheduling?",
    "How does Click manage blockchain transaction signing?",
    "How does Click implement a PostgreSQL database engine?",
    "How does Click train machine learning models?",
    "How does Click perform image classification using neural networks?",
]


def test_negative_queries_return_no_results(monkeypatch):
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection("code_chunks")
    collection.upsert(
        ids=["other-repository:src/example.py:1:1"],
        documents=["def example(): pass"],
        embeddings=[[0.1] * 384],
        metadatas=[
            {
                "repository": "other_repository",
                "file_path": "src/example.py",
                "name": "example",
                "start_line": 1,
                "end_line": 1,
            }
        ],
    )

    monkeypatch.setattr(retriever, "collection", collection)
    monkeypatch.setattr(retriever, "_managed_collection", collection)
    monkeypatch.setattr(
        retriever,
        "_managed_generation",
        retriever.chroma_store.get_storage_generation(),
    )
    monkeypatch.setattr(retriever, "_managed_query", collection.query)
    monkeypatch.setattr(retriever, "get_embedding", lambda text: [0.1] * 384)

    for query in NEGATIVE_QUERIES:
        result = retriever.search(
            query,
            repository="heldout_click",
            k=5,
            threshold=35.0,
        )

        assert result["results"] == [], (
            f"False positive for query: {query}\n"
            f"Retrieved: {result['results']}"
        )