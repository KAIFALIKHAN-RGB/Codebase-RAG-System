import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.retriever import search


NEGATIVE_QUERIES = [
    "How does Click implement CUDA kernel scheduling?",
    "How does Click manage blockchain transaction signing?",
    "How does Click implement a PostgreSQL database engine?",
    "How does Click train machine learning models?",
    "How does Click perform image classification using neural networks?",
]


def test_negative_queries_return_no_results():
    for query in NEGATIVE_QUERIES:
        result = search(
            query,
            repository="heldout_click",
            k=5,
            threshold=35.0,
        )

        assert result["results"] == [], (
            f"False positive for query: {query}\n"
            f"Retrieved: {result['results']}"
        )