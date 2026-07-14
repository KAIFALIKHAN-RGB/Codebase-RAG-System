from src.retrieval.retriever import search
from src.generation.context_builder import build_context

query = "subtract two numbers"

retrieval_response = search(query)

retrieved_chunks = retrieval_response["results"]
context = build_context(retrieved_chunks)

print("\nGenerated Context:\n")
print(context)

print(
    f"\nRetrieval Time: "
    f"{retrieval_response['retrieval_time_ms']} ms"
)