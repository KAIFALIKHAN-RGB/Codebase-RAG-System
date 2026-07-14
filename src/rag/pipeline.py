from src.retrieval.retriever import search
from src.generation.context_builder import build_context
from src.generation.llm_client import generate_answer


def run_rag_pipeline(question, k=3, threshold=30.0):
    """
    Run the complete RAG pipeline:
    retrieval -> context building -> answer generation
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Step 1: Retrieve relevant code chunks
    retrieval_output = search(
        query=question,
        k=k,
        threshold=threshold
    )

    retrieved_chunks = retrieval_output["results"]

    # Step 2: Stop if no relevant code was found
    if not retrieved_chunks:
        return {
            "answer": "No relevant code chunks found.",
            "sources": [],
            "retrieval_time_ms": retrieval_output["retrieval_time_ms"]
        }

    # Step 3: Build structured context for the LLM
    context = build_context(retrieved_chunks)

    # Step 4: Generate a grounded answer
    answer = generate_answer(question, context)

    # Step 5: Return structured pipeline output
    return {
        "answer": answer,
        "sources": retrieved_chunks,
        "retrieval_time_ms": retrieval_output["retrieval_time_ms"]
    }