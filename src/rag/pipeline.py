from src.retrieval.retriever import search
from src.generation.context_builder import build_context
from src.generation.llm_client import generate_answer
from src.relationships.import_graph import build_import_graph
from src.retrieval.graph_aware_retriever import graph_aware_search
from src.utils.repository_paths import get_repository_path
import src.retrieval.retriever as retriever_module


def run_rag_pipeline(question,repository=None, k=3, threshold=35.0):
    """
    Run the complete RAG pipeline:
    retrieval -> context building -> answer generation
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Step 1: Retrieve relevant code chunks
    repository_path = (
        get_repository_path(repository)
        if repository
        else None
    )

    if repository_path:
        try:
            graph = build_import_graph(repository_path)

            retrieval_output = graph_aware_search(
                retriever=retriever_module,
                graph=graph,
                query=question,
                repository=repository,
                k=k,
                threshold=threshold,
            )
        except Exception:
            retrieval_output = search(
                query=question,
                repository=repository,
                k=k,
                threshold=threshold,
            )
    else:
        retrieval_output = search(
            query=question,
            repository=repository,
            k=k,
            threshold=threshold,
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