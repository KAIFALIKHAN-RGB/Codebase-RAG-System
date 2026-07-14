import time
import re
import chromadb

from src.embeddings.embedder import get_embedding


# Connect to the persistent ChromaDB database
client = chromadb.PersistentClient(path="data/chroma_db")

# Load the existing code chunks collection
collection = client.get_collection("code_chunks")


def search(query, k=3, threshold=30.0):
    """
    Search the codebase for the most relevant code chunks.

    Args:
        query: User's search query.
        k: Maximum number of results to retrieve.
        threshold: Minimum similarity percentage required.

    Returns:
        A dictionary containing:
        - relevant search results
        - retrieval time in milliseconds
    """

    # Start measuring end-to-end retrieval time
    start_time = time.perf_counter()

    # Convert the user's query into an embedding
    query_embedding = get_embedding(query)

    #Normalize query into lowercase words for symbol matching
    query_words = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))

    # Retrieve the top-k nearest code chunks from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    relevant_results = []

    # Extract returned data
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Process and filter retrieved results
    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        # Convert cosine distance into similarity percentage
        similarity = max(0, (1 - distance) * 100)

        # Get and normalize the code symbol name
        symbol_name = metadata.get("name", "")

        #Check whether the symbol name appears in the user's query words
        symbol_words = set(re.findall(r"[a-zA-Z0-9]+", symbol_name.lower().replace("_", " ")))

        if symbol_words and symbol_words.issubset(query_words):
            similarity += 15  # Boost similarity if symbol matches query

        # Skip results below the similarity threshold
        if similarity < threshold:
            continue

        # Store clean, structured result
        relevant_results.append({
            "code": document,
            "metadata": metadata,
            "distance": distance,
            "similarity": round(similarity, 2)
        })

    # Calculate total retrieval time in milliseconds
    retrieval_time_ms = (time.perf_counter() - start_time) * 1000

    # Return API-ready structured data
    return {
        "results": relevant_results,
        "retrieval_time_ms": round(retrieval_time_ms, 2)
    }