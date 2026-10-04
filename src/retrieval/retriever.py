import time
import re
import chromadb

from src.embeddings.embedder import get_embedding


# Connect to the persistent ChromaDB database
client = chromadb.PersistentClient(path="data/chroma_db")

# Load the existing code chunks collection
collection = client.get_collection("code_chunks")

def _normalize_word(word):
    word = word.lower()

    # Common code/query alias
    if word == "args":
        return "argument"

    # Plural
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith("es") and len(word) > 4:
        return word[:-2]

    if word.endswith("s") and len(word) > 3:
        return word[:-1]

    # Verb forms
    # parsing -> parse
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]

        if base.endswith("s"):
            return base + "e"

        return base

    # parsed -> parse
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]

        if base.endswith("s"):
            return base + "e"

        return base

    return word


def search(query, repository=None, k=3, threshold=35.0):
    """
    Search the codebase for the most relevant code chunks.

    Args:
        query: User's search query.
        repository: Optional repository name used to restrict retrieval.
        k: Maximum number of final results to return.
        threshold: Minimum similarity percentage required.

    Returns:
        A dictionary containing:
        - results: Relevant code chunks ranked by similarity.
        - retrieval_time_ms: Retrieval time in milliseconds.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer.")

    # Start measuring end-to-end retrieval time
    start_time = time.perf_counter()

    # Convert the user's query into an embedding
    query_embedding = get_embedding(query)

    # Normalize query into lowercase words for symbol matching
    query_words = {
        _normalize_word(word)
        for word in re.findall(
            r"[a-zA-Z0-9]+",
            query.lower()
        )
    }

    # Apply repository filter when requested
    where_filter = None

    if repository is not None:
        where_filter = {
            "repository": repository
        }

    # Over-fetch candidates so that reranking and symbol boosting
    # can influence the final top-k results.
    candidate_count = max(k * 10, 50)

    # Retrieve a larger candidate pool from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        where=where_filter
    )

    relevant_results = []
    seen_chunks = set()

    # Extract returned data
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Process and filter retrieved candidates
    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        if not isinstance(metadata, dict):
            continue
        # Convert cosine distance into similarity percentage
        similarity = max(0, (1 - distance) * 100)

        # Get and normalize the code symbol name
        symbol_name = metadata.get("name", "")

        # Split symbol name into normalized words
        symbol_words = {
        _normalize_word(word)
        for word in re.findall(
            r"[a-zA-Z0-9]+",
            symbol_name.lower().replace("_", " ")
        )
    }

        # Boost results when the complete symbol appears
        # in the user's query
        if symbol_words and symbol_words.issubset(query_words):
            similarity += 15

        # Keep the similarity threshold based on the raw retrieval score.
        if similarity < threshold:
            continue

        # Prefer production code over test fixtures when ranking candidates.
        file_path = metadata.get("file_path", "").replace("\\", "/")

        ranking_score = similarity

        if file_path.startswith("src/") or "/src/" in file_path:
            ranking_score += 3
        elif file_path.startswith("tests/") or "/tests/" in file_path:
            ranking_score -= 3

        # Create a unique identity for each code chunk
        chunk_key = (
            metadata.get("repository"),
            metadata.get("file_path"),
            metadata.get("start_line"),
            metadata.get("end_line")
        )

        # Skip duplicate chunks
        if chunk_key in seen_chunks:
            continue

        seen_chunks.add(chunk_key)

        # Store clean, structured result
        relevant_results.append({
            "code": document,
            "metadata": metadata,
            "distance": distance,
            "similarity": round(similarity, 2),
            "ranking_score": round(ranking_score, 2)
            
         })

    # Re-rank after similarity calculation and symbol boosting
    relevant_results.sort(
        key=lambda result: (
            -result["ranking_score"],
            result["metadata"].get("repository") or "",
            result["metadata"].get("file_path") or "",
            result["metadata"].get("start_line", 0) or 0,
            result["metadata"].get("end_line", 0) or 0,
        )
    )

    # Return only the requested number of final results
    MAX_PER_SYMBOL_NAME = 2

    seen_names = {}
    diverse_results = []

    for result in relevant_results:
        name = result["metadata"].get("name", "")

        if name and seen_names.get(name, 0) >= MAX_PER_SYMBOL_NAME:
            continue

        diverse_results.append(result)

        if name:
            seen_names[name] = seen_names.get(name, 0) + 1

        if len(diverse_results) >= k:
            break

    relevant_results = diverse_results

    # Calculate total retrieval time in milliseconds
    retrieval_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    # Return API-ready structured data
    return {
        "results": relevant_results,
        "retrieval_time_ms": round(retrieval_time_ms, 2)
    }