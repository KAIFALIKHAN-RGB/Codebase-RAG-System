from src.retrieval.retriever import search


try:
    # Take query from the user
    query = input("Ask about the codebase: ").strip()

    # Validate empty query
    if not query:
        raise ValueError("Query cannot be empty.")

    # Search the codebase
    response = search(
        query=query,
        k=3,
        threshold=50.0
    )

    # Extract results and retrieval time
    results = response["results"]
    retrieval_time_ms = response["retrieval_time_ms"]

    # Check if any relevant result was found
    if not results:
        print("\nNo sufficiently relevant code found.")

    else:
        print("\nTop Results:\n")

        # Display each relevant result
        for i, result in enumerate(results, start=1):
            metadata = result["metadata"]

            print("=" * 60)
            print(f"Result {i}")
            print(f"Similarity: {result['similarity']:.2f}%")
            print(f"Name: {metadata.get('name', 'N/A')}")
            print(f"Type: {metadata.get('type', 'N/A')}")
            print(f"File: {metadata.get('file_path', 'N/A')}")

            # Display line range if available
            if "start_line" in metadata and "end_line" in metadata:
                print(
                    f"Lines: {metadata['start_line']} - "
                    f"{metadata['end_line']}"
                )

            print(
                f"Parent Class: "
                f"{metadata.get('parent_class', 'None')}"
            )

            print(
                f"Parameters: "
                f"{metadata.get('parameters', '')}"
            )

            print(
                f"Return Type: "
                f"{metadata.get('return_type', 'None')}"
            )

            print("\nCode Snippet:\n")
            print(result["code"])

    # Display retrieval performance
    print("\n" + "=" * 60)
    print(f"Retrieval Time: {retrieval_time_ms:.2f} ms")


except ValueError as e:
    print(f"\nInput Error: {e}")

except Exception as e:
    print(f"\nError: {e}")