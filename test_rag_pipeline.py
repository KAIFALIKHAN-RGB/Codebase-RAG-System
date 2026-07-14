from src.rag.pipeline import run_rag_pipeline


question = input("Ask about the codebase: ").strip()

try:
    result = run_rag_pipeline(question)

    print("\nAnswer:\n")
    print(result["answer"])

    sources = result["sources"]

    if sources:
        print("\nSources:\n")

        for i, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {})

            print(f"Source {i}:")
            print(f"  File: {metadata.get('file_path', 'Unknown')}")
            print(f"  Symbol: {metadata.get('name', 'Unknown')}")
            print(
                f"  Lines: {metadata.get('start_line', '?')} - "
                f"{metadata.get('end_line', '?')}"
            )
            print(f"  Relevance score: {source.get('similarity')}%")

    else:
        print("\nSources: None")

    print(f"\nRetrieval Time: {result['retrieval_time_ms']} ms")

except Exception as e:
    print(f"\nError: {e}")