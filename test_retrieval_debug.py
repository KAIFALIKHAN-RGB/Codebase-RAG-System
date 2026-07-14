from src.retrieval.retriever import search

query = "what functionality does the user provide?"

result = search(query=query,k=10,threshold=0.0)

print("\nRetrieved results for query:\n")

for i,item in enumerate(result["results"],start = 1):
    metadata = item.get("metadata", {})
    print(f"Result {i}:")
    print(f"File : {metadata.get('file_path', 'Unknown')}")
    print(f"Symbol: {metadata.get('name', 'Unknown')}")
    print(f"Similarity: {item.get('similarity')}%")
    print("-" * 40)