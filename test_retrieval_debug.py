from src.retrieval.retriever import search


TEST_CASES = [
    {
        "name": "Exact function match",
        "query": "add",
        "repository": "sample_repo",
    },
    {
        "name": "Exact class match",
        "query": "User",
        "repository": "sample_repo",
    },
    {
        "name": "Semantic query",
        "query": "what functionality does the user provide",
        "repository": "sample_repo",
    },
    {
        "name": "Login functionality",
        "query": "how does user login work",
        "repository": "sample_repo",
    },
    {
        "name": "User retrieval",
        "query": "how are users retrieved",
        "repository": "sample_repo",
    },
    {
        "name": "Admin functionality",
        "query": "what can an admin do",
        "repository": "sample_repo",
    },
    {
        "name": "Non-existent functionality",
        "query": "payment gateway processing and credit card validation",
        "repository": "sample_repo",
    },
    {
        "name": "Repo-specific query",
        "query": "user authentication",
        "repository": "sample_repo",
    },
]


def run_test(test_case):
    print("\n" + "=" * 70)
    print(f"TEST: {test_case['name']}")
    print(f"QUERY: {test_case['query']}")
    print(f"REPOSITORY: {test_case['repository']}")
    print("=" * 70)

    result = search(
        query=test_case["query"],
        repository=test_case["repository"],
        k=10,
        threshold=10,
    )

    results = result.get("results", [])

    if not results:
        print("NO RESULTS")
        return

    for i, item in enumerate(results, start=1):
        metadata = item.get("metadata", {})

        print(f"\nResult {i}:")
        print(f"File       : {metadata.get('file_path', 'Unknown')}")
        print(f"Symbol     : {metadata.get('name', 'Unknown')}")
        print(f"Repository : {metadata.get('repository', 'Unknown')}")
        print(f"Similarity : {item.get('similarity', 0):.2f}%")


def main():
    for test_case in TEST_CASES:
        run_test(test_case)


if __name__ == "__main__":
    main()