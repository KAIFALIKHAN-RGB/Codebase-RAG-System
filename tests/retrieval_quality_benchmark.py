import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.retriever import search


REPOSITORY = "python-dotenv"
K = 5
THRESHOLD = 10


test_cases = [
    {
        "query": "Where is load_dotenv implemented?",
        "expected_file": "src/dotenv/main.py",
        "expected_symbol": "load_dotenv",
    },
    {
        "query": "What does cli.py do?",
        "expected_file": "src/dotenv/cli.py",
        "expected_symbol": "cli",
    },
    {
        "query": "How are .env files parsed?",
        "expected_file": "src/dotenv/parser.py",
        "expected_symbol": "parse_stream",
    },
    {
        "query": "Where is command line functionality implemented?",
        "expected_file": "src/dotenv/cli.py",
        "expected_symbol": "cli",
    },
    {
        "query": "Which module is responsible for parsing?",
        "expected_file": "src/dotenv/parser.py",
        "expected_symbol": "parse_stream",
    },
    {
        "query": "How does the library read environment variables?",
        "expected_file": "src/dotenv/main.py",
        "expected_symbol": "load_dotenv",
    },
    {
        "query": "How are variables imported from .env?",
        "expected_file": "src/dotenv/main.py",
        "expected_symbol": "load_dotenv",
    },
]


def get_metadata(result):
    if isinstance(result, dict):
        return result.get("metadata", result)

    return getattr(result, "metadata", {})


def is_correct(result, test):
    metadata = get_metadata(result)

    file_path = metadata.get("file_path")
    symbol = metadata.get("symbol") or metadata.get("name")

    if not file_path or not symbol:
        return False

    normalized_file = str(file_path).replace("\\", "/").lower()
    expected_file = test["expected_file"].replace("\\", "/").lower()

    return (
        normalized_file.endswith(expected_file)
        and symbol == test["expected_symbol"]
    )


def reciprocal_rank(results, test):
    for rank, result in enumerate(results, start=1):
        if is_correct(result, test):
            return 1 / rank

    return 0


def main():
    hits = 0
    mrr_total = 0

    print("=" * 70)
    print("RETRIEVAL QUALITY BENCHMARK")
    print("=" * 70)

    for index, test in enumerate(test_cases, start=1):
        search_results = search(
            query=test["query"],
            repository=REPOSITORY,
            k=K,
            threshold=THRESHOLD,
        )

        results = search_results["results"]

        hit = any(is_correct(result, test) for result in results)
        rr = reciprocal_rank(results, test)

        if hit:
            hits += 1

        mrr_total += rr

        print(f"\nTest {index}")
        print(f"Query          : {test['query']}")
        print(f"Expected file  : {test['expected_file']}")
        print(f"Expected symbol: {test['expected_symbol']}")
        print(f"Hit@{K}        : {'PASS' if hit else 'FAIL'}")
        print(f"RR             : {rr:.3f}")

        print("Retrieved:")
        for rank, result in enumerate(results, start=1):
            metadata = get_metadata(result)
            file_path = metadata.get("file_path")
            symbol = metadata.get("symbol") or metadata.get("name")

            print(
                f"  {rank}. {file_path} | {symbol}"
            )

    total = len(test_cases)
    hit_rate = (hits / total) * 100
    mrr = mrr_total / total

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Hit@{K} : {hits}/{total} ({hit_rate:.2f}%)")
    print(f"MRR     : {mrr:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()