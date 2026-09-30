import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.retriever import search


REPOSITORY = "heldout_click"
K = 5
THRESHOLD = 35


test_cases = [
    {
        "query": "Where is the execution context for a Click command represented?",
        "expected_file": "src/click/core.py",
        "expected_symbol": "Context",
    },
    {
        "query": "Which class represents a CLI command in Click?",
        "expected_file": "src/click/core.py",
        "expected_symbol": "Command",
    },
    {
        "query": "How does Click organize multiple commands into a command group?",
        "expected_file": "src/click/core.py",
        "expected_symbol": "Group",
    },
    {
        "query": "Which decorator turns a Python function into a Click command?",
        "expected_file": "src/click/decorators.py",
        "expected_symbol": "command",
    },
    {
        "query": "How can code access the current Click execution context?",
        "expected_file": "src/click/globals.py",
        "expected_symbol": "get_current_context",
    },
    {
        "query": "How does Click prompt the user for interactive input?",
        "expected_file": "src/click/termui.py",
        "expected_symbol": "prompt",
    },
    {
        "query": "What utility is used to invoke Click commands during testing?",
        "expected_file": "src/click/testing.py",
        "expected_symbol": "CliRunner",
    },
    {
        "query": "How does Click write output to the terminal?",
        "expected_file": "src/click/utils.py",
        "expected_symbol": "echo",
    },
    {
        "query": "Where is Click's usage error handling defined?",
        "expected_file": "src/click/exceptions.py",
        "expected_symbol": "UsageError",
    },
    {
        "query": "Where are command-line arguments parsed?",
        "expected_file": "src/click/parser.py",
        "expected_symbol": "parse_args",
    },
]

def get_metadata(result):
    if isinstance(result, dict):
        return result.get("metadata", result)

    return getattr(result, "metadata", {})


def get_expected_chunks(test):
    if "expected_chunks" in test:
        return test["expected_chunks"]

    return [
        {
            "file": test["expected_file"],
            "symbol": test["expected_symbol"],
        }
    ]


def is_correct(result, test):
    metadata = get_metadata(result)

    file_path = metadata.get("file_path")
    symbol = metadata.get("symbol") or metadata.get("name")

    if not file_path or not symbol:
        return False

    normalized_file = str(file_path).replace("\\", "/").lower()

    for expected in get_expected_chunks(test):
        expected_file = expected["file"].replace("\\", "/").lower()

        if (
            normalized_file.endswith(expected_file)
            and symbol == expected["symbol"]
        ):
            return True

    return False


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
        print("Expected chunks:")
        for expected in get_expected_chunks(test):
           print(f"  - {expected['file']} | {expected['symbol']}")
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