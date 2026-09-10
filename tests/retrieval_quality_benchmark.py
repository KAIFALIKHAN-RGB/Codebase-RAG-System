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
    "expected_chunks": [
            {
                "file": "src/dotenv/cli.py",
                "symbol": "cli",
            },
            {
                "file": "src/dotenv/cli.py",
                "symbol": "run_command",
            },
            {
                "file": "src/dotenv/cli.py",
                "symbol": "run",
            },
        ],
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
    {
        "query": "How are environment variable values obtained?",
        "expected_chunks": [
            {
                "file": "src/dotenv/main.py",
                "symbol": "dotenv_values",
            },
        ],
    },
    {
    "query": "How does IPython integration work?",
    "expected_chunks": [
        {
            "file": "src/dotenv/ipython.py",
            "symbol": "dotenv",
        },
        {
            "file": "src/dotenv/ipython.py",
            "symbol": "load_ipython_extension",
        },
        {
            "file": "src/dotenv/ipython.py",
            "symbol": "IPythonDotEnv",
        },
    ],
},
    {
        "query": "How are environment variables enumerated?",
        "expected_chunks": [
            {
                "file": "src/dotenv/cli.py",
                "symbol": "enumerate_env",
            },
        ],
    },
    {
        "query": "Where is the command execution logic implemented?",
        "expected_chunks": [
            {
                "file": "src/dotenv/cli.py",
                "symbol": "run_command",
            },
            {
                "file": "src/dotenv/cli.py",
                "symbol": "run",
            },
        ],
    },
    {
        "query": "How does the library find the dotenv file?",
        "expected_chunks": [
            {
                "file": "src/dotenv/main.py",
                "symbol": "find_dotenv",
            },
        ],
    },
    {
        "query": "How are parsed values placed into environment variables?",
        "expected_chunks": [
            {
                "file": "src/dotenv/main.py",
                "symbol": "set_as_environment_variables",
            },
        ],
    },
    {
        "query": "How is the CLI string generated?",
        "expected_chunks": [
            {
                "file": "src/dotenv/__init__.py",
                "symbol": "get_cli_string",
            },
        ],
    },
    {
        "query": "How are variable references parsed?",
        "expected_chunks": [
            {
                "file": "src/dotenv/variables.py",
                "symbol": "parse_variables",
            },
        ],
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