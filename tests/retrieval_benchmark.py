import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag.pipeline import run_rag_pipeline
# -------------------------------
# Benchmark Test Cases
# -------------------------------
test_cases = [
        {
            "query": "Where is load_dotenv implemented?",
            "expected": "load_dotenv"
        },
        {
            "query": "What does cli.py do?",
            "expected": "cli"
        },
        {
            "query": "Explain parser.py.",
            "expected": "parser"
        },
        {
            "query": "How are .env files parsed?",
            "expected": [
              "parser",
              "DotEnv",
              "parse" ]
                       
        },
        {
            "query": "Which file loads environment variables?",
            "expected": "dotenv"
        },
        {
            "query": "How does the library read environment variables?",
            "expected": "dotenv"
        },
        {
            "query": "How is the env file processed?",
            "expected": ["DotEnv","processed","parser"]
        },
        {
            "query": "Where is command line functionality implemented?",
            "expected": "cli"
        },
        {
            "query": "How are variables imported from .env?",
            "expected": "dotenv"
        },
        {
            "query": "Which module is responsible for parsing?",
            "expected": "parser"
        },
        {
            "query": "What is JWT authentication?",
            "expected": None
        },
        {
            "query": "How does OAuth work?",
            "expected": None
        },
        {
            "query": "Explain binary search tree.",
            "expected": None
        },
        {
            "query": "What is an operating system?",
            "expected": None
        },
        {
            "query": "How is Redis caching implemented?",
            "expected": None
        },
        {
            "query": "How is parsing done?",
            "expected": "parser"
        },
        {
            "query": "Where are variables handled?",
            "expected": "variables"
        },
        {
            "query": "Which file reads configuration?",
            "expected": "dotenv"
        },
        {
            "query": "How does the project start?",
            "expected": "main"
        },
        {
            "query": "How are commands executed?",
            "expected": "cli"
        }

]
# -------------------------------
# Run Benchmark
# -------------------------------
passed = 0
failed = 0
print("\n========= Retrieval Benchmark ==========\n")
for i, test in enumerate(test_cases, start=1):

    print("=" * 60)
    print(f"Test {i}")
    print(f"Query    : {test['query']}")
    print(f"Expected : {test['expected']}")

    result = run_rag_pipeline(
             test["query"],
             repository="python-dotenv")
                          

    answer = result["answer"]

    print("\nAnswer:")
    print(answer)

    if test["expected"] is None:
        success = "No relevant code chunks found." in answer
    else:
        if isinstance(test["expected"], list):
            success = any(
                keyword.lower() in answer.lower()
            for keyword in test["expected"]
        )
        else:
            success = test["expected"].lower() in answer.lower()

    if success:
        print("\nResult : PASS ✅")
        passed += 1
    else:
        print("\nResult : FAIL ❌")
        failed += 1

print("\n" + "=" * 60)
print("FINAL RETRIEVAL REPORT")
print("=" * 60)

accuracy = (passed / len(test_cases)) * 100 if test_cases else 0
print(f"Total Tests      : {len(test_cases)}")
print(f"Passed           : {passed}")
print(f"Failed           : {failed}")
print(f"Accuracy         : {accuracy:.2f}%")

if accuracy >= 90:
    print("Status           : Excellent")
elif accuracy >= 80:
    print("Status           : Good")
else:
    print("Status           : Needs Improvement")

print("=" * 60)