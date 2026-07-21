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
        "query": "Tell me about the User class",
        "expected": "User"
    },
    {
        "query": "How does user login work?",
        "expected": "login"
    },
    {
        "query": "What is an array?",
        "expected": None
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

    result = run_rag_pipeline(test["query"])

    answer = result["answer"]

    print("\nAnswer:")
    print(answer)

    if test["expected"] is None:
        success = "No relevant code chunks found." in answer
    else:
        success = test["expected"].lower() in answer.lower()

    if success:
        print("\nResult : PASS ✅")
        passed += 1
    else:
        print("\nResult : FAIL ❌")
        failed += 1

print("\n" + "=" * 60)
print("Benchmark Summary")
print("=" * 60)

print(f"Total Tests : {len(test_cases)}")
print(f"Passed      : {passed}")
print(f"Failed      : {failed}")

accuracy = (passed / len(test_cases)) * 100

print(f"Accuracy    : {accuracy:.2f}%")
print("=" * 60)