from src.generation.llm_client import generate_answer


question = "What does the add function do?"

context = """
[Source 1]
File: sample_repo/app.py
Symbol: add
Lines: 1 - 3
Code:
def add(a: int, b: int) -> int:
    \"\"\"Adds two numbers.\"\"\"
    return a + b
"""

answer = generate_answer(question, context)

print("\nGenerated Answer:\n")
print(answer)