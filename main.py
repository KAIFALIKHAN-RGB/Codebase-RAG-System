import ast
from src.ingestion.loader import load_python_file
from src.ingestion.reader import read_python_file
from src.parser.ast_parser import parse_code

files = load_python_file("sample_repo")

for file in files:
    print("=" * 50)
    print(f"Reading : {file}")
    print("=" * 50)

    code = read_python_file(file)
    tree = parse_code(code)
    #print("\nAST Nodes Found:")

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            print(f"Class: {node.name}")
            print(f" Starts at :  Line {node.lineno}")

        elif isinstance(node, ast.FunctionDef):
            print(f"Function: {node.name}")
            print(f" Starts at :  Line {node.lineno}")

    # print(ast.dump(tree, indent=4))

   # print(code)