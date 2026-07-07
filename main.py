import ast
from src.ingestion.loader import load_python_file
from src.ingestion.reader import read_python_file
from src.parser.ast_parser import parse_code
from src.chunking.chunker import extract_chunks
from src.utils.storage import save_chunks

files = load_python_file("sample_repo")

all_chunks = []
for file in files:
    print("=" * 50)
    print(f"Reading : {file}")
    print("=" * 50)

    code = read_python_file(file)
    tree = parse_code(code)
    chunks = extract_chunks(tree, code, file)
    all_chunks.extend(chunks)

save_chunks(all_chunks, "data/chunks.json")
print(f"\nSaved {len(all_chunks)} chunks successfully.")
for chunk in all_chunks:
       # print("chunk keys:", chunk.keys())
        print("=" * 60)
        print(f"Type       : {chunk['type']}")
        print(f"Name       : {chunk['name']}")
        print(f"File       : {chunk['file_path']}")
        print(f"Start Line : {chunk['start_line']}")
        print(f"End Line   : {chunk['end_line']}")
        print(f"Parameters : {chunk['parameters']}")
        print(f"Docstring  : {chunk['docstring']}")
        print(f"Parent Class: {chunk['parent_class']}")
        print("\nCode:")
        print(chunk["code"])
    # print(ast.dump(tree, indent=4))

   # print(code)