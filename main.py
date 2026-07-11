import ast
from src.ingestion.loader import load_python_file
from src.ingestion.reader import read_python_file
from src.parser.ast_parser import parse_code
from src.chunking.chunker import extract_chunks
from src.utils.storage import save_chunks
from src.storage.chroma_store import store_chunk
from src.embeddings.embedder import get_embedding

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

for chunk in all_chunks:
    embedding = get_embedding(chunk["code"])
    store_chunk(chunk, embedding)
print(f"\nStored {len(all_chunks)} chunks in ChromaDB successfully.")

    # print(ast.dump(tree, indent=4))
   # print(code)
