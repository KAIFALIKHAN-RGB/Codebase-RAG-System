import sys
from src.indexing.indexer import index_codebase

if __name__ == "__main__":
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = "sample_repo"

    print(f"\nIndexing repository: {repo_path}\n")
    index_codebase(repo_path)