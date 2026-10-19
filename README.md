# Codebase RAG System

Codebase RAG System is a Python-focused retrieval-augmented generation application for asking natural-language questions about indexed repositories. It parses Python source code into AST-based semantic chunks, embeds those chunks for vector search, stores them in persistent ChromaDB, builds grounded context, and generates answers with Gemini.

The project includes a FastAPI service for indexing repositories, querying indexed code, checking index status, listing repositories, and removing repository data.

## What is implemented

### Repository ingestion and indexing

The indexing pipeline:

1. Recursively discovers Python files under a repository.
2. Skips `.venv`, `venv`, `__pycache__`, `.git`, and `tests` directories.
3. Parses each file with Python's `ast` module.
4. Extracts semantic chunks for classes, regular functions, and asynchronous functions.
5. Records chunk metadata including:
   - repository name
   - symbol name and AST type
   - file path
   - start and end line numbers
   - function parameters
   - docstring
   - containing class, when applicable
   - return annotation, when available
6. Generates embeddings with Sentence Transformers using `all-MiniLM-L6-v2`.
7. Stores code, embeddings, and metadata in the persistent ChromaDB collection `code_chunks`.

Chunks use stable IDs based on file path and line range. Indexing uses ChromaDB upserts and maintains repository file-hash state so unchanged files can be skipped. Repository indexing runs as a FastAPI background task and exposes `indexing`, `completed`, and `failed` status values.

### Retrieval

The vector retriever:

- embeds the natural-language query with the same Sentence Transformers model;
- retrieves an over-fetched candidate set from ChromaDB;
- converts cosine distance into a similarity percentage;
- applies the configured similarity threshold;
- boosts results when the complete normalized symbol name appears in the query;
- prefers production `src` files over test files;
- removes duplicate chunks;
- applies deterministic ordering and returns up to `k` results.

Each result contains the code plus metadata, vector distance, similarity, and `ranking_score`.

### RAG answer generation

The standard RAG pipeline is:

```text
Question
  -> vector retrieval
  -> structured context building
  -> Gemini answer generation
  -> answer, sources, and retrieval time
```

The context builder formats each retrieved chunk with its source number, file, symbol, line range, and Python code. The Gemini client uses the `GEMINI_API_KEY` environment variable and the `gemini-3.1-flash-lite` model. The prompt instructs Gemini to use only the supplied code context and to say when the context is insufficient.

### FastAPI service

The application is defined in `src/api/app.py` and can be run with Uvicorn. The service uses persistent local files and ChromaDB under `data/`; it is intended for local development and evaluation rather than a claim of production deployment readiness.

## API

### `GET /`

Returns a basic service message:

```json
{
  "message": "Codebase RAG API is running."
}
```

### `GET /health`

Checks whether the ChromaDB collection can be opened and counted.

Successful response:

```json
{
  "status": "healthy",
  "chromadb": "connected"
}
```

If ChromaDB is unavailable, the endpoint returns HTTP 503 with a `detail` message.

### `POST /query`

Queries an indexed repository and generates an answer from retrieved code.

Request body:

```json
{
  "query": "Where is the command execution logic implemented?",
  "repository": "python-dotenv"
}
```

Both fields are required non-empty strings. Whitespace-only values are rejected. The repository must already be indexed; otherwise the endpoint returns HTTP 404.

Successful response fields:

```json
{
  "answer": "Generated answer text",
  "sources": [
    {
      "code": "def example(): ...",
      "metadata": {
        "name": "example",
        "start_line": 1,
        "end_line": 2,
        "type": "FunctionDef",
        "repository": "python-dotenv",
        "file_path": "..."
      },
      "distance": 0.2,
      "similarity": 80.0,
      "ranking_score": 83.0
    }
  ]
}
```

The declared `QueryResponse` schema contains `answer` and `sources`. The pipeline also produces `retrieval_time_ms`; the service implementation returns the pipeline result, but this timing field is not declared in the Pydantic response model.

Normal API queries use vector-only retrieval. Graph-aware retrieval is not enabled by this endpoint.

### `POST /index`

Starts indexing for a local repository path as a background task.

Request body:

```json
{
  "repo_path": "C:\\Users\\you\\Projects\\my-repository"
}
```

The path must exist. The repository name is derived from the final directory name.

Successful response: HTTP 202

```json
{
  "success": true,
  "repositories": "my-repository",
  "message": "Repository indexing started."
}
```

The response field is named `repositories` in the implemented `IndexResponse` model, even though it contains one repository name.

### `GET /index/status/{repo_name}`

Returns the current indexing status:

```json
{
  "repository": "my-repository",
  "status": "completed"
}
```

If no status exists for the name, the endpoint returns HTTP 404.

### `GET /repositories`

Lists repository names recorded by the application:

```json
{
  "repositories": [
    "my-repository"
  ]
}
```

### `DELETE /repositories/{repo_name}`

Removes a repository from the application. The endpoint deletes its ChromaDB chunks, index state, repository registration, and saved repository path.

Successful response:

```json
{
  "success": true,
  "message": "Repository 'my-repository' deleted successfully."
}
```

An unknown repository returns HTTP 404. Deletion failures return HTTP 500 with a `detail` message.

## Windows PowerShell setup

From the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

If PowerShell blocks script activation for the current user, allow locally created scripts and activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

The runtime dependencies are pinned in `requirements.txt`. Development and test dependencies are pinned in `requirements-dev.txt`.

## Configure Gemini

The Gemini client loads environment variables with `python-dotenv` and requires `GEMINI_API_KEY` when an answer is generated.

For the current PowerShell session:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
```

Alternatively, create a local `.env` file in the repository root:

```text
GEMINI_API_KEY=your-api-key
```

Do not commit API keys or machine-specific environment files.

## Run the application

Start the FastAPI service with Uvicorn:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --reload
```

The service exposes the routes documented above. The project does not define a deployment URL in the repository.

For a simple command-line indexing run, `main.py` accepts an optional repository path and defaults to `sample_repo`:

```powershell
.\.venv\Scripts\python.exe main.py .\sample_repo
```

For an interactive command-line question-and-answer demo:

```powershell
.\.venv\Scripts\python.exe rag_pipeline_demo.py
```

## Run tests

Pytest is configured in `pytest.ini` to discover tests under `tests` with `test_*.py` filenames.

Run the complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run a focused test module:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_retriever.py
```

The repository also contains retrieval and graph benchmark scripts under `tests`. These are evaluation scripts, not API endpoints.

## Implemented, experimental, and planned work

### Implemented

- Python repository discovery and AST parsing
- Semantic class and function chunking
- Rich chunk metadata extraction
- Sentence Transformers embeddings
- Persistent ChromaDB storage
- Incremental indexing state and repository deletion
- Vector retrieval with thresholding and deterministic ranking
- Structured context construction
- Gemini-based grounded answer generation
- FastAPI indexing, querying, health, status, listing, and deletion endpoints
- Pytest-based unit and integration coverage

### Experimental and on hold

The repository contains an import-graph implementation and an optional graph-aware retriever. It can expand candidates through imported files and dependents, merge candidates, and rank them using existing retrieval scores when available.

Graph-aware retrieval is currently experimental and on hold. It is not enabled for normal API queries, and the current benchmark does not demonstrate an accuracy, MRR, Hit@K, or context-recall improvement over vector-only retrieval. It remains available as an explicit Python pipeline option with `graph_aware=True` for further relationship-focused evaluation.

### Future plans

Potential future work includes broader relationship-focused evaluation, improved API controls for explicitly selecting retrieval modes, and additional production hardening. These are plans, not currently implemented features.

## Project layout

```text
Codebase-RAG-System/
├── data/                    # Persistent local index and repository state
├── src/
│   ├── api/                 # FastAPI application and schemas
│   ├── chunking/            # AST-derived semantic chunks
│   ├── embeddings/          # Sentence Transformers embeddings
│   ├── generation/          # Context builder and Gemini client
│   ├── indexing/            # Repository indexing workflow
│   ├── ingestion/           # Python file discovery and reading
│   ├── parser/              # Python AST parsing
│   ├── rag/                 # End-to-end retrieval and generation pipeline
│   ├── relationships/       # Import graph and graph expansion
│   ├── retrieval/           # Vector and graph-aware retrieval
│   ├── storage/             # ChromaDB persistence
│   └── utils/               # Repository, state, and status utilities
├── tests/                   # Unit, API, retrieval, graph, and benchmark code
├── main.py                  # Command-line indexing entry point
├── rag_pipeline_demo.py     # Interactive pipeline demo
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # Development and test dependencies
├── pytest.ini               # Pytest discovery configuration
└── README.md
```

## Current limitations

- The system currently indexes Python source files only.
- ChromaDB, index state, repository registrations, and locks are stored locally under `data/`.
- Gemini answer generation requires a configured API key and network access to the Google GenAI service.
- The API has no authentication or authorization layer in the implemented code.
- Indexing is performed as a background task, but no persistent job queue is implemented.
- Graph-aware retrieval has not demonstrated a benefit in the current benchmark and is not part of the default query path.
- No deployment configuration or hosted service URL is provided by this repository.
