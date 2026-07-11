# 🚀 Codebase RAG System

An AI-powered Codebase Retrieval-Augmented Generation (RAG) system designed to understand and search Python repositories using Abstract Syntax Trees (AST), semantic chunking, embeddings, vector search, and metadata-aware retrieval.

The system currently provides an end-to-end pipeline for indexing Python source code and retrieving relevant code chunks using natural-language queries.

## ✨ Features

- 📂 Recursively loads Python source files
- 🌳 Parses source code using Python AST
- ✂️ Performs AST-based semantic code chunking
- 📝 Extracts rich metadata:
  - Function and class names
  - Symbol type
  - Parameters
  - Docstrings
  - Parent class
  - Return type
  - File path
  - Start and end line numbers
- ⚡ Supports asynchronous functions (async def)
- 🧠 Generates semantic embeddings using Sentence Transformers
- 🗄️ Stores embeddings and metadata in persistent ChromaDB storage
- 🔑 Uses stable chunk IDs for reliable re-indexing
- 🔄 Uses upsert() for safe chunk insertion and updates
- 🔎 Supports natural-language semantic code search
- 🎯 Retrieves Top-K relevant code chunks
- 📏 Calculates similarity scores from vector distance
- 🚦 Filters weak results using a configurable similarity threshold
- ⏱️ Tracks end-to-end retrieval latency
- 📦 Returns structured, metadata-rich retrieval results

## 🏗️ Current Pipeline

Python Repository
        ↓
Load Python Files
        ↓
Read Source Code
        ↓
AST Parsing
        ↓
Semantic Code Chunking
        ↓
Metadata Extraction
        ↓
Embedding Generation
        ↓
ChromaDB Vector Storage
        ↓
Natural-Language Query
        ↓
Query Embedding
        ↓
Top-K Vector Search
        ↓
Similarity Scoring
        ↓
Threshold Filtering
        ↓
Structured Code Results
## 🔎 Semantic Retrieval

The system accepts natural-language queries such as:

adds two numbers
The retrieval pipeline:

1. Converts the query into an embedding.
2. Searches ChromaDB for the Top-K nearest code chunks.
3. Converts vector distance into a readable similarity score.
4. Filters weak matches using a configurable threshold.
5. Returns relevant code with structured metadata.

Example output:

Result 1
Similarity: 74.76%
Name: add
Type: FunctionDef
File: sample_repo/app.py
Lines: 1 - 3
Parent Class: None
Parameters: a,b
Return Type: int

Code Snippet:

def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

Retrieval Time: 72.62 ms
> The displayed latency is from a small local test repository and is not a production benchmark.

## ⚙️ Retrieval Configuration

The retrieval function supports configurable search parameters:

search(
    query="adds two numbers",
    k=3,
    threshold=30.0
)
- `query → Natural-language search query
- k → Maximum number of candidate chunks to retrieve
- threshold → Minimum similarity score required for a result

The current threshold is an initial baseline and will be tuned later through retrieval evaluation on larger repositories.

## 📦 Structured Retrieval Results

The retrieval layer returns structured results containing:

- Similarity score
- Vector distance
- Symbol name
- Symbol type
- File path
- Start and end lines
- Parent class
- Parameters
- Return type
- Code snippet

This structure is designed for future integration with FastAPI, LLM pipelines, evaluation systems, and agent tools.

## 📁 Project Structure

``text
Codebase-RAG-System/
│
├── data/
│   ├── chunks.json
│   └── chroma_db/
│
├── sample_repo/
│
├── src/
│   ├── chunking/
│   ├── embeddings/
│   ├── ingestion/
│   ├── parser/
│   ├── retrieval/
│   ├── storage/
│   └── agents/
│
├── main.py
├── test_embedding.py
├── test_retrieval.py
├── requirements.txt
└── README.md

## 🧩 Retrieval Architecture

text
retriever.py
├── Query embedding
├── ChromaDB vector search
├── Similarity calculation
├── Threshold filtering
├── Structured result creation
└── Retrieval latency measurement
test_retrieval.py
├── User input
├── Calls the retrieval layer
└── Displays results in the terminal

The core retrieval logic is separated from testing and display logic, making it reusable for future backend APIs and RAG components.

## 🛠️ Tech Stack

- Python
- Abstract Syntax Tree (AST)
- Sentence Transformers
- ChromaDB
- JSON

## 🚧 Current Progress

- ✅ Repository Ingestion
- ✅ AST Parsing
- ✅ Semantic Code Chunking
- ✅ Metadata Extraction
- ✅ Embedding Generation
- ✅ ChromaDB Vector Indexing
- ✅ Stable Chunk IDs
- ✅ Safe Re-indexing with `upsert()`
- ✅ Semantic Code Retrieval
- ✅ Top-K Vector Search
- ✅ Similarity Scoring
- ✅ Configurable Relevance Filtering
- ✅ Structured Retrieval Results
- ✅ Retrieval Latency Tracking
- ⏳ Retrieval Evaluation
- ⏳ RAG Context Construction
- ⏳ LLM Integration
- ⏳ FastAPI Backend
- ⏳ Frontend
- ⏳ Deployment

## 🎯 Upcoming Features

- Retrieval evaluation on larger repositories
- RAG context construction
- AI-powered codebase question answering
- Grounded LLM responses using retrieved code
- FastAPI backend
- Interactive frontend
- Agentic code analysis tools
- Multi-language support
- Deployment

## 📚 Learning Goals

This project is being built from scratch to understand the complete lifecycle of a production-minded Codebase RAG system:

- AST Parsing
- Semantic Code Chunking
- Metadata Extraction
- Embeddings
- Vector Databases
- Semantic Retrieval
- Retrieval Evaluation
- Retrieval-Augmented Generation (RAG)
- Agentic AI
- Backend API Design
- Production-Ready System Design

## 📌 Current Status

The project currently has a working end-to-end code indexing and semantic retrieval pipeline:

text
Repository
→ Parse
→ Chunk
→ Extract Metadata
→ Embed
→ Store
→ Query
→ Retrieve
→ Score
→ Filter
→ Return Structured Results
`

The next major phase is to evaluate retrieval quality and use retrieved code chunks as grounded context for LLM-based codebase question answering.

## 🤝 Contributing

Suggestions and improvements are welcome. Feel free to open an issue or submit a pull request.

## ⭐ Star the Repository

If you find this project interesting, consider giving it a ⭐ to support the project.