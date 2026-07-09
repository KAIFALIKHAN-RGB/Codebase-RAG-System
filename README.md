# 🚀 Codebase RAG System

An AI-powered Codebase Retrieval-Augmented Generation (RAG) System that understands Python repositories using Abstract Syntax Trees (AST), semantic chunking, embeddings, and vector search.
This project indexes source code into a vector database, enabling semantic code search and laying the foundation for an intelligent AI coding assistant.

## ✨ Features

- 📂 Recursively loads Python source files
- 🌳 Parses source code using Python AST
- ✂️ Performs AST-based semantic code chunking
- 📝 Extracts rich metadata:
  - Function/Class names
  - Parameters
  - Docstrings
  - Parent Class
  - Return Types
  - File Path
  - Start & End Line Numbers
- ⚡ Supports asynchronous functions (async def)
- 🧠 Generates semantic embeddings using Sentence Transformers
- 🗄️ Stores embeddings and metadata in ChromaDB
- 🔑 Uses stable chunk IDs to prevent duplicate indexing

## 🏗️ Current Pipeline

Repository
      ↓
Load Python Files
      ↓
Read Source Code
      ↓
AST Parsing
      ↓
Semantic Chunking
      ↓
Metadata Extraction
      ↓
Embedding Generation
      ↓
ChromaDB Storage
## 📁 Project Structure

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
├── requirements.txt
└── README.md
## 🛠️ Tech Stack

- Python
- Abstract Syntax Tree (AST)
- Sentence Transformers
- ChromaDB
- JSON

## 🚧 Current Progress

- ✅ Repository Ingestion
- ✅ AST Parsing
- ✅ Semantic Chunking
- ✅ Metadata Extraction
- ✅ Embedding Generation
- ✅ ChromaDB Indexing
- ⏳ Semantic Retrieval
- ⏳ LLM Integration
- ⏳ FastAPI Backend
- ⏳ Frontend
- ⏳ Deployment

## 🎯 Upcoming Features

- Semantic Code Retrieval
- AI-powered Code Question Answering
- RAG Pipeline
- FastAPI Backend
- Interactive Frontend
- Multi-language Support
- Deployment

## 📌 Learning Goals

This project is being built from scratch to understand the complete lifecycle of a production-grade Codebase RAG System, including:

- AST Parsing
- Semantic Chunking
- Embeddings
- Vector Databases
- Retrieval
- Retrieval-Augmented Generation (RAG)
- Agentic AI
- Production-ready System Design

## 🤝 Contributing

Suggestions and improvements are welcome. Feel free to open an issue or submit a pull request.

## ⭐ Star the Repository

If you found this project interesting, consider giving it a ⭐ to support the project.