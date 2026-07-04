# Codebase RAG System

A simple Python project for exploring codebases and preparing them for retrieval-augmented generation (RAG) workflows.

## Project Goal
This project focuses on reading Python source files, parsing their structure, and extracting useful code information such as classes and functions for downstream search and reasoning systems.

## Day 1 (Yesterday)
- Initialized the project structure and created the main package layout.
- Added a sample repository under the sample_repo folder for testing.
- Implemented core ingestion modules to discover Python files and read their contents.
- Built the AST parser to extract Python classes and functions from source code.

## Day 2 (Today)
- Connected the ingestion and parsing flow in the main entry script.
- Verified that the system can read Python files, parse them into an AST, and print class/function definitions with their line numbers.
- Strengthened the project foundation for future RAG-related features such as chunking, embeddings, and retrieval.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the sample parser:
   ```bash
   python main.py
   ```

## Next Steps
- Add chunking support for code snippets.
- Integrate embeddings for semantic retrieval.
- Build a retrieval layer for querying codebase content.
