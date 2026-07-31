from fastapi import FastAPI, HTTPException
from src.rag.pipeline import run_rag_pipeline
from pydantic import BaseModel
from src.indexing.indexer import index_codebase
from src.utils.repositories import load_repositories

app = FastAPI(title = "Codebase RAG API", version = "1.0.0")

@app.get("/")
async def root():
  return {"message": "Codebase RAG API is running."}

@app.get("/health")
async def health():
  return {"message": "Codebase RAG API is healthy."}

class QueryRequest(BaseModel):
  query : str
  repository : str

class QueryResponse(BaseModel):
  answer : str
  sources : list

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):

  try:
    result = run_rag_pipeline(request.query, request.repository)
    return result

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

class IndexRequest(BaseModel):
  repo_path : str

@app.post("/index")
async def index_repository(request: IndexRequest):
  try:
    index_codebase(request.repo_path)
    return {"message": "Repository indexed successfully.", "repository": request.repo_path}

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@app.get("/repositories")
async def get_repositories():
    return {
        "repositories": load_repositories()
    }