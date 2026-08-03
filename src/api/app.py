import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from src.rag.pipeline import run_rag_pipeline
from pydantic import BaseModel, Field, field_validator
from pydantic import BaseModel
from src.indexing.indexer import index_codebase
from src.utils.repositories import load_repositories
from src.storage.chroma_store import get_collection

app = FastAPI(title = "Codebase RAG API", version = "1.0.0")

@app.get("/")
async def root():
  return {"message": "Codebase RAG API is running."}

@app.get("/health")
async def health():
    try:
        collection = get_collection()
        collection.count()

        return {
            "status": "healthy",
            "chromadb": "connected"
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"ChromaDB unavailable: {str(e)}"
        )

class QueryRequest(BaseModel):
  query : str = Field(..., min_length=1)
  repository : str = Field(..., min_length=1)

  @field_validator("query", "repository")
  @classmethod
  def validate_not_blank(cls, value):
    if not value.strip():
      raise ValueError("Field cannot be blank.")
    return value.strip()

class QueryResponse(BaseModel):
  answer : str
  sources : list

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):

  try:
    repositories = load_repositories()
    if request.repository not in repositories:
        raise HTTPException(status_code=404, detail=f"Repository {request.repository} is not indexed.")
    result = run_rag_pipeline(request.query, request.repository)
    return result

  except HTTPException:
    raise 

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

class IndexRequest(BaseModel):
  repo_path : str

class IndexResponse(BaseModel):
  success : bool
  repositories : str
  message : str

class RepositoryListResponse(BaseModel):
  repositories : list[str]

@app.post("/index", response_model=IndexResponse, status_code=202)
async def index_repository(request: IndexRequest, background_tasks: BackgroundTasks):
  try:
    if not os.path.exists(request.repo_path):
        raise HTTPException(status_code=400, detail="Invalid repository path.")
    
    background_tasks.add_task(index_codebase, request.repo_path)
    repository_name = os.path.basename(os.path.normpath(request.repo_path)) #reposi
    return {"success": True, "repositories": repository_name, "message": "Repository indexing started."}

  except HTTPException:
    raise

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@app.get("/repositories", response_model=RepositoryListResponse)
async def get_repositories():
    return {
        "repositories": load_repositories()
    }