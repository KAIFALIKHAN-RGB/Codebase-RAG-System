import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from src.rag.pipeline import run_rag_pipeline
from pydantic import BaseModel, Field, field_validator
from src.indexing.indexer import index_codebase
from src.utils.repositories import load_repositories, delete_repository
from src.storage.chroma_store import get_collection, delete_chunks_by_repository
from src.utils.index_status import set_index_status, get_index_status
from src.utils.index_state import delete_repository_state
from filelock import FileLock

def get_repo_lock(repo_name):
   lock_file = os.path.join(LOCK_DIR, f"{repo_name}.lock")
   return FileLock(lock_file)   

LOCK_DIR = "data/repo_locks"
os.makedirs(LOCK_DIR, exist_ok=True)

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

def run_indexing_task(repo_path, repo_name):
    try:
        with get_repo_lock(repo_name):  
          set_index_status(repo_name, "indexing")

          index_codebase(repo_path)

          set_index_status(repo_name, "completed")

    except Exception:
          set_index_status(repo_name, "failed")

class IndexRequest(BaseModel):
  repo_path : str

class IndexResponse(BaseModel):
  success : bool
  repositories : str
  message : str

class IndexStatusResponse(BaseModel):
  repository : str
  status : str

class RepositoryListResponse(BaseModel):
  repositories : list[str]

@app.post("/index", response_model=IndexResponse, status_code=202)
async def index_repository(request: IndexRequest, background_tasks: BackgroundTasks):
  try:
    if not os.path.exists(request.repo_path):
        raise HTTPException(status_code=400, detail="Invalid repository path.")
    
    repository_name = os.path.basename(os.path.normpath(request.repo_path))
    background_tasks.add_task(run_indexing_task, request.repo_path, repository_name)
    return {"success": True, "repositories": repository_name, "message": "Repository indexing started."}

  except HTTPException:
    raise

  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/index/status/{repo_name}",
    response_model=IndexStatusResponse
)
async def indexing_status(repo_name: str):
    status = get_index_status(repo_name)

    if status == "not_found":
        raise HTTPException(
            status_code=404,
            detail=f"No indexing status found for repository '{repo_name}'."
        )

    return {
        "repository": repo_name,
        "status": status
    }

@app.get("/repositories", response_model=RepositoryListResponse)
async def get_repositories():
    return {
        "repositories": load_repositories()
    }

@app.delete("/repositories/{repo_name}")
async def remove_repository(repo_name: str):
    repos = load_repositories()

    if repo_name not in repos:
        raise HTTPException(
            status_code=404,
            detail="Repository not found."
        )

    try:
        with get_repo_lock(repo_name):
            delete_chunks_by_repository(repo_name)
            delete_repository_state(repo_name)
            delete_repository(repo_name)

        return {
            "success": True,
            "message": f"Repository '{repo_name}' deleted successfully."
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete repository: {str(e)}"
        )