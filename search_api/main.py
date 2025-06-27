from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
import psycopg2

# Use relative imports since this is part of a package
from . import search_logic
from .database import connection_pool
from .models import SearchRequest, SearchResponse

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Code to run on startup ---
    print("Loading SentenceTransformer model...")
    app_state["model"] = SentenceTransformer('BAAI/bge-m3')
    print("Model loaded.")
    print("Database connection pool is ready.")
    
    yield
    
    # --- Code to run on shutdown ---
    print("Closing database connection pool...")
    if connection_pool:
        connection_pool.closeall()
    print("Shutdown complete.")

app = FastAPI(lifespan=lifespan)

def get_db_connection():
    """Dependency to get a database connection from the pool."""
    conn = None
    try:
        conn = connection_pool.getconn()
        yield conn
    finally:
        if conn:
            connection_pool.putconn(conn)

@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Search API is running."}

@app.post("/search/", response_model=SearchResponse)
def perform_search(request: SearchRequest, conn=Depends(get_db_connection)):
    """Accepts a query and k, returns the top k similar documents."""
    if "model" not in app_state or app_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        model = app_state["model"]
        search_results = search_logic.search_db(request.query, request.k, model, conn)
        return SearchResponse(**search_results)
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")