from fastapi import FastAPI, Depends, HTTPException, Request
from contextlib import asynccontextmanager
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import structlog
from . import search_logic
from . import database 
from .models import SearchRequest, SearchResponse, SearchResult, API_SearchResult 
from .logging_config import setup_logging, correlation_id_middleware, request_response_logging_middleware
from . import security
from sentence_transformers import SentenceTransformer
import os

setup_logging()
logger = structlog.get_logger(__name__)

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup sequence initiated.")
    
    logger.info("Loading BGE-M3 embedding model...")
    model_name = "BAAI/bge-m3"
    try:
        embedding_model = SentenceTransformer(model_name)
        app_state["embedding_model"] = embedding_model
        logger.info("BGE-M3 embedding model loaded successfully.")
    except Exception as e:
        logger.fatal("Failed to load embedding model", model=model_name, error=str(e), exc_info=True)
        app_state["embedding_model"] = None

    logger.info("Creating database connection pool...")
    try:
        database.connection_pool = SimpleConnectionPool(minconn=1, maxconn=2, dsn=database.CONNECTION_STRING)
        logger.info("Database connection pool created successfully.")
    except psycopg2.OperationalError as e:
        logger.fatal("Could not create database connection pool", error=str(e))
        database.connection_pool = None
    
    yield
    
    logger.info("Application shutdown sequence initiated.")
    logger.info("Closing database connection pool...")
    if database.connection_pool:
        database.connection_pool.closeall()
    app_state.clear()
    logger.info("Shutdown complete.")

app = FastAPI(lifespan=lifespan)
app.middleware("http")(correlation_id_middleware)
app.middleware("http")(request_response_logging_middleware)

def get_db_connection():
    """Dependency to get a database connection from the pool."""
    if database.connection_pool is None:
        logger.error("Database connection requested but pool is not available.")
        raise HTTPException(status_code=503, detail="Database connection pool is not available.")
    
    conn = None
    try:
        conn = database.connection_pool.getconn()
        yield conn
    finally:
        if conn:
            database.connection_pool.putconn(conn)

@app.get("/")
def read_root():
    """A simple health check endpoint."""
    logger.info("Health check endpoint was hit.")
    return {"status": "ok", "message": "Cross Dataset Discovery API is running."}

@app.post("/search/", response_model=SearchResponse)
def perform_search(
    request: SearchRequest, 
    claims: dict = Depends(security.require_role(["user", "dg_user"])),
    conn=Depends(get_db_connection)
):
    """Accepts a query and k, returns the top k similar documents using dense retrieval."""
    user_subject = claims.get("sub")
    log = logger.bind(query=request.query, k=request.k, user_subject=user_subject)
    log.info("Dense search request received.")

    model = app_state.get("embedding_model")
    if not model:
        log.error("Search request failed because the embedding model is not available.")
        raise HTTPException(status_code=503, detail="Search service is not available.")

    user_permissions = set(claims.get("datasets", []))
    try:
        search_results_data = search_logic.search_dense(request.query, request.k, model, conn)
        log.info(f"Used dense retrieval to find {len(search_results_data['results'])} results.", query_time=search_results_data["query_time"])
        
        authorized_results = []
        for result in search_results_data["results"]:
            required_permission = f"/dataset/group/{result.use_case}/search"
            if required_permission in user_permissions:
                authorized_results.append(API_SearchResult.model_validate(result.model_dump()))
            else:
                log.warning(
                    "Result filtered due to missing permission",
                    required_permission=required_permission,
                    source_id=result.source_id
                )
        final_response = {
            "query_time": search_results_data["query_time"],
            "results": authorized_results
        }
        log.info(
            "Search completed and filtered.",
            query_time=final_response["query_time"],
            initial_results_count=len(search_results_data["results"]),
            authorized_results_count=len(final_response["results"])
        )
        return SearchResponse(**final_response)

    except Exception as e:
        log.error("An unexpected error occurred during search.", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    
@app.get("/health")
def health_check(conn=Depends(get_db_connection)):
    """
    Performs a deep health check on the service's dependencies.
    1. Checks if the embedding model is loaded.
    2. Checks database connectivity and schema.
    """
    model = app_state.get("embedding_model")
    if not model:
        logger.error("Health check failed: Embedding model is not loaded.")
        raise HTTPException(status_code=503, detail="Embedding model is not available.")
    try:
        model.encode("health check")
    except Exception as e:
        logger.error("Health check failed: Embedding model dummy encoding failed.", error=str(e))
        raise HTTPException(status_code=503, detail=f"Embedding model is not responsive: {e}")

    try:
        search_logic.check_database_schema(conn)
        return {"status": "ok", "message": "All dependencies are healthy."}
    except (ConnectionError, ValueError) as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))