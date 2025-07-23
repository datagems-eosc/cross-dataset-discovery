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
from pyserini.search.lucene import LuceneSearcher
import os
setup_logging()
logger = structlog.get_logger(__name__)

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup sequence initiated.")
    index_path = os.getenv("PYSERINI_INDEX_PATH")
    if not index_path or not os.path.exists(index_path):
        logger.fatal("Pyserini index path not found or not configured. Set PYSERINI_INDEX_PATH.", path=index_path)
        app_state["searcher"] = None
    else:
        try:
            logger.info("Loading Pyserini LuceneSearcher...", index_path=index_path)
            searcher = LuceneSearcher(index_path)
            searcher.set_language('en')
            app_state["searcher"] = searcher
            logger.info("Pyserini LuceneSearcher loaded successfully.")
        except Exception as e:
            logger.fatal("Failed to load Pyserini LuceneSearcher", error=str(e), exc_info=True)
            app_state["searcher"] = None
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

app = FastAPI(
    lifespan=lifespan,
    title="Cross-Dataset Discovery API",
    description="An API for performing cross-dataset discovery using a BM25 index.",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",            
    redoc_url="/redoc"           
)

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
    claims: dict = Depends(security.require_role(["user", "dg_user"]))
):
    """Accepts a query and k, returns the top k similar documents using BM25."""
    user_subject = claims.get("sub")
    client_id = claims.get("clientid")
    
    log = logger.bind(query=request.query, k=request.k, UserId=user_subject)
    if client_id:
        log = log.bind(ClientId=client_id)

    log.info("Search request received.")

    searcher = app_state.get("searcher")
    if not searcher:
        log.error("Search request failed because Pyserini searcher is not available.")
        raise HTTPException(status_code=503, detail="Search service is not available.")

    user_permissions = set(claims.get("datasets", []))
    try:
        search_results_data = search_logic.search_bm25(request.query, request.k, searcher)
        log.info(f"Used BM25 and retireved {len(search_results_data['results'])} results.", query_time=search_results_data["query_time"])
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
    1. Checks if Pyserini searcher is loaded.
    2. Checks database connectivity and schema.
    """
    if not app_state.get("searcher"):
        logger.error("Health check failed: Pyserini searcher is not loaded.")
        raise HTTPException(status_code=503, detail="Pyserini searcher is not available.")
    try:
        app_state["searcher"].search("health check", k=1)
    except Exception as e:
        logger.error("Health check failed: Pyserini dummy search failed.", error=str(e))
        raise HTTPException(status_code=503, detail=f"Pyserini index is not accessible: {e}")
    try:
        search_logic.check_database_schema(conn)
        return {"status": "ok", "message": "All dependencies are healthy."}
    except (ConnectionError, ValueError) as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))