from fastapi import FastAPI, Depends, HTTPException, Request
from contextlib import asynccontextmanager
import os
from pyserini.search.lucene import LuceneSearcher 
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import structlog
from . import search_logic
from . import database 
from .models import SearchRequest, SearchResponse, SearchResult, API_SearchResult 
from .logging_config import setup_logging, correlation_id_middleware, request_response_logging_middleware
from . import security
setup_logging()
logger = structlog.get_logger(__name__)

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup sequence initiated.")
    
    #logger.info("Loading SentenceTransformer model...")
    #app_state["model"] = SentenceTransformer('BAAI/bge-m3')
    #logger.info("Model loaded successfully.")
    index_path = os.getenv("PYSERINI_INDEX_PATH")
    if not index_path:
        logger.fatal("PYSERINI_INDEX_PATH environment variable not set.")
        app_state["searcher"] = None
    else:
        try:
            logger.info("Loading Pyserini LuceneSearcher...", index_path=index_path)
            searcher = LuceneSearcher(index_path)
            searcher.set_language('en') # IMPORTANT: As per your example
            app_state["searcher"] = searcher
            logger.info("Pyserini LuceneSearcher loaded successfully.")
        except Exception as e:
            logger.fatal("Could not load Pyserini LuceneSearcher", error=str(e))
            app_state["searcher"] = None

    logger.info("Creating database connection pool...")
    try:
        database.connection_pool = SimpleConnectionPool(minconn=1, maxconn=10, dsn=database.CONNECTION_STRING)
        logger.info("Database connection pool created successfully.")
    except psycopg2.OperationalError as e:
        logger.fatal("Could not create database connection pool", error=str(e))
        database.connection_pool = None
    
    yield
    
    logger.info("Application shutdown sequence initiated.")
    logger.info("Closing database connection pool...")
    if database.connection_pool:
        database.connection_pool.closeall()
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
    #conn=Depends(get_db_connection),
    claims: dict = Depends(security.require_role(["user", "dg_user"])) # protect the endpoint with JWT authentication, checkkig for user or dg_user roles
):
    """Accepts a query and k, returns the top k similar documents."""
    user_subject = claims.get("sub")
    log = logger.bind(query=request.query, k=request.k, user_subject=user_subject)
    log.info("Search request received.")
    # the commented lines are related to the dense retrieval, hence commented out for the bm25 retrieval
    #if "model" not in app_state or app_state["model"] is None:
    #    log.error("Search request failed because model is not loaded.")
    #    raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    if "searcher" not in app_state or app_state["searcher"] is None:
        log.error("Search request failed because Pyserini searcher is not loaded.")
        raise HTTPException(status_code=503, detail="Searcher is not loaded yet.")

    user_permissions = set(claims.get("datasets", []))
    try:
        # CHANGED: Call the new Pyserini search function
        searcher = app_state["searcher"]
        search_results_data = search_logic.search_pyserini_index(request.query, request.k, searcher)
        
        # This authorization logic remains exactly the same, which is great!
        authorized_results = []
        for result in search_results_data["results"]:
            required_permission = f"/dataset/group/{result.use_case}/search"
            if required_permission in user_permissions:
                authorized_results.append(result)
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
    try:
        search_logic.check_database_schema(conn)
        if "searcher" not in app_state or app_state["searcher"] is None:
            raise ConnectionError("Pyserini searcher is not loaded.")
        return {"status": "ok", "message": "All dependencies are healthy."}
    except (ConnectionError, ValueError) as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))