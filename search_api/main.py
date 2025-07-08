from fastapi import FastAPI, Depends, HTTPException, Request
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import structlog
from . import search_logic
from . import database 
from .models import SearchRequest, SearchResponse
from .logging_config import setup_logging, correlation_id_middleware, request_response_logging_middleware
setup_logging()
logger = structlog.get_logger(__name__)

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup sequence initiated.")
    
    #logger.info("Loading SentenceTransformer model...")
    #app_state["model"] = SentenceTransformer('BAAI/bge-m3')
    #logger.info("Model loaded successfully.")

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
def perform_search(request: SearchRequest, conn=Depends(get_db_connection)):
    """Accepts a query and k, returns the top k similar documents."""
    log = logger.bind(query=request.query, k=request.k)
    log.info("Search request received.")
    # the commented lines are related to the dense retrieval, hence commented out for the bm25 retrieval
    #if "model" not in app_state or app_state["model"] is None:
    #    log.error("Search request failed because model is not loaded.")
    #    raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    try:
        #model = app_state["model"]
        model = None # this is placeholder and should change when we transition back to using dense retrieval
        search_results = search_logic.search_db(request.query, request.k, model, conn)
        log.info("Search completed successfully.", query_time=search_results["query_time"], results_count=len(search_results["results"]))
        return SearchResponse(**search_results)
    except psycopg2.Error as e:
        log.error("Database query failed during search.", error=str(e))
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")
    except Exception as e:
        log.error("An unexpected error occurred during search.", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    
@app.get("/health")
def health_check(conn=Depends(get_db_connection)):
    """
    Performs a deep health check on the service's dependencies.
    1. Checks database connectivity.
    2. Checks table and column schema.
    """
    try:
        # The get_db_connection dependency already checks connectivity.
        # Now, we check the schema.
        search_logic.check_database_schema(conn)
        return {"status": "ok", "message": "All dependencies are healthy."}
    except (ConnectionError, ValueError) as e:
        # Log the specific error for debugging
        logger.error("Health check failed", error=str(e))
        # Return a 503 Service Unavailable, which is standard for failing health checks
        raise HTTPException(status_code=503, detail=str(e))