from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from contextlib import asynccontextmanager
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import structlog
from . import search_logic
from . import database
from .models import SearchRequest, SearchResponse, API_SearchResult
from .logging_config import (
    setup_logging,
    correlation_id_middleware,
    request_response_logging_middleware,
    get_correlation_id,
)
from . import security
from pyserini.search.lucene import LuceneSearcher
import os
from typing import List, Optional, Any

setup_logging()
logger = structlog.get_logger(__name__)


class ErrorResponse(BaseModel):
    code: int
    error: str


class ValidationErrorDetail(BaseModel):
    Key: str
    Value: List[str]


class ValidationErrorResponse(BaseModel):
    code: int
    error: str
    message: List[ValidationErrorDetail]


class FailedDependencyMessage(BaseModel):
    statusCode: int
    source: str
    correlationId: Optional[str] = None
    payload: Optional[Any] = None


class FailedDependencyResponse(BaseModel):
    code: int
    error: str
    message: FailedDependencyMessage


class FailedDependencyException(HTTPException):
    def __init__(
        self,
        source: str,
        status_code: int,
        detail: str,
        correlation_id: Optional[str] = None,
        payload: Optional[Any] = None,
    ):
        self.source = source
        self.downstream_status_code = status_code
        self.correlation_id = correlation_id
        self.downstream_payload = payload
        super().__init__(status_code=status.HTTP_424_FAILED_DEPENDENCY, detail=detail)


app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup sequence initiated.")
    index_path = os.getenv("PYSERINI_INDEX_PATH")
    if not index_path or not os.path.exists(index_path):
        logger.fatal(
            "Pyserini index path not found or not configured. Set PYSERINI_INDEX_PATH.",
            path=index_path,
        )
        app_state["searcher"] = None
    else:
        try:
            logger.info("Loading Pyserini LuceneSearcher...", index_path=index_path)
            searcher = LuceneSearcher(index_path)
            searcher.set_language("en")
            app_state["searcher"] = searcher
            logger.info("Pyserini LuceneSearcher loaded successfully.")
        except Exception as e:
            logger.fatal(
                "Failed to load Pyserini LuceneSearcher", error=str(e), exc_info=True
            )
            app_state["searcher"] = None
    logger.info("Creating database connection pool...")
    try:
        database.connection_pool = SimpleConnectionPool(
            minconn=1, maxconn=2, dsn=database.CONNECTION_STRING
        )
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
    docs_url="/swagger",
    redoc_url="/redoc",
    root_path=os.getenv("ROOT_PATH", ""),
)


@app.exception_handler(FailedDependencyException)
async def failed_dependency_exception_handler(
    request: Request, exc: FailedDependencyException
):
    response_content = FailedDependencyResponse(
        code=104,
        error="error communicating with underpinning service",
        message=FailedDependencyMessage(
            statusCode=exc.downstream_status_code,
            source=exc.source,
            correlationId=exc.correlation_id,
            payload=exc.downstream_payload,
        ),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=response_content.model_dump(exclude_none=True),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = []
    for error in exc.errors():
        key = ".".join(map(str, error.get("loc", [])))
        if key.startswith("body."):  # Clean up the key for better readability
            key = key[5:]
        message = error.get("msg", "Invalid input")
        details.append(ValidationErrorDetail(Key=key, Value=[message]))

    response_content = ValidationErrorResponse(
        code=102,  # As per spec example for validation errors
        error="Validation Error",
        message=details,
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=response_content.model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    error_code_map = {
        401: 401,
        403: 403,
        404: 404,
        500: 500,
    }
    error_code = error_code_map.get(exc.status_code, exc.status_code)

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(code=error_code, error=exc.detail).model_dump(),
    )


app.middleware("http")(correlation_id_middleware)
app.middleware("http")(request_response_logging_middleware)


def get_db_connection():
    """Dependency to get a database connection from the pool."""
    if database.connection_pool is None:
        logger.error("Database connection requested but pool is not available.")
        raise FailedDependencyException(
            source="Database",
            status_code=503,
            detail="Database connection pool is not available.",
            correlation_id=get_correlation_id(),
        )

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
):
    """Accepts a query and k, returns the top k similar documents using BM25."""
    user_subject = claims.get("sub")
    client_id = claims.get("clientid")

    log = logger.bind(query=request.query, k=request.k, UserId=user_subject)
    if client_id:
        log = log.bind(ClientId=client_id)
    if request.dataset_ids:
        log = log.bind(dataset_ids=request.dataset_ids)

    log.info("Search request received.")

    searcher = app_state.get("searcher")
    if not searcher:
        log.error("Search request failed because Pyserini searcher is not available.")
        raise FailedDependencyException(
            source="PyseriniIndex",
            status_code=503,
            detail="Search service is not available.",
            correlation_id=get_correlation_id(),
        )

    user_permissions = set(claims.get("datasets", []))
    try:
        search_results_data = search_logic.search_bm25(
            request.query, request.k, searcher, dataset_ids=request.dataset_ids
        )
        log.info(
            f"Used BM25 and retireved {len(search_results_data['results'])} results.",
            query_time=search_results_data["query_time"],
        )
        authorized_results = []
        for result in search_results_data["results"]:
            required_permission = f"/dataset/group/{result.use_case}/search"
            if required_permission in user_permissions:
                authorized_results.append(
                    API_SearchResult.model_validate(result.model_dump())
                )
            else:
                log.warning(
                    "Result filtered due to missing permission",
                    required_permission=required_permission,
                    source_id=result.source_id,
                )
        final_response = {
            "query_time": search_results_data["query_time"],
            "results": authorized_results,
        }
        log.info(
            "Search completed and filtered.",
            query_time=final_response["query_time"],
            initial_results_count=len(search_results_data["results"]),
            authorized_results_count=len(final_response["results"]),
        )
        return SearchResponse(**final_response)

    except Exception as e:
        log.error(
            "An unexpected error occurred during search.", error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.get("/health")
def health_check(conn=Depends(get_db_connection)):
    """
    Performs a deep health check on the service's dependencies.
    1. Checks if Pyserini searcher is loaded.
    2. Checks database connectivity and schema.
    """
    if not app_state.get("searcher"):
        logger.error("Health check failed: Pyserini searcher is not loaded.")
        raise FailedDependencyException(
            source="PyseriniIndex",
            status_code=503,
            detail="Pyserini searcher is not available.",
            correlation_id=get_correlation_id(),
        )
    try:
        app_state["searcher"].search("health check", k=1)
    except Exception as e:
        logger.error("Health check failed: Pyserini dummy search failed.", error=str(e))
        raise FailedDependencyException(
            source="PyseriniIndex",
            status_code=503,
            detail=f"Pyserini index is not accessible: {e}",
            correlation_id=get_correlation_id(),
        )
    try:
        search_logic.check_database_schema(conn)
        return {"status": "ok", "message": "All dependencies are healthy."}
    except (ConnectionError, ValueError) as e:
        logger.error("Health check failed", error=str(e))
        raise FailedDependencyException(
            source="Database",
            status_code=503,
            detail=str(e),
            correlation_id=get_correlation_id(),
        )
