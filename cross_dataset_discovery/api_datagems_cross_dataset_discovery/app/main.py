import time
from contextlib import asynccontextmanager

import psycopg2
import structlog
from api_datagems_cross_dataset_discovery.app.config import settings
from api_datagems_cross_dataset_discovery.app.exceptions import (
    ErrorResponse,
    FailedDependencyException,
    FailedDependencyMessage,
    FailedDependencyResponse,
    ValidationErrorDetail,
    ValidationErrorResponse,
)
from api_datagems_cross_dataset_discovery.app.logging_config import (
    correlation_id_middleware,
    request_response_logging_middleware,
    setup_logging,
)
from api_datagems_cross_dataset_discovery.app.models import (
    API_SearchResult,
    SearchRequest,
    SearchResponse,
)
from nlp_retrieval.core.models import RetrievalResult
from nlp_retrieval.retrievers.bm25_retriever import PyseriniRetriever
from nlp_retrieval.searcher import Searcher
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from psycopg2.pool import SimpleConnectionPool
from typing import List

from . import database, security

setup_logging()
logger = structlog.get_logger(__name__)
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup sequence initiated.")

    # 1. Initialize the Searcher Component
    try:
        logger.info("Initializing search components...", index_path=settings.INDEX_PATH)
        # Configure the retriever we want to use
        bm25_retriever = PyseriniRetriever(enable_tqdm=False)
        # Assemble the main searcher orchestrator
        searcher = Searcher(retrievers=[bm25_retriever])
        app_state["searcher"] = searcher
        logger.info("Search components initialized successfully.")
    except Exception as e:
        logger.fatal(
            "Failed to initialize search components", error=str(e), exc_info=True
        )
        app_state["searcher"] = None

    # 2. Create Database Connection Pool
    logger.info("Creating database connection pool...")
    try:
        database.connection_pool = SimpleConnectionPool(
            minconn=1, maxconn=2, dsn=settings.DB_CONNECTION_STRING
        )
        logger.info("Database connection pool created successfully.")
    except psycopg2.OperationalError as e:
        logger.fatal("Could not create database connection pool", error=str(e))
        database.connection_pool = None

    yield

    logger.info("Application shutdown sequence initiated.")
    if database.connection_pool:
        database.connection_pool.closeall()
        logger.info("Database connection pool closed.")
    app_state.clear()
    logger.info("Shutdown complete.")


app = FastAPI(
    lifespan=lifespan,
    title="Cross-Dataset Discovery API",
    version="1.0.0",
    root_path=settings.ROOT_PATH,
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
    details = [
        ValidationErrorDetail(
            Key=".".join(map(str, err.get("loc", []))), Value=[err.get("msg", "")]
        )
        for err in exc.errors()
    ]
    response_content = ValidationErrorResponse(
        code=102, error="Validation Error", message=details
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST, content=response_content.model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(code=exc.status_code, error=exc.detail).model_dump(),
    )


# --- Middleware ---
app.middleware("http")(correlation_id_middleware)
app.middleware("http")(request_response_logging_middleware)


# --- Endpoints ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Cross Dataset Discovery API is running."}


@app.post("/search/", response_model=SearchResponse)
def perform_search(
    request: SearchRequest,
    claims: dict = Depends(security.require_role(["user", "dg_user"])),
):
    start_time = time.time()
    user_subject = claims.get("sub")
    log = logger.bind(query=request.query, k=request.k, UserId=user_subject)

    searcher: Searcher | None = app_state.get("searcher")
    if not searcher:
        log.error("Search request failed because searcher component is not available.")
        raise FailedDependencyException(
            source="SearchComponent",
            status_code=503,
            detail="Search service is not available.",
        )

    try:
        # Prepare filters for the component based on the request
        search_filters = {}
        if request.dataset_ids:
            log = log.bind(dataset_ids=request.dataset_ids)
            search_filters["source"] = request.dataset_ids

        # The component expects a list of queries and returns a list of lists of results
        search_results_batch: List[List[RetrievalResult]] = searcher.search(
            nlqs=[request.query],
            output_path=settings.INDEX_PATH,
            k=request.k,
            filters=search_filters,  # Pass the filters
        )

        # We only sent one query, so we only care about the first list of results
        component_results = search_results_batch[0]
        log.info(f"Component returned {len(component_results)} results.")

        # Authorize and transform results for the API response
        user_permissions = set(claims.get("datasets", []))
        authorized_results = []
        for result in component_results:
            # Metadata is in result.item.metadata
            use_case = result.item.metadata.get("use_case")
            required_permission = f"/dataset/group/{use_case}/search"

            if required_permission in user_permissions:
                # Map from RetrievalResult to API_SearchResult
                api_result = API_SearchResult.model_validate(
                    {
                        **result.item.metadata,
                        "score": result.score,
                        "content": result.item.content,
                    }
                )
                authorized_results.append(api_result)
            else:
                log.warning(
                    "Result filtered due to missing permission",
                    required_permission=required_permission,
                    source_id=result.item.metadata.get("source_id"),
                )

        query_time = (time.time() - start_time) * 1000
        final_response = SearchResponse(
            query_time=round(query_time, 2),
            results=authorized_results,
        )
        log.info(
            "Search completed and filtered.",
            authorized_results_count=len(authorized_results),
        )
        return final_response

    except Exception as e:
        log.error(
            "An unexpected error occurred during search.", error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.get("/health")
def health_check(conn=Depends(database.get_db_connection)):
    # 1. Check if searcher component is loaded
    if not app_state.get("searcher"):
        raise FailedDependencyException(
            source="SearchComponent",
            status_code=503,
            detail="Searcher is not available.",
        )

    # 2. Check database connectivity and schema
    try:
        database.check_database_schema(conn)
        return {"status": "ok", "message": "All dependencies are healthy."}
    except (ConnectionError, ValueError) as e:
        logger.error("Health check failed", error=str(e))
        raise FailedDependencyException(
            source="Database", status_code=503, detail=str(e)
        )
