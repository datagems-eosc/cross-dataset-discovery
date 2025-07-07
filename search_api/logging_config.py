import logging
import sys
import time
import uuid
from contextvars import ContextVar

import structlog
from fastapi import Request, Response
from structlog.types import EventDict, Processor

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default=None)

async def request_response_logging_middleware(request: Request, call_next):
    """
    FastAPI middleware to log the details of every incoming request and its response.
    """
    log = structlog.get_logger(__name__)
    start_time = time.time()
    
    try:
        response: Response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        status_code = response.status_code

        log_level = "info"
        if status_code >= 500:
            log_level = "error"
        elif status_code >= 400:
            log_level = "warning"
        
        log.log(
            getattr(logging, log_level.upper()),
            "http_request_finished",
            RequestMethod=request.method,
            RequestPath=str(request.url),
            StatusCode=status_code,
            ProcessTimeMS=round(process_time, 2)
        )

    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        log.error(
            "http_request_unhandled_exception",
            RequestMethod=request.method,
            RequestPath=str(request.url),
            ProcessTimeMS=round(process_time, 2),
            exc_info=e,
        )
        raise
        
    return response


def get_correlation_id() -> str | None:
    """Returns the current correlation ID."""
    return correlation_id_var.get()

async def correlation_id_middleware(request: Request, call_next):
    """
    FastAPI middleware to extract or generate a correlation ID
    and store it in a context variable for the request's lifetime.
    """
    correlation_id = request.headers.get("x-tracking-correlation")
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
    
    token = correlation_id_var.set(correlation_id)

    response: Response = await call_next(request)
    response.headers["x-tracking-correlation"] = get_correlation_id()
    
    correlation_id_var.reset(token)
    return response

def datagems_log_formatter(_, __, event_dict: EventDict) -> EventDict:
    formatted_event = {}
    formatted_event["@t"] = event_dict.pop("timestamp", None)
    formatted_event["@mt"] = event_dict.pop("event", None)
    level = event_dict.pop("log_level", "info")
    formatted_event["@l"] = level.capitalize()
    formatted_event["DGCorrelationId"] = event_dict.pop("correlation_id", None)
    formatted_event["SourceContext"] = event_dict.pop("logger", None)
    formatted_event.update(event_dict)
    return formatted_event


def setup_logging():
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        datagems_log_formatter,
    ]
    structlog.configure(
        processors=shared_processors + [
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    structlog.contextvars.bind_contextvars(
        correlation_id=get_correlation_id,
        Application="cross-dataset-discovery-api",
    )