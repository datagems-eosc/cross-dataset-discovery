# Logging & Accounting

The service uses `structlog` for structured, JSON-formatted logging.

## Log Structure

All log entries are formatted as JSON objects with a consistent structure, including:

- `@t`: ISO 8601 timestamp.
- `@mt`: The log message (event).
- `@l`: The log level (e.g., "Info", "Warning", "Error").
- `DGCorrelationId`: A unique ID that ties together all log entries for a single HTTP request.
- `SourceContext`: The name of the logger that produced the message.
- `Application`: The name of the application (`cross-dataset-discovery-api`).

Additional key-value pairs are added to provide context for specific events.

## Correlation ID

Every HTTP request is assigned a correlation ID.
- If the incoming request includes an `x-tracking-correlation` header, its value is used.
- Otherwise, a new UUID is generated.

This ID is included in every log message generated during the processing of that request and is also returned in the `x-tracking-correlation` response header. This allows for easy tracing of a request's entire lifecycle through the system and across different services.

## Request Logging

A middleware automatically logs the end of every HTTP request (except for health checks), capturing:

- `RequestMethod` (e.g., GET, POST)
- `RequestPath`
- `StatusCode`
- `ProcessTimeMS`
- `ResponseSize`
