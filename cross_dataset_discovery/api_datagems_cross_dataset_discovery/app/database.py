import psycopg2
import structlog
from api_datagems_cross_dataset_discovery.app.config import (
    FailedDependencyException,
    get_correlation_id,
    settings,
)
from psycopg2 import sql
from psycopg2.pool import SimpleConnectionPool

logger = structlog.get_logger(__name__)

connection_pool: SimpleConnectionPool | None = None


def get_db_connection():
    """Dependency to get a database connection from the pool."""
    if connection_pool is None:
        logger.error("Database connection requested but pool is not available.")
        raise FailedDependencyException(
            source="Database",
            status_code=503,
            detail="Database connection pool is not available.",
            correlation_id=get_correlation_id(),
        )

    conn = None
    try:
        conn = connection_pool.getconn()
        yield conn
    finally:
        if conn:
            connection_pool.putconn(conn)


def check_database_schema(conn):
    """
    Checks if the database is connected, the required table exists,
    and all necessary columns are present in the table.
    Raises an exception if any check fails.
    """
    required_columns = {
        "content",
        "use_case",
        "source",
        "source_id",
        "chunk_id",
        "language",
        "ts_content",
        "embedding",
    }

    try:
        with conn.cursor() as cur:
            query = sql.SQL("SELECT 1 FROM {table} LIMIT 1;").format(
                table=sql.Identifier(settings.TABLE_NAME)
            )
            cur.execute(query)

            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s;",
                (settings.TABLE_NAME,),
            )
            existing_columns = {row[0] for row in cur.fetchall()}
            missing_columns = required_columns - existing_columns
            if missing_columns:
                raise ValueError(
                    f"Schema validation failed. Missing columns in table '{settings.TABLE_NAME}': {', '.join(missing_columns)}"
                )
    except psycopg2.Error as e:
        raise ConnectionError(f"Database check failed: {e}") from e
