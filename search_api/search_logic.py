import time
import psycopg2
from pgvector.psycopg2 import register_vector
from .database import TABLE_NAME
from .models import SearchResult
from sentence_transformers import SentenceTransformer

def search_dense(query: str, k: int, model: SentenceTransformer, conn) -> dict:
    """
    Computes the embedding for the query and retrieves k most similar results from PostgreSQL using pgvector.
    """
    start_time = time.time()
    query_embedding = model.encode(query)   
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            sql_query = f"""
                SELECT 
                    content, 
                    use_case, 
                    source, 
                    source_id, 
                    chunk_id, 
                    language, 
                    embedding <-> %s::vector AS distance
                FROM {TABLE_NAME}
                ORDER BY 
                    distance ASC
                LIMIT %s;
            """
            
            cur.execute(sql_query, (query_embedding, k))
            rows = cur.fetchall()
            
            end_time = time.time()
            query_duration = end_time - start_time

            results_list = [
                SearchResult(
                    content=row[0],
                    use_case=row[1],
                    source=row[2],
                    source_id=row[3],
                    chunk_id=row[4],
                    language=row[5],
                    distance=row[6]
                )
                for row in rows
            ]
            
            return {"query_time": query_duration, "results": results_list}
            
    except psycopg2.Error as e:
        print(f"Database error during dense search: {e}")
        raise

def check_database_schema(conn):
    """
    Checks if the database is connected, the required table exists,
    and all necessary columns are present in the table.
    Raises an exception if any check fails.
    """
    required_columns = {
        "content", "use_case", "source", "source_id", 
        "chunk_id", "language", "embedding"
    }

    try:
        with conn.cursor() as cur:
            # 1. Check for table existence by querying it
            cur.execute(f"SELECT 1 FROM {TABLE_NAME} LIMIT 1;")

            # 2. Check for column existence
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s;
            """, (TABLE_NAME,))
            
            existing_columns = {row[0] for row in cur.fetchall()}

            missing_columns = required_columns - existing_columns

            if missing_columns:
                raise ValueError(f"Schema validation failed. Missing columns in table '{TABLE_NAME}': {', '.join(missing_columns)}")

    except psycopg2.Error as e:
        raise ConnectionError(f"Database check failed: {e}") from e