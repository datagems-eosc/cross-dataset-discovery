import time
import psycopg2
import json 
from pgvector.psycopg2 import register_vector
from pyserini.search.lucene import LuceneSearcher 
from .database import TABLE_NAME
from .models import SearchResult

def search_pyserini_index(query: str, k: int, searcher: LuceneSearcher) -> dict:
    """
    Performs a search on the pre-built Pyserini Lucene index.
    """
    start_time = time.time()
    hits = searcher.search(query, k=k)
    end_time = time.time()

    query_duration = end_time - start_time

    results_list = []
    for hit in hits:
        raw_doc_str = hit.lucene_document.get("raw")
        if raw_doc_str:
            doc_data = json.loads(raw_doc_str)
            results_list.append(
                SearchResult(
                    content=doc_data.get("content"),
                    use_case=doc_data.get("use_case"),
                    source=doc_data.get("source"),
                    source_id=doc_data.get("source_id"),
                    chunk_id=doc_data.get("chunk_id"),
                    language=doc_data.get("language"),
                    distance=hit.score 
                )
            )

    return {"query_time": query_duration, "results": results_list}

def search_db(query: str, k: int, model, conn):
    """
    Performs a BM25-style keyword search using PostgreSQL Full-Text Search.
    Retrieves the k most relevant results from the database.
    
    The 'model' parameter is accepted to maintain a consistent interface
    but is NOT used in this function.
    """
    try:
        # Clean and validate input
        query = query.strip()
        if not query:
            return {"query_time": 0, "results": []}
        
        # Extract words and create different query types
        query_words = re.findall(r'\w+', query.lower())
        if not query_words:
            return {"query_time": 0, "results": []}
        
        # Create OR query for individual words (broader match)
        or_query = " | ".join(query_words)
        
        # Use the original query for phrase search (exact phrase match)
        phrase_query = query
        
        # Create AND query for stricter matching
        and_query = " & ".join(query_words)

        with conn.cursor() as cur:
            start_time = time.time()
            
            # Use parameterized query to prevent SQL injection
            sql_query = f"""
            SELECT 
                content, 
                use_case, 
                source, 
                source_id, 
                chunk_id, 
                language, 
                (
                    -- Phrase match gets highest weight
                    CASE 
                        WHEN ts_content @@ phraseto_tsquery('english', %s) 
                        THEN ts_rank_cd(ts_content, phraseto_tsquery('english', %s)) * 10
                        ELSE 0 
                    END +
                    -- AND query gets medium weight (all words must be present)
                    CASE 
                        WHEN ts_content @@ to_tsquery('english', %s) 
                        THEN ts_rank_cd(ts_content, to_tsquery('english', %s)) * 3
                        ELSE 0 
                    END +
                    -- OR query gets base weight (any word can match)
                    CASE 
                        WHEN ts_content @@ to_tsquery('english', %s) 
                        THEN ts_rank_cd(ts_content, to_tsquery('english', %s)) * 1
                        ELSE 0 
                    END
                ) AS relevance
            FROM {TABLE_NAME}
            WHERE 
                ts_content @@ to_tsquery('english', %s) OR 
                ts_content @@ to_tsquery('english', %s) OR
                ts_content @@ phraseto_tsquery('english', %s)
            ORDER BY 
                relevance DESC
            LIMIT %s;
            """
            
            # Execute with proper parameter binding
            cur.execute(sql_query, (
                phrase_query, phrase_query,  # phrase match params
                and_query, and_query,        # and match params  
                or_query, or_query,          # or match params
                or_query, and_query, phrase_query,  # WHERE clause params
                k
            ))
            
            end_time = time.time()
            query_duration = end_time - start_time
            
            rows = cur.fetchall()
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
        print(f"Database error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error in search: {e}")
        raise

    
def search_db_embedding(query: str, k: int, model, conn):
    """
    Computes the embedding and retrieves k similar results from PostgreSQL.
    Assumes the connection and model are provided.
    """
    query_embedding = model.encode(query).tolist()
    
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SET LOCAL max_parallel_workers_per_gather = 8;")

            start_time = time.time()
            cur.execute(
                f"""
                SELECT content, use_case, source, source_id, chunk_id, language, embedding <-> %s::vector AS distance
                FROM {TABLE_NAME}
                ORDER BY distance
                LIMIT %s;
                """,
                (query_embedding, k)
            )
            end_time = time.time()
            
            query_duration = end_time - start_time
            
            rows = cur.fetchall()

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
        print(f"Database error: {e}")
        raise
    
def check_database_schema(conn):
    """
    Checks if the database is connected, the required table exists,
    and all necessary columns are present in the table.
    Raises an exception if any check fails.
    """
    required_columns = {
        "content", "use_case", "source", "source_id", 
        "chunk_id", "language", "ts_content", "embedding"
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
        # Re-raise database-specific errors to be caught by the health endpoint
        raise ConnectionError(f"Database check failed: {e}") from e