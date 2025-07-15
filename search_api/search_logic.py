import time
import psycopg2
from pgvector.psycopg2 import register_vector
from .database import TABLE_NAME
from .models import SearchResult
from pyserini.search.lucene import LuceneSearcher
import json 

def search_bm25(query: str, k: int, searcher: LuceneSearcher) -> dict:
    """
    Performs a BM25 search using a pre-initialized Pyserini LuceneSearcher.
    """
    start_time = time.time()
    
    hits = searcher.search(query, k=k)
    
    end_time = time.time()
    query_duration = end_time - start_time
    results_list = []
    for i, hit in enumerate(hits):
        raw_doc = json.loads(hit.lucene_document.get("raw"))
        if "contents" not in raw_doc or "use_case" not in raw_doc:
            continue
        result = SearchResult(
            content=raw_doc.get("contents"),
            use_case=raw_doc.get("use_case"),
            source=raw_doc.get("source"),
            source_id=raw_doc.get("source_id"),
            chunk_id=int(raw_doc.get("chunk_id", 0)), 
            language=raw_doc.get("language"),
            distance=hit.score 
        )
        results_list.append(result)

    return {"query_time": query_duration, "results": results_list}

def search_with_rerank(query: str, k: int, searcher: LuceneSearcher, reranker: MxbaiRerankV2) -> dict:
    """
    Performs a hybrid search:
    1. Retrieves an initial set of candidates using BM25.
    2. Reranks the candidates using a powerful cross-encoder model.
    """
    log = structlog.get_logger(__name__)
    total_start_time = time.time()

    # 1. Initial retrieval (fetch 5x more documents for the reranker)
    rerank_multiplier = 5
    initial_k = k * rerank_multiplier
    log.info("Step 1: Initial retrieval with BM25", initial_k=initial_k)
    bm25_results_data = search_bm25(query, initial_k, searcher)
    initial_results = bm25_results_data["results"]

    if not initial_results:
        log.info("BM25 returned no results. Skipping reranking.")
        return {"query_time": bm25_results_data["query_time"], "results": []}

    # 2. Rerank the results
    log.info("Step 2: Reranking candidates", num_candidates=len(initial_results))
    documents_to_rerank = [res.content for res in initial_results]
    
    reranked_scores = reranker.rank(
        query=query,
        documents=documents_to_rerank,
        return_documents=False,
        top_k=k
    )

    # 3. Construct final list from reranked results
    final_results = []
    for score_info in reranked_scores:
        reranked_result = initial_results[score_info['corpus_id']]
        reranked_result.distance = score_info['score'] # Update distance with the new, more accurate score
        final_results.append(reranked_result)

    total_duration = time.time() - total_start_time
    log.info("Hybrid search complete.", total_duration=total_duration, final_results_count=len(final_results))
    return {"query_time": total_duration, "results": final_results}

def search_db(query: str, k: int, model, conn):
    """
    Performs a BM25-style keyword search using PostgreSQL Full-Text Search.
    Retrieves the k most relevant results from the database.
    
    The 'model' parameter is accepted to maintain a consistent interface
    but is NOT used in this function.
    """
    try:
        with conn.cursor() as cur:
            start_time = time.time()

            sql_query = f"""
            SELECT 
                content, 
                use_case, 
                source, 
                source_id, 
                chunk_id, 
                language, 
                ts_rank_cd(ts_content, plainto_tsquery('english', %s)) AS relevance
            FROM {TABLE_NAME}
            WHERE 
                ts_content @@ plainto_tsquery('english', %s)
            ORDER BY 
                relevance DESC
            LIMIT %s;
            """
            
            cur.execute(sql_query, (query, query, k))
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