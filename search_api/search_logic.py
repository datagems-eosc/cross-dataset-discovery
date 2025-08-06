import time
import psycopg2
from pgvector.psycopg2 import register_vector
from .database import TABLE_NAME
from .models import SearchResult
from pyserini.search.lucene import LuceneSearcher, querybuilder
from pyserini.analysis import Analyzer, get_lucene_analyzer
import json
from typing import List, Optional
import structlog
from psycopg2 import sql

logger = structlog.get_logger(__name__)


def search_bm25(
    query: str,
    k: int,
    searcher: LuceneSearcher,
    dataset_ids: Optional[List[str]] = None,
) -> dict:
    """
    Performs a BM25 search using a Pyserini LuceneSearcher.
    This version uses the querybuilder API consistently for both filtered and unfiltered
    searches to ensure identical scoring logic for the text component.
    """
    start_time = time.time()

    # Create an analyzer to process the query string. This should match the analyzer used to build the index.
    analyzer = Analyzer(get_lucene_analyzer())

    # Define the boolean logic clauses we'll need from the querybuilder.
    should = querybuilder.JBooleanClauseOccur["should"].value
    must = querybuilder.JBooleanClauseOccur["must"].value
    # Define the FILTER clause, which matches documents but does not contribute to the score.
    filter_clause = querybuilder.JBooleanClauseOccur["filter"].value

    # This creates a "bag of words" query against the 'contents' field.
    text_query_terms = analyzer.analyze(query)
    text_query_builder = querybuilder.get_boolean_query_builder()
    for term in text_query_terms:
        term_query = querybuilder.get_term_query(term, field="contents")
        text_query_builder.add(term_query, should)
    text_query = text_query_builder.build()

    # The final query starts as just the text query.
    final_query = text_query

    if dataset_ids:
        logger.info(
            "Applying dataset filters to the search query.", dataset_ids=dataset_ids
        )

        # This outer builder will contain a clause for each dataset_id (OR logic).
        outer_filter_builder = querybuilder.get_boolean_query_builder()
        for did in dataset_ids:
            # For each UUID, we create an inner query that requires all its parts to be present (AND logic).
            inner_must_builder = querybuilder.get_boolean_query_builder()
            uuid_parts = did.split("-")
            for part in uuid_parts:
                term_query = querybuilder.get_term_query(part.lower(), field="source")
                inner_must_builder.add(term_query, must)
            outer_filter_builder.add(inner_must_builder.build(), should)
        filter_query = outer_filter_builder.build()

        final_query_builder = querybuilder.get_boolean_query_builder()
        final_query_builder.add(text_query, must)
        final_query_builder.add(filter_query, filter_clause)
        final_query = final_query_builder.build()

    # Perform the search using the final constructed query object.
    hits = searcher.search(final_query, k=k)

    end_time = time.time()
    query_duration = end_time - start_time
    results_list = []
    for i, hit in enumerate(hits):
        raw_doc = json.loads(hit.lucene_document.get("raw"))
        if "contents" not in raw_doc or "use_case" not in raw_doc:
            logger.warning(
                "Skipping hit due to missing required fields.",
                hit_number=i,
                doc_id=hit.docid,
            )
            continue
        result = SearchResult(
            content=raw_doc.get("contents"),
            use_case=raw_doc.get("use_case"),
            source=raw_doc.get("source"),
            source_id=raw_doc.get("source_id"),
            chunk_id=int(raw_doc.get("chunk_id", 0)),
            language=raw_doc.get("language"),
            distance=hit.score,
        )
        results_list.append(result)

    return {"query_time": query_duration, "results": results_list}


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

            query_template = sql.SQL("""
            SELECT
                content, use_case, source, source_id, chunk_id, language,
                ts_rank_cd(ts_content, plainto_tsquery('english', %s)) AS relevance
            FROM {table}
            WHERE
                ts_content @@ plainto_tsquery('english', %s)
            ORDER BY
                relevance DESC
            LIMIT %s;
            """)

            composed_query = query_template.format(table=sql.Identifier(TABLE_NAME))
            cur.execute(composed_query, (query, query, k))
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
                    distance=row[6],
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
            query_template = sql.SQL("""
                SELECT content, use_case, source, source_id, chunk_id, language, embedding <-> %s::vector AS distance
                FROM {table}
                ORDER BY distance
                LIMIT %s;
            """)
            composed_query = query_template.format(table=sql.Identifier(TABLE_NAME))
            cur.execute(composed_query, (query_embedding, k))
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
                    distance=row[6],
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
            # 1. Check for table existence by querying it
            query = sql.SQL("SELECT 1 FROM {table} LIMIT 1;").format(
                table=sql.Identifier(TABLE_NAME)
            )
            cur.execute(query)

            # 2. Check for column existence
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s;
            """,
                (TABLE_NAME,),
            )

            existing_columns = {row[0] for row in cur.fetchall()}

            missing_columns = required_columns - existing_columns
            if missing_columns:
                raise ValueError(
                    f"Schema validation failed. Missing columns in table '{TABLE_NAME}': {', '.join(missing_columns)}"
                )

    except psycopg2.Error as e:
        # Re-raise database-specific errors to be caught by the health endpoint
        raise ConnectionError(f"Database check failed: {e}") from e
