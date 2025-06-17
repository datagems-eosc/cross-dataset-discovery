import json
import os
import psycopg2
import psycopg2.extras
from typing import List, Any, Optional
from tqdm.auto import tqdm
import torch
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
import numpy as np
import asyncio

# Assuming 'BaseRetriever' and 'RetrievalResult' are defined in this path
# You might need to adjust the import path if your file structure is different.
try:
    from src.retrieval.base import BaseRetriever, RetrievalResult
except ImportError:
    # Define dummy classes if the import fails, so the code is self-contained for review
    class BaseRetriever:
        def index(self, *args, **kwargs):
            pass

        def retrieve(self, *args, **kwargs):
            pass

    class RetrievalResult:
        def __init__(self, score, object, metadata):
            self.score = score
            self.object = object
            self.metadata = metadata


from sentence_transformers import SentenceTransformer
from infinity_emb import AsyncEngineArray, EngineArgs


# --- HELPER FUNCTION FOR BATCH INSERT WITH PROGRESS BAR ---
def execute_batch_with_tqdm(
    cur,
    sql: str,
    argslist: List[Any],
    page_size: int = 100,
    desc: str = "Executing batch",
):
    """
    A tqdm-wrapped version of psycopg2.extras.execute_batch.

    Executes `sql` for each item in `argslist`, but sends them to the
    database in pages of `page_size` to reduce round-trips, showing a
    progress bar for the total number of records.
    """
    total = len(argslist)
    with tqdm(total=total, desc=desc, disable=total < page_size) as pbar:
        for i in range(0, total, page_size):
            page = argslist[i : i + page_size]
            # Use mogrify to create a query for each item in the page
            # This is the same safe, injection-proof method used by the original
            sqls = [cur.mogrify(sql, args) for args in page]

            # Join the individual queries with a semicolon and execute as one block
            cur.execute(b";".join(sqls))

            # Update the progress bar by the number of items processed in this page
            pbar.update(len(page))


class PgVectorDenseRetriever(BaseRetriever):
    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        enable_tqdm: bool = True,
        use_vllm_indexing: bool = False,
        use_infinity_indexing: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.enable_tqdm = enable_tqdm
        self.num_gpus = torch.cuda.device_count()
        self.embedding_dim: Optional[int] = None
        self.model = None
        self.infinity_engine_array: Optional[AsyncEngineArray] = None

        self.selected_backend = "sentence_transformer"
        if use_vllm_indexing:
            self.selected_backend = "vllm"
        elif use_infinity_indexing:
            self.selected_backend = "infinity"

        if self.selected_backend == "vllm":
            from vllm import LLM

            self.model = LLM(
                model=self.model_name_or_path,
                trust_remote_code=True,
                tensor_parallel_size=self.num_gpus or 1,
            )
            self.embedding_dim = self.model.llm_engine.model_config.get_hidden_size()
        elif self.selected_backend == "infinity":
            engine_args = EngineArgs(
                model_name_or_path=self.model_name_or_path, batch_size=2
            )
            self.infinity_engine_array = AsyncEngineArray.from_args([engine_args])
            self.embedding_dim = None  # Will be inferred after first encoding
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(
                self.model_name_or_path, device=device, trust_remote_code=True
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

        load_dotenv()

    def _get_db_connection(self):
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            dbname=os.getenv("DB_NAME"),
        )
        register_vector(conn)
        return conn

    def _setup_database(self, table_name: str, cursor):
        if self.embedding_dim is None:
            raise ValueError("Embedding dimension is not set. Cannot setup database.")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id BIGSERIAL PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            embedding VECTOR({self.embedding_dim})
        );
        """)
        print(
            f"Ensured table '{table_name}' exists. No index will be created for exact search."
        )

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalizes embeddings to unit length (L2 norm)."""
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Add a small epsilon to avoid division by zero
        return embeddings / (norm + 1e-8)

    def _encode(self, sentences: List[str]) -> np.ndarray:
        """Encodes a list of sentences using the selected backend."""
        if not sentences:
            # Return an empty array with the correct shape if possible
            shape = (0, self.embedding_dim) if self.embedding_dim else (0,)
            return np.empty(shape, dtype=np.float32)

        embeddings_np: Optional[np.ndarray] = None

        if self.selected_backend == "vllm":
            outputs = self.model.embed(sentences)
            embeddings_np = np.array(
                [out.outputs.embedding for out in outputs], dtype=np.float32
            )

        elif self.selected_backend == "infinity":
            engine = self.infinity_engine_array[0]

            async def _embed_texts_with_infinity(texts_to_embed):
                await engine.astart()
                all_raw_embeddings_list = []
                batch_size = 16384
                progress_bar = tqdm(
                    range(0, len(texts_to_embed), batch_size),
                    desc="Embedding with Infinity",
                    disable=not self.enable_tqdm or len(texts_to_embed) < batch_size,
                )
                for i in progress_bar:
                    batch = texts_to_embed[i : i + batch_size]
                    batch_embeds, _ = await engine.embed(sentences=batch)
                    all_raw_embeddings_list.extend(batch_embeds)
                await engine.astop()
                return all_raw_embeddings_list

            raw_embeddings_list = asyncio.run(_embed_texts_with_infinity(sentences))
            embeddings_np = np.array(raw_embeddings_list, dtype=np.float32)

            if self.embedding_dim is None and embeddings_np.size > 0:
                self.embedding_dim = embeddings_np.shape[1]

        else:  # sentence_transformer
            pool = self.model.start_multi_process_pool() if self.num_gpus > 1 else None
            embeddings_np = self.model.encode(
                sentences,
                batch_size=64 * max(1, self.num_gpus),
                show_progress_bar=self.enable_tqdm,
                convert_to_numpy=True,
                pool=pool,
            )
            if pool:
                self.model.stop_multi_process_pool(pool)

        return embeddings_np

    def index(
        self,
        input_jsonl_path: str,
        output_folder: str,  # For pgvector, this is used as the table name and a local directory name
        field_to_index: str,
        metadata_fields: List[str],
    ) -> None:
        table_name = output_folder

        # --- PHASE 1: NON-DATABASE WORK ---
        # Read data from file
        texts, metadata_list = [], []
        with open(input_jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(desc="Reading documents", disable=not self.enable_tqdm):
                data = json.loads(line.strip())
                text = data.get(field_to_index)
                if not text or not isinstance(text, str):
                    continue
                texts.append(text)
                entry = {
                    field: data[field] for field in metadata_fields if field in data
                }
                metadata_list.append(json.dumps(entry))

        if not texts:
            print("No valid documents to index.")
            return

        # Perform the long-running encoding process. No DB connection is open.
        embeddings = self._encode(texts)
        if embeddings.size == 0:
            print("Encoding resulted in empty embeddings. Aborting index.")
            return

        normalized_embeddings = self._normalize_embeddings(embeddings)

        ##############################################################
        ### START: ADDED CODE TO SAVE EMBEDDINGS AND METADATA LOCALLY ###
        ##############################################################

        print("\n--- Storing embeddings and metadata to a local file ---")
        try:
            # The 'output_folder' argument is used as a directory for the local file.
            os.makedirs(output_folder, exist_ok=True)
            local_file_path = os.path.join(
                output_folder, "embeddings_with_metadata.jsonl"
            )

            with open(local_file_path, "w", encoding="utf-8") as f:
                # Combine original texts, metadata, and their new embeddings
                data_to_save = zip(texts, metadata_list, normalized_embeddings)

                # Use tqdm for progress indication, consistent with the rest of the class
                iterator = tqdm(
                    data_to_save,
                    total=len(texts),
                    desc="Saving to local file",
                    disable=not self.enable_tqdm,
                )
                for text, metadata_json, embedding in iterator:
                    # The metadata is a JSON string, so we parse it back to a dict.
                    # The embedding is a numpy array, so we convert it to a list for JSON serialization.
                    record = {
                        "content": text,
                        "metadata": json.loads(metadata_json),
                        "embedding": embedding.tolist(),
                    }
                    f.write(json.dumps(record) + "\n")

            print(f"Successfully saved data to '{local_file_path}'")

        except Exception as e:
            print(f"An error occurred while saving to the local file: {e}")
            # The process will continue to the database insertion phase despite this error.

        ############################################################
        ### END: ADDED CODE TO SAVE EMBEDDINGS AND METADATA LOCALLY ###
        ############################################################

        # --- PHASE 2: DATABASE WORK ---
        print(f"\n--- Starting database insertion into table '{table_name}' ---")
        conn = None
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()

            cur.execute(f"SELECT to_regclass('{table_name}');")
            if cur.fetchone()[0]:
                cur.execute(f"SELECT COUNT(*) FROM {table_name};")
                if cur.fetchone()[0] > 0:
                    print(
                        f"Table '{table_name}' exists and is not empty. Skipping indexing."
                    )
                    return

            self._setup_database(table_name, cur)
            conn.commit()

            records_to_insert = list(
                zip(texts, metadata_list, normalized_embeddings.tolist())
            )

            if self.enable_tqdm:
                execute_batch_with_tqdm(
                    cur,
                    f"INSERT INTO {table_name} (content, metadata, embedding) VALUES (%s, %s, %s)",
                    records_to_insert,
                    page_size=500,
                    desc=f"Inserting records into '{table_name}'",
                )
            else:
                psycopg2.extras.execute_batch(
                    cur,
                    f"INSERT INTO {table_name} (content, metadata, embedding) VALUES (%s, %s, %s)",
                    records_to_insert,
                    page_size=500,
                )

            conn.commit()
            cur.close()
        finally:
            if conn:
                conn.close()

        torch.cuda.empty_cache()
        print("\nIndexing complete.")

    def retrieve(
        self, nlqs: List[str], output_folder: str, k: int
    ) -> List[List[RetrievalResult]]:
        table_name = output_folder

        query_embeddings = self._encode(nlqs)
        if query_embeddings.size == 0:
            return [[] for _ in nlqs]

        normalized_embeddings = self._normalize_embeddings(query_embeddings)

        all_batches: List[List[RetrievalResult]] = []
        conn = None
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()

            for query_embedding in tqdm(
                normalized_embeddings,
                desc="Retrieving from DB",
                disable=not self.enable_tqdm,
            ):
                cur.execute(
                    f"""
                    SELECT content, metadata, (embedding <#> %s) * -1 AS score
                    FROM {table_name}
                    ORDER BY embedding <#> %s
                    LIMIT %s
                    """,
                    (query_embedding.tolist(), query_embedding.tolist(), k),
                )
                batch_results = [
                    RetrievalResult(
                        score=float(score), object=content, metadata=metadata
                    )
                    for content, metadata, score in cur.fetchall()
                ]
                all_batches.append(batch_results)

            cur.close()
        finally:
            if conn:
                conn.close()

        torch.cuda.empty_cache()
        return all_batches
