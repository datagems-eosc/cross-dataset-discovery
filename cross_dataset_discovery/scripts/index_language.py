from cross_dataset_discovery.src.retrievers.bm25 import PyseriniBM25Retriever
from cross_dataset_discovery.src.retrievers.dense import FaissDenseRetriever
import time

if __name__ == "__main__":
    input_json = "cross_dataset_discovery/assets/language/collection/language_documents_chunked.jsonl"

    bm25_instance = PyseriniBM25Retriever()
    bm_25_start_time = time.time()
    bm25_instance.index(
        input_json,
        "cross_dataset_discovery/assets/language/indexes/bm25/",
        "contents",
        ["chunk_id", "id", "language", "source"],
    )
    bm_25_end_time = time.time()
    faiss_instance = FaissDenseRetriever()
    faiss_time_start = time.time()
    faiss_instance.index(
        input_jsonl_path=input_json,
        output_folder="cross_dataset_discovery/assets/language/indexes/dense/",
        field_to_index="contents",
        metadata_fields=["chunk_id", "id", "language", "source"],
    )
    faiss_time_end = time.time()
    print(f"BM25 indexing time: {bm_25_end_time - bm_25_start_time} seconds")
    print(f"Faiss indexing time: {faiss_time_end - faiss_time_start} seconds")
