import json
from typing import List, Tuple
from darelabdb.nlp_retrieval.benchmarking.benchmarker import Benchmarker
from darelabdb.nlp_retrieval.core.models import RetrievalResult, SearchableItem
from darelabdb.nlp_retrieval.evaluation.evaluator import RetrievalEvaluator
from darelabdb.nlp_retrieval.loaders.jsonl_loader import JsonlLoader
from darelabdb.nlp_retrieval.retrievers.bm25_retriever import PyseriniRetriever
from darelabdb.nlp_retrieval.retrievers.dense_retriever import FaissRetriever
from darelabdb.nlp_retrieval.searcher import Searcher
from darelabdb.nlp_retrieval.retrievers.colbert_retriever import PylateColbertRetriever
from darelabdb.nlp_retrieval.retrievers.ragatouille_colbert_retriever import (
    RagatouilleColbertRetriever,
)


DATA_FILE = "components/darelabdb/nlp_retrieval/test_data/multi_hop_rag.jsonl"
BENCHMARK_FILE = (
    "components/darelabdb/nlp_retrieval/test_data/benchmarks/multi_hop_rag.json"
)
DATA_WITH_IDS_FILE = (
    "components/darelabdb/nlp_retrieval/test_data/temp/data_with_ids.jsonl"
)
OUTPUT_PATH = "components/darelabdb/nlp_retrieval/test_data/multi_hop_rag"
WANDB_PROJECT = "retrieval-benchmark"
WANDB_ENTITY = "darelab"
K_VALUES_TO_EVALUATE = [1, 3, 5]


def prepare_data_for_loader(input_path: str, output_path: str):
    """
    Reads the raw data, creates a stable ID, and writes to a new file
    that the JsonlLoader can use.
    """
    print(f"Preparing data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            data = json.loads(line)
            # Create a stable, unique ID for each item
            item_id = f"{data['page_title']}_{data['source']}"
            data["id"] = item_id
            outfile.write(json.dumps(data) + "\n")


def prepare_benchmark_data(
    benchmark_file_path: str,
) -> Tuple[List[str], List[List[RetrievalResult]]]:
    """
    Loads the benchmark queries and constructs the gold standard list.
    """
    print(f"Loading benchmark queries and gold standard from {benchmark_file_path}...")
    queries = []
    gold_standard = []

    with open(benchmark_file_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    for record in benchmark_data:
        queries.append(record["query"])

        current_gold_list = []
        for doc_id_str in record["document_ids"]:
            if "_sentence_" in doc_id_str:
                parts = doc_id_str.split("_sentence_", 1)
                page_title = parts[0].replace("_", " ")
                source = "sentence_" + parts[1]
            else:
                parts = doc_id_str.split("_table_", 1)
                page_title = parts[0].replace("_", " ")
                source = "table_" + parts[1]

            gold_metadata = {"page_title": page_title, "source": source}

            gold_item = SearchableItem(
                item_id="gold_placeholder",
                content="placeholder",
                metadata=gold_metadata,
            )
            current_gold_list.append(RetrievalResult(item=gold_item, score=1.0))

        gold_standard.append(current_gold_list)

    return queries, gold_standard


def main():
    """
    Main function to configure and run the benchmark.
    """
    prepare_data_for_loader(DATA_FILE, DATA_WITH_IDS_FILE)
    queries, gold_standard = prepare_benchmark_data(BENCHMARK_FILE)
    loader = JsonlLoader(
        file_path=DATA_WITH_IDS_FILE,
        content_field="object",
        item_id_field="id",
        metadata_fields=["page_title", "source"],
    )

    sparse_retriever = PyseriniRetriever()
    bm25_searcher = Searcher(retrievers=[sparse_retriever])

    # Configuration 2: Faiss Dense Retriever
    dense_retriever = FaissRetriever()
    dense_searcher = Searcher(retrievers=[dense_retriever])

    # Configuration 3: ColBERT Retriever
    colbert_retriever = PylateColbertRetriever()
    colbert_searcher = Searcher(retrievers=[colbert_retriever])

    ragatouille_colbert_retriever = RagatouilleColbertRetriever()
    ragatouille_searcher = Searcher(retrievers=[ragatouille_colbert_retriever])

    # Configuration 4: MinHash LSH Retriever
    # minhash_lsh_retriever = MinHashLshRetriever()
    # minhash_lsh_searcher = Searcher(retrievers=[minhash_lsh_retriever])

    # Configuration 5: LightRAG Retriever

    # bge_reranker = BgeReranker()
    # mxbai_reranker = MxbaiCrossEncoderReranker(device_map="cuda:1")
    # llm_reranker = LLMReranker("Qwen/Qwen2.5-32B-Instruct-AWQ",
    #                           2,0.4)
    # bridge_reranker = BridgeReranker()
    # sentence_transformer_reranker = SentenceTransformerCrossEncoderReranker()

    # create one searcher for each reranker using the dense retriever
    # bge_searcher = Searcher(retrievers=[dense_retriever], reranker=bge_reranker)
    # llm_searcher = Searcher(retrievers=[dense_retriever], reranker=llm_reranker)
    # bridge_searcher = Searcher(retrievers=[dense_retriever], reranker=bridge_reranker)
    # sentence_transformer_searcher = Searcher(retrievers=[dense_retriever], reranker=sentence_transformer_reranker)

    # create one searcher for each query processor using the dense retriever
    # llm_decomposition_processor = DecompositionProcessor("gaunernst/gemma-3-27b-it-int4-awq",tensor_parallel_size=1,gpu_memory_utilization=0.80)
    # llm_keyword_extractor_processor = KeywordExtractorProcessor("gaunernst/gemma-3-27b-it-int4-awq",tensor_parallel_size=1,gpu_memory_utilization=0.8)
    # ner_query_processor = NERQueryProcessor()
    # ngram_query_processor = NgramQueryProcessor(3)
    # keybert_processor = KeyBERTProcessor(keyphrase_ngram_range=(1,3),top_n = 7)

    # llm_decomposition_searcher = Searcher(retrievers=[dense_retriever], query_processor=llm_decomposition_processor,reranker=mxbai_reranker)
    # llm_keyword_extractor_searcher = Searcher(retrievers=[dense_retriever], query_processor=llm_keyword_extractor_processor, reranker=mxbai_reranker)
    # ner_searcher = Searcher(retrievers=[dense_retriever], query_processor=ner_query_processor, reranker=mxbai_reranker)
    # ngram_searcher = Searcher(retrievers=[dense_retriever], query_processor=ngram_query_processor, reranker=mxbai_reranker)
    # keybert_searcher = Searcher(retrievers=[dense_retriever], query_processor=keybert_processor, reranker=mxbai_reranker)

    # hybrid_searcher = Searcher(
    #    retrievers=[sparse_retriever, dense_retriever],
    #    reranker=mxbai_reranker
    # )
    # List of all configurations to benchmark
    searcher_configs = [
        ("BM25", bm25_searcher),
        ("Dense", dense_searcher),
        ("ColBERT", colbert_searcher),
        ("RAGatouille ColBERT", ragatouille_searcher),
        # ("MinHash LSH", minhash_lsh_searcher),
        # ("BGE Reranker", bge_searcher),
        # ("MXBAI Reranker", mxbai_searcher),
        # ("LLM Reranker", llm_searcher),
        # ("Bridge Reranker", bridge_searcher),
        # ("Sentence Transformer Reranker", sentence_transformer_searcher),
        # ("LLM Decomposition Processor", llm_decomposition_searcher),
        # ("LLM Keyword Extractor Processor", llm_keyword_extractor_searcher),
        # ("NER Query Processor", ner_searcher),
        # ("Ngram Query Processor", ngram_searcher),
        # ("KeyBERT Query Processor", keybert_searcher),
        # ("Hybrid Retriever", hybrid_searcher),
    ]

    # --- 4. Initialize and Run the Benchmarker ---
    evaluator = RetrievalEvaluator()

    benchmarker = Benchmarker(
        searcher_configs=searcher_configs,
        evaluator=evaluator,
        loader=loader,
        queries=queries,
        gold_standard=gold_standard,
        k_values=K_VALUES_TO_EVALUATE,
        output_path=OUTPUT_PATH,
        use_wandb=True,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
    )

    benchmarker.run()


if __name__ == "__main__":
    main()
