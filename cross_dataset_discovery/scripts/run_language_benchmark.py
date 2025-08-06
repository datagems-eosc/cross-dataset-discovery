import torch
from cross_dataset_discovery.src.retrievers.bm25 import PyseriniBM25Retriever
from cross_dataset_discovery.src.retrievers.dense import FaissDenseRetriever
from cross_dataset_discovery.src.retrievers.dense_rerank import (
    DenseRetrieverWithReranker,
)
from cross_dataset_discovery.src.retrievers.base import RetrievalResult, BaseRetriever
from cross_dataset_discovery.src.utils.language_evaluator import LanguageEvaluator
import json
import time
from typing import List
# from cross_dataset_discovery.src.retrievers.colbert_pylate import PylateColbertRetriever

K = 50
evaluator = LanguageEvaluator(n_values=[1, 5, 10, 20])
BENCHMARK_FILE_PATH = "cross_dataset_discovery/assets/language/benchmark.json"
WANDB_PROJECT = "datagems"
WANDB_ENTITY = "darelab"

with open(BENCHMARK_FILE_PATH, "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)
all_nlqs = [
    record.get("question") for record in benchmark_data if record.get("question")
]

ALL_RETRIEVER_CLASSES = [
    PyseriniBM25Retriever,
    FaissDenseRetriever,
    DenseRetrieverWithReranker,
    # PylateColbertRetriever,
]


for retriever_instance in ALL_RETRIEVER_CLASSES:
    retriever: BaseRetriever = retriever_instance()
    print(f"Using retriever: {retriever.__class__.__name__}")
    if isinstance(retriever, PyseriniBM25Retriever):
        output_folder = "cross_dataset_discovery/assets/language/indexes/bm25"
    else:
        output_folder = "cross_dataset_discovery/assets/language/indexes/dense_bge_m3"

    start = time.time()
    retrieved_results: List[List[RetrievalResult]] = retriever.retrieve(
        nlqs=all_nlqs, output_folder=output_folder, k=K
    )
    end = time.time()
    wandb_group = "language"
    wandb_name = retriever.__class__.__name__
    if isinstance(retriever, FaissDenseRetriever):
        wandb_name += "bge"
    evaluator.evaluate(
        BENCHMARK_FILE_PATH,
        retrieved_results,
        end - start,
        enable_wandb=True,
        project_wandb=WANDB_PROJECT,
        entity_wandb=WANDB_ENTITY,
        group_wandb=wandb_group,
        name_wandb=wandb_name,
        verbose=False,
    )

    del retriever
    torch.cuda.empty_cache()
