import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from cross_dataset_discovery.src.retrievers.bm25 import PyseriniBM25Retriever
from cross_dataset_discovery.src.retrievers.dense import FaissDenseRetriever
from cross_dataset_discovery.src.retrievers.dense_rerank import (
    DenseRetrieverWithReranker,
)
from cross_dataset_discovery.src.retrievers.base import RetrievalResult
from typing import List

ALL_RETRIEVER_CLASSES = [
    PyseriniBM25Retriever,
    FaissDenseRetriever,
    DenseRetrieverWithReranker,
]

nlq = "Where did Johann Nepomuk von Fuchs become professor of mineralogy in 1826?"
for retriever_instance in ALL_RETRIEVER_CLASSES:
    retriever = retriever_instance()
    print(f"Using retriever: {retriever.__class__.__name__}")
    if isinstance(retriever, PyseriniBM25Retriever):
        output_folder = "cross_dataset_discovery/assets/language/indexes/bm25"
    else:
        output_folder = "cross_dataset_discovery/assets/language/indexes/dense"
    results: List[List[RetrievalResult]] = retriever.retrieve([nlq], output_folder, k=3)
    print(f"Results from {retriever.__class__.__name__}\n")
    for i, result in enumerate(results[0]):
        print(
            f"{i+1}) Document '{result.metadata['id']}', chunk content: '{result.object}'\n"
        )
    print("\n")
    del retriever
    torch.cuda.empty_cache()
