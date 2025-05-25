import torch
from cross_dataset_discovery.src.retrievers.react import ReActRetriever
from cross_dataset_discovery.src.retrievers.base import RetrievalResult
from typing import List

nlq = "How can I determine if a sequence is arithmetic?"

retriever = ReActRetriever()
print(f"Using retriever: {retriever.__class__.__name__}")
output_folder = "cross_dataset_discovery/assets/mathe/indexes/dense"
results: List[List[RetrievalResult]] = retriever.retrieve([nlq], output_folder, k=3)
print(f"Results from {retriever.__class__.__name__}\n")
for i, result in enumerate(results[0]):
    print(
        f"{i+1}) Document '{result.metadata['id']}', chunk content: '{result.object}'\n"
    )
print("\n")
del retriever
torch.cuda.empty_cache()
