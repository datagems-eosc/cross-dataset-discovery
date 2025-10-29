from typing import List, Union

import torch
from darelabdb.nlp_retrieval.core.models import RetrievalResult
from darelabdb.nlp_retrieval.rerankers.reranker_abc import BaseReranker
from FlagEmbedding import FlagLLMReranker, FlagReranker
from tqdm.auto import tqdm


class BgeReranker(BaseReranker):
    """
    A reranker that uses the BGE (BAAI General Embedding) reranker models.

    This class is a wrapper around the `FlagEmbedding` library, supporting both
    standard cross-encoders (like `bge-reranker-v2-m3`) and LLM-based rerankers
    (like `bge-reranker-v2-gemma`).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        normalize: bool = True,
        device: Union[str, int] = "auto",
    ):
        """
        Initializes the BgeReranker.

        Args:
            model_name: The name of the BGE reranker model from Hugging Face.
            use_fp16: Whether to use float16 precision for faster inference.
                      Defaults to True.
            normalize: Whether to normalize scores to a [0, 1] range using a
                       sigmoid function. Defaults to True.
            device: The device to run the model on (e.g., "auto", "cpu", "cuda:0").
        """
        self.model_name = model_name
        self.normalize = normalize

        # The 'gemma' model requires the LLM-specific reranker class
        if "gemma" in model_name.lower() or "minicpm" in model_name.lower():
            self.reranker = FlagLLMReranker(model_name, use_fp16=use_fp16)
        else:
            self.reranker = FlagReranker(model_name, use_fp16=use_fp16)
        if device != "auto" and hasattr(self.reranker, "model"):
            self.reranker.model.to(device)

    def rerank(
        self, nlqs: List[str], results_batch: List[List[RetrievalResult]], k: int
    ) -> List[List[RetrievalResult]]:
        """
        Reranks candidate results for a batch of queries using the BGE model.
        """
        final_batches = []
        progress_bar_desc = f"Reranking with {self.model_name}"

        for nlq, candidate_list in tqdm(
            zip(nlqs, results_batch),
            total=len(nlqs),
            desc=progress_bar_desc,
            disable=len(nlqs) < 5,
        ):
            if not candidate_list:
                final_batches.append([])
                continue

            # Prepare pairs of [query, document_content] for the reranker
            pairs_to_score = [[nlq, res.item.content] for res in candidate_list]

            try:
                # The reranker computes scores for all pairs in a single batch
                scores = self.reranker.compute_score(
                    pairs_to_score, normalize=self.normalize
                )

                # Create new RetrievalResult objects with the updated scores
                rescored_results = [
                    RetrievalResult(item=orig_res.item, score=float(score))
                    for orig_res, score in zip(candidate_list, scores)
                ]

                # Sort by the new scores and truncate to the top-k
                sorted_results = sorted(
                    rescored_results, key=lambda r: r.score, reverse=True
                )
                final_batches.append(sorted_results[:k])

            except Exception as e:
                print(
                    f"ERROR during BGE reranking for query '{nlq[:50]}...': {e}. "
                    "Falling back to initial results."
                )

                sorted_candidates = sorted(
                    candidate_list, key=lambda r: r.score, reverse=True
                )
                final_batches.append(sorted_candidates[:k])

            torch.cuda.empty_cache()

        return final_batches
