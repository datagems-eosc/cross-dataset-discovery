from typing import List

import torch
from darelabdb.nlp_retrieval.core.models import RetrievalResult
from darelabdb.nlp_retrieval.rerankers.reranker_abc import BaseReranker
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm


class SentenceTransformerCrossEncoderReranker(BaseReranker):
    """
    A reranker that uses a Cross-Encoder model from the sentence-transformers library.

    This component re-scores candidate documents by passing the query and document
    text through the model simultaneously
    """

    def __init__(
        self,
        model_name: str = "zeroentropy/zerank-1",
        batch_size: int = 1,
        device: str = "cuda",
        trust_remote_code: bool = False,
    ):
        """
        Initializes the SentenceTransformerCrossEncoderReranker.

        Args:
            model_name: The name of the Cross-Encoder model from Hugging Face.
            batch_size: The batch size for the prediction call.
            device: The device to run the model on (e.g., 'cpu', 'cuda').
                    If None, sentence-transformers will auto-detect.
            trust_remote_code: Whether to trust remote code when loading the model.
                               Set to True for models like 'zerank-1'.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(
            model_name, device=device, trust_remote_code=trust_remote_code
        )
        self.device = device

    def rerank(
        self, nlqs: List[str], results_batch: List[List[RetrievalResult]], k: int
    ) -> List[List[RetrievalResult]]:
        """
        Reranks candidate results for a batch of queries using a Cross-Encoder.
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

            pairs_to_score = [(nlq, res.item.content) for res in candidate_list]

            try:
                # The `predict` method handles batching internally
                scores = self.model.predict(
                    pairs_to_score,
                    batch_size=self.batch_size,
                    show_progress_bar=False,  # tqdm handles the outer loop
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
                    f"ERROR during Cross-Encoder reranking for query '{nlq[:50]}...': {e}. "
                    "Falling back to initial results."
                )

                sorted_candidates = sorted(
                    candidate_list, key=lambda r: r.score, reverse=True
                )
                final_batches.append(sorted_candidates[:k])

            if self.device == "cuda":
                torch.cuda.empty_cache()

        return final_batches
