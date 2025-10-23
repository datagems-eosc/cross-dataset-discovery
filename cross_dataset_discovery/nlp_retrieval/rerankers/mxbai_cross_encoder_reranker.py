from typing import Dict, List

import torch
from nlp_retrieval.core.models import RetrievalResult
from nlp_retrieval.rerankers.reranker_abc import BaseReranker
from mxbai_rerank import MxbaiRerankV2
from tqdm.auto import tqdm


class MxbaiCrossEncoderReranker(BaseReranker):
    """
    A reranker using a cross-encoder model from Mixedbread AI.

    This component uses a `MxbaiRerankV2` model to re-score a list of candidate
    documents against a query
    """

    def __init__(
        self,
        model_name: str = "mixedbread-ai/mxbai-rerank-large-v2",
        batch_size: int = 16,
        device_map: str | int = "cuda:0",
    ):
        """
        Initializes the MxbaiCrossEncoderReranker.

        Args:
            model_name: The name or path of the mixedbread-ai reranker model.
            batch_size: The batch size to use during the reranking process.
            device_map: The device to run the model on
        """
        self.reranker = MxbaiRerankV2(model_name, device_map=device_map)
        self.batch_size = batch_size
        self._model_name = model_name

    def rerank(
        self, nlqs: List[str], results_batch: List[List[RetrievalResult]], k: int
    ) -> List[List[RetrievalResult]]:
        """
        Reranks candidate results for a batch of queries using the Mxbai model.
        """
        final_batches = []
        progress_bar_desc = f"Reranking with {self._model_name}"

        for nlq, candidate_list in tqdm(
            zip(nlqs, results_batch),
            total=len(nlqs),
            desc=progress_bar_desc,
            disable=len(nlqs) < 5,
        ):
            if not candidate_list:
                final_batches.append([])
                continue

            # Prepare documents and a map to retrieve original metadata later
            docs_to_rerank = [res.item.content for res in candidate_list]

            # This map allows us to recover the full SearchableItem after reranking,
            # as the reranker only returns the text content.
            original_results_map: Dict[str, RetrievalResult] = {
                res.item.content: res for res in candidate_list
            }

            try:
                reranked_output = self.reranker.rank(
                    query=nlq,
                    documents=docs_to_rerank,
                    top_k=k,
                    return_documents=True,
                    batch_size=self.batch_size,
                )

                reranked_results: List[RetrievalResult] = []
                for reranked_item in reranked_output:
                    # Look up the original result to preserve its full item and metadata
                    original_result = original_results_map.get(reranked_item.document)
                    if original_result:
                        reranked_results.append(
                            RetrievalResult(
                                item=original_result.item,
                                score=float(reranked_item.score),
                            )
                        )
                final_batches.append(reranked_results)

            except Exception as e:
                # In case of an error, fall back to the original retriever's ranking
                print(
                    f"ERROR during reranking for query '{nlq[:50]}...': {e}. "
                    "Falling back to initial results."
                )

                sorted_candidates = sorted(
                    candidate_list, key=lambda r: r.score, reverse=True
                )
                final_batches.append(sorted_candidates[:k])

            torch.cuda.empty_cache()

        return final_batches
