from typing import List
from mxbai_rerank import MxbaiRerankV2
from cross_dataset_discovery.src.retrievers.base import RetrievalResult
from cross_dataset_discovery.src.retrievers.dense import FaissDenseRetriever
import numpy as np
from tqdm import tqdm
import torch


class DenseRetrieverWithReranker(FaissDenseRetriever):
    """
    A two-stage retriever that enhances dense retrieval with a subsequent reranking step.

    This class extends `FaissDenseRetriever` to first fetch an initial set of candidate
    documents using dense vector search (Stage 1). It then uses a powerful cross-encoder
    reranker model to re-score and re-order these candidates for each query,
    providing more accurate final results (Stage 2).
    """

    def __init__(
        self,
        embedding_model_name: str = "WhereIsAI/UAE-Large-V1",
        reranker_model_name: str = "mixedbread-ai/mxbai-rerank-large-v2",
        k_multiplier: int = 3,
    ):
        """
        Initializes the retriever and the reranker.

        Args:
            embedding_model_name (str): The name or path of the sentence-transformer model
                                        for the initial dense retrieval stage.
            reranker_model_name (str): The name or path of the mixedbread-ai reranker model.
            k_multiplier (int): A factor to determine how many initial candidates to retrieve
                                for reranking. The retriever will fetch `k * k_multiplier`
                                documents before reranking down to `k`.
        """
        super().__init__(model_name_or_path=embedding_model_name)
        # Initialize the reranker model, placing it on a specific GPU if available.
        self.reranker = MxbaiRerankV2(reranker_model_name, device_map="cuda:1")
        self._reranker_model_name = reranker_model_name
        self._k_multiplier = k_multiplier

    def retrieve(
        self, nlqs: List[str], output_folder: str, k: int
    ) -> List[List[RetrievalResult]]:
        # Stage 1: Retrieve an initial, larger set of candidate documents.
        initial_k = k * self._k_multiplier
        initial_batches = super().retrieve(nlqs, output_folder, initial_k)
        final_batches: List[List[RetrievalResult]] = []
        torch.cuda.empty_cache()

        # Stage 2: Rerank the candidates for each query.
        progress_bar_desc = f"Reranking with {self._reranker_model_name}"
        for i, (nlq, initial) in enumerate(
            tqdm(zip(nlqs, initial_batches), desc=progress_bar_desc)
        ):
            if not initial:
                final_batches.append([])
                continue

            valid_initial_results = [
                r for r in initial if isinstance(r.object, str) and r.object
            ]
            if not valid_initial_results:
                final_batches.append([])
                continue

            docs_to_rerank = [r.object for r in valid_initial_results]
            original_results_map = {r.object: r for r in valid_initial_results}

            try:
                reranked = self.reranker.rank(
                    query=nlq,
                    documents=docs_to_rerank,
                    top_k=k,
                    return_documents=True,
                    batch_size=8,
                )

                processed_results: List[RetrievalResult] = []
                if reranked:
                    for item in reranked:
                        if not all(
                            hasattr(item, attr) for attr in ["document", "score"]
                        ):
                            print(
                                f"Warning: Skipping unexpected item format from reranker: {item}"
                            )
                            continue

                        doc_text = item.document
                        score = item.score
                        original_result = original_results_map.get(doc_text)

                        if original_result:
                            final_score = (
                                float(score)
                                if isinstance(score, (int, float))
                                and np.isfinite(score)
                                else original_result.score
                            )
                            processed_results.append(
                                RetrievalResult(
                                    score=final_score,
                                    object=original_result.object,
                                    metadata=original_result.metadata,
                                )
                            )

                else:
                    print(
                        f"Warning: Reranker returned no results for query {i} with top_k={k}. Falling back."
                    )
                    processed_results = valid_initial_results

                processed_results.sort(key=lambda r: r.score, reverse=True)
                final_batches.append(processed_results)

            except Exception as e:
                print(
                    f"ERROR during reranking for query {i} ('{nlq[:50]}...'): {e}. Falling back to initial results."
                )
                valid_initial_results.sort(key=lambda r: r.score, reverse=True)
                final_batches.append(valid_initial_results[:k])

        return final_batches
