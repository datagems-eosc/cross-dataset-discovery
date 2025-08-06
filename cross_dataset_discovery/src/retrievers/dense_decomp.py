from typing import List, Optional, Dict
from cross_dataset_discovery.src.retrieval.base import RetrievalResult
from cross_dataset_discovery.src.retrieval.dense import FaissDenseRetriever
from cross_dataset_discovery.src.utils.query_decompostion import QueryDecomposer
from cross_dataset_discovery.src.utils.query_decomposition_vllm import (
    QueryDecomposer as VLLMQueryDecomposer,
)
from tqdm import tqdm


class DenseRetrieverWithDecomposition(FaissDenseRetriever):
    """
    A retriever that enhances dense search by first decomposing complex queries.

    This class extends `FaissDenseRetriever` to implement a multi-step retrieval process:
    1.  A large language model decomposes each complex query into several simpler,
        self-contained sub-queries.
    2.  It performs dense retrieval for the original query and all its sub-queries in a single batch.
    3.  The multiple lists of results are merged and re-ranked using Reciprocal Rank Fusion (RRF)
        to produce a final, more robust ranking.
    """

    def __init__(
        self,
        embedding_model_name: str = "WhereIsAI/UAE-Large-V1",
        model_name: str = "gaunernst/gemma-3-27b-it-int4-awq",
        decomposition_cache_folder: Optional[str] = None,
        use_vllm: bool = True,
    ):
        """
        Initializes the retriever and the query decomposer model.

        Args:
            embedding_model_name (str): The name or path of the sentence-transformer model
                                        for the dense retrieval stage.
            model_name (str): The name or path of the large language model used for
                              query decomposition.
            decomposition_cache_folder (Optional[str]): A path to a folder for caching
                                                        decomposition results to speed up
                                                        repeated runs.
            use_vllm (bool): If True, uses the VLLM-based query decomposer for faster
                             inference.
        """
        super().__init__(model_name_or_path=embedding_model_name)
        self.use_vllm = use_vllm
        if not self.use_vllm:
            self.decomposer = QueryDecomposer(
                model_name, output_folder=decomposition_cache_folder
            )
        else:
            self.decomposer = VLLMQueryDecomposer(
                model_name_or_path=model_name,
                output_folder=decomposition_cache_folder,
                gpu_memory_utilization=0.65,
            )

    def retrieve(
        self, nlqs: List[str], output_folder: str, k: int
    ) -> List[List[RetrievalResult]]:
        if not nlqs:
            return []
        # 1. Decompose queries and collect all unique queries (original + sub-queries) to retrieve.
        all_queries_to_retrieve: List[str] = []
        # This list tracks which original query each sub-query belongs to.
        original_query_indices: List[int] = []

        if self.use_vllm:
            decomposed_nlqs_batch: List[List[str]] = self.decomposer.decompose_batch(
                nlqs
            )
            for i, sub_queries in enumerate(decomposed_nlqs_batch):
                # Include the original query itself in the set for retrieval.
                queries_for_this_nlq = [nlqs[i]] + sub_queries
                for q in queries_for_this_nlq:
                    all_queries_to_retrieve.append(q)
                    original_query_indices.append(i)
        else:
            decomposer_name = getattr(self.decomposer, "ollama_model", "decomposer")
            for i, nlq in enumerate(
                tqdm(nlqs, desc=f"Decomposing with {decomposer_name}")
            ):
                sub_queries = self.decomposer.decompose(nlq) or []
                queries_for_this_nlq = [nlq] + sub_queries
                for q in queries_for_this_nlq:
                    all_queries_to_retrieve.append(q)
                    original_query_indices.append(i)

        if not all_queries_to_retrieve:
            return [[] for _ in nlqs]

        # 2. Retrieve for all queries in a single batch call to the parent retriever.
        all_retrieved_results_nested = super().retrieve(
            all_queries_to_retrieve, output_folder, k
        )

        # 3. Fuse results using Reciprocal Rank Fusion (RRF).
        rrf_k = 60  # RRF constant, balances influence of high vs. low ranks.
        grouped_results: Dict[int, Dict[str, Dict]] = {i: {} for i in range(len(nlqs))}

        for query_idx, single_query_results in enumerate(all_retrieved_results_nested):
            original_idx = original_query_indices[query_idx]

            for rank, r in enumerate(single_query_results, 1):
                # Use the document text as a unique key for aggregation.
                object_key = r.object
                rrf_score = 1 / (rrf_k + rank)

                if object_key not in grouped_results[original_idx]:
                    grouped_results[original_idx][object_key] = {
                        "score": rrf_score,
                        "result_obj": r,
                    }
                else:
                    # If document was found by multiple sub-queries, accumulate its RRF score.
                    grouped_results[original_idx][object_key]["score"] += rrf_score

        # 4. Sort the fused results by their final RRF score and select the top-k.
        final_batches: List[List[RetrievalResult]] = []
        for i in range(len(nlqs)):
            results_with_scores = list(grouped_results[i].values())

            sorted_by_score = sorted(
                results_with_scores, key=lambda item: item["score"], reverse=True
            )

            top_k_results = [item["result_obj"] for item in sorted_by_score[:k]]
            final_batches.append(top_k_results)

        return final_batches
