import os
from typing import Dict, List, Optional

from nlp_retrieval.core.models import RetrievalResult
from nlp_retrieval.loaders.loader_abc import BaseLoader
from nlp_retrieval.rerankers.reranker_abc import BaseReranker
from nlp_retrieval.retrievers.retriever_abc import BaseRetriever
from nlp_retrieval.user_query_processors.passthrough_processor import (
    PassthroughQueryProcessor,
)
from nlp_retrieval.user_query_processors.query_processor_abc import (
    BaseUserQueryProcessor,
)


class Searcher:
    """
    The main orchestrator for the retrieval pipeline.

    This class wires together the different components (loader, query processor,
    retrievers, reranker) to perform end-to-end indexing and searching. It supports
    hybrid search by aggregating results from multiple retrievers.
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        query_processor: Optional[BaseUserQueryProcessor] = None,
        reranker: Optional[BaseReranker] = None,
        reranker_multiplier: int = 3,
    ):
        """
        Initializes the search pipeline.

        Args:
            retrievers: A list of one or more initialized retriever instances.
            query_processor: An optional, initialized user query processor instance.
                             If None, a `PassthroughQueryProcessor` is used.
            reranker: An optional, initialized reranker instance.
            reranker_multiplier: The multiplier for the number of candidates to retrieve
                                 before reranking. This allows the reranker to work with a
                                 larger set of candidates.
        """
        if not retrievers:
            raise ValueError("At least one retriever must be provided.")
        self.retrievers = retrievers
        self.query_processor = query_processor or PassthroughQueryProcessor()
        self.reranker = reranker
        self.reranker_multiplier = reranker_multiplier

    def index(self, loader: BaseLoader, output_path: str) -> None:
        """
        Loads data using the provided loader and indexes it with all configured retrievers.

        Each retriever's index will be stored in a subdirectory named after its class.

        Args:
            loader: An initialized data loader instance.
            output_path: The root directory to save all index artifacts.
        """
        items = loader.load()
        if not items:
            print("Warning: No items loaded. Indexing will be skipped.")
            return
        for retriever in self.retrievers:
            # Use the retriever's class name for a unique, descriptive folder
            retriever_name = retriever.__class__.__name__
            retriever_path = os.path.join(output_path, retriever_name)
            os.makedirs(retriever_path, exist_ok=True)
            retriever.index(items, retriever_path)

    def search(
        self,
        nlqs: List[str],
        output_path: str,
        k: int,
        **kwargs,
    ) -> List[List[RetrievalResult]]:
        """
        Executes the full search pipeline for a batch of natural language queries.

        The pipeline is as follows:
        1. Process queries using the `UserQueryProcessor`.
        2. Retrieve candidates from all `Retrievers`.
        3. Combine and deduplicate the results from different retrievers.
        4. (Optional) Rerank the combined candidates using the `Reranker`.
        5. Return the final top-k results for each query.

        Args:
            nlqs: A list of natural language queries.
            output_path: The root directory where indexes are stored.
            k: The final number of results to return for each query.

        Returns:
            A list of lists of `RetrievalResult` objects, one list per input query.
            The results might be more than `k` if multiple retrievers return different results
        """
        if not nlqs:
            return []

        # 1. Process all NLQs in a batch
        processed_queries_batch = self.query_processor.process(nlqs)

        # 2. Retrieve from all retrievers and combine results
        # This pool will collect unique results for each query.
        # The dictionary structure handles deduplication by item_id automatically.
        hybrid_results_pool: List[Dict[str, RetrievalResult]] = [{} for _ in nlqs]

        for retriever in self.retrievers:
            retriever_name = retriever.__class__.__name__
            retriever_path = os.path.join(output_path, retriever_name)

            # The retriever handles its own sub-query logic and returns one list per NLQ
            # We retrieve more candidates to create a larger pool for the reranker.
            retriever_results_batch = retriever.retrieve(
                processed_queries_batch,
                retriever_path,
                k=k * self.reranker_multiplier,
                **kwargs,
            )

            # Combine results from this retriever into the hybrid pool
            for i, result_list in enumerate(retriever_results_batch):
                for result in result_list:
                    item_id = result.item.item_id
                    # Add the result if this item hasn't been seen yet for this query
                    if item_id not in hybrid_results_pool[i]:
                        hybrid_results_pool[i][item_id] = result

        # Convert the dictionary of unique results back into a list for each query
        candidate_results_batch: List[List[RetrievalResult]] = [
            list(pool.values()) for pool in hybrid_results_pool
        ]

        # 3. Rerank if a reranker is provided
        if self.reranker:
            # The reranker will handle sorting and truncating to k.
            final_results_batch = self.reranker.rerank(nlqs, candidate_results_batch, k)
        else:
            # If no reranker, sort the combined results by their original scores and truncate.
            final_results_batch = []
            for result_list in candidate_results_batch:
                sorted_results = sorted(
                    result_list, key=lambda r: r.score, reverse=True
                )
                final_results_batch.append(sorted_results[:k])
        return final_results_batch
