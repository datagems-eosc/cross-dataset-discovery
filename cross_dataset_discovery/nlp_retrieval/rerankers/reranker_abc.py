from abc import ABC, abstractmethod
from typing import List

from nlp_retrieval.core.models import RetrievalResult


class BaseReranker(ABC):
    """
    Abstract Base Class for reranking models.

    Rerankers take an initial list of candidate results from a retriever and
    re-score them
    """

    @abstractmethod
    def rerank(
        self, nlqs: List[str], results_batch: List[List[RetrievalResult]], k: int
    ) -> List[List[RetrievalResult]]:
        """
        Reranks a batch of retrieval results against their original queries.

        Args:
            nlqs: The list of original natural language queries.
            results_batch: A list where each inner list contains the candidate
                           `RetrievalResult` objects for the corresponding NLQ.
            k: The final number of results to return for each query after reranking.

        Returns:
            A list of lists, where each inner list contains the final, reranked
            and sorted `RetrievalResult` objects.
        """
        pass
