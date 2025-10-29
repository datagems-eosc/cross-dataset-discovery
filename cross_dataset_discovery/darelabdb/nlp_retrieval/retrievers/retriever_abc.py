from abc import ABC, abstractmethod
from typing import List

from darelabdb.nlp_retrieval.core.models import RetrievalResult, SearchableItem


class BaseRetriever(ABC):
    """
    Abstract Base Class for retrieval models.

    Retrievers are responsible for two main tasks:
    1. `index`: Building an efficient search index from a collection of `SearchableItem`s.
    2. `retrieve`: Using the index to find the most relevant items for a given set of queries.
    """

    @abstractmethod
    def index(self, items: List[SearchableItem], output_path: str) -> None:
        """
        Builds and saves an index from a list of items.

        Args:
            items: A list of `SearchableItem` objects to be indexed.
            output_path: The directory path where the index and any related
                         artifacts should be stored.
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        processed_queries_batch: List[List[str]],
        output_path: str,
        k: int,
        **kwargs,
    ) -> List[List[RetrievalResult]]:
        """
        Retrieves relevant items for a batch of processed queries.

        Each element in `processed_queries_batch` is a list of strings representing
        all the queries (e.g., original + sub-queries) for a single original NLQ.
        The implementation should retrieve candidates for all these sub-queries and
        return a single, aggregated list of results for that original NLQ.

        Args:
            processed_queries_batch: A list where each inner list contains the query strings
                                     for one original NLQ.
            output_path: The path to the directory containing the pre-built index.
            k: The number of candidate results to retrieve. Note that this `k` might
               be applied per sub-query, so the implementation should handle
               aggregation and deduplication to produce a final list for each original NLQ.

        Returns:
            A list of lists of `RetrievalResult` objects. Each outer list corresponds
            to one original NLQ, and the inner list contains the aggregated,
            deduplicated results from all its sub-queries.
            The structure must be `len(processed_queries_batch) == len(output)`.
        """
        pass
