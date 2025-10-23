from abc import ABC, abstractmethod
from typing import List


class BaseUserQueryProcessor(ABC):
    """
    Abstract Base Class for user query processors.

    Take a batch of raw natural language queries and transform them
    into a format suitable for the retriever
    """

    @abstractmethod
    def process(self, nlqs: List[str]) -> List[List[str]]:
        """
        Processes a batch of natural language queries.

        Args:
            nlqs: A list of raw natural language query strings.

        Returns:
            A list of lists, where each inner list contains the processed
            query strings (e.g., one or more sub-queries or extracted keywords)
            for the corresponding input NLQ.
        """
        pass
