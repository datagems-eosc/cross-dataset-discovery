from abc import ABC, abstractmethod
from typing import List

from darelabdb.nlp_retrieval.core.models import SearchableItem


class BaseLoader(ABC):
    """
    Abstract Base Class for data loaders.

    Loaders are responsible for reading data from a specific source (e.g., a JSONL file,
    a database, a directory of text files) and converting it into a standardized list
    of `SearchableItem` objects, ready for indexing.
    """

    @abstractmethod
    def load(self) -> List[SearchableItem]:
        """
        Loads data from the configured source.

        Returns:
            A list of `SearchableItem` objects.
        """
        pass
