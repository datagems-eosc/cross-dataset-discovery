from abc import ABC, abstractmethod
from typing import List


class Chunker(ABC):
    """Abstract base class for text chunking."""

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Processes raw text into a list of chunks."""
        pass
