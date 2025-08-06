import semchunk
from transformers import AutoTokenizer
from typing import List, Callable
from cross_dataset_discovery.src.utils.chunker_base import Chunker


class SemanticChunker(Chunker):
    """
    A chunker that chunks text semantically using the semchunk library.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", chunk_size: int = 128):
        """
        Initializes the SemanticChunker.

        Args:
            model_name: The Hugging Face model name or path for tokenization.
            chunk_size: The target chunk size in tokens.
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self._chunker: Callable[[str], List[str]] = self._create_chunker()

    def _create_chunker(self) -> Callable[[str], List[str]]:
        """Creates the internal semchunk chunker function."""
        chunker = semchunk.chunkerify(self.model_name, self.chunk_size)
        if chunker:
            return chunker

        # Fallback if direct name resolution fails
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        chunker = semchunk.chunkerify(tokenizer, self.chunk_size)
        return chunker

    def chunk(self, text: str) -> List[str]:
        """
        Chunks the input text into semantic segments.

        Args:
            text: The input string to chunk.

        Returns:
            A list of string chunks. Returns an empty list if input is empty
            or if chunking produces no results.
        """
        if not text or not text.strip():
            return []
        return self._chunker(text)
