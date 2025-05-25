from typing import List
from chonky import ParagraphSplitter
from chonky.markup_remover import MarkupRemover
from cross_dataset_discovery.src.utils.chunker_base import Chunker


class ChonkyChunker(Chunker):
    """
    A chunker that uses Chonky's MarkupRemover and ParagraphSplitter
    and can process a JSON file to produce a JSONL output.
    """

    def __init__(
        self, model_id: str = "mirth/chonky_modernbert_large_1", device: str = "cuda"
    ):
        """
        Initializes the ChonkyFileChunker.

        Args:
            model_id (str): The model ID for Chonky's ParagraphSplitter.
            device (str): The device to run the model on (e.g., "cpu", "cuda:0").
        """
        self.remover = MarkupRemover()
        self.splitter = ParagraphSplitter(model_id=model_id, device=device)

    def chunk(self, text: str) -> List[str]:
        """
        Processes raw text into a list of chunks using Chonky.
        Removes markup and then splits into paragraphs.

        Args:
            text (str): The raw text to chunk.

        Returns:
            List[str]: A list of text chunks.
        """
        if not text:
            return []
        plain_text = self.remover(text)
        chunks = self.splitter(plain_text)
        return chunks
