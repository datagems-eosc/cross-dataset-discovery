from typing import List, Optional

from darelabdb.nlp_retrieval.user_query_processors.query_processor_abc import (
    BaseUserQueryProcessor,
)
from keybert import KeyBERT


class KeyBERTProcessor(BaseUserQueryProcessor):
    """
    A query processor that uses KeyBERT to extract keywords and keyphrases.

    This method leverages sentence-transformer models to find the most relevant
    phrases in a query by comparing phrase embeddings to the full query embedding.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        keyphrase_ngram_range: tuple = (1, 2),
        stop_words: Optional[str] = "english",
        top_n: int = 5,
        use_mmr: bool = True,
        diversity: float = 0.5,
        **kwargs,
    ):
        """
        Initializes the KeyBERT-based processor.

        Args:
            model (str): The sentence-transformer model to use.
            keyphrase_ngram_range (tuple): The length of n-grams to consider for keyphrases.
            stop_words (Optional[str]): The language for stop words, e.g., 'english' or None.
            top_n (int): The number of keywords to extract.
            use_mmr (bool): Whether to use Maximal Marginal Relevance (MMR) to diversify results.
            diversity (float): The diversity factor for MMR (0 for no diversity, 1 for max).
            **kwargs: Additional arguments passed to `kw_model.extract_keywords`.
        """
        self.kw_model = KeyBERT(model)
        self.extract_kwargs = {
            "keyphrase_ngram_range": keyphrase_ngram_range,
            "stop_words": stop_words,
            "top_n": top_n,
            "use_mmr": use_mmr,
            "diversity": diversity,
            **kwargs,
        }

    def process(self, nlqs: List[str]) -> List[List[str]]:
        """
        Extracts keywords from a batch of queries using KeyBERT.
        """
        extracted_keywords_with_scores = self.kw_model.extract_keywords(
            nlqs, **self.extract_kwargs
        )
        if isinstance(extracted_keywords_with_scores[0], list):
            return [
                [keyword for keyword, score in kw_list]
                for kw_list in extracted_keywords_with_scores
            ]
        else:
            return [[keyword for keyword, score in extracted_keywords_with_scores]]
