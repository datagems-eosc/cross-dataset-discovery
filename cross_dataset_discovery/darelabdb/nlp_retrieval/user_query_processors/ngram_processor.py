from typing import List

from darelabdb.nlp_retrieval.user_query_processors.query_processor_abc import (
    BaseUserQueryProcessor,
)
from nltk import word_tokenize
from nltk.util import ngrams
from tqdm.auto import tqdm


class NgramQueryProcessor(BaseUserQueryProcessor):
    """
    A query processor that breaks down queries into n-grams.

    This processor generates all possible n-grams (from 1 to n) for each query,
    which can be useful for capturing multi-word entities in a simple,
    non-semantic way.
    """

    def __init__(self, n: int = 4):
        """
        Args:
            n (int): The maximum size of the n-grams to generate.
        """
        self.n = n

    def _extract_ngrams(self, text: str) -> List[str]:
        """Extracts n-grams for a single text string."""
        tokens = word_tokenize(text)
        all_ngrams = []
        for i in range(1, self.n + 1):
            n_grams_for_i = [" ".join(ngram) for ngram in ngrams(tokens, i)]
            all_ngrams.extend(n_grams_for_i)
        return list(set(all_ngrams))

    def process(self, nlqs: List[str]) -> List[List[str]]:
        """
        Processes a batch of queries by generating n-grams for each.
        """
        results = []
        for nlq in tqdm(nlqs, desc="Generating N-grams"):
            results.append(self._extract_ngrams(nlq))
        return results
