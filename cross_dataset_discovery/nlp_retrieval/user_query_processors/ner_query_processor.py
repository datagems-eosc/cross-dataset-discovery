import string
from typing import List

import spacy
from nlp_retrieval.user_query_processors.query_processor_abc import (
    BaseUserQueryProcessor,
)
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from tqdm.auto import tqdm


class NERQueryProcessor(BaseUserQueryProcessor):
    """
    A query processor using Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.

    This processor combines spaCy's NER with NLTK's POS tagging to identify
    potential keywords, filtering for nouns, proper nouns, and adjectives while
    removing common stopwords.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Args:
            spacy_model (str): The name of the spaCy model to use for NER.
        """
        self.nlp = spacy.load(spacy_model)
        self.stop_words = set(stopwords.words("english"))
        self.punctuation = set(string.punctuation)

    def _extract_ner_and_pos(self, text: str) -> List[str]:
        """Extracts entities and POS-filtered keywords for a single text."""
        # NER extraction
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents]

        # POS-based keyword extraction
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        keywords = [
            word
            for word, pos in pos_tags
            if pos
            in ["NN", "NNS", "NNP", "NNPS", "JJ"]  # Nouns, Proper Nouns, Adjectives
            and word.lower() not in self.stop_words
            and word not in self.punctuation
        ]
        return list(set(entities + keywords))

    def process(self, nlqs: List[str]) -> List[List[str]]:
        """
        Processes a batch of queries using NER and POS-tagging.
        """
        results = []
        for nlq in tqdm(nlqs, desc="Extracting NER & POS Keywords"):
            results.append(self._extract_ner_and_pos(nlq))
        return results
