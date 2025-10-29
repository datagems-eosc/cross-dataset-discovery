import difflib
from typing import List, Optional, Tuple

from darelabdb.nlp_retrieval.core.models import RetrievalResult
from darelabdb.nlp_retrieval.rerankers.reranker_abc import BaseReranker
from rapidfuzz import fuzz
from tqdm.auto import tqdm


class _Match:
    """A simple helper class to store the start and size of a string match."""

    def __init__(self, start: int, size: int) -> None:
        self.start = start
        self.size = size


class BridgeReranker(BaseReranker):
    """
    A reranker based on the string-matching logic from the BRIDGE system.

    This component re-scores candidate items by performing a complex, rule-based
    fuzzy string matching between the natural language query and the content of
    each candidate. It is not a neural reranker but is effective for cases
    where exact or near-exact string presence is a strong signal of relevance.
    """

    def __init__(
        self,
        m_theta: float = 0.85,
        s_theta: float = 0.85,
        include_substrings: bool = False,
    ):
        """
        Initializes the BridgeReranker.

        Args:
            m_theta: The matching threshold for the primary match.
            s_theta: The matching threshold for the secondary match.
            include_substrings: If True, grants a max score if the query is a
                                substring of the item content (case-insensitive).
        """
        self.include_substrings = include_substrings
        self.m_theta = m_theta
        self.s_theta = s_theta
        self._stopwords = {
            "a",
            "about",
            "above",
            "after",
            "again",
            "against",
            "all",
            "am",
            "an",
            "and",
            "any",
            "are",
            "aren't",
            "as",
            "at",
            "be",
            "because",
            "been",
            "before",
            "being",
            "below",
            "between",
            "both",
            "but",
            "by",
            "can't",
            "cannot",
            "could",
            "couldn't",
            "did",
            "didn't",
            "do",
            "does",
            "doesn't",
            "doing",
            "don't",
            "down",
            "during",
            "each",
            "few",
            "for",
            "from",
            "further",
            "had",
            "hadn't",
            "has",
            "hasn't",
            "have",
            "haven't",
            "having",
            "he",
            "he'd",
            "he'll",
            "he's",
            "her",
            "here",
            "here's",
            "hers",
            "herself",
            "him",
            "himself",
            "his",
            "how",
            "how's",
            "i",
            "i'd",
            "i'll",
            "i'm",
            "i've",
            "if",
            "in",
            "into",
            "is",
            "isn't",
            "it",
            "it's",
            "its",
            "itself",
            "let's",
            "me",
            "more",
            "most",
            "mustn't",
            "my",
            "myself",
            "no",
            "nor",
            "not",
            "of",
            "off",
            "on",
            "once",
            "only",
            "or",
            "other",
            "ought",
            "our",
            "ours",
            "ourselves",
            "out",
            "over",
            "own",
            "same",
            "shan't",
            "she",
            "she'd",
            "she'll",
            "she's",
            "should",
            "shouldn't",
            "so",
            "some",
            "such",
            "than",
            "that",
            "that's",
            "the",
            "their",
            "theirs",
            "them",
            "themselves",
            "then",
            "there",
            "there's",
            "these",
            "they",
            "they'd",
            "they'll",
            "they're",
            "they've",
            "this",
            "those",
            "through",
            "to",
            "too",
            "under",
            "until",
            "up",
            "very",
            "was",
            "wasn't",
            "we",
            "we'd",
            "we'll",
            "we're",
            "we've",
            "were",
            "weren't",
            "what",
            "what's",
            "when",
            "when's",
            "where",
            "where's",
            "which",
            "while",
            "who",
            "who's",
            "whom",
            "why",
            "why's",
            "with",
            "won't",
            "would",
            "wouldn't",
            "you",
            "you'd",
            "you'll",
            "you're",
            "you've",
            "your",
            "yours",
            "yourself",
            "yourselves",
        }
        self._commonwords = {"no", "yes", "many"}
        self._common_db_terms = {"id"}

    def rerank(
        self, nlqs: List[str], results_batch: List[List[RetrievalResult]], k: int
    ) -> List[List[RetrievalResult]]:
        final_batches = []
        for nlq, result_list in tqdm(
            zip(nlqs, results_batch),
            total=len(nlqs),
            desc="Reranking with BridgeReranker",
            disable=len(nlqs) < 5,
        ):
            if not result_list:
                final_batches.append([])
                continue

            rescored_results = []
            for res in result_list:
                item_content = res.item.content
                new_score = 0.0

                if self.include_substrings and nlq.lower() in item_content.lower():
                    new_score = 1.0
                else:
                    matched_entries = self._get_matched_entries(nlq, [item_content])
                    if matched_entries:
                        # Extract the match_score from the first (best) match
                        _, (_, _, match_score, _, _) = matched_entries[0]
                        new_score = match_score

                rescored_results.append(RetrievalResult(item=res.item, score=new_score))

            # Sort by the newly computed scores and truncate
            sorted_results = sorted(
                rescored_results, key=lambda r: r.score, reverse=True
            )
            final_batches.append(sorted_results[:k])

        return final_batches

    def _is_span_separator(self, c: str) -> bool:
        return c in "'\"()`,.?! "

    def _split_to_chars(self, s: str) -> List[str]:
        return [c.lower() for c in s.strip()]

    def _prefix_match(self, s1: str, s2: str) -> bool:
        i, j = 0, 0
        while i < len(s1) and self._is_span_separator(s1[i]):
            i += 1
        while j < len(s2) and self._is_span_separator(s2[j]):
            j += 1

        if i < len(s1) and j < len(s2):
            return s1[i] == s2[j]
        return i >= len(s1) and j >= len(s2)

    def _get_effective_match_source(
        self, s: str, start: int, end: int
    ) -> Optional[_Match]:
        _start = -1
        for i in range(start, start - 2, -1):
            if i < 0 or self._is_span_separator(s[i]):
                _start = i + 1 if i < 0 else i
                break
        if _start < 0:
            return None

        _end = -1
        for i in range(end - 1, end + 3):
            if i >= len(s) or self._is_span_separator(s[i]):
                _end = i - 1 if i >= len(s) else i
                break
        if _end < 0:
            return None

        while _start < len(s) and self._is_span_separator(s[_start]):
            _start += 1
        while _end >= 0 and self._is_span_separator(s[_end]):
            _end -= 1

        return _Match(_start, _end - _start + 1)

    def _get_matched_entries(
        self, s: str, field_values: List[str]
    ) -> Optional[List[Tuple[str, Tuple[str, str, float, float, int]]]]:
        n_grams = self._split_to_chars(s)
        matched = {}

        for field_value in field_values:
            if not isinstance(field_value, str):
                continue

            fv_tokens = self._split_to_chars(field_value)
            sm = difflib.SequenceMatcher(None, n_grams, fv_tokens)
            match = sm.find_longest_match(0, len(n_grams), 0, len(fv_tokens))

            if match.size > 0:
                source_match = self._get_effective_match_source(
                    n_grams, match.a, match.a + match.size
                )
                if not source_match:
                    continue

                match_str = field_value[match.b : match.b + match.size]
                source_match_str = s[
                    source_match.start : source_match.start + source_match.size
                ]

                c_match_str = match_str.lower().strip()
                c_source_match_str = source_match_str.lower().strip()
                c_field_value = field_value.lower().strip()

                if not c_match_str or c_match_str in self._common_db_terms:
                    continue
                if (
                    c_match_str in self._stopwords
                    or c_source_match_str in self._stopwords
                    or c_field_value in self._stopwords
                ):
                    continue

                if c_source_match_str.endswith(c_match_str + "'s"):
                    match_score = 1.0
                elif self._prefix_match(c_field_value, c_source_match_str):
                    match_score = fuzz.ratio(c_field_value, c_source_match_str) / 100.0
                else:
                    match_score = 0.0

                is_common = (
                    c_match_str in self._commonwords
                    or c_source_match_str in self._commonwords
                    or c_field_value in self._commonwords
                )
                if is_common and match_score < 1.0:
                    continue

                s_match_score = match_score
                if match_score >= self.m_theta and s_match_score >= self.s_theta:
                    if field_value.isupper() and (match_score * s_match_score) < 1.0:
                        continue
                    matched[match_str] = (
                        field_value,
                        source_match_str,
                        match_score,
                        s_match_score,
                        match.size,
                    )

        if not matched:
            return None
        return sorted(
            matched.items(),
            key=lambda x: (1e16 * x[1][2] + 1e8 * x[1][3] + x[1][4]),
            reverse=True,
        )
