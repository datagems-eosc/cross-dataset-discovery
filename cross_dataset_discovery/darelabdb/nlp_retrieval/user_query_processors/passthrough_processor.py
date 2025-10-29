from typing import List

from darelabdb.nlp_retrieval.user_query_processors.query_processor_abc import (
    BaseUserQueryProcessor,
)


class PassthroughQueryProcessor(BaseUserQueryProcessor):
    """

    A simple query processor that does not modify the original queries.

    It simply wraps each query in a list, conforming to the required
    `List[List[str]]` output format.
    """

    def process(self, nlqs: List[str]) -> List[List[str]]:
        """
        Wraps each query in a list.
        Args:
            nlqs: A list of raw natural language query strings.

        Returns:
            A list of lists, where each inner list contains only the original query.
            Example: `["q1", "q2"] -> [["q1"], ["q2"]]`
        """
        return [[nlq] for nlq in nlqs]
