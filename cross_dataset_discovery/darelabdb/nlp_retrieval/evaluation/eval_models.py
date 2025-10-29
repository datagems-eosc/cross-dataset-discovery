import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PerQueryMetrics(BaseModel):
    """
    Holds the detailed evaluation metrics for a single query.
    """

    query_index: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    perfect_recall: float = Field(
        description="1.0 if all gold items were found for this query, 0.0 otherwise."
    )
    recall_non_numerical: float = Field(
        description="Recall calculated after filtering out purely numerical gold items."
    )
    perfect_recall_non_numerical: float = Field(
        description="1.0 if all non-numerical gold items were found, 0.0 otherwise."
    )
    retrieved_items: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Details of all de-duplicated items that were retrieved for this query. Only populated if non-numerical recall is not perfect.",
    )
    missed_items: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Details of the gold items that were missed. Only populated if non-numerical recall is not perfect.",
    )


class EvaluationSummary(BaseModel):
    """
    A summary of the evaluation results across all queries.
    """

    num_queries: int
    overall_precision: float = Field(
        description="Precision calculated from the sum of all TPs and FPs (micro-average)."
    )
    overall_recall: float = Field(
        description="Recall calculated from the sum of all TPs and FNs (micro-average)."
    )
    overall_f1_score: float = Field(
        description="F1-score calculated from the overall precision and recall."
    )
    perfect_recall_rate: float = Field(
        description="The percentage of queries for which all gold standard items were found."
    )
    overall_recall_non_numerical: float = Field(
        description="Overall recall calculated after filtering out purely numerical gold items."
    )
    perfect_recall_rate_non_numerical: float = Field(
        description="The percentage of queries for which all non-numerical gold items were found."
    )
    per_query_details: List[PerQueryMetrics] = Field(
        description="A list containing detailed metrics for each individual query."
    )

    def save_failures_to_json(self, output_path: str):
        """
        Saves the details of queries with imperfect non-numerical recall to a JSON file.

        Args:
            output_path: The path to the JSON file where failures will be saved.
        """
        failure_cases = [
            {
                "query_index": query_metrics.query_index,
                "retrieved_items": query_metrics.retrieved_items,
                "missed_items": query_metrics.missed_items,
            }
            for query_metrics in self.per_query_details
            if query_metrics.perfect_recall_non_numerical == 0.0
        ]

        if failure_cases:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"evaluation_failures": failure_cases},
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
