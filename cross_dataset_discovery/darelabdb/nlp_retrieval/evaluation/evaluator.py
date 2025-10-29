import re
from typing import List, Set

import pandas as pd
from darelabdb.nlp_retrieval.core.models import RetrievalResult, SearchableItem
from darelabdb.nlp_retrieval.evaluation.eval_models import (
    EvaluationSummary,
    PerQueryMetrics,
)


class RetrievalEvaluator:
    """
    Calculates and visualizes retrieval evaluation metrics.

    This class compares a list of retrieved results against a gold standard
    list. The core matching logic is based on checking if the metadata of a gold
    item is a subset of the metadata of a retrieved item.

    It automatically handles evaluation granularity by inspecting the gold standard.
    For each query, it determines the granularity based on the metadata keys of the
    gold items and deduplicates the retrieved items accordingly before calculating metrics.
    """

    def _is_match(
        self, retrieved_item: SearchableItem, gold_item: SearchableItem
    ) -> bool:
        """
        Checks if a retrieved item is a match for a gold standard item.

        A match occurs if all key-value pairs in the gold item's metadata
        are present in the retrieved item's metadata. The comparison is
        case-insensitive for both keys and string values.
        """
        retrieved_meta_lower = {
            k.lower(): (v.lower() if isinstance(v, str) else v)
            for k, v in retrieved_item.metadata.items()
        }
        gold_meta_lower = {
            k.lower(): (v.lower() if isinstance(v, str) else v)
            for k, v in gold_item.metadata.items()
        }
        return gold_meta_lower.items() <= retrieved_meta_lower.items()

    def deduplicate_by_granularity(
        self, results: List[RetrievalResult], granularity_fields: List[str]
    ) -> List[RetrievalResult]:
        """
        Deduplicates a list of results based on a set of metadata fields.

        It preserves the original order and keeps the first occurrence of each
        unique entity defined by the granularity fields. The comparison is
        case-insensitive for both the metadata keys and their string values.
        """
        if not granularity_fields:
            return results

        seen_keys: Set[tuple] = set()
        deduplicated_results: List[RetrievalResult] = []
        for res in results:
            meta_lower_keys = {k.lower(): v for k, v in res.item.metadata.items()}
            key = tuple(
                (lambda v: v.lower() if isinstance(v, str) else v)(
                    meta_lower_keys.get(field.lower())
                )
                for field in granularity_fields
            )
            if key not in seen_keys:
                seen_keys.add(key)
                deduplicated_results.append(res)
        return deduplicated_results

    def _get_granularity_fields_for_query(
        self, gold_list: List[RetrievalResult]
    ) -> List[str]:
        """
        Determines evaluation granularity from the gold standard for a single query.
        """
        if not gold_list:
            return []
        return list(gold_list[0].item.metadata.keys())

    def _contains_alpha(self, text: str) -> bool:
        """Checks if a string contains any alphabetic characters."""
        return bool(re.search("[a-zA-Z]", text))

    def evaluate(
        self,
        retrieved_results: List[List[RetrievalResult]],
        gold_standard: List[List[RetrievalResult]],
    ) -> EvaluationSummary:
        """
        Calculates comprehensive evaluation metrics for a batch of queries.
        """
        if len(retrieved_results) != len(gold_standard):
            raise ValueError(
                f"Mismatch in number of queries. Retrieved results have {len(retrieved_results)}, "
                f"but gold standard has {len(gold_standard)}."
            )

        total_tp, total_fp, total_fn = 0, 0, 0
        total_perfect_recalls = 0.0
        total_tp_non_numerical, total_fn_non_numerical = 0, 0
        total_perfect_recalls_non_numerical = 0.0
        all_query_metrics: List[PerQueryMetrics] = []

        for i, (retrieved_list, gold_list) in enumerate(
            zip(retrieved_results, gold_standard)
        ):
            # --- Automatic Granularity Handling ---
            granularity_fields = self._get_granularity_fields_for_query(gold_list)
            effective_retrieved_list = self.deduplicate_by_granularity(
                retrieved_list, granularity_fields
            )

            # --- Per-Query Calculation (Standard) ---
            matched_gold_indices: Set[int] = set()
            matched_retrieved_indices: Set[int] = set()

            # Find all matches between the (deduplicated) retrieved list and the gold list
            for retrieved_idx, retrieved_res in enumerate(effective_retrieved_list):
                for gold_idx, gold_res in enumerate(gold_list):
                    if gold_idx in matched_gold_indices:
                        continue

                    if self._is_match(retrieved_res.item, gold_res.item):
                        matched_retrieved_indices.add(retrieved_idx)
                        matched_gold_indices.add(gold_idx)
                        break

            # Calculate TP, FP, FN using the length of the effective (deduplicated) list
            tp = len(matched_retrieved_indices)
            fp = len(effective_retrieved_list) - tp
            fn = len(gold_list) - len(matched_gold_indices)

            # Calculate metrics for this query
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            perfect_recall = 1.0 if fn == 0 else 0.0

            # --- Per-Query Calculation (Non-Numerical) ---
            gold_list_non_numerical = [
                (idx, gold_res)
                for idx, gold_res in enumerate(gold_list)
                if self._contains_alpha(str(gold_res.item.metadata.get("value", "")))
            ]

            matched_gold_indices_non_numerical: Set[int] = set()
            if gold_list_non_numerical:
                for _, retrieved_res in enumerate(effective_retrieved_list):
                    for gold_idx, gold_res in gold_list_non_numerical:
                        if gold_idx in matched_gold_indices_non_numerical:
                            continue
                        if self._is_match(retrieved_res.item, gold_res.item):
                            matched_gold_indices_non_numerical.add(gold_idx)
                            break

            tp_non_numerical = len(matched_gold_indices_non_numerical)
            fn_non_numerical = len(gold_list_non_numerical) - tp_non_numerical

            recall_non_numerical = (
                tp_non_numerical / len(gold_list_non_numerical)
                if gold_list_non_numerical
                else 1.0
            )
            perfect_recall_non_numerical = 1.0 if fn_non_numerical == 0 else 0.0

            # --- Logging Failure Cases ---
            retrieved_items_for_log = None
            missed_items_for_log = None
            if perfect_recall_non_numerical == 0.0:
                missed_items_for_log = [
                    gold_list[i].item.model_dump()
                    for i in range(len(gold_list))
                    if i not in matched_gold_indices
                ]
                retrieved_items_for_log = [
                    res.model_dump() for res in effective_retrieved_list
                ]

            all_query_metrics.append(
                PerQueryMetrics(
                    query_index=i,
                    true_positives=tp,
                    false_positives=fp,
                    false_negatives=fn,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    perfect_recall=perfect_recall,
                    recall_non_numerical=recall_non_numerical,
                    perfect_recall_non_numerical=perfect_recall_non_numerical,
                    retrieved_items=retrieved_items_for_log,
                    missed_items=missed_items_for_log,
                )
            )

            # Update total counts
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_perfect_recalls += perfect_recall

            # Update new total counts
            total_tp_non_numerical += tp_non_numerical
            total_fn_non_numerical += fn_non_numerical
            total_perfect_recalls_non_numerical += perfect_recall_non_numerical

        # --- Overall Aggregation ---
        num_queries = len(retrieved_results)
        overall_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        overall_recall = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        )
        overall_f1 = (
            2
            * (overall_precision * overall_recall)
            / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0
            else 0.0
        )
        perfect_recall_rate = (
            total_perfect_recalls / num_queries if num_queries > 0 else 0.0
        )

        overall_recall_non_numerical = (
            total_tp_non_numerical / (total_tp_non_numerical + total_fn_non_numerical)
            if (total_tp_non_numerical + total_fn_non_numerical) > 0
            else 0.0
        )
        perfect_recall_rate_non_numerical = (
            total_perfect_recalls_non_numerical / num_queries
            if num_queries > 0
            else 0.0
        )

        return EvaluationSummary(
            num_queries=num_queries,
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            overall_f1_score=overall_f1,
            perfect_recall_rate=perfect_recall_rate,
            overall_recall_non_numerical=overall_recall_non_numerical,
            perfect_recall_rate_non_numerical=perfect_recall_rate_non_numerical,
            per_query_details=all_query_metrics,
        )

    def to_dataframe(self, summary: EvaluationSummary) -> pd.DataFrame:
        """
        Converts the EvaluationSummary object into a pandas DataFrame for easy viewing.
        """
        records = [
            query_metrics.model_dump() for query_metrics in summary.per_query_details
        ]
        df = pd.DataFrame.from_records(records)
        df = df.set_index("query_index")
        return df
