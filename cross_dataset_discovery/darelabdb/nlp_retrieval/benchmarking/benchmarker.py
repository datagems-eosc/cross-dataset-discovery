import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from darelabdb.nlp_retrieval.core.models import RetrievalResult
from darelabdb.nlp_retrieval.evaluation.evaluator import RetrievalEvaluator
from darelabdb.nlp_retrieval.loaders.loader_abc import BaseLoader
from darelabdb.nlp_retrieval.searcher import Searcher

import wandb


class Benchmarker:
    """
    A class to run, evaluate, and log benchmarks for multiple
    retrieval pipeline configurations.
    """

    def __init__(
        self,
        searcher_configs: List[Tuple[str, Searcher]],
        evaluator: RetrievalEvaluator,
        loader: BaseLoader,
        queries: List[str],
        gold_standard: List[List[RetrievalResult]],
        k_values: List[int],
        output_path: str,
        use_wandb: bool = True,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        retrieval_depth_multiplier: int = 3,
    ):
        """
        Initializes the Benchmarker.

        Args:
            searcher_configs: A list of tuples, where each tuple contains a unique
                              run name and an initialized `Searcher` instance.
                              Run names are just used to log with a specific name.
            evaluator: An initialized `RetrievalEvaluator` instance.
            loader: An initialized `BaseLoader` for loading data for indexing.
            queries: A list of natural language queries to run.
            gold_standard: A parallel list of lists of gold standard `RetrievalResult`s.
            k_values: A list of `k` values (e.g., [1, 5, 10]) to evaluate metrics at.
            output_path: The root directory to store all generated indexes.
            use_wandb: If True, logs results to Weights & Biases.
            wandb_project: The W&B project name. Required if `use_wandb` is True.
            wandb_entity: The W&B entity (username or team). Required if `use_wandb` is True.
            retrieval_depth_multiplier: Multiplier for the maximum retrieval depth.
                                        Default is 3, meaning if max_k is 10, it retrieves up to 30 items
                                        ensuring enough results for evaluation after deduplication.
        """
        self.searcher_configs = searcher_configs
        self.evaluator = evaluator
        self.loader = loader
        self.queries = queries
        self.gold_standard = gold_standard
        self.k_values = sorted(
            k_values, reverse=True
        )  # Sort descending to get max_k first
        self.max_k = self.k_values[0] if self.k_values else 10
        self.output_path = output_path

        self.use_wandb = use_wandb
        if use_wandb and (not wandb_project or not wandb_entity):
            raise ValueError(
                "`wandb_project` and `wandb_entity` must be provided when `use_wandb` is True."
            )
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.retrieval_depth_multiplier = retrieval_depth_multiplier
        os.makedirs(self.output_path, exist_ok=True)
        self.summary_data = []

    def _get_config_dict(self, searcher: Searcher) -> Dict[str, Any]:
        """Creates a JSON-serializable dictionary of the Searcher's configuration."""
        config = {
            "searcher_class": searcher.__class__.__name__,
            "query_processor": {
                "class": searcher.query_processor.__class__.__name__,
                "params": searcher.query_processor.__dict__,
            },
            "retrievers": [
                {
                    "class": retriever.__class__.__name__,
                    "params": retriever.__dict__,
                }
                for retriever in searcher.retrievers
            ],
            "reranker": None,
        }
        if searcher.reranker:
            config["reranker"] = {
                "class": searcher.reranker.__class__.__name__,
                "params": searcher.reranker.__dict__,
            }

        # Clean up non-serializable objects from the params
        def cleanup_dict(d):
            if isinstance(d, dict):
                return {
                    k: cleanup_dict(v)
                    for k, v in d.items()
                    if not callable(v) and not k.startswith("_")
                }
            if isinstance(d, list):
                return [cleanup_dict(i) for i in d]
            if hasattr(d, "__dict__"):
                # Fallback for complex objects, just get their class name
                return d.__class__.__name__
            return d

        return cleanup_dict(config)

    def run(self):
        """
        Executes the full benchmarking pipeline for all configured searchers.
        """
        for run_name, searcher in self.searcher_configs:
            print(f"\n{'='*20} Running Benchmark for: {run_name} {'='*20}")

            # --- 1. Indexing ---
            run_output_path = os.path.join(self.output_path, run_name)
            os.makedirs(run_output_path, exist_ok=True)
            searcher.index(self.loader, run_output_path)

            retrieval_depth = self.max_k * self.retrieval_depth_multiplier
            print(
                f"Retrieving up to {retrieval_depth} items to evaluate at k={self.k_values}..."
            )

            # --- 2. Searching ---
            start_time = time.time()
            # Retrieve once with the largest k value for efficiency
            predicted_results = searcher.search(
                self.queries, run_output_path, retrieval_depth
            )
            end_time = time.time()

            search_duration = end_time - start_time
            qps = (
                len(self.queries) / search_duration
                if search_duration > 0
                else float("inf")
            )
            print(f"Search completed in {search_duration:.2f}s ({qps:.2f} QPS)")

            # --- 3. W&B Initialization ---
            if self.use_wandb:
                wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    name=run_name,
                    config=self._get_config_dict(searcher),
                    reinit=True,
                )
                wandb.summary["queries_per_second"] = qps
                wandb.summary["total_search_time_s"] = search_duration

            print(
                f"--- Evaluating on FULL retrieved set (up to k={retrieval_depth}) ---"
            )
            full_summary = self.evaluator.evaluate(
                predicted_results, self.gold_standard
            )
            failures_path = os.path.join(run_output_path, "evaluation_failures.json")
            full_summary.save_failures_to_json(failures_path)
            if os.path.exists(failures_path):
                print(f"Saved failure analysis to {failures_path}")

            print(f"  Full Precision: {full_summary.overall_precision:.4f}")
            print(f"  Full Recall: {full_summary.overall_recall:.4f}")
            print(f"  Full F1 Score: {full_summary.overall_f1_score:.4f}")
            print(f"  Full Perfect Recall Rate: {full_summary.perfect_recall_rate:.4f}")
            print(
                f"  Full Recall (Non-Numerical): {full_summary.overall_recall_non_numerical:.4f}"
            )
            print(
                f"  Full Perfect Recall Rate (Non-Numerical): {full_summary.perfect_recall_rate_non_numerical:.4f}"
            )

            if self.use_wandb:
                wandb.summary["FullResults_Precision"] = full_summary.overall_precision
                wandb.summary["FullResults_Recall"] = full_summary.overall_recall
                wandb.summary["FullResults_F1_Score"] = full_summary.overall_f1_score
                wandb.summary["FullResults_Perfect_Recall_Rate"] = (
                    full_summary.perfect_recall_rate
                )
                wandb.summary["FullResults_Recall_Non_Numerical"] = (
                    full_summary.overall_recall_non_numerical
                )
                wandb.summary["FullResults_Perfect_Recall_Rate_Non_Numerical"] = (
                    full_summary.perfect_recall_rate_non_numerical
                )

            # --- 4. Evaluation at different k values ---
            metrics_table_data = []
            for k in sorted(self.k_values):
                print(f"--- Evaluating @k={k} ---")
                correctly_prepared_predictions_at_k = []
                for i, pred_list_for_query in enumerate(predicted_results):
                    # Get the corresponding gold standard to determine granularity
                    gold_list_for_query = self.gold_standard[i]

                    # Step A: Determine the granularity for this specific query.
                    granularity_fields = (
                        self.evaluator._get_granularity_fields_for_query(
                            gold_list_for_query
                        )
                    )
                    # Step B: Deduplicate the *entire* list of predictions for this query.
                    deduplicated_list = self.evaluator.deduplicate_by_granularity(
                        pred_list_for_query, granularity_fields
                    )
                    # Step C: slice the *deduplicated* list to k.
                    sliced_list = deduplicated_list[:k]
                    correctly_prepared_predictions_at_k.append(sliced_list)
                summary = self.evaluator.evaluate(
                    correctly_prepared_predictions_at_k, self.gold_standard
                )

                failures_path_k = os.path.join(
                    run_output_path, f"evaluation_failures_at_k_{k}.json"
                )
                summary.save_failures_to_json(failures_path_k)

                metrics_at_k = {
                    "@k": k,
                    "Precision": summary.overall_precision,
                    "Recall": summary.overall_recall,
                    "F1 Score": summary.overall_f1_score,
                    "Perfect Recall Rate": summary.perfect_recall_rate,
                    "Recall (Non-Numerical)": summary.overall_recall_non_numerical,
                    "Perfect Recall Rate (Non-Numerical)": summary.perfect_recall_rate_non_numerical,
                }
                metrics_table_data.append(list(metrics_at_k.values()))

                if self.use_wandb:
                    wandb.summary[f"Precision@{k}"] = summary.overall_precision
                    wandb.summary[f"Recall@{k}"] = summary.overall_recall
                    wandb.summary[f"F1_Score@{k}"] = summary.overall_f1_score
                    wandb.summary[f"Perfect_Recall_Rate@{k}"] = (
                        summary.perfect_recall_rate
                    )
                    wandb.summary[f"Recall_Non_Numerical@{k}"] = (
                        summary.overall_recall_non_numerical
                    )
                    wandb.summary[f"Perfect_Recall_Rate_Non_Numerical@{k}"] = (
                        summary.perfect_recall_rate_non_numerical
                    )

            if self.use_wandb:
                columns = [
                    "@k",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Perfect Recall Rate",
                    "Recall (Non-Numerical)",
                    "Perfect Recall Rate (Non-Numerical)",
                ]
                metrics_table = wandb.Table(columns=columns, data=metrics_table_data)
                wandb.log({"performance_metrics_at_k": metrics_table})

                self.summary_data.append(
                    [
                        run_name,
                        qps,
                        metrics_table_data[-1][1],  # Precision at max_k
                        metrics_table_data[-1][2],  # Recall at max_k
                        metrics_table_data[-1][3],  # F1 at max_k
                        metrics_table_data[-1][4],  # PRR at max_k
                        metrics_table_data[-1][5],  # Recall (Non-Num) at max_k
                        metrics_table_data[-1][6],  # PRR (Non-Num) at max_k
                    ]
                )

                wandb.finish()

        # --- 5. Final Aggregated Report ---
        if self.use_wandb and self.summary_data:
            print(f"\n{'='*20} Logging Aggregated Benchmark Summary {'='*20}")
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name="Benchmark_Summary_Report",
                reinit=True,
            )

            summary_columns = [
                "Run Name",
                "QPS",
                f"Precision@{self.max_k}",
                f"Recall@{self.max_k}",
                f"F1_Score@{self.max_k}",
                f"Perfect_Recall_Rate@{self.max_k}",
                f"Recall_Non_Numerical@{self.max_k}",
                f"Perfect_Recall_Rate_Non_Numerical@{self.max_k}",
            ]
            summary_table = wandb.Table(columns=summary_columns, data=self.summary_data)
            wandb.log({"benchmark_summary_table": summary_table})

            summary_df = pd.DataFrame(self.summary_data, columns=summary_columns)
            print("\nBenchmark Summary:")
            print(summary_df.to_markdown(index=False))

            wandb.finish()

        print("\nBenchmarking finished.")
