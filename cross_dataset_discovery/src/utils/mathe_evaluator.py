import pandas as pd
from typing import List, Dict, Optional, Tuple
import json
import wandb
from src.utils.evaluator import EvaluationMetrics
from src.retrieval.base import RetrievalResult

try:
    from IPython.display import display, HTML

    _IPYTHON_DISPLAY_AVAILABLE = True
except ImportError:
    _IPYTHON_DISPLAY_AVAILABLE = False

    def display(x):
        print(str(x))

    def HTML(x):
        print(x)


# --- MatheEvaluator Class ---
class MatheEvaluator(EvaluationMetrics):
    """
    Evaluates retriever performance against a benchmark JSON file for a math use case.
    Calculates overall metrics and optionally logs results to Weights & Biases.
    """

    def __init__(self, n_values: List[int]):
        """
        Initializes the MatheEvaluator.

        Args:
            n_values: A list of integers representing the 'n' values for
                      calculating metrics@n (e.g., [1, 5, 10]).
        """
        super().__init__(n_values)

    def _load_and_prepare_data(
        self, benchmark_json_path: str, retrieved_results: List[List[RetrievalResult]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Loads benchmark data and aligns it with retrieved results.

        Args:
            benchmark_json_path: Path to the JSON benchmark file. Each item in the
                                 JSON list should be a dictionary with a
                                 "source_document" key (List[str]).
            retrieved_results: The list of lists of RetrievalResult from the retriever.

        Returns:
            A tuple containing two lists:
            1. All ground truth ID lists.
            2. All predicted ID lists.
        """
        all_gt_ids: List[List[str]] = []
        all_pred_ids: List[List[str]] = []

        benchmark_data = []
        with open(benchmark_json_path, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)
            if not isinstance(benchmark_data, list):
                raise TypeError(
                    f"Expected a JSON list in {benchmark_json_path}, but got {type(benchmark_data)}"
                )

        if len(benchmark_data) != len(retrieved_results):
            raise ValueError(
                f"Mismatch between number of benchmark queries ({len(benchmark_data)}) "
                f"and number of retrieved result sets ({len(retrieved_results)})."
            )

        for i, benchmark_item in enumerate(benchmark_data):
            gt_ids_instance_raw = benchmark_item.get(
                "source_document"
            )  # Use a temporary variable

            if gt_ids_instance_raw is None:
                print(
                    f"Benchmark item {i+1} (index {i}): 'source_document' field is missing. "
                    f"Assuming empty list of ground truth IDs for this instance."
                )
                gt_ids_instance = []
            elif isinstance(gt_ids_instance_raw, str):
                gt_ids_instance = [gt_ids_instance_raw]
            elif isinstance(gt_ids_instance_raw, list):
                gt_ids_instance = [str(item) for item in gt_ids_instance_raw]
            else:
                print(
                    f"Benchmark item {i+1} (index {i}): 'source_document' is not a list or a string "
                    f"(type: {type(gt_ids_instance_raw)}). Assuming empty list of ground truth IDs for this instance."
                )
                gt_ids_instance = []

            pred_instance_results = retrieved_results[i]
            pred_ids_instance = []
            for result in pred_instance_results:
                pred_id = result.metadata.get("id")
                if pred_id is not None:
                    pred_ids_instance.append(str(pred_id))
                else:
                    print(
                        f"Query {i+1} (index {i}): A retrieved result is missing 'id' in metadata."
                    )

            all_gt_ids.append(gt_ids_instance)
            all_pred_ids.append(pred_ids_instance)

        return all_gt_ids, all_pred_ids

    def evaluate(
        self,
        benchmark_json_path: str,
        retrieved_results: List[List[RetrievalResult]],
        total_seconds: float,
        enable_wandb: bool = False,
        project_wandb: Optional[str] = None,
        entity_wandb: Optional[str] = None,
        group_wandb: Optional[str] = None,
        name_wandb: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Performs the evaluation.

        Args:
            benchmark_json_path: Path to the JSON benchmark file.
            retrieved_results: The list of lists of RetrievalResult from the retriever.
            total_seconds: Total time taken for the retrieval experiment.
            enable_wandb: If True, log results to Weights & Biases.
            project_wandb: W&B project name (required if enable_wandb=True).
            entity_wandb: W&B entity (user or team) name (required if enable_wandb=True).
            group_wandb: W&B group name (optional).
            name_wandb: W&B run name (optional).
            verbose: If True, print results to console.

        Returns:
            A dictionary with a single key "Overall" and its value as a
            Pandas DataFrame containing the evaluation metrics.
        """
        wandb_run = None
        if enable_wandb:
            if not project_wandb or not entity_wandb:
                print("W&B project and entity must be specified to enable W&B logging.")
                enable_wandb = False
            else:
                try:
                    wandb_run = wandb.init(
                        project=project_wandb,
                        entity=entity_wandb,
                        group=group_wandb,
                        name=name_wandb,
                        config={
                            "n_values": self.n_values,
                            "benchmark_file": benchmark_json_path,
                        },
                        job_type="evaluation",
                    )
                except Exception as e:
                    print(f"Failed to initialize W&B: {e}")
                    enable_wandb = False  # Disable W&B logging if init fails

        results_tables: Dict[str, pd.DataFrame] = {}

        try:
            # 1. Load and prepare data
            all_gt_ids, all_pred_ids = self._load_and_prepare_data(
                benchmark_json_path, retrieved_results
            )

            # 2. Calculate overall metrics
            overall_metrics_df = self.calculate_metrics(all_gt_ids, all_pred_ids)
            results_tables["Overall"] = overall_metrics_df

            # 3. Display results
            if verbose:
                print("\n--- Evaluation Results ---")
                print(f"Total Retrieval Time: {total_seconds:.2f} seconds")

                title = "--- Overall Metrics ---"
                if _IPYTHON_DISPLAY_AVAILABLE:
                    display(HTML(f"<h3>{title}</h3>"))
                else:
                    print(f"\n{title}")

                if overall_metrics_df.empty:
                    print("No results calculated.")
                else:
                    # Select float columns for formatting
                    float_cols = overall_metrics_df.select_dtypes(
                        include=["float"]
                    ).columns
                    if _IPYTHON_DISPLAY_AVAILABLE:
                        display(
                            overall_metrics_df.style.format(
                                "{:.4f}", subset=pd.IndexSlice[:, float_cols]
                            )
                        )
                    else:
                        print(overall_metrics_df.to_string(float_format="%.4f"))

            # 4. Log to W&B (if enabled and initialized)
            if enable_wandb and wandb_run:
                wandb_log_data = {}
                if not overall_metrics_df.empty:
                    wandb_log_data["eval/Overall_metrics"] = wandb.Table(
                        dataframe=overall_metrics_df.reset_index()
                    )
                wandb_log_data["eval/Total_Retrieval_Time_seconds"] = total_seconds
                wandb_log_data["eval/Number_of_Queries"] = len(all_gt_ids)

                wandb.log(wandb_log_data)
                print("Results logged to Weights & Biases.")

        except Exception:
            print("An error occurred during evaluation.")
            if enable_wandb and wandb_run:
                wandb.finish(exit_code=1)  # Indicate failure
            raise
        finally:
            if enable_wandb and wandb_run:
                wandb.finish()

        return results_tables
