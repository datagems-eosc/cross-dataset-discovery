# Evaluating and Benchmarking Retrieval Pipelines

This document provides a guide to using the evaluation and benchmarking tools within the `nlp_retrieval` framework. These tools allow you to measure the performance of different retrieval configurations.

## Table of Contents
1.  [Core](#core)
2.  [Key Classes](#key-classes)
3.  [The Matching Logic: Granularity and Metadata](#the-matching-logic-granularity-and-metadata)
4.  [How to Run a Benchmark: A Step-by-Step Guide](#how-to-run-a-benchmark-a-step-by-step-guide)
    *   [Step 1: Prepare the Corpus Loader](#step-1-prepare-the-corpus-loader)
    *   [Step 2: Prepare the Queries and Gold Standard Lists](#step-2-prepare-the-queries-and-gold-standard-lists)
    *   [Step 3: Configure Your `Searcher` Pipelines](#step-3-configure-your-searcher-pipelines)
    *   [Step 4: Initialize the `RetrievalEvaluator`](#step-4-initialize-the-retrievalevaluator)
    *   [Step 5: Initialize and Run the `Benchmarker`](#step-5-initialize-and-run-the-benchmarker)
5.  [Interpreting the Output](#interpreting-the-output)
    *   [Console Output](#console-output)
    *   [Weights & Biases (W&B) Dashboard](#weights--biases-wb-dashboard)

## Core

-    Measure standard IR metrics like Precision, Recall, and F1-Score, Perfect Recall.
-    The matching logic is based on metadata, allowing for evaluation at different levels of granularity (e.g., item-level, document-level, table-level).
-   Log detailed configurations, metrics, and summary reports to Weights & Biases (W&B)

## Key Classes

Two main classes drive the evaluation process:

-   **`RetrievalEvaluator`** (`darelabdb/nlp_retrieval/evaluation/evaluator.py`): The core engine for calculating metrics. It takes predicted results and a gold standard and returns a structured summary of performance.
-   **`Benchmarker`** (`darelabdb/nlp_retrieval/benchmarking/benchmarker.py`): The high-level orchestrator. It manages the entire process: running different `Searcher` configurations, calling the `RetrievalEvaluator`, and logging everything.

## The Matching Logic: Granularity and Metadata

This is the most important concept in our evaluation system.

A predicted item is considered a **"correct hit"** if its metadata contains all the key-value pairs from a gold standard item's metadata.

**Example:**
-   **Gold Standard Item Metadata:** `{'page_title': 'Mashable', 'source': 'some_sentence'}`
-   **Prediction 1 Metadata:** `{'page_title': 'Mashable', 'source': 'some_sentence', 'value': 'xyz'}` -> **MATCH** (It's a superset)
-   **Prediction 2 Metadata:** `{'page_title': 'Mashable', 'source': 'another_sentence'}` -> **NO MATCH**

Furthermore, the evaluator **automatically handles granularity**. When you evaluate metrics `@k=5`, the `k` refers to the number of *unique entities* based on the gold standard's metadata keys.

-   If your gold standard metadata is `{'page_title': 'A'}`, the evaluator will first deduplicate the prediction list to keep only the first item found for each unique `page_title`. It then calculates metrics on this deduplicated list.

## How to Run a Benchmark: A Step-by-Step Guide

### Step 1: Prepare the Corpus Loader

The `Benchmarker` does not care about your data's format (e.g., CSV, JSONL, database). It only requires an initialized **loader object** that inherits from `BaseLoader` and whose `.load()` method returns a `List[SearchableItem]`.

You can use one of the provided loaders or create your own.

**Example using `JsonlLoader`:**
```python
from darelabdb.nlp_retrieval.loaders.jsonl_loader import JsonlLoader

# The loader is configured to read your specific file format.
loader = JsonlLoader(
    file_path="path/to/my_corpus.jsonl",
    content_field="object",
    item_id_field="id",
    metadata_fields=["page_title", "source"]
)
```

**Example using `DatabaseLoader`:**
```python
from darelabdb.nlp_retrieval.loaders.database_loader import DatabaseLoader, SerializationStrategy

# Assume 'db_connection' is an active database connection object.
loader = DatabaseLoader(
    db=db_connection,
    strategy=SerializationStrategy.ROW_LEVEL
)
```

### Step 2: Prepare the Queries and Gold Standard Lists

Similarly, the `Benchmarker` does not read a specific benchmark file. Your responsibility is to create two Python lists in memory:

1.  `queries: List[str]`: A list of the natural language query strings.
2.  `gold_standard: List[List[RetrievalResult]]`: A parallel list, where each inner list contains the `RetrievalResult` objects that are the correct answers for the corresponding query.

**Important:** For the `gold_standard`, you only need to populate the `metadata` of each `SearchableItem`. The `item_id` and `content` can be placeholders, as they are not used in the matching logic.

**Example of preparing these lists:**
```python
import json
from darelabdb.nlp_retrieval.core.models import RetrievalResult, SearchableItem

def prepare_benchmark_data_from_file(file_path: str):
    """An example function to load queries and gold standards from a JSON file."""
    queries = []
    gold_standard = []

    with open(file_path, "r") as f:
        benchmark_tasks = json.load(f)

    for task in benchmark_tasks:
        queries.append(task["query"])

        current_gold_list = []
        for doc_id_str in task["document_ids"]:
            # This parsing logic is specific to your benchmark file's format
            page_title, source = parse_my_custom_id(doc_id_str)
            gold_metadata = {"page_title": page_title, "source": source}

            # Create a gold item with only the necessary metadata
            gold_item = SearchableItem(item_id="placeholder", content="placeholder", metadata=gold_metadata)
            current_gold_list.append(RetrievalResult(item=gold_item, score=1.0))

        gold_standard.append(current_gold_list)

    return queries, gold_standard
```

### Step 3: Configure Your `Searcher` Pipelines

For each retrieval configuration you want to test, create an initialized `Searcher` instance.

```python
# Configuration 1: Sparse Retriever
sparse_retriever = PyseriniRetriever()
bm25_searcher = Searcher(retrievers=[sparse_retriever])

# Configuration 2: Dense Retriever with a Reranker
dense_retriever = FaissRetriever()
reranker = MxbaiCrossEncoderReranker()
dense_reranked_searcher = Searcher(retrievers=[dense_retriever], reranker=reranker)

# Create a list of named configurations for the Benchmarker
searcher_configs = [
    ("BM25_Only", bm25_searcher),
    ("Dense_with_Reranker", dense_reranked_searcher),
]
```

### Step 4: Initialize the `RetrievalEvaluator`

This is straightforward, as the evaluator is stateless.

```python
from darelabdb.nlp_retrieval.evaluation.evaluator import RetrievalEvaluator
evaluator = RetrievalEvaluator()
```

### Step 5: Initialize and Run the `Benchmarker`

Pass all the prepared objects and parameters to the `Benchmarker` and call the `.run()` method.

```python
from darelabdb.nlp_retrieval.benchmarking.benchmarker import Benchmarker

# Assume 'loader', 'queries', and 'gold_standard' have been prepared.
benchmarker = Benchmarker(
    searcher_configs=searcher_configs,
    evaluator=evaluator,
    loader=loader,
    queries=queries,
    gold_standard=gold_standard,
    k_values=[1, 5, 10],
    output_path="./benchmark_results",
    use_wandb=True,
    wandb_project="my-retrieval-project",
    wandb_entity="my-username",
)

benchmarker.run()
```

---

## Interpreting the Output

### Console Output

The script will print progress for each stage. For each configuration, you will see:
1.  Indexing status.
2.  Search speed (Queries Per Second).
3.  Metrics for the **full retrieved set** (the best possible recall).
4.  A breakdown of metrics **@k** for each value in `k_values`.
5.  A final summary table comparing all runs at the maximum `k` value.

### Weights & Biases (W&B) Dashboard

If `use_wandb=True`, the `Benchmarker` creates two types of runs:
1.  **Individual Runs:** One for each named configuration (e.g., "BM25_Only").
    -   **Config Tab:** Shows the detailed parameters of all components in the pipeline.
    -   **Summary Tab:** Contains all the final metrics (`Precision@k`, `Recall@k`, QPS, etc.). This is great for a quick overview.
    -   **Tables Tab:** Contains a table named `performance_metrics_at_k` with the P/R/F1 values for each `k`.
2.  **Summary Run:** A single run named "Benchmark_Summary_Report".
    -   **Tables Tab:** Contains a table named `benchmark_summary_table` that aggregates the key performance indicators from all individual runs, making it easy to see which pipeline performed best at a glance.
