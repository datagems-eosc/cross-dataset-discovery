# Adding new feautures to the Retrieval Framework

This document provides guidelines to extend the `nlp_retrieval` framework by adding new components like loaders, query processors, retrievers, or rerankers. Following these guidelines will ensure that new components integrate into the existing pipeline.

## Table of Contents
1.  [Core](#core)
2.  [Directory Structure Overview](#directory-structure-overview)
3.  [Core Data Models](#core-data-models)
4.  [How to Add a New Component (General Steps)](#how-to-add-a-new-component-general-steps)
5.  [Component-Specific Guides](#component-specific-guides)
    *   [Creating a New Loader](#creating-a-new-loader)
    *   [Creating a New User Query Processor](#creating-a-new-user-query-processor)
    *   [Creating a New Retriever](#creating-a-new-retriever)
    *   [Creating a New Reranker](#creating-a-new-reranker)
6.  [Putting It All Together: Assembling a Pipeline](#putting-it-all-together-assembling-a-pipeline)

## Core

The framework is built on a few key principles:

-   **Modularity:** Each component is a self-contained, swappable Python class.
-   **Clear Interfaces:** Components inherit from an Abstract Base Class (ABC) that defines a strict contract for its methods.
-   **Standardized Data Flow:** We use Pydantic models (`SearchableItem`, `RetrievalResult`) to pass data between components.
-   **Batch-First Design:** All components are designed to operate on batches (`List`) of queries or results, enabling batch processing if supported by the component.

## Directory Structure Overview

All retrieval-related code resides in `darelabdb/nlp_retrieval/`.

```
nlp_retrieval/
├── core/                  # Pydantic models
├── loaders/               # Reads data from sources
├── user_query_processors/ # Processes raw user queries
├── retrievers/            # Indexes data and retrieves candidates
├── rerankers/             # Re-scores and ranks candidates
└── searcher.py            # The main orchestrator
```

## Core Data Models

Before you start, familiarize yourself with the two fundamental data models defined in `darelabdb/nlp_retrieval/core/models.py`:

-   `SearchableItem`: Represents a single piece of data to be indexed.
    -   `item_id` (str): A unique identifier for the item.
    -   `content` (str): The main text content used for searching.
    -   `metadata` (Dict): A dictionary for any other associated data.

-   `RetrievalResult`: Represents a single search hit.
    -   `item` (SearchableItem): The retrieved item.
    -   `score` (float): The relevance score assigned by the retriever or reranker.

## How to Add a New Component (General Steps)

1.  **Identify the Component Type:** Determine which category your new class falls into: `loaders`, `user_query_processors`, `retrievers`, or `rerankers`.
2.  **Create the File:** Add a new Python file in the appropriate directory (e.g., `darelabdb/nlp_retrieval/retrievers/my_new_retriever.py`).
3.  **Inherit from the ABC:** Your new class must inherit from the correct Abstract Base Class (e.g., `BaseRetriever` from `retriever_abc.py`).
4.  **Implement All Abstract Methods:** Your IDE or Python itself will require you to implement all methods marked with `@abstractmethod` in the parent ABC. Ensure your method signatures match the ABC exactly.
5.  **Use Core Data Models:** Your component must accept and/or return the Pydantic models (`SearchableItem`, `RetrievalResult`) as defined in the ABC contract.


---

## Component-Specific Guides

### Creating a New Loader

Loaders convert external data into `SearchableItem` objects.

-   **File:** `darelabdb/nlp_retrieval/loaders/my_loader.py`
-   **Inherit from:** `BaseLoader`
-   **Implement:**
    -   `__init__(self, ...)`: Accept any necessary parameters, like file paths or database connection details.
    -   `load(self) -> List[SearchableItem]`: This is the core method. It should contain the logic to read your data source, iterate through it, and create a list of `SearchableItem` objects.


### Creating a New User Query Processor

Query processors transform raw user queries into a format ready for retrieval.

-   **File:** `darelabdb/nlp_retrieval/user_query_processors/my_processor.py`
-   **Inherit from:** `BaseUserQueryProcessor`
-   **Implement:**
    -   `__init__(self, ...)`: Accept any configuration for your processing logic (e.g., model names, thresholds).
    -   `process(self, nlqs: List[str]) -> List[List[str]]`:
        -   This method receives a list of raw query strings.
        -   It must return a list of lists. Each inner list corresponds to an input query and contains the processed strings (e.g., sub-queries, keywords).
        -   The length of the output list must equal the length of the input list (`len(output) == len(nlqs)`).

### Creating a New Retriever

Retrievers are the most complex components. They handle both indexing and searching.

-   **File:** `darelabdb/nlp_retrieval/retrievers/my_retriever.py`
-   **Inherit from:** `BaseRetriever`
-   **Implement:**
    -   `__init__(self, ...)`: Initialize your retrieval model, parameters, etc.
    -   `index(self, items: List[SearchableItem], output_path: str) -> None`:
        -   Receives a list of `SearchableItem` objects.
        -   Build your search index using the `content` of each item.
        -   Store the `item_id` and any necessary metadata to reconstruct a `SearchableItem` at retrieval time.
        -   Save all index files inside the provided `output_path` directory.
    -   `retrieve(self, processed_queries_batch: List[List[str]], output_path: str, k: int) -> List[List[RetrievalResult]]`:
        -   Load the index from `output_path`.
        -   The input `processed_queries_batch` is a list where each element is a list of sub-queries/keywords for one original NLQ.
        -   You must process **all** sub-queries. A common pattern is to flatten them, perform one large batch search, and then regroup the results.
        -   For each original NLQ, you must return a **single, aggregated, and deduplicated** list of `RetrievalResult` objects.
        -  The length of the output list must equal `len(processed_queries_batch)`.

### Creating a New Reranker

Rerankers refine the results from the retrievers.

-   **File:** `darelabdb/nlp_retrieval/rerankers/my_reranker.py`
-   **Inherit from:** `BaseReranker`
-   **Implement:**
    -   `__init__(self, ...)`: Initialize your reranking model (e.g., a cross-encoder).
    -   `rerank(self, nlqs: List[str], results_batch: List[List[RetrievalResult]], k: int) -> List[List[RetrievalResult]]`:
        -   Receives the original queries and a batch of candidate results.
        -   For each query and its corresponding list of candidates, you must compute new scores.
        -   Return a new list of `RetrievalResult` objects, sorted by the new scores in descending order.
        -   The final list for each query should be truncated to the top `k` results.
        -   The length of the output list must equal `len(nlqs)`.

---

## Putting It All Together: Assembling a Pipeline

Once you have created your custom component, you can use it in a `Searcher` pipeline.

```python
from darelabdb.nlp_retrieval.searcher import Searcher
from darelabdb.nlp_retrieval.loaders.jsonl_loader import JsonlLoader
from darelabdb.nlp_retrieval.retrievers.dense_retriever import FaissRetriever
# Import your new custom component
from darelabdb.nlp_retrieval.rerankers.my_reranker import MyReranker

# 1. Initialize the components
my_loader = JsonlLoader(file_path="path/to/data.jsonl", content_field="text")
my_retriever = FaissRetriever(model_name_or_path="BAAI/bge-m3")
my_custom_reranker = MyReranker(model_name="some-model") # Your new reranker

# 2. Assemble the Searcher
searcher = Searcher(
    retrievers=[my_retriever],
    reranker=my_custom_reranker
)

# 3. Run the pipeline
INDEX_DIR = "./my_index"
searcher.index(loader=my_loader, output_path=INDEX_DIR)

# 4. Search
queries = ["what is the capital of france?", "what is the best gpu for gaming?"]
results = searcher.search(nlqs=queries, output_path=INDEX_DIR, k=5)

print(results)
```
