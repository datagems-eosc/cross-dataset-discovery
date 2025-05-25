# Cross-Dataset Discovery in DataGEMS

This repository provides the implementation and tools for exploring **cross-dataset discovery** using the DataGEMS framework. It includes everything needed to download and preprocess the **MathE** and **language use-case** datasets, and to prepare them for compatibility with the discovery pipeline.

You'll also find a simple guide for dataset preparation and example scripts demonstrating how to run discovery tasks across datasets.

## 📥 Downloading the Datasets

To get started, run the following scripts to download the datasets:

```
python cross_dataset_discovery/scripts/download_mathe.py
python cross_dataset_discovery/scripts/download_language.py
```

After downloading, the datasets will be located in:

- `cross_dataset_discovery/assets/mathe/`
- `cross_dataset_discovery/assets/language/`

### 🧮 MathE Dataset

The source documents are stored in:

- `cross_dataset_discovery/assets/mathe/materials_pdfs/`

These include `.docx`, `.ipynb`, and `.xlsx` files that have been manually converted to PDFs for consistency.

OCR results are precomputed and available at:

- `cross_dataset_discovery/assets/mathe/data.json`

Each entry in this file contains:
- `id`: relative path to the corresponding PDF
- `contents`: serialized text extracted from the PDF

#### Running OCR (Optional)

For the OCR, [olmOCR](https://github.com/allenai/olmocr) was used. To regenerate OCR output yourself, install `olmOCR` with GPU support:

```
pip install olmocr[gpu] --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
```

Then run:

```
python -m olmocr.pipeline cross_dataset_discovery/assets/mathe/materiald_md/ --markdown --pdfs cross_dataset_discovery/assets/mathe/materials/*.pdf
python cross_dataset_discovery/scripts/mathe_ocr.py
```

### 🌐 Language Use-Case Dataset

All documents are combined into a single unified JSON file:

- `cross_dataset_discovery/assets/language/collection/language_documents.json`

Each entry has the following fields:
- `id`: unique document identifier
- `contents`: full document text
- `language`: language code (e.g., `en`, `fr`, `de`)
- `source`: source name (e.g., `german_encyclopedia`, `britannica`)

## 🧪 Generating Benchmarks

To generate benchmark files for evaluation, run the following scripts:

```
python cross_dataset_discovery/scripts/create_mathe_benchmark.py
python cross_dataset_discovery/scripts/create_language_benchmark.py
```

After running the scripts, the benchmark files will be located at:

- `cross_dataset_discovery/assets/mathe/benchmark.json`
- `cross_dataset_discovery/assets/language/benchmark.json`

### ℹ️ Benchmark Format Notes

- For the **MathE** use case, the benchmark includes document references and associated query information.

- For the **Language** use case, the benchmark additionally includes:
  - The **language of the query**
  - The **language of the ground truth document**

This structure supports multilingual evaluations and cross-lingual document retrieval tasks.

## 🧩 Document Chunking

To enable more effective retrieval, documents can be split into smaller chunks using the `chonky` library. This helps handle long documents and improves granularity during retrieval.

Run the following commands to chunk the documents:

```
python cross_dataset_discovery/scripts/perform_chunking.py \
    cross_dataset_discovery/assets/mathe/collection/mathe_documents.json \
    cross_dataset_discovery/assets/mathe/collection/mathe_documents_chunked.jsonl

python cross_dataset_discovery/scripts/perform_chunking.py \
    cross_dataset_discovery/assets/language/collection/language_documents.json \
    cross_dataset_discovery/assets/language/collection/language_documents_chunked.jsonl
```

The resulting files:

- `cross_dataset_discovery/assets/mathe/collection/mathe_documents_chunked.jsonl`
- `cross_dataset_discovery/assets/language/collection/language_documents_chunked.jsonl`

These are the chunked versions of the original document collections and are the expected inputs for downstream retrieval tasks.

> 📦 Chunking is performed using [chonky](https://github.com/mirth/chonky), a nice framework for semantic chunking.

> ⚠️ **Note on Performance**:  
> For the **Language Use-Case**, the dataset contains a large number of documents, so chunking can be time-consuming. To speed this up, you can use parallel processing with the following script:
>
> ```
> python cross_dataset_discovery/scripts/perform_chunking_parallel.py \
>     cross_dataset_discovery/assets/language/collection/language_documents.json \
>     cross_dataset_discovery/assets/language/collection/language_documents_chunked.jsonl
> ```

## 🎛️  Retriever Interface

The retrievers used for cross-dataset discovery follow a unified interface defined in [`cross_dataset_discovery/src/retrievers/base.py`](cross_dataset_discovery/src/retrievers/base.py). This interface ensures consistent input/output formats across different retrieval backends.

### 🔄 `BaseRetriever` (Abstract Class)

Every retriever must implement the following two methods:

#### `index(...)`

Indexes a collection of documents from a `.jsonl` file.

**Arguments:**
- `input_jsonl_path` *(str)*: Path to a JSON Lines file, where each line is a document (JSON object).
- `output_folder` *(str)*: Directory to store the generated index files.
- `field_to_index` *(str)*: Field from the JSON to index (e.g., `"contents"`).
- `metadata_fields` *(List[str])*: List of fields to store as retrievable metadata.

#### `retrieve(...)`

Performs top-k retrieval over previously indexed data for a list of natural language queries.

**Arguments:**
- `nlqs` *(List[str])*: A list of natural language queries.
- `output_folder` *(str)*: Path to the folder containing the index files. The actual stored files inside the output folder and their manpipulation is perfomed iternally.
- `k` *(int)*: Number of top results to return per query.

**Returns:**  
A `List[List[RetrievalResult]]`, where each sublist contains the top-k results for one query.

---

### 📦 `RetrievalResult` (Data Structure)

Each retrieval result is returned as a `RetrievalResult` object with the following fields:

- `score` *(float)*: Relevance score assigned to the result. Depending on the retrieval, this might have different scale.
- `object` *(str)*: The retrieved text (from the indexed field).
- `metadata` *(Dict[str, Any])*: Dictionary containing additional fields (e.g., document ID, source, language).

This structure allows easy formatting, comparison, and downstream evaluation of retrieval quality.


## 🧠 Core Component: Indexing & Retrieval

To prepare the datasets for retrieval, you need to build the indexes (both sparse and dense). Use the following scripts:

```
python cross_dataset_discovery/scripts/index_mathe.py
python cross_dataset_discovery/scripts/index_language.py
```

These scripts will generate the corresponding index files in:

- `cross_dataset_discovery/assets/mathe/indexes/`
- `cross_dataset_discovery/assets/language/indexes/`

### ▶️ Running Example Retrievals

Once indexing is complete, you can test the retrieval pipeline using:

```
python cross_dataset_discovery/scripts/run_mathe_example.py
python cross_dataset_discovery/scripts/run_language_example.py
```

These scripts demonstrate how to query the indexed datasets and display top retrieval results.

> 💡 **Note:** If you plan to use the **ReAct retriever**, make sure to download the required LLM model first. The default model can be fetched with:

```
wget https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
     -O cross_dataset_discovery/assets/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf
```

This model is in `GGUF` format and can be replaced with any [llama.cpp](https://github.com/ggerganov/llama.cpp)-compatible model of your choice.

Then you can run
```
python cross_dataset_discovery/scripts/run_mathe_react_example.py
python cross_dataset_discovery/scripts/run_language_react_example.py
```

## ⚙️ Technical Components

Below are the core components and libraries used:

### 🔍 Retrieval Backends

- **Dense Retrieval**:  
  Implemented using [**FAISS**](https://github.com/facebookresearch/faiss).

- **Sparse Retrieval (BM25)**:  
  Powered by [**Pyserini**](https://github.com/castorini/pyserini), a Python interface to the Lucene-based Anserini toolkit.

### 🧬 Embedding Generation

- **Sentence Transformers**:  
  Embeddings are generated using models from [**sentence-transformers**](https://huggingface.co/sentence-transformers). You can use any compatible model.  
  The default model used is:  
  [`Snowflake/snowflake-arctic-embed-l-v2.0`](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)

### 🔁 Reranking

- **Cross-Encoder Reranker**:  
  In case of the dense retriever with reranker, the cross encoder used is: 
  [`mixedbread-ai/mxbai-rerank-large-v2`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2)

### 🤖 ReAct Retriever (LLM + Tool Use)

- The **ReAct retriever** is built on LLM reasoning over actions (retrieve, answer).
- It uses the [**Guidance**](https://github.com/guidance-ai/guidance) framework to constrain the language model’s output, enforcing the decoding of the predefined actions.
