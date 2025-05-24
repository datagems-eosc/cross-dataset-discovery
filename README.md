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

### MathE Dataset

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

### Language Use-Case Dataset

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

