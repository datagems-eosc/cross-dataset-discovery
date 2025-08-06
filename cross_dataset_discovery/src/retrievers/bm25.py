from pyserini.search.lucene import LuceneSearcher
import json
import os
import subprocess
import shutil
import tempfile
from typing import List
from cross_dataset_discovery.src.retrieval.base import BaseRetriever, RetrievalResult
from tqdm import tqdm


class PyseriniBM25Retriever(BaseRetriever):
    """
    A sparse retriever implementation using the BM25 algorithm provided by Pyserini.

    This class handles the end-to-end process of:
    1.  Taking a standard JSONL file as input.
    2.  Converting it into the format required by Pyserini's `JsonCollection`.
    3.  Calling the Pyserini command-line tool to build a Lucene index.
    4.  Using the generated index to perform batch retrieval for a list of queries.
    """

    def __init__(self, enable_tqdm: bool = True):
        """
        Initializes the PyseriniBM25Retriever.

        Args:
            enable_tqdm (bool): If True, displays a tqdm progress bar during retrieval.
        """
        super().__init__()
        self.enable_tqdm = enable_tqdm

    def index(
        self,
        input_jsonl_path: str,
        output_folder: str,
        field_to_index: str,
        metadata_fields: List[str],
    ) -> None:
        if os.path.exists(output_folder):
            print(f"Index already exists in '{output_folder}'. Skipping indexing.")
            return

        temp_dir = None
        # Pyserini's JsonCollection requires the input to be a directory of files.
        temp_dir = tempfile.mkdtemp()
        prepared_jsonl_path = os.path.join(temp_dir, "prepared_data.jsonl")
        line_count = 0
        with open(input_jsonl_path, "r", encoding="utf-8") as infile, open(
            prepared_jsonl_path, "w", encoding="utf-8"
        ) as outfile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue

                try:
                    original_data = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"Skipping invalid JSON line {line_count + 1} in {input_jsonl_path}"
                    )
                    continue

                text_content = original_data.get(field_to_index)
                if not text_content or not isinstance(text_content, str):
                    print(
                        f"Skipping line {line_count + 1}: Missing or invalid field '{field_to_index}'"
                    )
                    continue

                pyserini_doc = {"id": f"doc_{line_count}", "contents": text_content}
                # Embed requested metadata fields into the document to be stored.
                for meta_field in metadata_fields:
                    if meta_field in original_data:
                        pyserini_doc[meta_field] = original_data[meta_field]

                outfile.write(json.dumps(pyserini_doc, ensure_ascii=False) + "\n")
                line_count += 1

            if line_count == 0:
                print("No valid documents found to index.")
                return

            os.makedirs(output_folder, exist_ok=True)

            num_threads = os.cpu_count() or 1
            cmd = [
                "python",
                "-m",
                "pyserini.index.lucene",
                "--collection",
                "JsonCollection",
                "--input",
                temp_dir,
                "--index",
                output_folder,
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                str(num_threads),
                "--storePositions",
                "--storeDocvectors",
                "--storeRaw",
            ]

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                shell=False,
            )  # nosec B603

        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def retrieve(
        self, nlqs: List[str], output_folder: str, k: int
    ) -> List[List[RetrievalResult]]:
        if not os.path.exists(output_folder):
            print(f"Index directory not found: {output_folder}")
            return [[] for _ in nlqs]

        searcher = LuceneSearcher(output_folder)
        num_threads = os.cpu_count() or 1

        batch_hits = searcher.batch_search(
            queries=nlqs,
            qids=[f"q{i}" for i in range(len(nlqs))],
            k=k,
            threads=num_threads,
        )

        all_batches: List[List[RetrievalResult]] = []
        for i, nlq in enumerate(
            tqdm(nlqs, desc="Retrieving with BM25", disable=not self.enable_tqdm)
        ):
            qid = f"q{i}"
            hits = batch_hits.get(qid, [])
            results: List[RetrievalResult] = []
            for hit in hits:
                if hasattr(hit, "raw_doc"):
                    raw_doc_str = hit.raw_doc
                elif hasattr(hit, "lucene_document"):
                    raw_doc_str = hit.lucene_document.get("raw")
                else:
                    raw_doc_str = None

                if raw_doc_str:
                    stored_data = json.loads(raw_doc_str)
                    retrieved_object = stored_data.get("contents", "")
                    retrieved_metadata = {
                        key: value
                        for key, value in stored_data.items()
                        if key not in ["id", "contents"]
                    }
                    results.append(
                        RetrievalResult(
                            score=hit.score,
                            object=retrieved_object,
                            metadata=retrieved_metadata,
                        )
                    )

            results.sort(key=lambda r: r.score, reverse=True)
            all_batches.append(results[:k])

        return all_batches
