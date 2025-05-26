import json
import os
from typing import List, Dict, Any, Optional
from cross_dataset_discovery.src.retrievers.base import BaseRetriever, RetrievalResult
from ragatouille import RAGPretrainedModel
from tqdm.auto import tqdm


class ColbertRAGatouilleRetriever(BaseRetriever):
    RAGATOUILLE_INDEX_NAME = "ragatouille_colbert_index"

    def __init__(
        self,
        model_name_or_path: str = "lightonai/Reason-ModernColBERT",
        enable_tqdm: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.rag_model: Optional[RAGPretrainedModel] = None
        self.enable_tqdm = enable_tqdm

    def _get_ragatouille_internal_index_path(self, base_output_folder: str) -> str:
        return os.path.join(
            base_output_folder,
            ".ragatouille",
            "colbert",
            "indexes",
            self.RAGATOUILLE_INDEX_NAME,
        )

    def index(
        self,
        input_jsonl_path: str,
        output_folder: str,
        field_to_index: str,
        metadata_fields: List[str],
    ) -> None:
        os.makedirs(output_folder, exist_ok=True)

        ragatouille_specific_index_dir = self._get_ragatouille_internal_index_path(
            output_folder
        )

        if os.path.exists(ragatouille_specific_index_dir) and os.listdir(
            ragatouille_specific_index_dir
        ):
            print(
                f"RAGatouille index '{self.RAGATOUILLE_INDEX_NAME}' already exists and is not empty in '{output_folder}'. Skipping indexing."
            )
            return

        self.rag_model = RAGPretrainedModel.from_pretrained(
            self.model_name_or_path, index_root=output_folder
        )

        documents_to_index: List[str] = []
        document_metadatas: List[Dict[str, Any]] = []

        file_line_count = 0
        if self.enable_tqdm:
            with open(input_jsonl_path, "r", encoding="utf-8") as f_count:
                file_line_count = sum(1 for _ in f_count)

        with open(input_jsonl_path, "r", encoding="utf-8") as infile:
            iterator = infile
            if self.enable_tqdm and file_line_count > 0:
                iterator = tqdm(
                    infile,
                    total=file_line_count,
                    desc="Reading documents for RAGatouille indexing",
                    unit="docs",
                )

            for line in iterator:
                data = json.loads(line.strip())
                text_content = data.get(field_to_index)

                if not text_content or not isinstance(text_content, str):
                    continue

                documents_to_index.append(text_content)

                meta = {}
                for field_name in metadata_fields:
                    if field_name in data:
                        meta[field_name] = data[field_name]
                document_metadatas.append(meta)

        if not documents_to_index:
            print("No valid documents found to index with RAGatouille.")
            return

        print(
            f"Starting RAGatouille indexing for '{self.RAGATOUILLE_INDEX_NAME}' in root '{output_folder}' with {len(documents_to_index)} documents..."
        )

        actual_index_path = self.rag_model.index(
            collection=documents_to_index,
            index_name=self.RAGATOUILLE_INDEX_NAME,
            document_metadatas=document_metadatas
            if any(m for m in document_metadatas)
            else None,
        )
        print(f"RAGatouille index created successfully at: {actual_index_path}")
        self.rag_model = None

    def retrieve(
        self, nlqs: List[str], output_folder: str, k: int
    ) -> List[List[RetrievalResult]]:
        ragatouille_specific_index_dir = self._get_ragatouille_internal_index_path(
            output_folder
        )

        if not os.path.exists(ragatouille_specific_index_dir) or not os.listdir(
            ragatouille_specific_index_dir
        ):
            print(
                f"RAGatouille index not found or empty at '{ragatouille_specific_index_dir}'. Ensure indexing was run for output_folder '{output_folder}'."
            )
            return [[] for _ in nlqs]

        self.rag_model = RAGPretrainedModel.from_index(ragatouille_specific_index_dir)

        if not nlqs:
            return []

        print(
            f"RAGatouille: Retrieving top-{k} for {len(nlqs)} queries from index '{ragatouille_specific_index_dir}'..."
        )
        rag_search_results = self.rag_model.search(
            query=nlqs,
            k=k,
        )

        all_query_results: List[List[RetrievalResult]] = []

        if (
            len(nlqs) == 1
            and rag_search_results
            and isinstance(rag_search_results[0], dict)
        ):
            rag_search_results = [rag_search_results]

        iterator = range(len(nlqs))
        if self.enable_tqdm and len(nlqs) > 0:
            iterator = tqdm(
                iterator,
                desc="Processing RAGatouille search results",
                unit="query",
                total=len(nlqs),
            )

        for i in iterator:
            if i >= len(rag_search_results):
                all_query_results.append([])
                continue

            current_query_rag_docs = rag_search_results[i]
            current_query_retrieved_items: List[RetrievalResult] = []

            if not isinstance(current_query_rag_docs, list):
                all_query_results.append([])
                continue

            for res_dict in current_query_rag_docs:
                text_content = res_dict.get("content")
                score = res_dict.get("score")
                metadata = res_dict.get("document_metadata", {})

                if not isinstance(text_content, str) or score is None:
                    continue

                if metadata is None:
                    metadata = {}

                current_query_retrieved_items.append(
                    RetrievalResult(
                        score=float(score), object=text_content, metadata=metadata
                    )
                )
            all_query_results.append(current_query_retrieved_items)

        self.rag_model = None
        return all_query_results
