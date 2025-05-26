import json
import os
import pickle
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm
import torch
from pylate.utils import iter_batch
from pylate import models as pylate_models
from pylate import indexes as pylate_indexes
from pylate import retrieve as pylate_retrieve

from cross_dataset_discovery.src.retrievers.base import BaseRetriever, RetrievalResult


class PylateColbertRetriever(BaseRetriever):
    PLAID_INDEX_DIR_NAME = "plaid_colbert_index"
    DOC_STORE_FILENAME = "doc_store.pkl"

    def __init__(
        self,
        model_name_or_path: str = "lightonai/GTE-ModernColBERT-v1",
        plaid_nbits: int = 2,
        plaid_kmeans_niters: int = 4,
        plaid_index_bsize: int = 128,
        plaid_ndocs: int = 8192,
        plaid_centroid_score_threshold: float = 0.35,
        plaid_ncells: int = 8,
        plaid_search_batch_size: int = 2**18,
        encode_batch_size: int = 32,
        enable_tqdm: bool = True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pylate_model = pylate_models.ColBERT(
            model_name_or_path=model_name_or_path, device=self.device
        )
        self.embedding_size = self.pylate_model.get_sentence_embedding_dimension()

        self.plaid_config_params = {
            "embedding_size": self.embedding_size,
            "nbits": plaid_nbits,
            "kmeans_niters": plaid_kmeans_niters,
            "index_bsize": plaid_index_bsize,
            "ndocs": plaid_ndocs,
            "centroid_score_threshold": plaid_centroid_score_threshold,
            "ncells": plaid_ncells,
            "search_batch_size": plaid_search_batch_size,
        }
        self.encode_batch_size = encode_batch_size
        self.enable_tqdm = enable_tqdm

        self.plaid_index_instance: Optional[pylate_indexes.PLAID] = None
        self.pylate_retriever_instance: Optional[pylate_retrieve.ColBERT] = None
        self.doc_store: Optional[Dict[str, Dict[str, Any]]] = None
        self.num_gpus = torch.cuda.device_count()

    def _get_plaid_index_parent_folder(self, output_folder: str) -> str:
        return output_folder

    def _get_plaid_index_name(self) -> str:
        return self.PLAID_INDEX_DIR_NAME

    def _get_plaid_full_index_path(self, output_folder: str) -> str:
        return os.path.join(
            self._get_plaid_index_parent_folder(output_folder),
            self._get_plaid_index_name(),
        )

    def _get_doc_store_path(self, output_folder: str) -> str:
        return os.path.join(output_folder, self.DOC_STORE_FILENAME)

    def _initialize_plaid_components(self, output_folder: str, override: bool = False):
        plaid_parent_folder = self._get_plaid_index_parent_folder(output_folder)
        plaid_index_name = self._get_plaid_index_name()

        self.plaid_index_instance = pylate_indexes.PLAID(
            index_folder=plaid_parent_folder,
            index_name=plaid_index_name,
            override=override,
            **self.plaid_config_params,
        )
        self.pylate_retriever_instance = pylate_retrieve.ColBERT(
            index=self.plaid_index_instance
        )

    def index(
        self,
        input_jsonl_path: str,
        output_folder: str,
        field_to_index: str,
        metadata_fields: List[str],
    ) -> None:
        os.makedirs(output_folder, exist_ok=True)
        doc_store_path = self._get_doc_store_path(output_folder)
        plaid_full_index_path = self._get_plaid_full_index_path(output_folder)

        plaid_metadata_file = os.path.join(plaid_full_index_path, "metadata.json")

        should_reindex = True
        if os.path.exists(doc_store_path) and os.path.exists(plaid_metadata_file):
            print(f"Index components found in '{output_folder}'. Skipping indexing.")
            should_reindex = False
            self._initialize_plaid_components(output_folder, override=False)
            with open(doc_store_path, "rb") as f:
                self.doc_store = pickle.load(f)
        else:
            if os.path.exists(doc_store_path):
                print(
                    f"Doc store exists at '{doc_store_path}' but PLAID index metadata not found at '{plaid_metadata_file}'. Re-indexing."
                )
                os.remove(doc_store_path)
            self._initialize_plaid_components(output_folder, override=True)

        if not should_reindex:
            return

        texts_to_encode = []
        doc_store_build: Dict[str, Dict[str, Any]] = {}

        total_lines = 0
        with open(input_jsonl_path, "r", encoding="utf-8") as infile:
            for _ in infile:
                total_lines += 1

        doc_idx_counter = 0
        with open(input_jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(
                f,
                total=total_lines,
                desc="Reading documents for indexing",
                disable=not self.enable_tqdm,
            ):
                data = json.loads(line.strip())
                text_content = data.get(field_to_index)

                if not text_content or not isinstance(text_content, str):
                    continue

                texts_to_encode.append(text_content)

                current_metadata: Dict[str, Any] = {}
                for meta_field in metadata_fields:
                    if meta_field in data:
                        current_metadata[meta_field] = data[meta_field]

                plaid_doc_id = f"doc_{doc_idx_counter}"
                doc_store_build[plaid_doc_id] = {
                    "text": text_content,
                    "metadata": current_metadata,
                }
                doc_idx_counter += 1

        if not texts_to_encode:
            print("No valid documents to index.")
            self.doc_store = {}
            with open(doc_store_path, "wb") as f:
                pickle.dump(self.doc_store, f)
            return

        print(f"Encoding {len(texts_to_encode)} documents using multi-process...")

        target_devices = (
            [f"cuda:{i}" for i in range(self.num_gpus)]
            if self.num_gpus > 0
            else ["cpu"]
        )
        pool = self.pylate_model.start_multi_process_pool(target_devices=target_devices)

        # --- START OF CHANGES FOR TQDM WITH MULTI-PROCESS DOCUMENT ENCODING ---
        all_doc_embeddings_list = []
        # Define a size for macro-chunks for the tqdm progress bar
        # This determines how many documents are processed by encode_multi_process
        # before the tqdm bar updates.
        EXTERNAL_MACRO_CHUNK_SIZE = 20480  # Adjust as needed

        for text_macro_batch in iter_batch(
            texts_to_encode,
            batch_size=EXTERNAL_MACRO_CHUNK_SIZE,
            tqdm_bar=self.enable_tqdm,
            desc="Encoding document macro-chunks",
        ):
            if (
                not text_macro_batch
            ):  # Should not happen if texts_to_encode is not empty
                continue

            current_batch_embeddings_list = self.pylate_model.encode_multi_process(
                text_macro_batch,
                pool=pool,
                batch_size=self.encode_batch_size,  # This is batch_size per worker
                is_query=False,
            )
            all_doc_embeddings_list.extend(current_batch_embeddings_list)

        doc_embeddings = all_doc_embeddings_list
        # --- END OF CHANGES FOR TQDM WITH MULTI-PROCESS DOCUMENT ENCODING ---

        self.pylate_model.stop_multi_process_pool(pool)

        plaid_doc_ids_list = list(doc_store_build.keys())

        print("Adding documents to PLAID index...")
        self.plaid_index_instance.add_documents(
            documents_ids=plaid_doc_ids_list, documents_embeddings=doc_embeddings
        )

        self.doc_store = doc_store_build
        with open(doc_store_path, "wb") as f:
            pickle.dump(self.doc_store, f)

        print(f"Indexing complete. Index and metadata saved in '{output_folder}'.")

    def retrieve(
        self, nlqs: List[str], output_folder: str, k: int
    ) -> List[List[RetrievalResult]]:
        if self.pylate_retriever_instance is None or self.doc_store is None:
            doc_store_path = self._get_doc_store_path(output_folder)
            plaid_full_index_path = self._get_plaid_full_index_path(output_folder)
            plaid_metadata_file = os.path.join(plaid_full_index_path, "metadata.json")

            if not os.path.exists(doc_store_path) or not os.path.exists(
                plaid_metadata_file
            ):
                print(
                    f"Index or doc_store not found in '{output_folder}'. Please run index() first."
                )
                return [[] for _ in nlqs]

            self._initialize_plaid_components(output_folder, override=False)
            with open(doc_store_path, "rb") as f:
                self.doc_store = pickle.load(f)

        if not nlqs:
            return []

        print(f"Encoding {len(nlqs)} queries...")
        query_embeddings = self.pylate_model.encode(
            nlqs,
            batch_size=self.encode_batch_size,
            is_query=True,
            show_progress_bar=self.enable_tqdm,
            convert_to_numpy=True,
        )

        print("Retrieving documents with PyLate/PLAID...")
        retrieved_batches_pylate = self.pylate_retriever_instance.retrieve(
            queries_embeddings=query_embeddings, k=k
        )

        all_results: List[List[RetrievalResult]] = []
        for i, pylate_batch_results in enumerate(
            tqdm(
                retrieved_batches_pylate,
                desc="Processing retrieval results",
                disable=not self.enable_tqdm,
            )
        ):
            current_query_results: List[RetrievalResult] = []
            for res_dict in pylate_batch_results:
                plaid_doc_id = res_dict["id"]
                score = res_dict["score"]

                if plaid_doc_id in self.doc_store:
                    doc_info = self.doc_store[plaid_doc_id]
                    current_query_results.append(
                        RetrievalResult(
                            score=float(score),
                            object=str(doc_info["text"]),
                            metadata=doc_info["metadata"],
                        )
                    )
                else:
                    print(
                        f"Warning: PLAID ID '{plaid_doc_id}' not found in doc_store for query '{nlqs[i]}'."
                    )
            all_results.append(current_query_results)

        return all_results
