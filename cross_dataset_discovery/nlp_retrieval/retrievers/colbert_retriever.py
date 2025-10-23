import os
import pickle
from typing import Dict, List

import torch
from nlp_retrieval.core.models import RetrievalResult, SearchableItem
from nlp_retrieval.retrievers.retriever_abc import BaseRetriever
from pylate import indexes as pylate_indexes
from pylate import models as pylate_models
from pylate import retrieve as pylate_retrieve
from tqdm.auto import tqdm


class PylateColbertRetriever(BaseRetriever):
    """
    A retriever for ColBERT models using the `pylate` library with PLAID indexing.
    Optimized for maximum accuracy regardless of latency.
    Note that `pylate` is not installed in the monorepo due to its dependency
    on `sentence-transformers == 4.0.2`. This is an open issue that must be
    addressed when the dependency changes.
    """

    ITEMS_FILENAME = "items.pkl"
    PLAID_INDEX_DIR_NAME = "plaid_index"

    def __init__(
        self,
        model_name_or_path: str = "lightonai/Reason-ModernColBERT",
        plaid_nbits: int = 8,
        plaid_kmeans_niters: int = 256,
        plaid_index_bsize: int = 64,
        plaid_ncells: int = 1,
        encode_batch_size: int = 8,
        enable_tqdm: bool = True,
        use_fp16: bool = False,
        compile_model: bool = False,
        max_length: int = 8192,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pylate_model = pylate_models.ColBERT(
            model_name_or_path=model_name_or_path,
            device=self.device,
        )

        if not compile_model:
            if not use_fp16:
                self.pylate_model.float()
        else:
            self.pylate_model = torch.compile(self.pylate_model)

        self.embedding_size = self.pylate_model.get_sentence_embedding_dimension()
        self.use_fp16 = use_fp16
        self.max_length = max_length

        self.plaid_config_params = {
            "embedding_size": self.embedding_size,
            "nbits": plaid_nbits,
            "kmeans_niters": plaid_kmeans_niters,
            "index_bsize": plaid_index_bsize,
            "ncells": plaid_ncells,
        }
        self.encode_batch_size = encode_batch_size
        self.enable_tqdm = enable_tqdm

        self._items_cache: Dict[str, Dict[str, SearchableItem]] = {}
        self._retriever_cache: Dict[str, pylate_retrieve.ColBERT] = {}

    def _get_plaid_index_path(self, output_path: str) -> str:
        return os.path.join(output_path, self.PLAID_INDEX_DIR_NAME)

    def _get_items_path(self, output_path: str) -> str:
        return os.path.join(output_path, self.ITEMS_FILENAME)

    def index(self, items: List[SearchableItem], output_path: str) -> None:
        """Builds a PLAID index and saves the corresponding item data."""
        items_path = self._get_items_path(output_path)
        plaid_index_path = self._get_plaid_index_path(output_path)
        plaid_metadata_file = os.path.join(plaid_index_path, "metadata.json")

        if os.path.exists(items_path) and os.path.exists(plaid_metadata_file):
            print(f"Index already exists in '{output_path}'. Skipping indexing.")
            return

        if not items:
            return

        os.makedirs(output_path, exist_ok=True)

        contents_to_encode = [item.content for item in items]

        self.pylate_model.eval()

        with torch.no_grad():
            doc_embeddings = self.pylate_model.encode(
                contents_to_encode,
                batch_size=self.encode_batch_size,
                is_query=False,  # Mark as document encoding
                show_progress_bar=self.enable_tqdm,
                convert_to_numpy=False,  # Keep as tensors for better precision
            )

        plaid_index = pylate_indexes.PLAID(
            index_folder=output_path,
            index_name=self.PLAID_INDEX_DIR_NAME,
            override=True,
            **self.plaid_config_params,
        )
        plaid_index.add_documents(
            documents_ids=[item.item_id for item in items],
            documents_embeddings=doc_embeddings,
        )

        with open(items_path, "wb") as f:
            pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_retriever_and_items(
        self, output_path: str
    ) -> tuple[pylate_retrieve.ColBERT, Dict[str, SearchableItem]]:
        """Loads the PLAID retriever and item map, caching them in memory."""
        if output_path in self._retriever_cache:
            return self._retriever_cache[output_path], self._items_cache[output_path]

        items_path = self._get_items_path(output_path)
        plaid_index_path = self._get_plaid_index_path(output_path)

        if not os.path.exists(items_path) or not os.path.exists(plaid_index_path):
            raise FileNotFoundError(f"Index or item file not found in '{output_path}'")

        plaid_index = pylate_indexes.PLAID(
            index_folder=output_path,
            index_name=self.PLAID_INDEX_DIR_NAME,
            override=False,
            **self.plaid_config_params,
        )
        retriever = pylate_retrieve.ColBERT(index=plaid_index)

        with open(items_path, "rb") as f:
            items_list = pickle.load(f)
        items_map = {item.item_id: item for item in items_list}

        self._retriever_cache[output_path] = retriever
        self._items_cache[output_path] = items_map
        return retriever, items_map

    def retrieve(
        self, processed_queries_batch: List[List[str]], output_path: str, k: int
    ) -> List[List[RetrievalResult]]:
        """Retrieves items using the Pylate/PLAID index."""
        if not os.path.exists(output_path):
            print(f"ColBERT index not found for in {output_path}. Skipping.")
            return [[] for _ in processed_queries_batch]
        retriever, items_map = self._load_retriever_and_items(output_path)

        flat_queries = []
        query_to_original_idx_map = []
        for i, sub_queries in enumerate(processed_queries_batch):
            for sub_query in sub_queries:
                flat_queries.append(sub_query)
                query_to_original_idx_map.append(i)

        if not flat_queries:
            return [[] for _ in processed_queries_batch]

        self.pylate_model.eval()

        with torch.no_grad():
            query_embeddings = self.pylate_model.encode(
                flat_queries,
                batch_size=self.encode_batch_size,
                is_query=True,
                show_progress_bar=self.enable_tqdm,
                convert_to_numpy=False,
            )

        pylate_results_batch = retriever.retrieve(
            queries_embeddings=query_embeddings, k=k
        )

        aggregated_results: List[Dict[str, RetrievalResult]] = [
            {} for _ in processed_queries_batch
        ]

        pbar_desc = "Processing ColBERT Results"
        for i in tqdm(
            range(len(flat_queries)), desc=pbar_desc, disable=not self.enable_tqdm
        ):
            original_nlq_idx = query_to_original_idx_map[i]
            pylate_results = pylate_results_batch[i]

            for res_dict in pylate_results:
                item_id = res_dict["id"]
                score = res_dict["score"]

                if item_id in items_map:
                    item = items_map[item_id]
                    result = RetrievalResult(item=item, score=float(score))

                    if (
                        item_id not in aggregated_results[original_nlq_idx]
                        or result.score
                        > aggregated_results[original_nlq_idx][item_id].score
                    ):
                        aggregated_results[original_nlq_idx][item_id] = result

        final_batches = []
        for res_dict in aggregated_results:
            sorted_res = sorted(res_dict.values(), key=lambda r: r.score, reverse=True)
            final_batches.append(sorted_res[:k])

        return final_batches
