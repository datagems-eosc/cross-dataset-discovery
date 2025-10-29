import os
from typing import Dict, List, Union

from darelabdb.nlp_retrieval.core.models import RetrievalResult, SearchableItem
from darelabdb.nlp_retrieval.retrievers.retriever_abc import BaseRetriever
from ragatouille import RAGPretrainedModel
from tqdm.auto import tqdm


class RagatouilleColbertRetriever(BaseRetriever):
    """
    A late-interaction retriever using ColBERT models via the RAGatouille library.
    It is preferable to use the ColBERT implementation using pylate, using
    the `PylateColbertRetriever` class found in `colbert_retriever.py`
    """

    def __init__(
        self,
        model_name_or_path: str = "jinaai/jina-colbert-v2",
        enable_tqdm: bool = True,
    ):
        """
        Initializes the RagatouilleColbertRetriever.

        Args:
            model_name_or_path: The name or path of the pretrained ColBERT model
            enable_tqdm: If True, displays tqdm progress bars during operations.
        """
        self.model_name_or_path = model_name_or_path
        self.enable_tqdm = enable_tqdm
        self._rag_model_cache: Dict[str, RAGPretrainedModel] = {}

    def index(self, items: List[SearchableItem], output_path: str) -> None:
        """
        Builds and saves a RAGatouille (ColBERT) index.
        """
        # RAGatouille manages index paths internally relative to an `index_root`.
        # The `output_path` from our framework will serve as this root.
        index_name = f"ragatouille_index_{os.path.basename(output_path)}"
        full_index_path = os.path.join(
            output_path, ".ragatouille", "colbert", "indexes", index_name
        )

        if os.path.exists(full_index_path):
            print(
                f"Index '{index_name}' appears to exist in '{output_path}'. Skipping indexing."
            )
            return

        if not items:
            print("Warning: No items provided to index.")
            return

        # Initialize the RAG model, setting the root for all its indexes.
        rag_model = RAGPretrainedModel.from_pretrained(
            self.model_name_or_path, index_root=output_path
        )

        collection = [item.content for item in items]
        document_ids = [item.item_id for item in items]
        document_metadatas = [item.metadata for item in items]

        rag_model.index(
            collection=collection,
            document_ids=document_ids,
            document_metadatas=document_metadatas,
            index_name=index_name,
            overwrite_index=True,
            use_faiss=True,
        )

    def _load_rag_model(self, output_path: str) -> RAGPretrainedModel:
        """Loads a RAGatouille model from an index path, using an in-memory cache."""
        # We use the output_path as the key, as it's the root for all indexes within.
        if output_path in self._rag_model_cache:
            return self._rag_model_cache[output_path]

        # RAGatouille needs the path to the specific index directory to load.
        # We assume there's only one index per retriever output_path.
        index_root_path = os.path.join(
            output_path, ".ragatouille", "colbert", "indexes"
        )
        if not os.path.exists(index_root_path):
            raise FileNotFoundError(
                f"RAGatouille index directory not found at: {index_root_path}. "
                "Please ensure `index()` has been run."
            )

        try:
            # Get the first (and likely only) index name in the directory
            index_name = os.listdir(index_root_path)[0]
            full_index_path = os.path.join(index_root_path, index_name)
        except (FileNotFoundError, IndexError):
            raise FileNotFoundError(
                f"No RAGatouille index found under the root: {output_path}. "
                "Please ensure `index()` has been run."
            )

        # from_index is the correct way to load a model for searching
        rag_model = RAGPretrainedModel.from_index(full_index_path)
        self._rag_model_cache[output_path] = rag_model
        return rag_model

    def retrieve(
        self, processed_queries_batch: List[List[str]], output_path: str, k: int
    ) -> List[List[RetrievalResult]]:
        """
        Retrieves items for a batch of queries using the RAGatouille index.
        """
        rag_model = self._load_rag_model(output_path)

        # 1. Flatten all sub-queries for a single, efficient batch search call
        flat_queries = []
        query_to_original_idx_map = []
        for i, sub_queries in enumerate(processed_queries_batch):
            for sub_query in sub_queries:
                flat_queries.append(sub_query)
                query_to_original_idx_map.append(i)

        if not flat_queries:
            return [[] for _ in processed_queries_batch]

        # 2. Perform one large batch search using RAGatouille
        ragatouille_results_batch: Union[List[Dict], List[List[Dict]]]
        ragatouille_results_batch = rag_model.search(query=flat_queries, k=k)

        # RAGatouille returns a single list for a single query, but a list of lists for multiple.
        # We normalize to always be a list of lists for consistent processing.
        if flat_queries and len(flat_queries) == 1:
            ragatouille_results_batch = [ragatouille_results_batch]

        # 3. Aggregate results, keeping the highest score for each unique item
        aggregated_results: List[Dict[str, RetrievalResult]] = [
            {} for _ in processed_queries_batch
        ]

        pbar_desc = "Processing RAGatouille Results"
        for i in tqdm(
            range(len(flat_queries)), desc=pbar_desc, disable=not self.enable_tqdm
        ):
            original_nlq_idx = query_to_original_idx_map[i]
            results_for_sub_query = ragatouille_results_batch[i]

            for res_dict in results_for_sub_query:
                item_id = res_dict.get("document_id")
                if not item_id:
                    continue

                # Reconstruct our SearchableItem and RetrievalResult models
                item = SearchableItem(
                    item_id=item_id,
                    content=res_dict.get("content", ""),
                    metadata=res_dict.get("document_metadata", {}),
                )
                result = RetrievalResult(
                    item=item, score=float(res_dict.get("score", 0.0))
                )

                # If item is new for this query or has a better score, add/update it
                if (
                    item_id not in aggregated_results[original_nlq_idx]
                    or result.score
                    > aggregated_results[original_nlq_idx][item_id].score
                ):
                    aggregated_results[original_nlq_idx][item_id] = result

        # 4. Convert the aggregated dictionaries to sorted lists
        final_batches = []
        for res_dict in aggregated_results:
            sorted_by_score = sorted(
                res_dict.values(), key=lambda r: r.score, reverse=True
            )
            final_batches.append(sorted_by_score)

        return final_batches
