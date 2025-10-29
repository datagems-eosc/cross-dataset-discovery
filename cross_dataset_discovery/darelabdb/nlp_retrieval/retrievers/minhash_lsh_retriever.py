import os
import pickle
from typing import Dict, List, Set

from darelabdb.nlp_retrieval.core.models import RetrievalResult, SearchableItem
from darelabdb.nlp_retrieval.retrievers.retriever_abc import BaseRetriever
from datasketch import MinHash, MinHashLSHForest
from tqdm import tqdm


class MinHashLshRetriever(BaseRetriever):
    """
    A retriever based on MinHash and Locality Sensitive Hashing (LSH).

    This retriever is effective at finding items with high Jaccard similarity
    (i.e., high token overlap) to the query in a scalable manner. It is not a
    semantic retriever but excels at near-duplicate detection and set-based matching.
    """

    LSH_INDEX_FILENAME = "minhash_lsh.pkl"
    ITEMS_FILENAME = "items.pkl"

    def __init__(self, num_perm: int = 128, enable_tqdm: bool = True):
        """
        Initializes the MinHashLshRetriever.

        Args:
            num_perm: The number of permutation functions to use for the MinHash
                      signatures. A larger number increases accuracy but also
                      memory usage and processing time.
            enable_tqdm: If True, displays tqdm progress bars.
        """
        self.num_perm = num_perm
        self.enable_tqdm = enable_tqdm

        # Caches to avoid reloading from disk for the same index path
        self._lsh_cache: Dict[str, tuple[MinHashLSHForest, Dict[str, MinHash]]] = {}
        self._items_cache: Dict[str, Dict[str, SearchableItem]] = {}

    def _create_minhash(self, text: str) -> MinHash:
        """Creates a MinHash signature for a given text."""
        minhash = MinHash(num_perm=self.num_perm)
        # Simple whitespace tokenization
        for token in text.split():
            minhash.update(token.encode("utf8"))
        return minhash

    def index(self, items: List[SearchableItem], output_path: str) -> None:
        """
        Builds a MinHash LSH Forest index from a list of items.
        """
        lsh_index_path = os.path.join(output_path, self.LSH_INDEX_FILENAME)
        items_path = os.path.join(output_path, self.ITEMS_FILENAME)

        if os.path.exists(lsh_index_path) and os.path.exists(items_path):
            print(f"Index already exists in '{output_path}'. Skipping indexing.")
            return

        if not items:
            return

        forest = MinHashLSHForest(num_perm=self.num_perm)
        item_id_to_minhash_map: Dict[str, MinHash] = {}

        seen_item_ids: Set[str] = set()

        pbar_desc = "Creating MinHash Signatures"
        for item in tqdm(items, desc=pbar_desc, disable=not self.enable_tqdm):
            if item.item_id in seen_item_ids:
                # Skip if we've already processed this item
                continue

            minhash = self._create_minhash(item.content)
            item_id_to_minhash_map[item.item_id] = minhash
            forest.add(item.item_id, minhash)

            # Add the item ID to the seen set
            seen_item_ids.add(item.item_id)

        forest.index()

        with open(lsh_index_path, "wb") as f:
            pickle.dump((forest, item_id_to_minhash_map), f)

        with open(items_path, "wb") as f:
            pickle.dump(items, f)

    def _load_index_and_items(
        self, output_path: str
    ) -> tuple[MinHashLSHForest, Dict[str, MinHash], Dict[str, SearchableItem]]:
        """Loads the LSH Forest, MinHash map, and item map, caching them."""
        if output_path in self._lsh_cache:
            lsh_forest, minhash_map = self._lsh_cache[output_path]
            items_map = self._items_cache[output_path]
            return lsh_forest, minhash_map, items_map

        lsh_index_path = os.path.join(output_path, self.LSH_INDEX_FILENAME)
        items_path = os.path.join(output_path, self.ITEMS_FILENAME)

        if not os.path.exists(lsh_index_path) or not os.path.exists(items_path):
            raise FileNotFoundError(f"Index or item file not found in '{output_path}'")

        with open(lsh_index_path, "rb") as f:
            lsh_forest, minhash_map = pickle.load(f)

        with open(items_path, "rb") as f:
            items_list = pickle.load(f)

        items_map = {item.item_id: item for item in items_list}

        self._lsh_cache[output_path] = (lsh_forest, minhash_map)
        self._items_cache[output_path] = items_map
        return lsh_forest, minhash_map, items_map

    def retrieve(
        self, processed_queries_batch: List[List[str]], output_path: str, k: int
    ) -> List[List[RetrievalResult]]:
        """
        Retrieves items by finding nearest neighbors in the MinHash LSH Forest.
        """
        if not os.path.exists(output_path):
            print(f"MinHash LSH index not found for in {output_path}. Skipping.")
            return [[] for _ in processed_queries_batch]
        lsh_forest, minhash_map, items_map = self._load_index_and_items(output_path)

        # 1. Flatten queries for processing
        flat_queries = []
        query_to_original_idx_map = []
        for i, sub_queries in enumerate(processed_queries_batch):
            for sub_query in sub_queries:
                flat_queries.append(sub_query)
                query_to_original_idx_map.append(i)

        if not flat_queries:
            return [[] for _ in processed_queries_batch]

        # 2. Create MinHash for all queries
        query_minhashes = [self._create_minhash(q) for q in flat_queries]

        # 3. Query the LSH Forest and aggregate results
        aggregated_results: List[Dict[str, RetrievalResult]] = [
            {} for _ in processed_queries_batch
        ]

        pbar_desc = "Processing MinHash LSH Results"
        for i in tqdm(
            range(len(flat_queries)), desc=pbar_desc, disable=not self.enable_tqdm
        ):
            original_nlq_idx = query_to_original_idx_map[i]
            query_minhash = query_minhashes[i]

            # Find candidate item IDs from the LSH Forest
            retrieved_ids = lsh_forest.query(query_minhash, k)

            for item_id in retrieved_ids:
                if item_id in minhash_map and item_id in items_map:
                    # Calculate the Jaccard similarity as the score
                    item_minhash = minhash_map[item_id]
                    score = query_minhash.jaccard(item_minhash)

                    item = items_map[item_id]
                    result = RetrievalResult(item=item, score=score)

                    # Add to results, keeping the one with the highest score
                    if (
                        item_id not in aggregated_results[original_nlq_idx]
                        or result.score
                        > aggregated_results[original_nlq_idx][item_id].score
                    ):
                        aggregated_results[original_nlq_idx][item_id] = result

        # 4. Convert dictionaries to sorted lists
        final_batches = []
        for res_dict in aggregated_results:
            sorted_res = sorted(res_dict.values(), key=lambda r: r.score, reverse=True)
            final_batches.append(sorted_res)

        return final_batches
