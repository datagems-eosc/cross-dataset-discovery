import asyncio
import os
import pickle
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import torch
from nlp_retrieval.core.models import RetrievalResult, SearchableItem
from nlp_retrieval.retrievers.retriever_abc import BaseRetriever

# from infinity_emb import AsyncEngineArray, EngineArgs
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from vllm import LLM


class FaissRetriever(BaseRetriever):
    """
    A powerful dense retriever using FAISS for indexing and search.

    This class supports multiple embedding backends:
    - 'sentence-transformers': Standard, easy-to-use library (default).
    - 'vllm': High-throughput embedding generation using VLLM.
    - 'infinity': High-throughput embedding generation using Infinity-Embed.
    """

    INDEX_FILENAME = "faiss.index"
    ITEMS_FILENAME = "items.pkl"

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        embedding_backend: str = "sentence-transformers",
        batch_size: int = 128,
        normalize_embeddings: bool = True,
        enable_tqdm: bool = True,
    ):
        """
        Initializes the FaissRetriever.

        Args:
            model_name_or_path: The name or path of the embedding model.
            embedding_backend: The backend to use for generating embeddings.
                               Options: 'sentence-transformers', 'vllm', 'infinity'.
            batch_size: The batch size for encoding operations.
            normalize_embeddings: Whether to L2-normalize embeddings. Essential for
                                  Inner Product (IP) search.
            enable_tqdm: If True, displays tqdm progress bars.
        """
        self.model_name_or_path = model_name_or_path
        self.embedding_backend = embedding_backend
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.enable_tqdm = enable_tqdm
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_gpus = torch.cuda.device_count()

        self._model: Optional[Any] = None
        self._embedding_dim: Optional[int] = None
        self._items_cache: Dict[str, List[SearchableItem]] = {}
        self._faiss_index_cache: Dict[str, faiss.Index] = {}

        self._initialize_model()

    def _initialize_model(self):
        """Lazy initialization of the embedding model based on the selected backend."""
        if self._model:
            return

        if self.embedding_backend == "sentence-transformers":
            self._model = SentenceTransformer(
                self.model_name_or_path, device=self.device
            )
            self._embedding_dim = self._model.get_sentence_embedding_dimension()

        elif self.embedding_backend == "vllm":
            self._model = LLM(
                model=self.model_name_or_path,
                tensor_parallel_size=self.num_gpus or 1,
                trust_remote_code=True,
                enforce_eager=True,  # Required for embed mode
            )
            self._embedding_dim = self._model.llm_engine.model_config.get_hidden_size()

        # elif self.embedding_backend == "infinity":
        #    engine_args = EngineArgs(
        #        model_name_or_path=self.model_name_or_path,
        #        device=self.device,
        #        batch_size=self.batch_size,
        #    )
        #    self._model = AsyncEngineArray.from_args([engine_args])

        else:
            raise ValueError(f"Unknown embedding_backend: '{self.embedding_backend}'")

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of texts into embeddings using the configured backend."""
        self._initialize_model()

        if self.embedding_backend == "sentence-transformers":
            return self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=self.enable_tqdm,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )

        if self.embedding_backend == "vllm":
            outputs = self._model.embed(texts)
            embeddings = np.array([out.outputs.embedding for out in outputs])
            if self.normalize_embeddings:
                faiss.normalize_L2(embeddings)
            return embeddings

        if self.embedding_backend == "infinity":

            async def _embed_async():
                engine = self._model[0]
                await engine.astart()
                embeds, _ = await engine.embed(sentences=texts)
                await engine.astop()
                return np.array(embeds)

            embeddings = asyncio.run(_embed_async())
            if self._embedding_dim is None:  # Infer embedding dim on first run
                self._embedding_dim = embeddings.shape[1]
            if self.normalize_embeddings:
                faiss.normalize_L2(embeddings)
            return embeddings

        raise RuntimeError("Encoding failed due to uninitialized or unknown backend.")

    def index(self, items: List[SearchableItem], output_path: str) -> None:
        """Builds and saves a FAISS index and the corresponding item data."""
        index_file = os.path.join(output_path, self.INDEX_FILENAME)
        items_file = os.path.join(output_path, self.ITEMS_FILENAME)

        if os.path.exists(index_file) and os.path.exists(items_file):
            print(f"Index already exists in '{output_path}'. Skipping indexing.")
            return

        if not items:
            return

        contents = [item.content for item in items]
        embeddings = self._encode(contents)

        if self._embedding_dim is None:
            raise RuntimeError("Embedding dimension could not be determined.")

        # Using IndexFlatIP for normalized embeddings (dot product is equivalent to cosine similarity)
        index = faiss.IndexFlatIP(self._embedding_dim)
        index.add(embeddings.astype(np.float32))

        faiss.write_index(index, index_file)
        with open(items_file, "wb") as f:
            pickle.dump(items, f)

    def _load_index_and_items(
        self, output_path: str
    ) -> tuple[faiss.Index, List[SearchableItem]]:
        """Loads the FAISS index and item list, caching them in memory."""
        if output_path in self._faiss_index_cache:
            return self._faiss_index_cache[output_path], self._items_cache[output_path]

        index_file = os.path.join(output_path, self.INDEX_FILENAME)
        items_file = os.path.join(output_path, self.ITEMS_FILENAME)

        if not os.path.exists(index_file) or not os.path.exists(items_file):
            raise FileNotFoundError(f"Index or item file not found in '{output_path}'")

        index = faiss.read_index(index_file)
        with open(items_file, "rb") as f:
            items = pickle.load(f)

        # Move index to GPU if available
        if self.device == "cuda":
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                print("Failed to move FAISS index to GPU. Using CPU instead.")
                pass  # Keep on CPU if GPU transfer fails

        self._faiss_index_cache[output_path] = index
        self._items_cache[output_path] = items
        return index, items

    def retrieve(
        self, processed_queries_batch: List[List[str]], output_path: str, k: int
    ) -> List[List[RetrievalResult]]:
        """
        Retrieves items for a batch of processed queries, handling sub-query aggregation.

        This method retrieves k items for each sub-query, combines them into a
        single, deduplicated list for each original query, and annotates each
        result with the keyword that retrieved it.
        """
        if not os.path.exists(output_path):
            print(f"Dense index not found for in {output_path}. Skipping.")
            return [[] for _ in processed_queries_batch]
        index, items = self._load_index_and_items(output_path)

        # 1. Flatten all sub-queries for a single, efficient batch encoding call
        flat_queries = []
        query_to_original_idx_map = []
        for i, sub_queries in enumerate(processed_queries_batch):
            for sub_query in sub_queries:
                flat_queries.append(sub_query)
                query_to_original_idx_map.append(i)

        if not flat_queries:
            return [[] for _ in processed_queries_batch]

        # 2. Encode all queries in one batch
        query_embeddings = self._encode(flat_queries).astype(np.float32)

        # 3. Perform FAISS search
        scores_batch, indices_batch = index.search(query_embeddings, k)

        # 4. Aggregate results, keeping the highest score and tracking the keyword
        aggregated_results: List[Dict[str, RetrievalResult]] = [
            {} for _ in processed_queries_batch
        ]

        pbar_desc = f"Processing FAISS Results ({self.embedding_backend})"
        for i in tqdm(
            range(len(flat_queries)), desc=pbar_desc, disable=not self.enable_tqdm
        ):
            original_nlq_idx = query_to_original_idx_map[i]
            keyword_used = flat_queries[i]  #
            scores = scores_batch[i]
            indices = indices_batch[i]

            for score, doc_idx in zip(scores, indices):
                if doc_idx == -1:  # FAISS uses -1 for no result
                    continue

                retrieved_item = items[doc_idx].copy(deep=True)
                item_id = retrieved_item.item_id

                # If item is new or has a better score, add/update it
                if (
                    item_id not in aggregated_results[original_nlq_idx]
                    or score > aggregated_results[original_nlq_idx][item_id].score
                ):
                    retrieved_item.metadata["retrieved_by_keyword"] = keyword_used

                    aggregated_results[original_nlq_idx][item_id] = RetrievalResult(
                        item=retrieved_item, score=float(score)
                    )

        # 5. Convert the aggregated dictionaries to sorted lists
        final_batches = []
        for res_dict in aggregated_results:
            # Sort final results by score
            sorted_by_score = sorted(
                res_dict.values(), key=lambda r: r.score, reverse=True
            )
            final_batches.append(sorted_by_score)
        return final_batches
