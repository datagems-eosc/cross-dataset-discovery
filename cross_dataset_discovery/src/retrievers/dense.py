import json
import os
import pickle
from typing import List, Dict, Any, Optional
from tqdm.auto import tqdm
import faiss
import torch
from cross_dataset_discovery.src.retrievers.base import BaseRetriever, RetrievalResult
import numpy as np
from vllm import LLM
from sentence_transformers import SentenceTransformer as ImportedSentenceTransformer
import asyncio
from infinity_emb import AsyncEngineArray, EngineArgs


class FaissDenseRetriever(BaseRetriever):
    """
    A retriever that uses dense embeddings and a FAISS index for efficient similarity search.

    This class supports multiple embedding backends:
    1.  sentence-transformers (default)
    2.  vLLM (for high-throughput inference on supported models)
    3.  Infinity-Embed (for high-throughput async inference)

    It handles the creation of a FAISS index from a corpus, saving it to disk,
    and then using that index to retrieve relevant documents for a given set of queries.
    """

    INDEX_FILENAME = "index.faiss"
    METADATA_FILENAME = "metadata.pkl"

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        enable_tqdm: bool = True,
        use_vllm_indexing: bool = False,
        use_infinity_indexing: bool = False,
    ):
        """
        Initializes the FaissDenseRetriever.

        Args:
            model_name_or_path (str): The name or path of the sentence-transformer model to use.
            enable_tqdm (bool): Whether to display tqdm progress bars during indexing and retrieval.
            use_vllm_indexing (bool): If True, uses the vLLM backend for embedding generation.
            use_infinity_indexing (bool): If True, uses the Infinity-Embed backend for embedding generation. (recommended)
        """
        self.model_name_or_path = model_name_or_path
        self.enable_tqdm = enable_tqdm
        self.use_vllm_indexing = use_vllm_indexing
        self.use_infinity_indexing = use_infinity_indexing
        self.num_gpus = torch.cuda.device_count()

        self.st_model = None
        self.model = None
        self.vllm_sampling_params = None
        self.infinity_engine_array: Optional[AsyncEngineArray] = None
        self.embedding_dim: Optional[int] = None

        self.selected_backend = "sentence_transformer"
        if self.use_vllm_indexing:
            self.selected_backend = "vllm"
        elif self.use_infinity_indexing:
            self.selected_backend = "infinity"

        if self.selected_backend == "vllm":
            self.model = LLM(
                model=self.model_name_or_path,
                trust_remote_code=True,
                tensor_parallel_size=self.num_gpus if self.num_gpus > 0 else 1,
                task="embed",
                enforce_eager=True,
                dtype="float16",
                tokenizer_pool_size=4,
                max_num_batched_tokens=8192,
                gpu_memory_utilization=0.7,
                max_model_len=8192,
            )
            self.embedding_dim = self.model.llm_engine.model_config.get_hidden_size()
        elif self.selected_backend == "infinity":
            engine_args = EngineArgs(model_name_or_path=self.model_name_or_path)
            self.infinity_engine_array = AsyncEngineArray.from_args([engine_args])
            self.embedding_dim = (
                None  # Will be determined from the first batch of embeddings
            )
        else:
            # For multi-GPU with sentence-transformers, we initialize on CPU and let the library handle distribution.
            device = (
                "cpu"
                if self.num_gpus > 1
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            if self.model_name_or_path == "Qwen/Qwen3-Embedding-0.6B":
                self.model = ImportedSentenceTransformer(
                    self.model_name_or_path,
                    device=device,
                    trust_remote_code=True,
                    cache_folder="assets/cache",
                    revision="refs/pr/2",
                )
            else:
                self.model = ImportedSentenceTransformer(
                    self.model_name_or_path,
                    device=device,
                    trust_remote_code=True,
                    cache_folder="assets/cache",
                )
            self.model.max_seq_length = 8192
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

        self.faiss_index_cpu: Optional[faiss.Index] = None
        self.doc_metadata_list: Optional[List[Dict[str, Any]]] = None
        self.text_to_id_map: Optional[Dict[str, int]] = None
        self._loaded_output_folder: Optional[str] = None

    def index(
        self,
        input_jsonl_path: str,
        output_folder: str,
        field_to_index: str,
        metadata_fields: List[str],
    ) -> None:
        os.makedirs(output_folder, exist_ok=True)
        index_path = os.path.join(output_folder, self.INDEX_FILENAME)
        metadata_path = os.path.join(output_folder, self.METADATA_FILENAME)
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            print(f"Index and metadata exist in '{output_folder}', skipping.")
            return

        texts, current_metadata_list = [], []
        with open(input_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                text = data[field_to_index]
                texts.append(text)
                entry = {"_text": text}
                for field in metadata_fields:
                    if field in data:
                        entry[field] = data[field]
                current_metadata_list.append(entry)

        embeddings_np: Optional[np.ndarray] = None
        final_metadata_list = current_metadata_list

        if self.selected_backend == "vllm":
            request_outputs = self.model.embed(texts, truncate_prompt_tokens=8192)
            processed_embeddings = [
                output.outputs.embedding for output in request_outputs
            ]
            embeddings_np = np.array(processed_embeddings, dtype=np.float32)
        elif self.selected_backend == "infinity":
            engine = self.infinity_engine_array[0]

            async def _embed_texts_with_infinity(texts_to_embed):
                await engine.astart()
                all_raw_embeddings_list = []
                if self.enable_tqdm and len(texts_to_embed) > 0:
                    batch_size = 512
                    for i in tqdm(
                        range(0, len(texts_to_embed), batch_size),
                        desc="Embedding with Infinity",
                    ):
                        batch = texts_to_embed[i : i + batch_size]
                        batch_embeds, _ = await engine.embed(sentences=batch)
                        all_raw_embeddings_list.extend(batch_embeds)
                elif len(texts_to_embed) > 0:
                    all_raw_embeddings_list, _ = await engine.embed(
                        sentences=texts_to_embed
                    )
                await engine.astop()
                return all_raw_embeddings_list

            if len(texts) > 0:
                raw_embeddings_list = asyncio.run(_embed_texts_with_infinity(texts))
                embeddings_np = np.array(raw_embeddings_list, dtype=np.float32)
                if self.embedding_dim is None:
                    self.embedding_dim = embeddings_np.shape[1]
            else:
                embeddings_np = np.array([], dtype=np.float32)
        else:  # sentence_transformer backend
            if self.num_gpus > 1:
                print(
                    f"--- Using {self.num_gpus} GPUs for indexing via sentence-transformers ---"
                )
                pool = self.model.start_multi_process_pool()
                effective_batch_size = 16 * self.num_gpus
                embeddings_np = self.model.encode_multi_process(
                    texts,
                    pool=pool,
                    batch_size=effective_batch_size,
                    show_progress_bar=self.enable_tqdm,
                    normalize_embeddings=False,
                )
                self.model.stop_multi_process_pool(pool)
            else:
                embeddings_np = self.model.encode(
                    texts,
                    batch_size=64,
                    show_progress_bar=self.enable_tqdm,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    truncation=True,
                )

        # Normalize embeddings for Inner Product (IP) search, which is equivalent to cosine similarity on normalized vectors.
        faiss.normalize_L2(embeddings_np)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings_np)

        faiss.write_index(index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(final_metadata_list, f)

        del embeddings_np
        del texts
        del current_metadata_list
        if "processed_embeddings" in locals():
            del processed_embeddings

        torch.cuda.empty_cache()

    def retrieve(
        self, nlqs: List[str], output_folder: str, k: int
    ) -> List[List[RetrievalResult]]:
        index_path = os.path.join(output_folder, self.INDEX_FILENAME)
        metadata_path = os.path.join(output_folder, self.METADATA_FILENAME)

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print(f"Index or metadata not found in '{output_folder}'")
            return [[] for _ in nlqs]

        try:
            index_cpu = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                doc_metadata_list: List[Dict[str, Any]] = pickle.load(f)
        except Exception as e:
            print(f"Error loading index/metadata from {output_folder}: {e}")
            return [[] for _ in nlqs]

        # Move index to GPU for faster search if available.
        index_to_search = index_cpu
        res = None
        if faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                index_to_search = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            except Exception as e:
                print(f"GPU move failed ({e}), using CPU index.")
        else:
            print("Using FAISS CPU index.")

        query_embeddings = self.model.encode(
            nlqs, batch_size=128, show_progress_bar=False, convert_to_numpy=True
        )
        faiss.normalize_L2(query_embeddings)
        scores, indices = index_to_search.search(query_embeddings, k)

        all_batches: List[List[RetrievalResult]] = []
        num_docs_in_index = len(doc_metadata_list)

        for qi, nlq in enumerate(
            tqdm(nlqs, desc="Retrieving with FAISS", disable=not self.enable_tqdm)
        ):
            batch_results: List[RetrievalResult] = []
            if qi >= len(indices) or qi >= len(scores):
                print(
                    f"Warning: Mismatch in FAISS search results length for query index {qi}. Skipping."
                )
                all_batches.append(batch_results)
                continue

            query_indices = indices[qi]
            query_scores = scores[qi]

            for rank, doc_idx in enumerate(query_indices):
                if (
                    doc_idx < 0
                    or doc_idx >= num_docs_in_index
                    or rank >= len(query_scores)
                ):
                    continue

                raw_score = query_scores[rank]
                current_score: Optional[float] = None
                if np.isnan(raw_score):
                    current_score = -float("inf")
                elif np.isinf(raw_score):
                    current_score = float(1e12) if raw_score > 0 else -float("inf")
                else:
                    current_score = float(raw_score)

                meta = doc_metadata_list[doc_idx]
                text_obj = meta.get("_text")

                if not isinstance(text_obj, str) or not text_obj:
                    print(
                        f"Warning: Retrieved object is not a valid string for doc_idx {doc_idx}. Skipping result."
                    )
                    continue
                text = text_obj

                extra = {key: val for key, val in meta.items() if key != "_text"}

                batch_results.append(
                    RetrievalResult(score=current_score, object=text, metadata=extra)
                )
            all_batches.append(batch_results)

        torch.cuda.empty_cache()
        if res:
            del res
        return all_batches
