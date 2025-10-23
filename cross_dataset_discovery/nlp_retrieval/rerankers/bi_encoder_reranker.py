from typing import List

import torch
import torch.nn.functional as F
from nlp_retrieval.core.models import RetrievalResult
from nlp_retrieval.rerankers.reranker_abc import BaseReranker
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


class BiEncoderReranker(BaseReranker):
    """
    A reranker that uses a bi-encoder model to re-score candidates.

    This component independently encodes the query and each candidate document,
    then computes their cosine similarity
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initializes the BiEncoderReranker.

        Args:
            model_name: The name of the sentence-transformer model to use.
            batch_size: The batch size for encoding documents.
            device: The device to run the model on (e.g., "cuda" or "cpu").
        """
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def _cosine_similarity(
        self, query_emb: torch.Tensor, doc_embs: torch.Tensor
    ) -> torch.Tensor:
        """Computes cosine similarity between a single query and multiple docs."""
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        doc_embs = F.normalize(doc_embs, p=2, dim=-1)
        # Squeeze to convert shape [1, N] to [N]
        return torch.mm(query_emb, doc_embs.T).squeeze(0)

    def rerank(
        self, nlqs: List[str], results_batch: List[List[RetrievalResult]], k: int
    ) -> List[List[RetrievalResult]]:
        final_batches = []
        for nlq, result_list in tqdm(
            zip(nlqs, results_batch),
            total=len(nlqs),
            desc="Reranking with BiEncoder",
            disable=len(nlqs) < 5,
        ):
            if not result_list:
                final_batches.append([])
                continue

            query_to_embed = [f"search_query: {nlq}"]
            docs_to_embed = [
                f"search_document: {res.item.content}" for res in result_list
            ]

            with torch.no_grad():
                query_emb = self.model.encode(query_to_embed, convert_to_tensor=True)
                doc_embs = self.model.encode(
                    docs_to_embed, batch_size=self.batch_size, convert_to_tensor=True
                )

            similarities = self._cosine_similarity(query_emb, doc_embs)

            rescored_results = [
                RetrievalResult(item=orig_res.item, score=float(score))
                for orig_res, score in zip(result_list, similarities)
            ]

            # Sort by the newly computed scores and truncate
            sorted_results = sorted(
                rescored_results, key=lambda r: r.score, reverse=True
            )
            final_batches.append(sorted_results[:k])

        return final_batches
