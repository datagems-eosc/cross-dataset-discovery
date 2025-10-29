import json
import os
import subprocess
import tempfile
from typing import Dict, List, Optional

from darelabdb.nlp_retrieval.core.models import RetrievalResult, SearchableItem
from darelabdb.nlp_retrieval.retrievers.retriever_abc import BaseRetriever
from pyserini.analysis import Analyzer, get_lucene_analyzer
from pyserini.search.lucene import LuceneSearcher, querybuilder
from tqdm import tqdm


class PyseriniRetriever(BaseRetriever):
    """
    A sparse retriever implementation using the BM25 algorithm via Pyserini.
    """

    def __init__(self, k1: float = 0.9, b: float = 0.4, enable_tqdm: bool = True):
        """
        Initializes the Pyserini-based BM25 retriever.

        Args:
            k1: The BM25 k1 parameter. Controls term frequency saturation.
            b: The BM25 b parameter. Controls document length normalization.
            enable_tqdm: If True, displays tqdm progress bars during operations.
        """
        self.k1 = k1
        self.b = b
        self.enable_tqdm = enable_tqdm
        self._searcher_cache: Dict[str, LuceneSearcher] = {}

    def index(self, items: List[SearchableItem], output_path: str) -> None:
        """
        Builds a Pyserini/Lucene index from a list of SearchableItem objects.
        """
        if os.path.exists(output_path) and os.listdir(output_path):
            print(f"Index already exists in '{output_path}'. Skipping indexing.")
            return

        # Pyserini's JsonCollection requires a directory of .jsonl files.
        # We create a temporary directory to stage the data.
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_jsonl_path = os.path.join(temp_dir, "corpus.jsonl")
            with open(prepared_jsonl_path, "w", encoding="utf-8") as outfile:
                for item in items:
                    pyserini_doc = {
                        "id": item.item_id,
                        "contents": item.content,
                        **item.metadata,
                    }
                    outfile.write(json.dumps(pyserini_doc, ensure_ascii=False) + "\n")
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
                output_path,
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                str(num_threads),
                "--storePositions",
                "--storeDocvectors",
                "--storeRaw",
            ]
            subprocess.run(
                cmd, check=True, capture_output=True, text=True, encoding="utf-8"
            )  # nosec B603

    def retrieve(
        self,
        processed_queries_batch: List[List[str]],
        output_path: str,
        k: int,
        **kwargs,
    ) -> List[List[RetrievalResult]]:
        """
        Retrieves items using Pyserini. If filters are provided, it builds complex
        queries; otherwise, it uses the default high-performance batch search.
        """
        analyzer = Analyzer(get_lucene_analyzer())
        if not os.path.exists(output_path):
            print(f"BM25 index not found in {output_path}. Skipping.")
            return [[] for _ in processed_queries_batch]

        filters: Optional[Dict] = kwargs.get("filters")
        if output_path not in self._searcher_cache:
            searcher = LuceneSearcher(output_path)
            searcher.set_bm25(self.k1, self.b)
            self._searcher_cache[output_path] = searcher
        searcher = self._searcher_cache[output_path]
        flat_queries = []
        query_to_original_idx_map = []
        for i, sub_queries in enumerate(processed_queries_batch):
            for sub_query in sub_queries:
                if sub_query and isinstance(sub_query, str):
                    flat_queries.append(sub_query)
                    query_to_original_idx_map.append(i)

        if not flat_queries:
            return [[] for _ in processed_queries_batch]

        q_ids = [f"q{i}" for i in range(len(flat_queries))]
        if filters and filters.get("source"):
            query_objects = []
            for query_text in flat_queries:
                analyzed_terms = analyzer.analyze(query_text)

                should = querybuilder.JBooleanClauseOccur["should"].value
                must = querybuilder.JBooleanClauseOccur["must"].value
                filter_clause = querybuilder.JBooleanClauseOccur["filter"].value

                text_query_builder = querybuilder.get_boolean_query_builder()
                for term in analyzed_terms:
                    term_query = querybuilder.get_term_query(term, field="contents")
                    text_query_builder.add(term_query, should)
                text_query = text_query_builder.build()

                dataset_ids = filters["source"]
                filter_builder = querybuilder.get_boolean_query_builder()
                for did in dataset_ids:
                    term_query = querybuilder.get_term_query(did, field="source")
                    filter_builder.add(term_query, should)

                combined_builder = querybuilder.get_boolean_query_builder()
                combined_builder.add(text_query, must)
                combined_builder.add(filter_builder.build(), filter_clause)
                query_objects.append(combined_builder.build())

            batch_hits = searcher.batch_search(
                queries=query_objects, qids=q_ids, k=k, threads=os.cpu_count() or 1
            )
        else:
            batch_hits = searcher.batch_search(
                queries=flat_queries, qids=q_ids, k=k, threads=os.cpu_count() or 1
            )

        aggregated_results: List[Dict[str, RetrievalResult]] = [
            {} for _ in processed_queries_batch
        ]

        pbar_desc = "Processing BM25 Results"
        for i in tqdm(
            range(len(flat_queries)), desc=pbar_desc, disable=not self.enable_tqdm
        ):
            original_nlq_idx = query_to_original_idx_map[i]
            qid = q_ids[i]
            hits = batch_hits.get(qid, [])

            for hit in hits:
                raw_doc = json.loads(hit.lucene_document.get("raw"))
                item_id = raw_doc.get("id")
                if not item_id:
                    continue

                item = SearchableItem(
                    item_id=item_id,
                    content=raw_doc.get("contents", ""),
                    metadata={
                        k: v for k, v in raw_doc.items() if k not in ["id", "contents"]
                    },
                )
                result = RetrievalResult(item=item, score=hit.score)

                # If item is new or has a better score, add/update it
                if (
                    item_id not in aggregated_results[original_nlq_idx]
                    or result.score
                    > aggregated_results[original_nlq_idx][item_id].score
                ):
                    aggregated_results[original_nlq_idx][item_id] = result

        # Convert aggregated dicts to sorted lists
        final_batches = [
            sorted(res_dict.values(), key=lambda r: r.score, reverse=True)
            for res_dict in aggregated_results
        ]
        return final_batches
