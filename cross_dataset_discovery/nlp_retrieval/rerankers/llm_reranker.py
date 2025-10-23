import re
from typing import Dict, List, Optional

from nlp_retrieval.core.models import RetrievalResult
from nlp_retrieval.rerankers.reranker_abc import BaseReranker
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class LLMReranker(BaseReranker):
    """
    A reranker with a Large Language Model (LLM) via VLLM for re-scoring.

    This reranker constructs a detailed prompt for each query, presenting the candidate
    documents with temporary integer IDs. It uses few-shot examples to instruct the
    LLM to return a sorted list of the IDs corresponding to the most relevant documents.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        cache_dir="/data/hdd1/vllm_models/",
        **kwargs,
    ):
        """
        Initializes the VLLM-based reranker.

        Args:
            model_name_or_path: The name or path of the model to be loaded by VLLM.
            tensor_parallel_size: The number of GPUs to use for tensor parallelism.
            gpu_memory_utilization: The fraction of GPU memory to reserve for the model.
            **kwargs: Additional arguments for VLLM's LLM class (e.g., quantization)
                      or SamplingParams (e.g., temperature, max_tokens).
        """
        self.model_name_or_path = model_name_or_path

        vllm_args = {
            "trust_remote_code": kwargs.pop("trust_remote_code", True),
            "max_model_len": kwargs.pop("max_model_len", 32768),
            "quantization": kwargs.pop("quantization", None),
        }
        sampling_args = {
            "temperature": kwargs.pop("temperature", 0.0),
            "top_p": kwargs.pop("top_p", 1.0),
            "max_tokens": kwargs.pop("max_tokens", 1024),
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=vllm_args["trust_remote_code"],
            cache_dir=cache_dir,
        )
        self.llm = LLM(
            model=self.model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            download_dir=cache_dir,
            **vllm_args,
        )
        self.sampling_params = SamplingParams(**sampling_args)

        self._build_prompt_components()

    def _build_prompt_components(self):
        """Constructs the static parts of the prompt (system message and few-shot examples)."""
        self.system_message = {
            "role": "system",
            "content": "You are an expert search result reranker. Your task is to analyze a user's query and a list of retrieved documents. You must identify the most relevant documents that directly answer the query.\n\nRespond ONLY with a comma-separated list of the integer IDs, sorted from most relevant to least relevant. Do not include any explanation, preamble, or formatting.",
        }

        # Few-shot examples to guide the model's output format and reasoning
        few_shot_user_1 = """User Query: "What is the capital of France?"

Documents:
[1] Paris is the capital and most populous city of France.
[2] The Eiffel Tower is a famous landmark in Paris.
[3] Berlin is the capital of Germany.

Instructions:
Identify the top 2 most relevant documents and respond with their IDs."""

        few_shot_assistant_1 = "1,2"

        few_shot_user_2 = """User Query: "Which GPU is better for gaming, the RTX 4090 or the RX 7900 XTX?"

Documents:
[1] The Nvidia RTX 4090 offers top-tier performance in 4K gaming.
[2] A detailed review concludes that the RTX 4090 has a slight edge in most games, while the 7900 XTX offers better value.
[3] The AMD RX 7900 XTX provides excellent rasterization performance for modern titles.
[4] The RTX 3060 is a popular mid-range GPU from the previous generation.

Instructions:
Identify the top 3 most relevant documents and respond with their IDs."""

        few_shot_assistant_2 = "2,1,3"

        self.few_shot_examples = [
            {"role": "user", "content": few_shot_user_1},
            {"role": "assistant", "content": few_shot_assistant_1},
            {"role": "user", "content": few_shot_user_2},
            {"role": "assistant", "content": few_shot_assistant_2},
        ]

    def _parse_llm_output(self, raw_output: str, max_id: int) -> List[int]:
        """Cleans and parses the raw text output from the LLM into a list of integer IDs."""
        try:
            # Find all numbers in the string
            ids_str = re.findall(r"\d+", raw_output)
            # Convert to integers and filter out any invalid IDs
            parsed_ids = [
                int(id_str) for id_str in ids_str if 1 <= int(id_str) <= max_id
            ]
            return parsed_ids
        except (ValueError, TypeError):
            return []

    def rerank(
        self, nlqs: List[str], results_batch: List[List[RetrievalResult]], k: int
    ) -> List[List[RetrievalResult]]:
        if not nlqs:
            return []

        prompts_to_generate: List[str] = []
        result_maps_for_batch: List[Dict[int, RetrievalResult]] = []
        indices_to_process: List[int] = []
        final_results_batch: List[Optional[List[RetrievalResult]]] = [None] * len(nlqs)

        # --- 1. Prepare prompts for all queries in the batch ---
        for i, (nlq, candidate_list) in enumerate(
            tqdm(
                zip(nlqs, results_batch),
                total=len(nlqs),
                desc="Preparing VLLM Reranking Prompts",
            )
        ):
            if not candidate_list:
                final_results_batch[i] = []
                continue

            id_to_result_map = {idx + 1: res for idx, res in enumerate(candidate_list)}
            formatted_docs = "\n".join(
                f"[{doc_id}] {res.item.content}"
                for doc_id, res in id_to_result_map.items()
            )

            # Construct the final user prompt for the current query
            final_user_prompt = f"""User Query: "{nlq}"

Documents:
{formatted_docs}

Instructions:
Identify the top {k} most relevant documents and respond with their IDs."""

            # Combine system message, few-shot examples, and the final user prompt
            messages = (
                [self.system_message]
                + self.few_shot_examples
                + [{"role": "user", "content": final_user_prompt}]
            )

            chat_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            prompts_to_generate.append(chat_prompt)
            result_maps_for_batch.append(id_to_result_map)
            indices_to_process.append(i)

        # --- 2. Generate responses from VLLM in a single batch call ---
        if prompts_to_generate:
            vllm_outputs = self.llm.generate(prompts_to_generate, self.sampling_params)

            # --- 3. Process the batch results ---
            for i, output in enumerate(tqdm(vllm_outputs, desc="Reranking with VLLM")):
                original_batch_idx = indices_to_process[i]
                id_map = result_maps_for_batch[i]
                raw_output = output.outputs[0].text

                reranked_ids = self._parse_llm_output(raw_output, max_id=len(id_map))

                reranked_results = [
                    id_map[doc_id] for doc_id in reranked_ids if doc_id in id_map
                ]

                # Assign new scores based on the rank from the LLM for consistency
                for rank, res in enumerate(reranked_results):
                    res.score = 1.0 / (rank + 1)

                final_results_batch[original_batch_idx] = reranked_results[:k]

        return [res if res is not None else [] for res in final_results_batch]
