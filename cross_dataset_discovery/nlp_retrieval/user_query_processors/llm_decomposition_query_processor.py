import json
import os
import re
from typing import Dict, List, Optional

from nlp_retrieval.user_query_processors.query_processor_abc import (
    BaseUserQueryProcessor,
)
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class DecompositionProcessor(BaseUserQueryProcessor):
    """
    A query processor that decomposes complex queries into simpler, self-contained
    sub-queries using a VLLM-powered large language model.

    This processor is optimized for high-throughput decomposition, using batching
    and an optional file-based cache to avoid redundant computations.
    """

    def __init__(
        self,
        model_name_or_path: str,
        cache_folder: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        cache_dir="/data/hdd1/vllm_models/",
        **kwargs,
    ):
        """
        Initializes the VLLM-based decomposition processor.

        Args:
            model_name_or_path (str): The name or path of the model for VLLM.
            cache_folder (Optional[str]): A path to a folder for caching results.
                                          If None, caching is disabled.
            tensor_parallel_size (int): The number of GPUs for tensor parallelism.
            gpu_memory_utilization (float): Fraction of GPU memory for VLLM.
            **kwargs: Additional arguments for VLLM's LLM class or SamplingParams.
        """
        self.model_name_or_path = model_name_or_path
        self.cache_folder = cache_folder
        self.decompositions_cache: Optional[Dict[str, List[str]]] = None

        if self.cache_folder:
            os.makedirs(self.cache_folder, exist_ok=True)
            self.cache_file = os.path.join(
                self.cache_folder, "decompositions_cache.json"
            )
            self.decompositions_cache = self._load_cache()

        # Pop VLLM and SamplingParams specific args from kwargs
        vllm_args = {
            "quantization": kwargs.pop("quantization", None),
            "enable_prefix_caching": kwargs.pop("enable_prefix_caching", True),
            "trust_remote_code": kwargs.pop("trust_remote_code", True),
            "download_dir": kwargs.pop("download_dir", cache_dir),
            "max_model_len": kwargs.pop("max_model_len", 2048),
            "max_seq_len_to_capture": kwargs.pop("max_seq_len_to_capture", 1024),
        }
        sampling_args = {
            "temperature": kwargs.pop("temperature", 0.0),
            "top_p": kwargs.pop("top_p", 0.95),
            "top_k": kwargs.pop("top_k", -1),
            "max_tokens": kwargs.pop("max_tokens", 512),
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
            **vllm_args,
        )
        self.sampling_params = SamplingParams(**sampling_args)

        self._build_prompt_template()

    def _build_prompt_template(self):
        """Constructs the few-shot prompt for the language model."""
        examples_data = [
            {
                "input": "What's the difference between web-based search and reflection agents? Do they use similar graph-based approaches?",
                "output": "What is the difference between web-based search and reflection agents\nDo web-based search and reflection agents use graph-based approaches\nWhat are web-based search agents\nWhat are reflection agents",
            },
            {
                "input": "How can I build a multi-agent system and stream intermediate steps from it?",
                "output": "How to build a multi-agent system\nHow to stream intermediate steps from a multi-agent system",
            },
        ]

        self.formatted_examples = []
        for ex in examples_data:
            self.formatted_examples.extend(
                [
                    {"role": "user", "content": ex["input"]},
                    {"role": "assistant", "content": ex["output"]},
                ]
            )

        system_prompt = """You are an expert at query decomposition. Your goal is to break down a user's question into a set of specific, answerable sub-questions that are all necessary to fully answer the original question.

Follow these rules:
1.  **Single Focus:** Each sub-question must target only one specific fact, concept, or entity.
2.  **Necessity:** Only generate sub-questions whose answers are strictly required. Do not add questions for general context if not asked.
3.  **Completeness:** The set of sub-questions must collectively cover all parts of the original question.
4.  **Preserve Terms:** Retain all acronyms, technical terms, and proper nouns from the original question.

Respond ONLY with the list of sub-questions, each on a new line. Do NOT include any introduction, explanation, numbering, or bullet points."""

        self.system_message = {"role": "system", "content": system_prompt}

    def _load_cache(self) -> Dict[str, List[str]]:
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        if self.cache_file and self.decompositions_cache is not None:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.decompositions_cache, f, indent=2, ensure_ascii=False)

    def _parse_llm_output(self, raw_output: str) -> List[str]:
        """Cleans and parses the raw text output from the LLM."""
        decomposed_queries = []
        if raw_output:
            lines = raw_output.strip().split("\n")
            for line in lines:
                # Clean up potential markdown, numbering, or extra whitespace
                cleaned_line = re.sub(r"^\s*[-*]?\s*\d*\.\s*", "", line).strip()
                if cleaned_line:
                    decomposed_queries.append(cleaned_line)
        return decomposed_queries

    def process(self, nlqs: List[str]) -> List[List[str]]:
        """Decomposes a batch of natural language queries into sub-queries."""
        if not nlqs:
            return []

        final_results: List[Optional[List[str]]] = [None] * len(nlqs)
        prompts_to_generate, indices_to_generate, nlqs_to_generate = [], [], []

        # First pass: check cache and prepare prompts for non-cached queries.
        for i, nlq in enumerate(tqdm(nlqs, desc="Preparing decomposition prompts")):
            if (
                self.decompositions_cache is not None
                and nlq in self.decompositions_cache
            ):
                final_results[i] = self.decompositions_cache[nlq]
                continue

            messages = (
                [self.system_message]
                + self.formatted_examples
                + [{"role": "user", "content": nlq}]
            )
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts_to_generate.append(prompt_text)
            indices_to_generate.append(i)
            nlqs_to_generate.append(nlq)

        # Second pass: generate decompositions for non-cached queries in a single batch.
        if prompts_to_generate:
            vllm_outputs = self.llm.generate(prompts_to_generate, self.sampling_params)

            for i, output in enumerate(
                tqdm(vllm_outputs, desc="Generating decompositions")
            ):
                original_nlq_index = indices_to_generate[i]
                current_nlq = nlqs_to_generate[i]
                raw_output = output.outputs[0].text
                decomposed_queries = self._parse_llm_output(raw_output)

                # Include the original query as one of the queries to search for.
                final_decomposed_list = [current_nlq] + decomposed_queries
                final_results[original_nlq_index] = list(
                    dict.fromkeys(final_decomposed_list)
                )  # Deduplicate

                if self.decompositions_cache is not None:
                    self.decompositions_cache[current_nlq] = final_results[
                        original_nlq_index
                    ]

        # Save the cache if it was modified.
        if self.decompositions_cache is not None and prompts_to_generate:
            self._save_cache()

        # Ensure the final output matches the required format.
        return [
            res if res is not None else [nlqs[i]] for i, res in enumerate(final_results)
        ]
