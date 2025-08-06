import os
import json
import re
from typing import Optional, List, Dict

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class QueryDecomposer:
    """
    Decomposes complex natural language queries into simpler sub-queries using a VLLM-powered model.

    This class is optimized for high-throughput query decomposition. It constructs a
    few-shot prompt to guide the language model, sends requests in batches to the VLLM
    inference engine, and parses the text-based output. It also includes an optional
    file-based caching mechanism to store and retrieve previously computed decompositions,
    avoiding redundant processing.
    """

    def __init__(
        self,
        model_name_or_path: str = "gaunernst/gemma-3-27b-it-int4-awq",
        output_folder: Optional[str] = None,
        tensor_parallel_size: int = 2,
        quantization: Optional[str] = None,
        gpu_memory_utilization: float = 0.70,
        enable_prefix_caching: bool = True,
        trust_remote_code: bool = True,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 20,
    ):
        """
        Initializes the VLLM-based QueryDecomposer.

        Args:
            model_name_or_path (str): The name or path of the model to be loaded by VLLM.
            output_folder (Optional[str]): A path to a folder for caching decomposition results.
                                           If None, caching is disabled.
            tensor_parallel_size (int): The number of GPUs to use for tensor parallelism.
            quantization (Optional[str]): The quantization method to use (e.g., "awq", "gptq").
            gpu_memory_utilization (float): The fraction of GPU memory to reserve for the model.
            enable_prefix_caching (bool): Enables VLLM's prefix caching for faster processing
                                          of shared prompt prefixes.
            trust_remote_code (bool): Whether to trust remote code when loading the model.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The temperature for sampling. 0.0 means deterministic output.
            top_p (float): The nucleus sampling probability.
            top_k (int): The number of top tokens to consider for sampling.
        """
        self.model_name_or_path = model_name_or_path
        self.output_folder = output_folder
        self.cache_file = None
        self.decompositions_cache: Optional[Dict[str, List[str]]] = None

        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)
            self.cache_file = os.path.join(self.output_folder, "decompositions.json")
            self.decompositions_cache = self._load_cache()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir="/data/hdd1/vllm_models/",
        )
        self.llm = LLM(
            model=self.model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            trust_remote_code=trust_remote_code,
            download_dir="/data/hdd1/vllm_models/",
            max_model_len=4096,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens
        )

        examples_data = [
            {
                "input": "What's chat langchain, is it a langchain template?",
                "tool_calls": [
                    {"sub_query": "What is chat langchain"},
                    {"sub_query": "What is a langchain template"},
                ],
            },
            {
                "input": "How would I use LangGraph to build an automaton",
                "tool_calls": [
                    {"sub_query": "How to build automaton with LangGraph"},
                ],
            },
            {
                "input": "How to build multi-agent system and stream intermediate steps from it",
                "tool_calls": [
                    {"sub_query": "How to build multi-agent system"},
                    {"sub_query": "How to stream intermediate steps"},
                    {
                        "sub_query": "How to stream intermediate steps from multi-agent system"
                    },
                ],
            },
            {
                "input": "What's the difference between LangChain agents and LangGraph?",
                "tool_calls": [
                    {
                        "sub_query": "What's the difference between LangChain agents and LangGraph?"
                    },
                    {"sub_query": "What are LangChain agents"},
                    {"sub_query": "What is LangGraph"},
                ],
            },
            {
                "input": "what's the difference between web voyager and reflection agents? do they use langgraph?",
                "tool_calls": [
                    {
                        "sub_query": "What's the difference between web voyager and reflection agents"
                    },
                    {"sub_query": "Do web voyager and reflection agents use LangGraph"},
                    {"sub_query": "What is web voyager"},
                    {"sub_query": "What are reflection agents"},
                ],
            },
        ]

        self.formatted_examples_for_vllm = []
        for ex in examples_data:
            input_text = ex["input"]
            output_text = "\n".join([sq["sub_query"] for sq in ex["tool_calls"]])
            self.formatted_examples_for_vllm.extend(
                [
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": output_text},
                ]
            )

        self.system_prompt_content = """You are an expert at query decomposition. Your goal is to break down a user's question into the smallest possible set of specific, answerable sub-questions that are *all necessary* to fully answer the original question.

Follow these rules:
1.  **Identify Core Components:** Break the original question into distinct pieces of information required for a complete answer.
2.  **Single Focus:** Each sub-question must target only *one* specific fact, concept, or entity.
3.  **Necessity:** Only generate sub-questions whose answers are *strictly required* to answer the original question. Do not add questions for general context if not asked.
4.  **Completeness:** Ensure the set of sub-questions *collectively covers all parts* of the original question.
5.  **No Redundancy:** Do *not* create multiple sub-questions asking for the same information, even if phrased differently.
6.  **Independence:** Sub-questions should ideally be answerable independently.
7.  **Preserve Terms:** Retain all acronyms, technical terms, and proper nouns from the original question.

Respond ONLY with the list of sub-questions, each on a new line. Do NOT include any introduction, explanation, numbering, or bullet points preceding the sub-questions."""

        self.system_message_for_vllm = {
            "role": "system",
            "content": self.system_prompt_content,
        }

    def _load_cache(self) -> Dict[str, List[str]]:
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        if self.cache_file and self.decompositions_cache is not None:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.decompositions_cache, f, indent=4, ensure_ascii=False)

    def decompose(self, nlq: str) -> List[str]:
        """
        Decomposes a single Natural Language Query (NLQ) into a list of sub-queries.

        This method is a convenience wrapper around `decompose_batch` for a single query.

        Args:
            nlq (str): The natural language query string to decompose.

        Returns:
            List[str]: A list of strings, where each string is a sub-query.
        """
        if self.decompositions_cache is not None:
            cached_result = self.get_cached_decompositions(nlq)
            if cached_result:
                return cached_result

        messages = (
            [self.system_message_for_vllm]
            + self.formatted_examples_for_vllm
            + [{"role": "user", "content": nlq}]
        )

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        vllm_outputs = self.llm.generate([prompt_text], self.sampling_params)
        raw_output = vllm_outputs[0].outputs[0].text

        decomposed_queries = []
        if raw_output:
            lines = raw_output.strip().split("\n")
            for line in lines:
                cleaned_line = re.sub(r"\s+", " ", line).strip()
                if cleaned_line:
                    decomposed_queries.append(cleaned_line)

        if self.decompositions_cache is not None and decomposed_queries:
            self.decompositions_cache[nlq] = decomposed_queries
            self._save_cache()

        return decomposed_queries

    def get_cached_decompositions(self, nlq: str) -> Optional[List[str]]:
        if self.decompositions_cache is None:
            return None
        return self.decompositions_cache.get(nlq)

    def decompose_batch(self, nlqs: List[str]) -> List[List[str]]:
        """
        Decomposes a batch of Natural Language Queries (NLQs) into sub-queries.

        This method efficiently processes multiple queries by first checking the cache
        for each one. For queries not found in the cache, it constructs prompts, sends
        them to the VLLM engine in a single batch request, parses the results, and
        updates the cache.

        Args:
            nlqs (List[str]): A list of natural language query strings to decompose.

        Returns:
            List[List[str]]: A list of lists, where each inner list contains the
                             sub-queries for the corresponding input NLQ.
        """
        final_results: List[Optional[List[str]]] = [None] * len(nlqs)
        prompts_for_vllm: List[str] = []
        indices_for_vllm: List[int] = []
        nlqs_for_vllm: List[str] = []

        if not nlqs:
            return []

        # First pass: check cache and prepare prompts for non-cached queries.
        for i, nlq in enumerate(nlqs):
            if self.decompositions_cache is not None:
                cached_result = self.get_cached_decompositions(nlq)
                if cached_result is not None:
                    final_results[i] = cached_result
                    continue

            messages = (
                [self.system_message_for_vllm]
                + self.formatted_examples_for_vllm
                + [{"role": "user", "content": nlq}]
            )

            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts_for_vllm.append(prompt_text)
            indices_for_vllm.append(i)
            nlqs_for_vllm.append(nlq)

        if prompts_for_vllm:
            vllm_outputs = self.llm.generate(prompts_for_vllm, self.sampling_params)

            for i, output_obj in enumerate(vllm_outputs):
                original_nlq_index = indices_for_vllm[i]
                current_nlq = nlqs_for_vllm[i]

                raw_output = output_obj.outputs[0].text
                decomposed_queries = []
                if raw_output:
                    lines = raw_output.strip().split("\n")
                    for line in lines:
                        cleaned_line = re.sub(r"\s+", " ", line).strip()
                        if cleaned_line:
                            decomposed_queries.append(cleaned_line)

                final_results[original_nlq_index] = decomposed_queries

                if self.decompositions_cache is not None and decomposed_queries:
                    self.decompositions_cache[current_nlq] = decomposed_queries

            if self.decompositions_cache is not None and prompts_for_vllm:
                self._save_cache()

        return [res if res is not None else [] for res in final_results]
