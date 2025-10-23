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


class KeywordExtractorProcessor(BaseUserQueryProcessor):
    """
    A query processor that extracts key terms and entities from queries
    using a VLLM-powered large language model.

    This processor is optimized for high-throughput keyword extraction, using batching
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
        Initializes the VLLM-based keyword extraction processor.

        Args:
            model_name_or_path (str): The name or path of the model for VLLM.
            cache_folder (Optional[str]): A path to a folder for caching results.
            tensor_parallel_size (int): The number of GPUs for tensor parallelism.
            gpu_memory_utilization (float): Fraction of GPU memory for VLLM.
            **kwargs: Additional arguments for VLLM's LLM class or SamplingParams.
        """
        self.model_name_or_path = model_name_or_path
        self.cache_folder = cache_folder
        self.keywords_cache: Optional[Dict[str, List[str]]] = None

        if self.cache_folder:
            os.makedirs(self.cache_folder, exist_ok=True)
            self.cache_file = os.path.join(self.cache_folder, "keywords_cache.json")
            self.keywords_cache = self._load_cache()

        self.vllm_args = {
            "quantization": kwargs.pop("quantization", None),
            "trust_remote_code": kwargs.pop("trust_remote_code", True),
            "max_model_len": kwargs.pop("max_model_len", 4096),
        }
        sampling_args = {
            "temperature": kwargs.pop("temperature", 0.0),
            "top_p": kwargs.pop("top_p", 1.0),
            "max_tokens": kwargs.pop("max_tokens", 256),
        }
        self.cache_dir = cache_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tokenizer = None
        self.llm = None
        self.sampling_params = SamplingParams(**sampling_args)

        self._build_prompt_template()

    def _build_prompt_template(self):
        """Constructs the few-shot prompt for the language model."""
        examples_data = [
            {
                "input": "Compare the performance of the GeForce RTX 4090 and the Radeon RX 7900 XTX in Cyberpunk 2077.",
                "output": "GeForce RTX 4090\nRadeon RX 7900 XTX\nCyberpunk 2077\nperformance",
            },
            {
                "input": "Who was the prime minister of the United Kingdom during the Falklands War?",
                "output": "prime minister\nUnited Kingdom\nFalklands War",
            },
            {
                "input": "How do I implement a thread-safe singleton pattern in Java?",
                "output": "thread-safe\nsingleton pattern\nJava",
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

        system_prompt = """You are an expert at keyword extraction. Your goal is to identify and list the most important terms, entities, and concepts from the user's question. These keywords should be suitable for a search engine.

Follow these rules:
1.  **Extract Core Concepts:** Identify the main subjects of the query.
2.  **Identify Named Entities:** Extract all specific names of people, organizations, products, locations, etc.
3.  **Preserve Multi-Word Terms:** Keep phrases like "machine learning" or "GeForce RTX 4090" together.
4.  **Be Concise:** Do not include stopwords (like 'the', 'is', 'a') or generic verbs (like 'compare', 'find') unless they are part of a key phrase.
5.  ** This is the most inportant part, since the keywords you provide will be used for retriveing values from a database, make sure that they are exhaustive, meaning that you provide all the keywords from the query that might actually be a database value, so perhaps for a keyword with multiuple tokens you could also return the ngrams of it if you think that these could be database values on their own.

Respond ONLY with the list of keywords, each on a new line. Do NOT include any introduction, explanation, numbering, or bullet points."""

        self.system_message = {"role": "system", "content": system_prompt}

    def _load_cache(self) -> Dict[str, List[str]]:
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        if self.cache_file and self.keywords_cache is not None:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.keywords_cache, f, indent=2, ensure_ascii=False)

    def _parse_llm_output(self, raw_output: str) -> List[str]:
        """Cleans and parses the raw text output from the LLM into a list of keywords."""
        keywords = []
        if raw_output:
            lines = raw_output.strip().split("\n")
            for line in lines:
                cleaned_line = re.sub(r"^\s*[-*]?\s*\d*\.\s*", "", line).strip()
                if cleaned_line:
                    keywords.append(cleaned_line)
        return keywords

    def process(self, nlqs: List[str]) -> List[List[str]]:
        """Extracts keywords from a batch of natural language queries."""
        if not nlqs:
            return []

        final_results: List[Optional[List[str]]] = [None] * len(nlqs)
        prompts_to_generate, indices_to_generate, nlqs_to_generate = [], [], []

        for i, nlq in enumerate(
            tqdm(nlqs, desc="Preparing keyword extraction prompts")
        ):
            if self.keywords_cache is not None and nlq in self.keywords_cache:
                final_results[i] = self.keywords_cache[nlq]
                continue
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=self.vllm_args["trust_remote_code"],
                    cache_dir=self.cache_dir,
                )
            if self.llm is None:
                self.llm = LLM(
                    model=self.model_name_or_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    download_dir=self.cache_dir,
                    **self.vllm_args,
                )
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

        if prompts_to_generate:
            vllm_outputs = self.llm.generate(prompts_to_generate, self.sampling_params)

            for i, output in enumerate(tqdm(vllm_outputs, desc="Extracting keywords")):
                original_nlq_index = indices_to_generate[i]
                current_nlq = nlqs_to_generate[i]
                raw_output = output.outputs[0].text
                extracted_keywords = self._parse_llm_output(raw_output)

                final_results[original_nlq_index] = extracted_keywords

                if self.keywords_cache is not None:
                    self.keywords_cache[current_nlq] = extracted_keywords

        if self.keywords_cache is not None and prompts_to_generate:
            self._save_cache()

        # Ensure the final output is a list of lists, with empty lists for failures.
        return [res if res is not None else [] for res in final_results]
