import os
import json
import re
from typing import Optional, List, Dict, Any

from pydantic.v1 import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

class SubQueryExample(BaseModel):
    """A Pydantic model to structure few-shot examples before formatting."""
    sub_query: str

class QueryDecomposer:
    """
    Decomposes a complex natural language query (NLQ) into simpler sub-queries.

    This class uses a large language model served via Ollama, guided by a few-shot
    prompt constructed with LangChain. It relies on simple string parsing of the
    model's output rather than structured tool calling. It also features an
    optional file-based caching system to avoid re-computing decompositions for
    the same query, speeding up repeated runs.
    """
    def __init__(self, ollama_model: str, output_folder: Optional[str] = None):
        """
        Initializes the QueryDecomposer.

        Args:
            ollama_model (str): The name of the Ollama model to use (e.g., "llama3.1:8b").
            output_folder (Optional[str]): A path to a folder where 'decompositions.json'
                                           will be stored for caching. If None, caching is disabled.
        """
        self.ollama_model = ollama_model
        self.output_folder = output_folder
        self.cache_file = None
        self.decompositions_cache: Optional[Dict[str, List[str]]] = None

        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)
            self.cache_file = os.path.join(self.output_folder, "decompositions.json")
            self.decompositions_cache = self._load_cache()

        # --- Define Few-Shot Examples ---
        examples_data = [
            {
                "input": "What's chat langchain, is it a langchain template?",
                "tool_calls": [
                    SubQueryExample(sub_query="What is chat langchain"),
                    SubQueryExample(sub_query="What is a langchain template"),
                ]
            },
            {
                "input": "How would I use LangGraph to build an automaton",
                "tool_calls": [
                    SubQueryExample(sub_query="How to build automaton with LangGraph"),
                ]
            },
            {
                "input": "How to build multi-agent system and stream intermediate steps from it",
                "tool_calls": [
                    SubQueryExample(sub_query="How to build multi-agent system"),
                    SubQueryExample(sub_query="How to stream intermediate steps"),
                    SubQueryExample(sub_query="How to stream intermediate steps from multi-agent system"),
                ]
            },
            {
                 "input": "What's the difference between LangChain agents and LangGraph?",
                 "tool_calls": [
                     SubQueryExample(sub_query="What's the difference between LangChain agents and LangGraph?"),
                     SubQueryExample(sub_query="What are LangChain agents"),
                     SubQueryExample(sub_query="What is LangGraph"),
                 ]
            },
            {
                "input": "what's the difference between web voyager and reflection agents? do they use langgraph?",
                "tool_calls": [
                    SubQueryExample(sub_query="What's the difference between web voyager and reflection agents"),
                    SubQueryExample(sub_query='Do web voyager and reflection agents use LangGraph'),
                    SubQueryExample(sub_query='What is web voyager'),
                    SubQueryExample(sub_query='What are reflection agents')
                ]
            }
        ]

        # Convert structured examples into the human message format for the prompt.
        simple_examples = []
        for ex in examples_data:
            input_text = ex["input"]
            output_text = "\n".join([sq.sub_query for sq in ex["tool_calls"]])
            simple_examples.extend([
                HumanMessage(content=input_text),
                AIMessage(content=output_text)
            ])

        # --- LangChain Setup ---
        system = """You are an expert at query decomposition. Your goal is to break down a user's question into the smallest possible set of specific, answerable sub-questions that are *all necessary* to fully answer the original question.

Follow these rules:
1.  **Identify Core Components:** Break the original question into distinct pieces of information required for a complete answer.
2.  **Single Focus:** Each sub-question must target only *one* specific fact, concept, or entity.
3.  **Necessity:** Only generate sub-questions whose answers are *strictly required* to answer the original question. Do not add questions for general context if not asked.
4.  **Completeness:** Ensure the set of sub-questions *collectively covers all parts* of the original question.
5.  **No Redundancy:** Do *not* create multiple sub-questions asking for the same information, even if phrased differently.
6.  **Independence:** Sub-questions should ideally be answerable independently.
7.  **Preserve Terms:** Retain all acronyms, technical terms, and proper nouns from the original question.

Respond ONLY with the list of sub-questions, each on a new line. Do NOT include any introduction, explanation, numbering, or bullet points preceding the sub-questions."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                *simple_examples,
                ("human", "{question}"),
            ]
        )

        llm = ChatOllama(model=self.ollama_model, temperature=0)
        parser = StrOutputParser()
        self.query_analyzer = prompt | llm | parser

    def _load_cache(self) -> Dict[str, List[str]]:
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load cache file {self.cache_file}. Error: {e}")
        return {}

    def _save_cache(self):
        if self.cache_file and self.decompositions_cache is not None:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.decompositions_cache, f, indent=4, ensure_ascii=False)
            except IOError as e:
                print(f"Warning: Could not save cache file {self.cache_file}. Error: {e}")

    def decompose(self, nlq: str) -> List[str]:
        """
        Decomposes a single Natural Language Query (NLQ) into a list of sub-queries.

        This method first checks the cache for a pre-existing decomposition. If not
        found, it invokes the LangChain pipeline to generate a raw string from the LLM,
        which is then parsed into a clean list of sub-query strings. The result is
        cached for future use if caching is enabled.

        Args:
            nlq (str): The natural language query string to decompose.

        Returns:
            List[str]: A list of strings, where each string is a sub-query. Returns
                       an empty list if the decomposition fails or returns no results.
        """
        if self.decompositions_cache is not None:
            cached_result = self.get_cached_decompositions(nlq)
            if cached_result:
                return cached_result

        decomposed_queries = []
        try:
            raw_output: str = self.query_analyzer.invoke({"question": nlq})

            if raw_output:
                # Parse the raw string output by splitting by newline and cleaning whitespace.
                decomposed_queries = [
                    line.strip() for line in raw_output.strip().split('\n')
                    if line.strip()
                ]

            if not decomposed_queries:
                 print(f"Warning: Model returned empty or non-parseable output for '{nlq}'. Raw output: '{raw_output}'")

        except Exception as e:
            print(f"Error during decomposition or parsing for '{nlq}': {e}")
            return []

        if self.decompositions_cache is not None and decomposed_queries:
            self.decompositions_cache[nlq] = decomposed_queries
            self._save_cache()
        elif self.decompositions_cache is not None and not decomposed_queries:
             # Current behavior: do not cache failures to allow for retries.
             pass

        return decomposed_queries

    def get_cached_decompositions(self, nlq: str) -> Optional[List[str]]:
        if self.decompositions_cache is None:
            return None
        return self.decompositions_cache.get(nlq)