import json
import random
from tqdm import tqdm
from langchain.chains.qa_generation.prompt import PROMPT_SELECTOR
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langdetect import detect, LangDetectException
from collections import Counter


def detect_question_language(text, fallback_lang):
    try:
        lang_code = detect(text)
        if lang_code in ["en", "fr", "de"]:
            return lang_code
        if lang_code == "es" and fallback_lang == "fr":
            return fallback_lang
        return fallback_lang
    except LangDetectException:
        return fallback_lang


def initialize_qa_chain():
    llm = Ollama(model="llama3.3", temperature=0.3, num_predict=512)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    qa_generation_prompt = PROMPT_SELECTOR.get_prompt(llm)
    qa_chain_per_doc = (
        RunnableLambda(lambda doc: {"text": doc.page_content})
        | qa_generation_prompt
        | llm
        | JsonOutputParser()
    )
    split_and_map_chain = (
        RunnableLambda(lambda text: text_splitter.create_documents([text]))
        | qa_chain_per_doc.map()
    )

    chain = RunnableParallel(text=RunnablePassthrough(), questions=split_and_map_chain)

    return chain


def generate_benchmark():
    benchmark = []

    language_config = {
        "en": {"sample_size": 100, "source_prefixes": ["eb_en", "wiki_en"]},
        "fr": {"sample_size": 50, "source_prefixes": ["diderot_fr"]},
        "de": {"sample_size": 50, "source_prefixes": ["ger_de"]},
    }

    with open(
        "cross_dataset_discovery/assets/language/collection/language_documents.json",
        "r",
        encoding="utf-8",
    ) as f:
        documents = json.load(f)

    print("Initializing QA generation chain...")
    qa_chain = initialize_qa_chain()
    print("QA chain initialized.")

    total_qas_generated = 0

    for lang, config in language_config.items():
        print(f"\nProcessing {lang} documents...")

        lang_docs = [
            doc
            for doc in documents
            if doc.get("language") == lang
            and any(
                doc.get("id", "").startswith(prefix)
                for prefix in config["source_prefixes"]
            )
        ]
        print(f"Found {len(lang_docs)} documents for {lang} matching prefixes.")

        filtered_docs = [
            doc
            for doc in lang_docs
            if doc.get("contents") and len(doc["contents"].split()) > 50
        ]
        print(f"Filtered down to {len(filtered_docs)} documents with > 50 words.")

        sample_size = min(config["sample_size"], len(filtered_docs))
        if sample_size == 0:
            print(f"No documents to sample for {lang}.")
            continue

        print(f"Sampling {sample_size} documents for {lang}.")
        sampled_docs = random.sample(filtered_docs, sample_size)

        lang_success_count = 0
        lang_qa_count = 0
        for doc in tqdm(sampled_docs, desc=f"Generating {lang} QAs", unit="doc"):
            doc_id = doc.get("id", "unknown_id")
            context = doc.get("contents")
            doc_language = lang

            if not context:
                print(f"Warning: Document {doc_id} has no 'contents'. Skipping.")
                continue

            result = qa_chain.invoke(context)

            if "questions" in result and isinstance(result["questions"], list):
                num_qas_for_doc = 0
                for chunk_qa_list in result["questions"]:
                    if isinstance(chunk_qa_list, dict):
                        chunk_qa_list = [chunk_qa_list]

                    if isinstance(chunk_qa_list, list):
                        for q_idx_in_chunk, qa_pair in enumerate(chunk_qa_list):
                            if (
                                isinstance(qa_pair, dict)
                                and "question" in qa_pair
                                and "answer" in qa_pair
                            ):
                                question_id = (
                                    f"{doc_id}_q{lang_qa_count + num_qas_for_doc}"
                                )
                                question_text = qa_pair["question"]

                                detected_question_lang = detect_question_language(
                                    question_text, doc_language
                                )

                                print(
                                    f"Generated QA pair for {doc_id}: {question_text} -> {qa_pair['answer']}"
                                )
                                benchmark.append(
                                    {
                                        "question_id": question_id,
                                        "question": question_text,
                                        "answer": qa_pair["answer"],
                                        "language": doc_language,
                                        "question_language": detected_question_lang,
                                        "language_mismatch": detected_question_lang
                                        != doc_language,
                                        "source_document": doc_id,
                                    }
                                )
                                num_qas_for_doc += 1
                            else:
                                print(
                                    f"Warning: Invalid QA pair format received for {doc_id} (in chunk list): {qa_pair}"
                                )
                    else:
                        print(
                            f"Warning: Expected list of QAs for chunk in {doc_id}, but got: {type(chunk_qa_list)}"
                        )

                if num_qas_for_doc > 0:
                    lang_success_count += 1
                    lang_qa_count += num_qas_for_doc
            else:
                print(
                    f"Warning: No 'questions' list found or invalid format in result for {doc_id}. Result: {result}"
                )

        print(
            f"Successfully processed {lang_success_count} documents for {lang}, generating {lang_qa_count} QA pairs."
        )
        total_qas_generated += lang_qa_count

    print(f"\nTotal QA pairs generated across all languages: {total_qas_generated}")
    return benchmark


if __name__ == "__main__":
    print("Starting QA benchmark generation...")
    benchmark_data = generate_benchmark()

    if benchmark_data:
        output_filename = "cross_dataset_discovery/assets/language/benchmark.json"
        print(
            f"\nSaving benchmark with {len(benchmark_data)} QA pairs to {output_filename}..."
        )
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
            print("Benchmark saved successfully.")

            print("\nLanguage distribution of generated questions:")
            lang_counts = Counter(qa["question_language"] for qa in benchmark_data)
            print(lang_counts)

            mismatch_count = sum(1 for qa in benchmark_data if qa["language_mismatch"])
            print(
                f"\nNumber of QA pairs where question language mismatches document language: {mismatch_count}"
            )
            if mismatch_count > 0:
                print("Mismatch details (document lang -> question lang):")
                mismatch_distribution = Counter(
                    (qa["language"], qa["question_language"])
                    for qa in benchmark_data
                    if qa["language_mismatch"]
                )
                for (doc_lang, q_lang), count in mismatch_distribution.items():
                    print(f"  {doc_lang} -> {q_lang}: {count}")

        except IOError as e:
            print(f"Error saving benchmark file: {str(e)}")
    else:
        print("\nNo benchmark data was generated.")

    print("\nBenchmark generation process finished.")
