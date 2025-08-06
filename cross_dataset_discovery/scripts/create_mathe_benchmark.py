import json
import random
import os
import re
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama

TOPIC_QUESTION_PROMPT_TEMPLATE = """
Given the following text excerpt, please generate a list of 2-4 questions that a user might ask if they were searching for the main topics, concepts, methods, or procedures discussed.

Focus on questions that seek understanding or "how-to" information related to the core subjects. The questions should reflect someone trying to *find* or *learn about* the information in the text, not asking about tiny specific details already present.

Examples of desired question types:
- What are the basic concepts of [Topic]?
- How can I use [Method] to solve [Problem]?
- What is the process for [Task]?
- Explain the main ideas behind [Concept].
- What are the different types of [Item]?

AVOID questions like:
- What specific number/date/name was mentioned?
- What was the exact phrase used in paragraph 3?

Text Excerpt:
{text}

IMPORTANT: Respond ONLY with the JSON object, starting with '{{' and ending with '}}'. Do NOT include any introductory text, explanations, or markdown formatting around the JSON.

Example JSON Output:
{{"questions": ["What is linear optimization?", "How does the graphical method work for solving linear problems?", "What are constraints in linear programming?"]}}
"""


def extract_json_from_string(text: str) -> str:
    """Tries to extract the JSON part from a string, handling surrounding text."""
    start_index = text.find("{")
    end_index = text.rfind("}")

    if start_index != -1 and end_index != -1 and end_index > start_index:
        potential_json = text[start_index : end_index + 1]
        if potential_json.startswith("{") and potential_json.endswith("}"):
            return potential_json

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        potential_json = match.group(0)
        return potential_json
    return "{}"


# --- Modified Chain Initialization ---
def initialize_topic_question_chain():
    """Initializes the LangChain chain for Topic Question generation."""

    print("Initializing Topic Question generation chain...")
    llm = Ollama(model="llama3.3", temperature=0, num_predict=512)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    topic_question_prompt = PromptTemplate(
        template=TOPIC_QUESTION_PROMPT_TEMPLATE,
        input_variables=["text"],
    )

    question_chain_per_doc = (
        RunnableLambda(lambda doc: {"text": doc.page_content})
        | topic_question_prompt
        | llm
        | StrOutputParser()  # Get the raw string output first
        | RunnableLambda(extract_json_from_string)  # Clean the string
        | JsonOutputParser()  # NOW parse the cleaned string
    )

    split_and_map_chain = (
        RunnableLambda(lambda text: text_splitter.create_documents([text]))
        | question_chain_per_doc.map()
    )

    chain = RunnableParallel(topic_questions_results=split_and_map_chain)
    print("Topic Question chain initialized.")
    return chain


def generate_topic_questions_from_merged(
    input_filename="merged_simplified_data.json", sample_size=100
):
    """Generates topic-seeking questions from the merged simplified data."""
    generated_questions_list = []

    print(f"Loading documents from {input_filename}...")
    if not os.path.exists(input_filename):
        print(f"Error: Input file not found - {input_filename}")
        return []

    try:
        with open(input_filename, "r", encoding="utf-8") as f:
            documents = json.load(f)
        print(f"Loaded {len(documents)} documents.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_filename}.")
        return []
    except Exception as e:
        print(f"Error loading file {input_filename}: {e}")
        return []

    if not isinstance(documents, list):
        print(
            f"Error: Expected a JSON list in {input_filename}, but got {type(documents)}."
        )
        return []

    # Initialize QA chain
    topic_question_chain = initialize_topic_question_chain()

    # Filter out short documents
    print("Filtering documents...")
    filtered_docs = [
        doc
        for doc in documents
        if isinstance(doc, dict)
        and doc.get("contents")
        and isinstance(doc.get("contents"), str)
        and len(doc["contents"].split()) > 30
    ]
    print(f"Filtered down to {len(filtered_docs)} documents with > 30 words.")

    # Sample documents
    actual_sample_size = min(sample_size, len(filtered_docs))
    if actual_sample_size == 0:
        print("No documents to sample after filtering.")
        return []
    elif actual_sample_size < len(filtered_docs):
        print(f"Sampling {actual_sample_size} documents from the filtered list.")
        sampled_docs = random.sample(filtered_docs, actual_sample_size)
    else:
        print(f"Processing all {len(filtered_docs)} filtered documents.")
        sampled_docs = filtered_docs

    # Generate questions
    total_questions_generated = 0
    success_doc_count = 0
    global_question_counter = 0

    for doc in tqdm(sampled_docs, desc="Generating Topic Questions", unit="doc"):
        doc_id = doc.get("Source-File", "unknown_source_file")
        context = doc.get("contents")

        if not context:
            continue

        try:
            result = topic_question_chain.invoke(context)

            if "topic_questions_results" in result and isinstance(
                result["topic_questions_results"], list
            ):
                num_questions_for_doc = 0
                for chunk_result in result["topic_questions_results"]:
                    if isinstance(chunk_result, dict) and "questions" in chunk_result:
                        questions_from_chunk = chunk_result["questions"]
                        if isinstance(questions_from_chunk, list):
                            for question_text in questions_from_chunk:
                                if (
                                    isinstance(question_text, str)
                                    and question_text.strip()
                                ):
                                    question_id = f"{os.path.basename(doc_id)}_q{global_question_counter}"
                                    generated_questions_list.append(
                                        {
                                            "question_id": question_id,
                                            "question": question_text.strip(),
                                            "source_document": doc_id,
                                        }
                                    )
                                    print(question_text.strip())
                                    num_questions_for_doc += 1
                                    global_question_counter += 1
                if num_questions_for_doc > 0:
                    success_doc_count += 1
        except Exception as e:
            print(
                f"\nError processing document '{doc_id}' during chain invocation: {str(e)}"
            )
            import traceback

            traceback.print_exc()
            continue

    print(
        f"\nSuccessfully generated questions from {success_doc_count}/{actual_sample_size} sampled documents."
    )
    print(f"Total topic questions generated: {total_questions_generated}")
    return generated_questions_list


if __name__ == "__main__":
    print("Starting Topic Question generation from merged data...")

    INPUT_JSON_FILE = (
        "cross_dataset_discovery/assets/mathe/collection/mathe_documents.json"
    )
    OUTPUT_QUESTIONS_FILE = "cross_dataset_discovery/assets/mathe/benchmark.json"
    SAMPLE_SIZE = 200

    generated_questions = generate_topic_questions_from_merged(
        input_filename=INPUT_JSON_FILE, sample_size=SAMPLE_SIZE
    )

    if generated_questions:
        print(
            f"\nSaving {len(generated_questions)} topic questions to {OUTPUT_QUESTIONS_FILE}..."
        )
        with open(OUTPUT_QUESTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(generated_questions, f, ensure_ascii=False, indent=2)
        print("Topic questions saved successfully.")
    else:
        print("\nNo topic questions were generated.")

    print("Topic question generation process finished.")
