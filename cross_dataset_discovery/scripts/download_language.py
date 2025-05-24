import os
import subprocess
import glob
import re
import json
import tarfile
import requests
from lxml import etree
from tqdm import tqdm

BASE_DOWNLOAD_DIR = "cross_dataset_discovery/assets/language"
OUTPUT_COLLECTION_DIR = "cross_dataset_discovery/assets/language/collection"
UNIFIED_JSON_FILENAME = os.path.join(OUTPUT_COLLECTION_DIR, "language_documents.json")

GERMAN_DATA_URL = (
    "https://zenodo.org/records/4554112/files/encyclopedias_transformed.tar.gz"
)
GERMAN_DATA_ARCHIVE_NAME = "encyclopedias_transformed.tar.gz"
GERMAN_DATA_EXTRACT_DIR = os.path.join(BASE_DOWNLOAD_DIR, "german")
GERMAN_ENCYC_PATH_PATTERN = os.path.join(
    GERMAN_DATA_EXTRACT_DIR, "encyclopedias_transformed", "*.xml"
)

ENGLISH_DATA_REPO_URL = "https://github.com/TU-plogan/kp-editions.git"
ENGLISH_DATA_REPO_DIR = os.path.join(BASE_DOWNLOAD_DIR, "kp-editions")
EB09_PATH_PATTERN = os.path.join(ENGLISH_DATA_REPO_DIR, "eb09", "XML_v1", "*", "*")
EB07_PATH_PATTERN = os.path.join(ENGLISH_DATA_REPO_DIR, "eb07", "XML_v3", "*", "*")

DIDEROT_JSON_URL = (
    "https://github.com/ThoraHagen/languagedatagems/releases/download/v1.0/diderot.json"
)
WIKI_JSON_URL = (
    "https://github.com/ThoraHagen/languagedatagems/releases/download/v1.0/wiki.json"
)

DIDEROT_DATA_DIR = os.path.join(BASE_DOWNLOAD_DIR, "diderot")
WIKIPEDIA_DATA_DIR = os.path.join(BASE_DOWNLOAD_DIR, "wikipedia")

DIDEROT_JSON_PATH = os.path.join(DIDEROT_DATA_DIR, "diderot.json")
WIKI_JSON_PATH = os.path.join(WIKIPEDIA_DATA_DIR, "wiki.json")


def download_file(url, dest_folder, filename=None):
    os.makedirs(dest_folder, exist_ok=True)
    if not filename:
        filename = url.split("/")[-1]
    dest_path = os.path.join(dest_folder, filename)

    if os.path.exists(dest_path):
        print(f"File {dest_path} already exists. Skipping download.")
        return dest_path

    print(f"Downloading {url} to {dest_path}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in tqdm(
                r.iter_content(chunk_size=8192), desc=f"Downloading {filename}"
            ):
                f.write(chunk)
    print(f"Successfully downloaded {filename}.")
    return dest_path


def download_and_extract_tar_gz(url, extract_to_dir, archive_name):
    os.makedirs(extract_to_dir, exist_ok=True)
    current_archive_path = os.path.join(extract_to_dir, archive_name)

    expected_extracted_folder_name = archive_name.replace(".tar.gz", "")
    target_extracted_path = os.path.join(extract_to_dir, expected_extracted_folder_name)

    if os.path.exists(target_extracted_path) and os.path.isdir(target_extracted_path):
        print(
            f"Data in {target_extracted_path} seems to be already extracted. Skipping download and extraction."
        )
        return True

    if not os.path.exists(current_archive_path):
        current_archive_path = download_file(url, extract_to_dir, archive_name)
        if (
            not current_archive_path
        ):  # If download_file could return a falsy value on failure
            return False
    else:
        print(f"Archive {current_archive_path} already exists.")

    print(f"Extracting {current_archive_path} to {extract_to_dir}...")
    with tarfile.open(current_archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to_dir)
    print(f"Successfully extracted {archive_name}.")
    return True


def clone_git_repo(repo_url, dest_dir):
    if os.path.exists(dest_dir):
        print(
            f"Directory {dest_dir} already exists. Assuming repo is cloned. Skipping clone."
        )
        return True

    print(f"Cloning {repo_url} into {dest_dir}...")
    subprocess.run(
        ["git", "clone", repo_url, dest_dir], check=True, capture_output=True
    )
    print(f"Successfully cloned {repo_url}.")
    return True


def entry_gen(path):
    NS = {"tei": "http://www.tei-c.org/ns/1.0"}
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(path, parser)
    entries = tree.xpath("//tei:entry[not(contains(@xml:id, '-app'))]", namespaces=NS)
    for e in entries:
        ID = e.xpath("@xml:id", namespaces=NS)[0]
        headword_elements = e.xpath("./tei:form/tei:term/text()", namespaces=NS)
        headword = headword_elements[0] if headword_elements else "N/A"
        t = e.xpath(".//tei:sense//text()", namespaces=NS)
        t = "".join(t)
        t = re.sub("\n +|\n", " ", t)
        t = re.sub("\x96", "—", t)
        yield (ID, headword, t)


def get_german_documents(path_pattern):
    documents = []
    year_list = []
    german_files = glob.glob(path_pattern)
    if not german_files:
        print(f"Warning: No files found for pattern {path_pattern}")
        return [], []

    print(f"Processing German documents from {len(german_files)} files...")
    for file in tqdm(german_files, desc="Processing German XMLs"):
        year_match = re.findall(r"-(\d{4})", file)
        if not year_match:
            print(
                f"Warning: Could not extract year from filename {file}, skipping this file."
            )
            continue
        year = year_match[0]

        generator = entry_gen(file)
        for entry_id, headword, text in generator:
            documents.append(text)
            year_list.append(year)
    print(f"Loaded {len(documents)} German documents.")
    return documents, year_list


def get_french_documents(path):
    if not os.path.exists(path):
        print(f"Error: Diderot JSON file not found at {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    documents = [v for k, v in loaded_data.items()]
    print(f"Loaded {len(documents)} French documents.")
    return documents


def get_wiki_documents(path, languages=["de", "fr", "en"]):
    if not os.path.exists(path):
        print(f"Error: Wikipedia JSON file not found at {path}")
        return []
    result = []
    with open(path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    documents_data = [
        v["languages"]
        for k, v in loaded_data.items()
        if "languages" in v and isinstance(v["languages"], dict)
    ]
    for doc in documents_data:
        if "de" in languages and doc.get("de"):
            result.append(doc["de"])
        if "en" in languages and doc.get("en"):
            result.append(doc["en"])
        if "fr" in languages and doc.get("fr"):
            result.append(doc["fr"])
    print(
        f"Loaded {len(result)} Wikipedia documents for languages: {', '.join(languages)}."
    )
    return result


def extract_body_text(file_path):
    tree = etree.parse(file_path)
    body = tree.find(".//TEI:body", namespaces={"TEI": "http://www.tei-c.org/ns/1.0"})
    if body is not None:
        text_content = " ".join(body.itertext())
        return text_content.strip()
    else:
        return None


def extract_from_date(file_path):
    tree = etree.parse(file_path)
    date_element = tree.find(
        ".//TEI:publicationStmt/TEI:date[@from]",
        namespaces={"TEI": "http://www.tei-c.org/ns/1.0"},
    )
    if date_element is not None:
        return date_element.get("from")
    else:
        return None


def get_english_documents(path_pattern):
    entries = []
    dates = []
    paths = glob.glob(path_pattern)
    if not paths:
        print(f"Warning: No files found for pattern {path_pattern}")
        return [], []

    print(
        f"Processing English documents from {len(paths)} files using pattern {path_pattern}..."
    )

    desc_detail_path_base = os.path.dirname(os.path.dirname(path_pattern))
    desc_detail = os.path.basename(desc_detail_path_base)

    for p in tqdm(paths, desc=f"Processing English XMLs ({desc_detail})"):
        doc = extract_body_text(p)
        if doc:
            doc = re.sub(r"\n+|\t+| +", " ", doc)
            entries.append(doc)
            date_from = extract_from_date(p)
            dates.append(int(date_from) if date_from and date_from.isdigit() else 0)
    print(f"Loaded {len(entries)} English documents from {path_pattern}.")
    return entries, dates


def clean_docs(docs):
    if not isinstance(docs, list):
        raise ValueError("Input documents must be a list")
    cleaned_docs = [str(doc).strip() for doc in docs if doc and str(doc).strip()]
    cleaned_docs = [re.sub("\n+|\t+", " ", text) for text in cleaned_docs]
    if (
        not docs and not cleaned_docs
    ):  # if input was empty, cleaned is empty, that's fine
        return []
    if not cleaned_docs and any(
        str(d).strip() for d in docs if d
    ):  # If input had content but cleaned is empty
        raise ValueError("No valid documents found after cleaning!")
    return cleaned_docs


def create_unified_dataset(en_docs_all, wiki_docs_en, fr_docs, ger_docs):
    documents = []

    for idx, doc in tqdm(
        enumerate(en_docs_all), total=len(en_docs_all), desc="Unifying Britannica (EN)"
    ):
        documents.append(
            {
                "id": f"eb_en_{idx}",
                "contents": doc,
                "language": "en",
                "source": "britannica",
            }
        )

    for idx, doc in tqdm(
        enumerate(wiki_docs_en), total=len(wiki_docs_en), desc="Unifying Wikipedia (EN)"
    ):
        documents.append(
            {
                "id": f"wiki_en_{idx}",
                "contents": doc,
                "language": "en",
                "source": "wikipedia",
            }
        )

    for idx, doc in tqdm(
        enumerate(fr_docs), total=len(fr_docs), desc="Unifying Diderot (FR)"
    ):
        documents.append(
            {
                "id": f"diderot_fr_{idx}",
                "contents": doc,
                "language": "fr",
                "source": "diderot",
            }
        )

    for idx, doc in tqdm(
        enumerate(ger_docs),
        total=len(ger_docs),
        desc="Unifying German Encyclopedias (DE)",
    ):
        documents.append(
            {
                "id": f"ger_de_{idx}",
                "contents": doc,
                "language": "de",
                "source": "german_encyclopedia",
            }
        )

    return documents


def save_json(documents, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved {len(documents)} documents to {filename}")


if __name__ == "__main__":
    print("--- Starting Data Download and Processing Script ---")

    os.makedirs(BASE_DOWNLOAD_DIR, exist_ok=True)
    print(
        f"All downloads will be stored in subdirectories of: {os.path.abspath(BASE_DOWNLOAD_DIR)}"
    )

    print("\n--- Downloading German Data ---")
    if not download_and_extract_tar_gz(
        GERMAN_DATA_URL, GERMAN_DATA_EXTRACT_DIR, GERMAN_DATA_ARCHIVE_NAME
    ):
        print("Failed to download or extract German data. Exiting.")
        exit(1)

    print("\n--- Downloading English Data (kp-editions) ---")
    if not clone_git_repo(ENGLISH_DATA_REPO_URL, ENGLISH_DATA_REPO_DIR):
        print("Failed to clone English data repository. Exiting.")
        exit(1)

    print("\n--- Downloading Diderot and Wikipedia JSONs ---")
    diderot_download_path = download_file(DIDEROT_JSON_URL, DIDEROT_DATA_DIR)
    if not diderot_download_path:
        print("Failed to download Diderot JSON. Exiting.")
        exit(1)
    wiki_download_path = download_file(WIKI_JSON_URL, WIKIPEDIA_DATA_DIR)
    if not wiki_download_path:
        print("Failed to download Wikipedia JSON. Exiting.")
        exit(1)

    print("\n--- All downloads complete. Starting data processing. ---")

    en_docs1, _ = get_english_documents(EB07_PATH_PATTERN)
    en_docs2, _ = get_english_documents(EB09_PATH_PATTERN)

    raw_en_docs = en_docs1 + en_docs2
    if not raw_en_docs:
        print(
            "Warning: No English documents loaded from Britannica, final JSON might be missing them."
        )
        en_docs_all = []
    else:
        en_docs_all = clean_docs(raw_en_docs)

    raw_wiki_docs_en = get_wiki_documents(WIKI_JSON_PATH, languages=["en"])
    if not raw_wiki_docs_en:
        print(
            "Warning: No English documents loaded from Wikipedia, final JSON might be missing them."
        )
        wiki_docs_en = []
    else:
        wiki_docs_en = clean_docs(raw_wiki_docs_en)

    raw_fr_docs = get_french_documents(DIDEROT_JSON_PATH)
    if not raw_fr_docs:
        print(
            "Warning: No French documents loaded from Diderot, final JSON might be missing them."
        )
        fr_docs = []
    else:
        fr_docs = clean_docs(raw_fr_docs)

    raw_ger_docs, _ = get_german_documents(GERMAN_ENCYC_PATH_PATTERN)
    if not raw_ger_docs:
        print("Warning: No German documents loaded, final JSON might be missing them.")
        ger_docs = []
    else:
        ger_docs = clean_docs(raw_ger_docs)

    print("\n--- Creating Unified Dataset ---")
    unified_docs = create_unified_dataset(
        en_docs_all=en_docs_all,
        wiki_docs_en=wiki_docs_en,
        fr_docs=fr_docs,
        ger_docs=ger_docs,
    )

    print("\n--- Saving Unified Dataset ---")
    os.makedirs(OUTPUT_COLLECTION_DIR, exist_ok=True)
    save_json(unified_docs, UNIFIED_JSON_FILENAME)

    print("\n--- Script Finished ---")
    print(
        f"Created dataset with {len(unified_docs)} documents at {os.path.abspath(UNIFIED_JSON_FILENAME)}"
    )
