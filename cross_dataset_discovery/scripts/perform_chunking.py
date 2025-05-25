from cross_dataset_discovery.src.utils.chonky_chunker import ChonkyChunker
import sys
import json
from tqdm import tqdm


def main():
    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]

    chunker_instance = ChonkyChunker("mirth/chonky_modernbert_large_1")

    with open(input_json_path, "r", encoding="utf-8") as f_in:
        input_data = json.load(f_in)

    output_records = []
    global_chunk_id = 0

    for item in tqdm(input_data, desc="Processing records", unit="record"):
        content_to_chunk = item["contents"]
        generated_chunks = chunker_instance.chunk(content_to_chunk)

        for chunk_text in generated_chunks:
            new_record = item.copy()
            new_record["contents"] = chunk_text
            new_record["chunk_id"] = global_chunk_id
            output_records.append(new_record)
            global_chunk_id += 1

    with open(output_json_path, "w", encoding="utf-8") as f_out:
        for record in output_records:
            f_out.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
