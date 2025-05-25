import json


def find_eb_en_0():
    input_path = "/data/hdd1/users/akouk/ARM/ARM/assets/datagems/language_documents_semchunk.jsonl"
    target_id = "eb_en_96"

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # skip invalid JSON
                continue

            if obj.get("id") == target_id:
                print(f"[Line {line_num}] {line}\n")


if __name__ == "__main__":
    find_eb_en_0()
