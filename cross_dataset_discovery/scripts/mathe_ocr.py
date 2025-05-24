import os
import glob
import json


def main():
    pdf_base_dir_for_id = "cross_dataset_discovery/assets/mathe/"
    pdf_input_dir_glob_path = "cross_dataset_discovery/assets/mathe/materials/*.pdf"

    markdown_files_location_base = (
        "cross_dataset_discovery/assets/mathe/materiald_md/markdown/"
    )

    output_json_file = (
        "cross_dataset_discovery/assets/mathe/collection/mathe_documents.json"
    )

    original_pdf_file_paths = glob.glob(pdf_input_dir_glob_path)

    if not original_pdf_file_paths:
        print(
            f"No original PDF files found matching pattern: {pdf_input_dir_glob_path} to base the JSON on."
        )
        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        print(f"Empty results saved to {output_json_file}")
        return

    results = []

    for pdf_full_path in original_pdf_file_paths:
        path_component_for_md = pdf_full_path.replace(".pdf", ".md")

        actual_markdown_file_path = os.path.join(
            markdown_files_location_base, path_component_for_md
        )
        actual_markdown_file_path = os.path.normpath(actual_markdown_file_path)

        content = ""
        try:
            with open(actual_markdown_file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            print(
                f"Warning: Markdown file not found for {pdf_full_path} at {actual_markdown_file_path}. Content will be empty."
            )
        except Exception as e:
            print(
                f"Warning: Error reading markdown file {actual_markdown_file_path}: {e}. Content will be empty."
            )

        id_value_path_relative_to_pdf_base = os.path.relpath(
            pdf_full_path, pdf_base_dir_for_id
        )
        id_value = f"./{id_value_path_relative_to_pdf_base.replace(os.sep, '/')}"

        results.append({"contents": content, "id": id_value})

    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Processing complete. Results saved to {output_json_file}")


if __name__ == "__main__":
    main()
