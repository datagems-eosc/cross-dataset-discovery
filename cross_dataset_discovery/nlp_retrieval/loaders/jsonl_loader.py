import json
from typing import List, Optional

from nlp_retrieval.core.models import SearchableItem
from nlp_retrieval.loaders.loader_abc import BaseLoader
from tqdm import tqdm


class JsonlLoader(BaseLoader):
    """
    Loads data from a JSON Lines (.jsonl) file.

    Each line in the file is expected to be a valid JSON object. This loader
    extracts a specified content field and a selection of metadata fields
    to create a list of `SearchableItem` objects.
    """

    def __init__(
        self,
        file_path: str,
        content_field: str,
        metadata_fields: Optional[List[str]] = None,
        item_id_field: Optional[str] = None,
    ):
        """
        Initializes the JsonlLoader.

        Args:
            file_path: The path to the .jsonl file.
            content_field: The key in the JSON object to be used as the main `content`.
            metadata_fields: A list of keys to be included in the `metadata` dictionary.
                             If None, all fields other than the content and ID fields
                             will be included.
            item_id_field: An optional key in the JSON object to use as the `item_id`.
                           If None, a new UUID will be generated for each item.
        """
        self.file_path = file_path
        self.content_field = content_field
        self.metadata_fields = metadata_fields
        self.item_id_field = item_id_field

    def load(self) -> List[SearchableItem]:
        """
        Reads the .jsonl file and converts each line into a `SearchableItem`.

        A progress bar will be displayed. Lines that are not valid JSON or
        are missing the specified `content_field` will be skipped.

        Returns:
            A list of `SearchableItem` objects.
        """
        items: List[SearchableItem] = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading from {self.file_path}"):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue

                content = data.pop(self.content_field, None)
                if not content or not isinstance(content, str):
                    continue

                item_id = (
                    data.pop(self.item_id_field, None) if self.item_id_field else None
                )

                metadata: dict
                if self.metadata_fields is None:
                    # If no specific fields are requested, use all remaining data as metadata
                    metadata = data
                else:
                    metadata = {
                        key: data[key] for key in self.metadata_fields if key in data
                    }

                items.append(
                    SearchableItem(
                        item_id=item_id,  # Pydantic will generate a UUID if item_id is None
                        content=content,
                        metadata=metadata,
                    )
                )
        return items
