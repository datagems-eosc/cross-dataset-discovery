import uuid
from typing import Any, Dict

from pydantic import BaseModel, Field


def _generate_uuid_str() -> str:
    return str(uuid.uuid4())


class SearchableItem(BaseModel):
    """
    A standard representation of a single item to be indexed and retrieved.

    This is the core data object that flows into the indexing part of the pipeline.
    """

    item_id: str = Field(
        default_factory=_generate_uuid_str,
        description="A unique identifier for the item. If not provided, a UUID will be generated.",
    )
    content: str = Field(
        description="The main text content of the item that will be used for searching."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary to hold any additional metadata associated with the item.",
    )


class RetrievalResult(BaseModel):
    """
    A standard representation of a single retrieval result.

    Consists of a retrieved item (SearchableItem) and its
    associated relevance score.
    """

    item: SearchableItem
    score: float

    def __hash__(self) -> int:
        return hash(self.item.item_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RetrievalResult):
            return NotImplemented
        return self.item.item_id == other.item.item_id
