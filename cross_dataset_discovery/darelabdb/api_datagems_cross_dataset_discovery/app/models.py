from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=5, gt=0, le=100)
    dataset_ids: Optional[List[str]] = Field(
        default=None,
        description="A list of dataset identifiers (UUIDs) to restrict the search to.",
    )

    @field_validator("dataset_ids")
    @classmethod
    def check_dataset_ids_not_empty(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None and not v:
            raise ValueError("dataset_ids cannot be an empty list.")
        return v


class API_SearchResult(BaseModel):
    content: str
    dataset_id: str = Field(validation_alias="source")
    object_id: str = Field(validation_alias="source_id")
    similarity: float = Field(validation_alias="score")


class SearchResponse(BaseModel):
    query_time: float
    results: List[API_SearchResult]
