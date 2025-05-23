from abc import ABC, abstractmethod
import os
import json
from typing import Optional, List, Dict


class BaseQueryDecomposer(ABC):
    def __init__(self, output_folder: Optional[str] = None):
        self.output_folder = output_folder
        self.cache_file: Optional[str] = None
        self.decompositions_cache: Optional[Dict[str, List[str]]] = None

        if self.output_folder:
            os.makedirs(self.output_folder, exist_ok=True)
            self.cache_file = os.path.join(self.output_folder, "decompositions.json")
            self.decompositions_cache = self._load_cache()

    def _load_cache(self) -> Dict[str, List[str]]:
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        if self.cache_file and self.decompositions_cache is not None:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.decompositions_cache, f, indent=4, ensure_ascii=False)

    @abstractmethod
    def decompose(self, nlq: str) -> List[str]:
        pass

    def get_cached_decompositions(self, nlq: str) -> Optional[List[str]]:
        if self.decompositions_cache is None:
            return None
        return self.decompositions_cache.get(nlq)

    def decompose_batch(self, nlqs: List[str]) -> List[List[str]]:
        return [self.decompose(nlq) for nlq in nlqs]
