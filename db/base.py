from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any
import numpy as np


@dataclass
class FaceRecord:
    id: Any
    name: str
    embedding: np.ndarray
    created_at: str
    metadata: dict | None = None


class BaseFaceRepository(ABC):
    """Abstract Base Class for face storage backends (SQLite, MongoDB, etc.)."""

    @abstractmethod
    def add_face(self, name: str, embedding: np.ndarray, metadata: dict | None = None) -> Any:
        """Saves a new face name and its 128D embedding vector."""
        pass

    @abstractmethod
    def get_all_faces(self) -> list[FaceRecord]:
        """Retrieves all registered face records."""
        pass

    @abstractmethod
    def get_faces_by_name(self, name: str) -> list[FaceRecord]:
        """Retrieves face records matching a given name."""
        pass

    @abstractmethod
    def delete_face(self, name: str) -> int:
        """Deletes face record(s) by name. Returns the number of records removed."""
        pass

    @abstractmethod
    def delete_by_id(self, record_id: Any) -> bool:
        """Deletes a specific face record by its unique ID."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns total number of registered faces."""
        pass
