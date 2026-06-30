"""Contract for a vector database: store embeddings, then query for nearest neighbors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StoredChunk:
    id: str
    text: str
    metadata: dict


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
        """Persist chunks and their embeddings."""

    @abstractmethod
    def query(self, embedding: list[float], top_k: int) -> list[StoredChunk]:
        """Return the top_k chunks whose embeddings are closest to the given embedding."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored data."""
