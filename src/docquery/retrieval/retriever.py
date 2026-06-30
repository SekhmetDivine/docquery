"""Finds the chunks most relevant to a question."""

from docquery.embedding.base import EmbeddingProvider
from docquery.store.base import StoredChunk, VectorStore


class Retriever:
    def __init__(self, embedder: EmbeddingProvider, store: VectorStore, top_k: int = 4) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k

    def retrieve(self, question: str) -> list[StoredChunk]:
        [question_embedding] = self._embedder.embed([question])
        return self._store.query(question_embedding, top_k=self._top_k)
