"""Local embedding provider using sentence-transformers (no API call needed)."""

from sentence_transformers import SentenceTransformer

from docquery.embedding.base import EmbeddingProvider


class SentenceTransformerEmbedder(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()
