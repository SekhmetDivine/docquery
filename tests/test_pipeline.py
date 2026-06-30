from pathlib import Path

from docquery.embedding.base import EmbeddingProvider
from docquery.generation.base import LLMClient
from docquery.ingestion.chunker import Chunker
from docquery.ingestion.loader import Document, Loader
from docquery.pipeline import Pipeline
from docquery.store.base import StoredChunk, VectorStore


class FakeLoader(Loader):
    def supports(self, path: Path) -> bool:
        return path.suffix == ".txt"

    def load(self, path: Path) -> Document:
        return Document(source_path=path, text="The sky is blue. Water is wet.")


class FakeEmbedder(EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]


class FakeStore(VectorStore):
    def __init__(self) -> None:
        self.data: list[StoredChunk] = []

    def add(self, ids, texts, embeddings, metadatas) -> None:
        for id_, text, meta in zip(ids, texts, metadatas):
            self.data.append(StoredChunk(id=id_, text=text, metadata=meta))

    def query(self, embedding, top_k):
        return self.data[:top_k]

    def clear(self) -> None:
        self.data = []


class FakeLLM(LLMClient):
    def generate(self, question: str, context: str) -> str:
        return f"answer using: {context}"


def make_pipeline(tmp_path: Path) -> tuple[Pipeline, Path]:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("The sky is blue. Water is wet.")
    pipeline = Pipeline(
        loaders=[FakeLoader()],
        chunker=Chunker(chunk_size=100, overlap=10),
        embedder=FakeEmbedder(),
        store=FakeStore(),
        llm=FakeLLM(),
    )
    return pipeline, folder


def test_ingest_returns_chunk_count(tmp_path):
    pipeline, folder = make_pipeline(tmp_path)
    count = pipeline.ingest(folder)
    assert count == 1


def test_ask_returns_answer_and_sources(tmp_path):
    pipeline, folder = make_pipeline(tmp_path)
    pipeline.ingest(folder)
    answer, sources = pipeline.ask("What color is the sky?")
    assert "blue" in answer
    assert str(folder / "a.txt") in sources


def test_clear_removes_all_data(tmp_path):
    pipeline, folder = make_pipeline(tmp_path)
    pipeline.ingest(folder)
    pipeline.clear()
    answer, sources = pipeline.ask("anything")
    assert sources == []
