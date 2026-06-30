"""Orchestrates ingestion and question-answering using the components below it."""

from pathlib import Path

from docquery.embedding.base import EmbeddingProvider
from docquery.generation.base import LLMClient
from docquery.ingestion.chunker import Chunker
from docquery.ingestion.loader import Loader
from docquery.retrieval.retriever import Retriever
from docquery.store.base import VectorStore


class Pipeline:
    def __init__(
        self,
        loaders: list[Loader],
        chunker: Chunker,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LLMClient,
        top_k: int = 4,
    ) -> None:
        self._loaders = loaders
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._retriever = Retriever(embedder, store, top_k=top_k)

    def ingest(self, folder: Path) -> int:
        """Load, chunk, embed, and store every supported file in folder. Returns chunk count."""
        chunk_count = 0
        for file_path in sorted(folder.rglob("*")):
            if not file_path.is_file():
                continue
            loader = next((l for l in self._loaders if l.supports(file_path)), None)
            if loader is None:
                continue
            document = loader.load(file_path)
            chunks = self._chunker.split(document.text, file_path)
            if not chunks:
                continue
            ids = [f"{file_path.name}-{c.chunk_index}" for c in chunks]
            texts = [c.text for c in chunks]
            metadatas = [{"source": str(c.source_path), "chunk_index": c.chunk_index} for c in chunks]
            embeddings = self._embedder.embed(texts)
            self._store.add(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
            chunk_count += len(chunks)
        return chunk_count

    def ask(self, question: str) -> tuple[str, list[str]]:
        """Retrieve relevant chunks, ask the LLM, and return (answer, sources)."""
        results = self._retriever.retrieve(question)
        context = "\n\n".join(r.text for r in results)
        answer = self._llm.generate(question, context)
        sources = sorted({r.metadata["source"] for r in results})
        return answer, sources

    def clear(self) -> None:
        self._store.clear()
