"""Command-line interface for docquery."""

from pathlib import Path

import typer

from docquery.embedding.sentence_transformer import SentenceTransformerEmbedder
from docquery.generation.ollama import OllamaClient
from docquery.ingestion.chunker import Chunker
from docquery.ingestion.loader import MarkdownLoader, TextLoader
from docquery.pipeline import Pipeline
from docquery.store.chroma import ChromaVectorStore

app = typer.Typer()


def _build_pipeline() -> Pipeline:
    return Pipeline(
        loaders=[TextLoader(), MarkdownLoader()],
        chunker=Chunker(),
        embedder=SentenceTransformerEmbedder(),
        store=ChromaVectorStore(),
        llm=OllamaClient(),
    )


@app.command()
def ingest(folder: Path) -> None:
    """Load, chunk, embed, and store every supported document in FOLDER."""
    pipeline = _build_pipeline()
    count = pipeline.ingest(folder)
    typer.echo(f"Ingested {count} chunks from {folder}")


@app.command()
def ask(question: str) -> None:
    """Ask a question about the ingested documents."""
    pipeline = _build_pipeline()
    answer, sources = pipeline.ask(question)
    typer.echo(answer)
    typer.echo("\nSources:")
    for source in sources:
        typer.echo(f"  - {source}")


@app.command()
def clear() -> None:
    """Remove all ingested data."""
    pipeline = _build_pipeline()
    pipeline.clear()
    typer.echo("Cleared.")


if __name__ == "__main__":
    app()
