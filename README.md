# docquery

A command-line RAG (Retrieval-Augmented Generation) tool. Ask questions about your own documents and get answers grounded in their actual content, with source citations.

## How it works

1. **Ingest** — documents are loaded, split into overlapping chunks, converted into vector embeddings (locally, via `sentence-transformers`), and stored in a local ChromaDB vector database.
2. **Ask** — your question is embedded the same way, the database returns the most relevant chunks, and an LLM generates an answer using only that retrieved context.

By default the LLM is a local [Ollama](https://ollama.com) model (`llama3.2:1b`) — free, no API key, runs on a laptop CPU. A `ClaudeClient` implementation (Anthropic API) is also included behind the same `LLMClient` interface and can be swapped in by editing `cli.py`.

## Architecture

Each pipeline stage is an abstract interface with a concrete implementation, following SOLID principles — e.g. `LLMClient` is implemented by both `ClaudeClient` and `OllamaClient`, and `VectorStore` could gain a FAISS implementation without touching the rest of the code.

```
src/docquery/
├── ingestion/   # Document loaders + text chunker
├── embedding/   # Text -> vector (sentence-transformers)
├── store/       # Vector database (ChromaDB)
├── retrieval/   # Nearest-neighbor chunk lookup
├── generation/  # LLM answer generation (Ollama / Claude)
├── pipeline.py  # Orchestrates the above
└── cli.py       # Typer CLI entrypoint
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

Requires [Ollama](https://ollama.com) installed locally with a model pulled:

```powershell
ollama pull llama3.2:1b
```

## Usage

```powershell
docquery ingest path\to\your\documents
docquery ask "your question here"
docquery clear
```

## Tests

```powershell
pip install pytest
pytest
```
