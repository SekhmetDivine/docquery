from pathlib import Path

import pytest

from docquery.ingestion.chunker import Chunker


def test_split_produces_overlapping_chunks():
    chunker = Chunker(chunk_size=10, overlap=2)
    chunks = chunker.split("a" * 25, Path("test.txt"))
    assert len(chunks) > 1
    assert all(c.source_path == Path("test.txt") for c in chunks)


def test_split_empty_text_returns_no_chunks():
    chunker = Chunker(chunk_size=10, overlap=2)
    assert chunker.split("   ", Path("test.txt")) == []


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        Chunker(chunk_size=10, overlap=10)
