"""Splits document text into overlapping chunks suitable for embedding."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    text: str
    source_path: Path
    chunk_index: int


class Chunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, source_path: Path) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = start + self.chunk_size
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, source_path=source_path, chunk_index=index))
                index += 1
            start += self.chunk_size - self.overlap
        return chunks
