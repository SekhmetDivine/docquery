"""Loaders read raw documents from disk and return their plain text content."""

from abc import ABC, abstractmethod
from pathlib import Path


class Document:
    """A single loaded document: where it came from and its raw text."""

    def __init__(self, source_path: Path, text: str) -> None:
        self.source_path = source_path
        self.text = text


class Loader(ABC):
    """Base contract for anything that can read a file and produce a Document."""

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Return True if this loader knows how to read the given file."""

    @abstractmethod
    def load(self, path: Path) -> Document:
        """Read the file at path and return its content as a Document."""


class TextLoader(Loader):
    """Loads plain .txt files."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".txt"

    def load(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8")
        return Document(source_path=path, text=text)


class MarkdownLoader(Loader):
    """Loads .md files. Markdown is read as plain text — no special parsing."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".md"

    def load(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8")
        return Document(source_path=path, text=text)
