"""Contract for an LLM that can generate an answer given a question and context."""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        """Return an answer to the question, grounded in the given context."""
