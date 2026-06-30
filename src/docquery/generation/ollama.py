"""Generates answers using a local Ollama model (free, no API key needed)."""

import requests

from docquery.generation.base import LLMClient

SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    "If the context doesn't contain the answer, say so clearly instead of guessing."
)


class OllamaClient(LLMClient):
    def __init__(self, model: str = "llama3.2:1b", host: str = "http://localhost:11434") -> None:
        self._model = model
        self._host = host

    def generate(self, question: str, context: str) -> str:
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
        response = requests.post(
            f"{self._host}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]
