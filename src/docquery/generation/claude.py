"""Generates answers using the Claude API."""

import os

from anthropic import Anthropic

from docquery.generation.base import LLMClient

SYSTEM_PROMPT = (
    "You answer questions using only the provided context. "
    "If the context doesn't contain the answer, say so clearly instead of guessing."
)


class ClaudeClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set the ANTHROPIC_API_KEY environment variable.")
        self._client = Anthropic(api_key=key)
        self._model = model

    def generate(self, question: str, context: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        return message.content[0].text
