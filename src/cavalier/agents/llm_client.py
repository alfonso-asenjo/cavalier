"""Clientes de LLM. Interfaz mínima: devolver JSON que siga un esquema.

`OpenAICompatibleClient` sirve para Ollama, llama.cpp server, LM Studio y vLLM
(todos exponen /v1/chat/completions con `response_format` de tipo json_schema).
La decodificación restringida por esquema es lo que hace viables los modelos pequeños.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class LLMClient(Protocol):
    model_id: str

    def complete_json(self, system: str, user: str, schema: dict) -> str: ...


class OpenAICompatibleClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "local",
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int = 1200,
    ):
        from openai import OpenAI

        self.model_id = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=self.max_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": "decision", "schema": schema}},
        )
        return response.choices[0].message.content or ""


class FakeLLMClient:
    """Cliente de pruebas: devuelve respuestas programadas y registra los prompts."""

    def __init__(self, responder: Callable[[str, str], str] | list[str], model_id: str = "fake"):
        self.model_id = model_id
        self._responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        self.calls.append((system, user))
        if callable(self._responder):
            return self._responder(system, user)
        return self._responder[min(len(self.calls), len(self._responder)) - 1]
