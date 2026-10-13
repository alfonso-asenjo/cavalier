"""Clientes de LLM. Interfaz mínima: devolver JSON que siga un esquema.

`OpenAICompatibleClient` sirve para Ollama, llama.cpp server, LM Studio y vLLM
(todos exponen /v1/chat/completions con `response_format` de tipo json_schema).
La decodificación restringida por esquema es lo que hace viables los modelos pequeños.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
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
        base_url = base_url.rstrip("/")
        self._client = OpenAI(base_url=base_url if base_url.endswith("/v1") else base_url + "/v1", api_key=api_key)

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


class OllamaClient:
    """API nativa de Ollama (/api/chat). Frente a la compatible con OpenAI permite desactivar el
    modo "thinking" (en una GPU de 8 GB, generar es lo caro), fijar el contexto y medir tokens.
    Cada llamada se registra en `metrics` (latencia, tokens de entrada y salida, velocidad)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        seed: int | None = None,
        num_ctx: int = 8192,
        max_tokens: int = 1200,
        think: bool = False,
        timeout: float = 600.0,
    ):
        self.model_id = model
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.think = think
        self.timeout = timeout
        self.metrics: list[dict] = []

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(f"{self.base_url}/api/chat", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        options = {"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": self.max_tokens}
        if self.seed is not None:
            options["seed"] = self.seed
        body = {
            "model": self.model_id, "stream": False, "format": schema, "think": self.think, "options": options,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        start = time.perf_counter()
        try:
            data = self._post(body)
        except urllib.error.HTTPError as exc:  # modelos sin modo "thinking" pueden rechazar el parámetro
            if exc.code == 400 and "think" in exc.read().decode("utf-8", "ignore").lower():
                body.pop("think")
                data = self._post(body)
            else:
                raise
        elapsed = time.perf_counter() - start
        eval_s = data.get("eval_duration", 0) / 1e9
        self.metrics.append({
            "latency_s": elapsed,
            "load_s": data.get("load_duration", 0) / 1e9,
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "tokens_per_s": data.get("eval_count", 0) / eval_s if eval_s else 0.0,
            "done_reason": data.get("done_reason"),
        })
        return data.get("message", {}).get("content", "")


class FakeLLMClient:
    """Cliente de pruebas: devuelve respuestas programadas y registra los prompts."""

    def __init__(self, responder: Callable[[str, str], str] | list[str], model_id: str = "fake"):
        self.model_id = model_id
        self._responder = responder
        self.calls: list[tuple[str, str]] = []
        self.schemas: list[dict] = []

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        self.calls.append((system, user))
        self.schemas.append(schema)
        if callable(self._responder):
            return self._responder(system, user)
        return self._responder[min(len(self.calls), len(self._responder)) - 1]
