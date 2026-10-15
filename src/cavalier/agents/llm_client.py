"""Clientes de LLM. Interfaz mínima: devolver JSON que siga un esquema.

`OpenAICompatibleClient` sirve para Ollama, llama.cpp server, LM Studio y vLLM
(todos exponen /v1/chat/completions con `response_format` de tipo json_schema).
La decodificación restringida por esquema es lo que hace viables los modelos pequeños.
`OpenAIClient` y `AnthropicClient` usan las API comerciales (clave en OPENAI_API_KEY y
ANTHROPIC_API_KEY); `make_client` elige el cliente por nombre de backend.
"""

from __future__ import annotations

import copy
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


# Palabras clave que las salidas estructuradas de las API comerciales no admiten (o no admiten en
# modo estricto). Se quitan del esquema enviado; LLMAgent ya valida y recorta en el cliente.
_UNSUPPORTED = {"default", "title", "minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems"}


def strict_schema(schema: dict) -> dict:
    """Esquema apto para el modo estricto de OpenAI y para Anthropic: todo objeto cierra sus
    propiedades (`additionalProperties: false`) y las declara todas obligatorias; las opcionales
    ya admiten `null`, así que el modelo puede dejarlas vacías."""
    def walk(node):
        if isinstance(node, dict):
            node = {k: walk(v) for k, v in node.items() if k not in _UNSUPPORTED}
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            return node
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(copy.deepcopy(schema))


class _Metered:
    """Registro de llamadas en `metrics`, con los mismos campos que OllamaClient."""

    metrics: list[dict]

    def _record(self, start: float, prompt_tokens: int, output_tokens: int, done_reason: str | None) -> None:
        elapsed = time.perf_counter() - start
        self.metrics.append({
            "latency_s": elapsed, "load_s": 0.0, "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
            "tokens_per_s": output_tokens / elapsed if elapsed else 0.0, "done_reason": done_reason,
        })


class OpenAIClient(_Metered):
    """API de OpenAI (modelos GPT-5.x, que razonan antes de responder). No admiten `temperature`;
    el razonamiento se cobra como salida, así que el esfuerzo por defecto es bajo."""

    def __init__(self, model: str = "gpt-5.6-terra", seed: int | None = None, effort: str = "low",
                 max_tokens: int = 16000):
        from openai import OpenAI

        self.model_id = model
        self.seed = seed
        self.effort = effort
        self.max_tokens = max_tokens
        self.metrics: list[dict] = []
        self._client = OpenAI()

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            seed=self.seed,
            reasoning_effort=self.effort,
            max_completion_tokens=self.max_tokens,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "decision", "strict": True, "schema": strict_schema(schema)}},
        )
        choice = response.choices[0]
        usage = response.usage
        self._record(start, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0,
                     choice.finish_reason)
        return choice.message.content or ""


class AnthropicClient(_Metered):
    """API de Anthropic (modelos Claude). Los modelos actuales no admiten `temperature` ni `seed`
    y razonan siempre; el esfuerzo se controla con `output_config.effort` (bajo por defecto).
    No se activan los modelos de respaldo ante un rechazo: cambiarían el modelo a mitad de partida.
    Un rechazo devuelve texto vacío, que LLMAgent trata como respuesta inválida."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", max_tokens: int = 16000):
        import anthropic

        self.model_id = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.metrics: list[dict] = []
        self._client = anthropic.Anthropic()

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        start = time.perf_counter()
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": strict_schema(schema)}},
        )
        self._record(start, response.usage.input_tokens, response.usage.output_tokens, response.stop_reason)
        if response.stop_reason == "refusal":
            return ""
        return next((b.text for b in response.content if b.type == "text"), "")


BACKENDS = ("ollama", "compat", "openai", "anthropic")
DEFAULT_MODELS = {"ollama": "qwen3.5:9b", "compat": "qwen3.5:9b", "openai": "gpt-5.6-terra",
                  "anthropic": "claude-opus-5-5"}


def make_client(backend: str, model: str | None = None, *, base_url: str = "http://localhost:11434",
                temperature: float = 0.7, seed: int | None = None, num_ctx: int = 8192,
                effort: str = "low") -> LLMClient:
    """ollama: API nativa local · compat: servidor compatible con OpenAI (llama.cpp, LM Studio, vLLM)
    · openai / anthropic: API comerciales (ahí no se aplican la temperatura ni el contexto, pero sí
    `effort`, el esfuerzo de razonamiento: el razonamiento se cobra como salida)."""
    model = model or DEFAULT_MODELS.get(backend)
    if backend == "ollama":
        return OllamaClient(model, base_url=base_url, temperature=temperature, seed=seed, num_ctx=num_ctx)
    if backend == "compat":
        return OpenAICompatibleClient(model, base_url=base_url, temperature=temperature, seed=seed)
    if backend == "openai":
        return OpenAIClient(model, seed=seed, effort=effort)
    if backend == "anthropic":
        return AnthropicClient(model, effort=effort)
    raise ValueError(f"Backend desconocido: {backend!r} (usa {', '.join(BACKENDS)})")


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

    def _post(self, body: dict, endpoint: str = "chat") -> dict:
        req = urllib.request.Request(f"{self.base_url}/api/{endpoint}", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)

    def unload(self) -> None:
        """Saca el modelo de la GPU (p. ej. antes de sintetizar voces con XTTS)."""
        self._post({"model": self.model_id, "keep_alive": 0}, endpoint="generate")

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
