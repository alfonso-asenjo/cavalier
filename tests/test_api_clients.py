import json
from types import SimpleNamespace

from cavalier.agents.llm import LLMAgent
from cavalier.agents.llm_client import AnthropicClient, FakeLLMClient, OpenAIClient, strict_schema
from cavalier.cli import _agent_builder
from cavalier.engine import Game
from cavalier.agents.scripted import ReplayAgent


def _obs(bundle, side="USA"):
    game = Game(bundle, {s: ReplayAgent({}) for s in bundle.scenario.sides}, seed=0, anonymized=True)
    return game.observe(side)


def _objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            yield node
        for v in node.values():
            yield from _objects(v)
    elif isinstance(node, list):
        for v in node:
            yield from _objects(v)


def test_strict_schema_closes_objects_and_drops_unsupported_keywords(bundle):
    agent = LLMAgent(bundle, "USA", FakeLLMClient([]))
    schema = agent.schema_for(_obs(bundle))
    strict = strict_schema(schema)
    text = json.dumps(strict)
    for kw in ("maxItems", "minItems", "maxLength", "minimum", "maximum", '"default"'):
        assert kw not in text
    objs = list(_objects(strict))
    assert len(objs) == 3  # decisión, acción y término
    for o in objs:
        assert o["additionalProperties"] is False and set(o["required"]) == set(o["properties"])
    # los códigos cerrados del turno se conservan
    assert strict["$defs"]["LLMAction"]["properties"]["code"]["enum"] == \
        schema["$defs"]["LLMAction"]["properties"]["code"]["enum"]
    assert "maxItems" in json.dumps(schema)  # el original no se toca


def _fake_anthropic(stop_reason="end_turn", text='{"ok": true}'):
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)],
                               usage=SimpleNamespace(input_tokens=2500, output_tokens=800))
    client = AnthropicClient.__new__(AnthropicClient)
    client.model_id, client.effort, client.max_tokens, client.metrics = "claude-test", "low", 16000, []
    client._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return client, calls


def test_anthropic_client_sends_structured_output_and_records_tokens():
    client, calls = _fake_anthropic()
    assert client.complete_json("sys", "user", {"type": "object", "properties": {"a": {"type": "string"}}}) \
        == '{"ok": true}'
    kw = calls[0]
    assert kw["system"] == "sys" and kw["messages"] == [{"role": "user", "content": "user"}]
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert "temperature" not in kw
    assert client.metrics[0]["prompt_tokens"] == 2500 and client.metrics[0]["output_tokens"] == 800


def test_anthropic_refusal_returns_empty_text():
    client, _ = _fake_anthropic(stop_reason="refusal")
    assert client.complete_json("sys", "user", {"type": "object"}) == ""


def test_openai_client_uses_strict_schema_and_reasoning_effort():
    calls = []

    def create(**kw):
        calls.append(kw)
        msg = SimpleNamespace(content='{"ok": true}')
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=2400, completion_tokens=700))
    client = OpenAIClient.__new__(OpenAIClient)
    client.model_id, client.seed, client.effort, client.max_tokens, client.metrics = "gpt-test", 1, "low", 16000, []
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert client.complete_json("sys", "user", {"type": "object", "properties": {}}) == '{"ok": true}'
    kw = calls[0]
    assert kw["response_format"]["json_schema"]["strict"] is True
    assert kw["reasoning_effort"] == "low" and "temperature" not in kw and "max_tokens" not in kw
    assert client.metrics[0]["prompt_tokens"] == 2400


def test_each_side_can_use_its_own_provider(bundle, monkeypatch):
    built = []

    def fake_make_client(backend, model=None, **kw):
        built.append((backend, model))
        return SimpleNamespace(model_id=model or backend)
    monkeypatch.setattr("cavalier.agents.llm_client.make_client", fake_make_client)
    args = SimpleNamespace(backend="ollama", model="qwen3.5:9b", base_url="x", temperature=0.7, num_ctx=4096,
                           corpus=None, anon=True)
    usa = _agent_builder(args, "llm:historico@openai:gpt-5.6-terra", "USA")(bundle, 1)
    ussr = _agent_builder(args, "llm:historico@anthropic", "USSR")(bundle, 2)
    local = _agent_builder(args, "llm:historico", "USA")(bundle, 3)
    assert built == [("openai", "gpt-5.6-terra"), ("anthropic", None), ("ollama", "qwen3.5:9b")]
    assert usa.name.startswith("llm:gpt-5.6-terra:historico") and ussr.name.startswith("llm:anthropic")
    assert local.name.startswith("llm:qwen3.5:9b")
