import json
from collections import Counter

import pytest

from cavalier.agents.advisors import AdvisorAgent, load_advisors
from cavalier.agents.llm import LLMAgent
from cavalier.agents.llm_client import FakeLLMClient
from cavalier.eval.advisor_probe import js_divergence, load_records, report, run_probe, select_states
from cavalier.montecarlo import play, scripted_factory

from test_llm_agent import _obs

REAL_NAMES = ("McNamara", "Taylor", "Thompson", "Stevenson", "Mikoyán", "Malinovski", "Dobrynin", "Moscú", "Washington")


@pytest.fixture(scope="module")
def personas():
    return load_advisors("cuba1962")


def _say(agent, kind, **extra):
    return json.dumps({"recommendation": agent.leader._action_to_code[kind], "confidence": 0.7,
                       "argument": "a", "risk": "r", "would_change_if": "c", **extra})


def test_cards_are_consistent_with_the_rules(bundle, personas):
    sections = set(LLMAgent(bundle, "USA", FakeLLMClient([])).sections(_obs(bundle), []))
    for p in personas.values():
        assert p.side in bundle.scenario.sides
        assert set(p.anchor) <= set(bundle.rules.actions), p.id
        assert set(p.briefing) <= sections, p.id
        assert p.stance and p.red_line and p.changes_if and p.evidence


def test_anonymized_advisors_use_office_and_do_not_leak(bundle, personas):
    obs = _obs(bundle)
    for p in (personas["mcnamara"], personas["thompson"]):
        agent = AdvisorAgent(bundle, p, FakeLLMClient([]), anonymized=True)
        text = agent.system_prompt(obs) + agent.user_prompt(obs)
        assert not [n for n in REAL_NAMES if n in text], p.id
        assert not agent.anon.leaks(text)
    real = AdvisorAgent(bundle, personas["mcnamara"], FakeLLMClient([]), anonymized=False)
    assert "Robert McNamara" in real.system_prompt(obs)


def test_private_briefing_versus_label_only(bundle, personas):
    obs = _obs(bundle, turn_steps=2)
    full = AdvisorAgent(bundle, personas["taylor"], FakeLLMClient([]), mode="completo")
    label = AdvisorAgent(bundle, personas["taylor"], FakeLLMClient([]), mode="etiqueta")
    assert "### Términos negociables" not in full.user_prompt(obs)
    assert "### Términos negociables" in label.user_prompt(obs)
    assert "Tu postura" in full.system_prompt(obs) and "Tu postura" not in label.system_prompt(obs)
    assert "### Acciones disponibles" in full.user_prompt(obs)


def test_advice_is_validated_and_sycophancy_mode_states_the_intent(bundle, personas):
    obs = _obs(bundle)
    client = FakeLLMClient([])
    agent = AdvisorAgent(bundle, personas["mcnamara"], client, mode="complacencia")
    client._responder = ["no es json", _say(agent, "gather_intel")]
    res = agent.advise(obs, leader_intent="impose_quarantine")
    assert res["valid"] and res["recommendation"] == "gather_intel" and res["attempts"] == 2
    assert "Se inclina por" in client.calls[0][1]

    client._responder = [json.dumps({"recommendation": "A99", "confidence": 1, "argument": "", "risk": "",
                                     "would_change_if": ""})] * 2
    assert not agent.advise(obs)["valid"]


def test_js_divergence():
    assert js_divergence(Counter(a=3), Counter(a=5)) == 0
    assert js_divergence(Counter(a=1), Counter(b=1)) == pytest.approx(1)


def test_probe_end_to_end_with_fake_model(bundle, personas, tmp_path):
    runs = tmp_path / "base"
    runs.mkdir()
    for seed in range(3):
        play(bundle, scripted_factory({"USA": "pragmatica", "USSR": "defensiva"})(bundle, seed), seed,
             log_path=runs / f"s{seed:03d}.jsonl")
    states = select_states(runs, n_fork=2, n_key=1)
    assert 2 <= len(states) <= 3 and states[0]["kind"] == "bifurcacion"

    chosen = {"mcnamara": "raise_alert", "taylor": "mobilize"}
    mine = {k: personas[k] for k in chosen}
    codes = LLMAgent(bundle, "USA", FakeLLMClient([]))._action_to_code
    client = FakeLLMClient(lambda system, user: json.dumps({
        "recommendation": codes[chosen["taylor" if "Estado Mayor" in system else "mcnamara"]],
        "confidence": 0.6, "argument": "texto", "risk": "r", "would_change_if": "c"}))
    out = tmp_path / "probe.jsonl"
    run_probe(bundle, client, mine, states, out, k=2, k_sycophancy=1)
    n = len(load_records(out))
    assert n == len(states) * 2 * (2 + 2 + 1)
    run_probe(bundle, client, mine, states, out, k=2, k_sycophancy=1)
    assert len(load_records(out)) == n  # reanudable: no repite nada

    rep = report(load_records(out))
    assert rep["criterio"]["cumple"]
    assert rep["modes"]["completo"]["pairs"]["mcnamara/taylor"]["js_media"] == pytest.approx(1)
