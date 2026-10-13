import json

import pytest

from cavalier.agents.llm import LLMAgent
from cavalier.agents.llm_client import FakeLLMClient
from cavalier.eval import fidelity
from cavalier.eval.stats import fisher_exact


def _client(bundle):
    wait = LLMAgent(bundle, "USA", FakeLLMClient([]))._action_to_code["wait"]
    client = FakeLLMClient([json.dumps({"reasoning": "r", "confidence": 0.5, "actions": [{"code": wait}]})])
    client.metrics, client.seed = [], None
    return client


def test_fisher_exact_matches_known_values():
    assert fisher_exact(7, 3, 0, 10) == pytest.approx(0.003096, abs=1e-5)  # 7/10 frente a 0/10
    assert fisher_exact(5, 5, 5, 5) == pytest.approx(1.0)
    assert fisher_exact(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-4)  # té de Fisher


def test_fidelity_interleaves_samples_and_resumes(bundle):
    saved = []
    records = fidelity.run_fidelity(_client(bundle), k=2, log=lambda *_: None, on_progress=lambda r: saved.append(list(r)))
    cells = 4 * len(fidelity.LLM_CONDITIONS)
    assert len(records) == 2 * cells
    assert {r["seed"] for r in saved[cells - 1]} == {0}  # la primera muestra de todas las celdas va antes
    assert all(r["choice"] == "wait" and not r["fallback"] for r in records)
    assert {r["persona"] for r in records} == {"historico", "ninguna"}

    client = _client(bundle)
    resumed = fidelity.run_fidelity(client, k=2, log=lambda *_: None, done=saved[cells + 2])
    assert len(client.calls) == 2 * cells - (cells + 3)  # solo lo que faltaba
    assert [fidelity._key(r) for r in resumed] == [fidelity._key(r) for r in records]


def test_report_includes_references_and_leakage_contrast(bundle):
    records = fidelity.run_fidelity(_client(bundle), k=1, log=lambda *_: None)
    scripted = fidelity.scripted_choices("cuba1962", k=3)
    rep = fidelity.report(records + scripted)
    cp = rep["usa_respuesta_inicial"]
    assert cp["uniform"] == pytest.approx(1 / 6)
    assert {"llm:historico:real", "llm:ninguna:anon", "scripted:halcon"} <= set(cp["conditions"])
    assert cp["conditions"]["scripted:halcon"]["n"] == 3
    assert cp["real_vs_anon"]["historico"]["fisher_p"] == pytest.approx(1.0)
