import json

import pytest

from cavalier import experiments
from cavalier.montecarlo import ScriptedFactory
from cavalier.scenario import load_scenario

FACTORY = ScriptedFactory({"USA": "pragmatica", "USSR": "defensiva"})
CELLS = ["base", "sin_canal_secreto", "mando_debil"]


def test_counterfactual_variants():
    secret = load_scenario("cuba1962", "sin_canal_secreto")
    assert list(secret.rules.channels) == ["official"]
    assert not secret.scenario.deal_terms["withdraw_jupiters"].secretable
    assert "backchannel" in load_scenario("cuba1962").rules.channels  # la base no cambia
    assert load_scenario("cuba1962", "mando_debil").rules.incident_prob_multiplier == 2.0
    with pytest.raises(ValueError):
        load_scenario("cuba1962", overrides={"rules": {"disabled_channels": ["telegrafo"]}})


def test_games_fork_from_the_checkpoint_state(tmp_path):
    results = experiments.run_experiment("cuba1962", "usa_sabado_negro", CELLS, 2, tmp_path, FACTORY,
                                         log=lambda *_: None)
    for cell in CELLS:
        assert len(results[cell]) == 2 and all(g.turns > 23 for g in results[cell])
        first = json.loads((tmp_path / cell / "s000.jsonl").read_text(encoding="utf-8").splitlines()[1])
        assert first["turn"] == 23  # empieza en el Sábado Negro, no al principio
    assert experiments.load_results(tmp_path).keys() == set(CELLS)


def test_experiment_skips_finished_games_and_resumes(tmp_path):
    full = experiments.run_experiment("cuba1962", "usa_sabado_negro", CELLS, 2, tmp_path / "a", FACTORY,
                                      log=lambda *_: None)
    out = tmp_path / "b"
    experiments.run_experiment("cuba1962", "usa_sabado_negro", CELLS, 1, out, FACTORY, log=lambda *_: None)
    played = []
    resumed = experiments.run_experiment("cuba1962", "usa_sabado_negro", CELLS, 2, out, FACTORY,
                                         log=played.append)
    assert len(played) == len(CELLS)  # solo la semilla 1 de cada celda
    assert {c: [g.outcome for g in v] for c, v in resumed.items()} == {c: [g.outcome for g in v] for c, v in full.items()}


def test_report_contrasts_each_cell_against_base(tmp_path):
    results = experiments.run_experiment("cuba1962", "usa_sabado_negro", CELLS, 3, tmp_path, FACTORY,
                                         log=lambda *_: None)
    rep = experiments.report(results)
    assert set(rep["contrasts"]) == {f"{c}/{m}" for c in CELLS[1:] for m in ("acuerdo", "escalada_militar")}
    for c in rep["contrasts"].values():
        assert c["holm_p"] >= c["fisher_p"]
    assert rep["cells"]["base"]["n"] == 3


def test_holm_is_monotone():
    adj = experiments._holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
