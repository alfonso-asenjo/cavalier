"""Invariantes cualitativos del motor (plan, fase 1). Con n=1 histórico no se puede
calibrar, pero sí exigir que el motor se comporte de forma coherente."""

from cavalier.models import Outcome
from cavalier.montecarlo import replay_factory, run_batch, scripted_factory
from cavalier.scenario import load_scenario

N = 200
WAR = (Outcome.CONVENTIONAL, Outcome.NUCLEAR)


def _war_rate(result):
    return sum(result.proportion(o) for o in WAR)


def test_hawks_escalate_more_than_doves(bundle):
    hawks = run_batch(bundle, scripted_factory({"USA": "halcon", "USSR": "halcon"}), N)
    doves = run_batch(bundle, scripted_factory({"USA": "paloma", "USSR": "paloma"}), N)
    assert _war_rate(hawks) > _war_rate(doves) + 0.3


def test_weaker_command_control_means_more_incidents():
    def incidents(control):
        b = load_scenario("cuba1962", overrides={"scenario": {"sides": {
            "USA": {"initial": {"control": control}}, "USSR": {"initial": {"control": control}}}}})
        res = run_batch(b, scripted_factory({"USA": "defensiva", "USSR": "defensiva"}), N)
        return sum(len(s.incidents) for s in res.summaries) / res.n
    assert incidents(0.4) > incidents(0.95)


def test_tactical_weapons_raise_nuclear_share_of_wars(bundle):
    def nuclear_share(b):
        res = run_batch(b, scripted_factory({"USA": "halcon", "USSR": "defensiva"}), N)
        wars = [s for s in res.summaries if s.outcome in {o.value for o in WAR}]
        return sum(s.outcome == Outcome.NUCLEAR.value for s in wars) / max(1, len(wars))
    assert nuclear_share(bundle) > nuclear_share(load_scenario("cuba1962", "sin_armas_tacticas"))


def test_historical_replay_is_neither_doomed_nor_trivial(bundle):
    res = run_batch(bundle, replay_factory("cuba1962"), N)
    negotiated = res.proportion(Outcome.NEGOTIATED)
    assert 0.5 < negotiated < 0.99, negotiated
    assert sum(s.invalid_actions for s in res.summaries) == 0


def test_parallel_batch_matches_sequential(bundle):
    from concurrent.futures import ProcessPoolExecutor

    factory = scripted_factory({"USA": "impredecible", "USSR": "pragmatica"})
    sequential = run_batch(bundle, factory, 40, seed=7)
    with ProcessPoolExecutor(max_workers=4) as ex:
        parallel = run_batch(bundle, factory, 40, seed=7, executor=ex, chunks=4)
    assert parallel.summaries == sequential.summaries


def test_morris_runs_and_ranks_parameters():
    from cavalier.sensitivity import load_ranges, morris

    ranges = load_ranges("cuba1962")
    subset = {k: ranges[k] for k in ("rules.incidents.1.nuclear_risk", "rules.observation_noise")}
    res = morris("cuba1962", replay_factory("cuba1962"), n=60, trajectories=2, ranges=subset)
    nuclear = {row["param"]: row for row in res["metrics"]["nuclear"]}
    # con acciones fijas, el ruido de observación no puede influir; el riesgo del submarino sí
    assert nuclear["rules.observation_noise"]["mu_star"] == 0.0
    assert nuclear["rules.incidents.1.nuclear_risk"]["mu_star"] > 0.0


def test_ranges_cover_base_values(bundle):
    from cavalier.sensitivity import _get, load_ranges

    raw = {"rules": bundle.rules.model_dump()}
    for path, rg in load_ranges("cuba1962").items():
        assert rg.low <= _get(raw, path.split(".")) <= rg.high, path
