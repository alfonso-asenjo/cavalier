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
