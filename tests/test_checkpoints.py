from cavalier.agents.scripted import ScriptedAgent
from cavalier.eval.checkpoints import evaluate_checkpoint, reconstruct_state
from cavalier.eval.stats import runs_for_margin, wilson_interval
from cavalier.history import load_checkpoints


def test_every_checkpoint_can_be_reconstructed(bundle):
    for cp in load_checkpoints("cuba1962"):
        state = reconstruct_state(bundle, "cuba1962", cp)
        assert state.turn == cp.turn and state.outcome is None


def test_acceptance_checkpoint_has_the_us_offer_open(bundle):
    cp = next(c for c in load_checkpoints("cuba1962") if c.id == "ussr_aceptacion")
    state = reconstruct_state(bundle, "cuba1962", cp)
    open_from_usa = [d for d in state.deals.values() if d.status == "open" and d.proposer == "USA"]
    assert open_from_usa and open_from_usa[0].deliver_turn <= cp.turn


def test_evaluate_with_scripted_baseline(bundle):
    cp = next(c for c in load_checkpoints("cuba1962") if c.id == "ussr_aceptacion")
    res = evaluate_checkpoint(bundle, "cuba1962", cp, lambda seed: ScriptedAgent(bundle, cp.side, "pragmatica", seed), k=20)
    assert res["k"] == 20 and sum(res["choices"].values()) == 20
    assert 0 <= res["ci95"][0] <= res["p_historical"] <= res["ci95"][1] <= 1


def test_stats():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    assert runs_for_margin(0.02, 0.01) > 700  # eventos raros: hacen falta muchas partidas
