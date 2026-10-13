from cavalier.engine import initial_state
from cavalier.engine import state_ops
from cavalier.history import load_checkpoints, load_historical_script
from cavalier.scenario import load_scenario


def test_all_rule_paths_resolve(bundle):
    state = initial_state(bundle)
    for kind, rule in bundle.rules.actions.items():
        for path in [*rule.effects, *rule.sets, *rule.requires]:
            state_ops.get(state, path, actor="USA")
    for drift in bundle.rules.drifts:
        for path in [drift.target, *drift.when, *drift.when_gte]:
            state_ops.get(state, path)
    for inc in bundle.rules.incidents:
        assert inc.actor in bundle.scenario.sides
        for path in inc.effects:
            state_ops.get(state, path, actor=inc.actor)
    for term in bundle.scenario.deal_terms.values():
        assert term.obligor in bundle.scenario.sides
        for path in term.sets:
            state_ops.get(state, path)


def test_variants_override(bundle):
    assert bundle.scenario.variables["tactical_nukes_cuba"].initial == 1.0
    v = load_scenario("cuba1962", "sin_armas_tacticas")
    assert v.scenario.variables["tactical_nukes_cuba"].initial == 0.0
    assert not load_scenario("cuba1962", "sin_jupiter").scenario.deal_terms["withdraw_jupiters"].enabled


def test_overrides_for_sensitivity():
    b = load_scenario("cuba1962", overrides={"rules": {"tension_decay": 9.0}})
    assert b.rules.tension_decay == 9.0
    assert b.rules.actions["mobilize"].effects  # el resto se conserva


def test_historical_files_are_consistent(bundle):
    script = load_historical_script("cuba1962")
    for side, turns in script.items():
        assert side in bundle.scenario.sides
        for actions in turns.values():
            assert all(a.kind in bundle.rules.actions for a in actions)
    for cp in load_checkpoints("cuba1962"):
        assert cp.historical in cp.considered
        assert all(k in bundle.rules.actions for k in cp.considered)
