from cavalier.agents.scripted import ReplayAgent, ScriptedAgent
from cavalier.engine import Game, state_ops
from cavalier.models import Action, DealTerm, Outcome, WorldState
from cavalier.montecarlo import play
from cavalier.scenario import load_scenario


def _idle(bundle):
    return {s: ReplayAgent({}) for s in bundle.scenario.sides}


def test_same_seed_same_game(bundle):
    def agents(seed):
        return {"USA": ScriptedAgent(bundle, "USA", "halcon", seed), "USSR": ScriptedAgent(bundle, "USSR", "impredecible", seed + 1)}
    assert play(bundle, agents(5), 5) == play(bundle, agents(5), 5)


def test_idle_game_ends_unresolved_or_by_incident(bundle):
    state = Game(bundle, _idle(bundle), seed=1, stochastic_events=False).run()
    assert state.outcome == Outcome.UNRESOLVED
    assert state.vars["construction_progress"] == 1.0  # las obras avanzan si nadie hace nada


def test_state_roundtrip_allows_fork(bundle):
    game = Game(bundle, _idle(bundle), seed=1)
    for _ in range(3):
        game.step()
    restored = WorldState.model_validate_json(game.state.model_dump_json())
    fork = Game(bundle, _idle(bundle), seed=99, state=restored)
    assert fork.state.turn == 3 and fork.state is not game.state


def test_message_latency_by_channel():
    slow = load_scenario("cuba1962", "comunicaciones_lentas")  # official +2, backchannel +1
    usa = ReplayAgent({0: [Action(kind="private_message", channel="official", text="lento"),
                           Action(kind="private_message", channel="backchannel", text="rápido")]})
    game = Game(slow, {"USA": usa, "USSR": ReplayAgent({})}, stochastic_events=False)
    inboxes = []
    for _ in range(4):
        inboxes.append([m.text for m in game.observe("USSR").inbox])
        game.step()
    assert inboxes == [[], [], ["rápido"], ["lento"]]


def _deal_game(bundle, terms, usa_extra=None, ussr_extra=None, stochastic=False, seed=0):
    usa = ReplayAgent({0: [Action(kind="propose_deal", channel="backchannel", terms=terms)], **(usa_extra or {})})
    ussr = ReplayAgent({1: [Action(kind="accept_deal")], **(ussr_extra or {})})
    return Game(bundle, {"USA": usa, "USSR": ussr}, seed=seed, stochastic_events=stochastic)


def test_accepting_opens_implementation_phase(bundle):
    terms = [DealTerm(term="withdraw_missiles"), DealTerm(term="withdraw_jupiters", secret=True)]
    game = _deal_game(bundle, terms)
    assert game.observe("USSR").open_deals == []
    game.step()
    assert len(game.observe("USSR").open_deals) == 1
    game.step()
    assert game.state.outcome is None  # aceptar ya no termina la crisis
    assert {c.term: c.status for c in game.state.commitments} == {
        "withdraw_missiles": "pending", "withdraw_jupiters": "pending"}
    assert game.state.vars["missiles_withdrawn"] == 0.0  # aún no se ha cumplido
    state = game.run()
    assert state.outcome == Outcome.NEGOTIATED
    assert state.vars["missiles_withdrawn"] == 1.0 and state.vars["jupiters_turkey"] == 0.0
    assert state.turn >= 1 + bundle.rules.implementation_turns


def test_secret_terms_cost_less_credibility(bundle):
    def usa_credibility_after_accept(secret):
        game = _deal_game(bundle, [DealTerm(term="withdraw_jupiters", secret=secret)])
        game.step(); game.step()
        return game.state.sides["USA"].credibility
    assert usa_credibility_after_accept(True) > usa_credibility_after_accept(False)


def test_conceding_at_low_tension_costs_more(bundle):
    def credibility_loss(tension):
        game = _deal_game(bundle, [DealTerm(term="withdraw_missiles")])
        game.step()
        game.state.tension = tension
        before = game.state.sides["USSR"].credibility
        game.step()
        return before - game.state.sides["USSR"].credibility
    assert credibility_loss(20.0) > credibility_loss(90.0)


def test_renege_breaks_the_deal_and_the_crisis_continues(bundle):
    game = _deal_game(bundle, [DealTerm(term="withdraw_missiles")],
                      ussr_extra={2: [Action(kind="renege_deal", deal_id="P1")]})
    for _ in range(3):
        game.step()
    assert game.state.deals["P1"].status == "broken"
    assert all(c.status == "broken" for c in game.state.commitments)
    assert game.state.outcome is None


def test_observer_only_sees_compliance_once_verified(bundle):
    game = _deal_game(bundle, [DealTerm(term="withdraw_missiles")])
    game.state.sides["USA"].intel_quality = 0.0  # sin inteligencia no se verifica nada
    for _ in range(2 + bundle.rules.implementation_turns):
        game.step()
    real = game.state.commitments[0]
    assert real.status == "fulfilled" and not real.verified
    assert game.observe("USA").commitments[0].status == "pending"
    assert game.observe("USSR").commitments[0].status == "fulfilled"
    assert game.state.outcome is None


def test_inspection_guarantees_verification(bundle):
    game = _deal_game(bundle, [DealTerm(term="withdraw_missiles"), DealTerm(term="un_inspection")])
    game.state.sides["USA"].intel_quality = 0.0
    state = game.run()
    assert state.outcome == Outcome.NEGOTIATED


def test_invalid_actions_are_counted_not_applied(bundle):
    ussr = ReplayAgent({0: [Action(kind="impose_quarantine"), Action(kind="accept_deal", deal_id="nope")]})
    game = Game(bundle, {"USA": ReplayAgent({}), "USSR": ussr}, stochastic_events=False)
    game.step()
    assert game.state.invalid_actions["USSR"] == 2
    assert game.state.vars["quarantine"] == 0.0


def test_fog_of_war(bundle):
    game = Game(bundle, _idle(bundle), seed=3)
    usa, ussr = game.observe("USA"), game.observe("USSR")
    assert "tactical_nukes_cuba" not in usa.vars
    assert ussr.vars["tactical_nukes_cuba"] == 1.0
    assert usa.vars["soviet_troops_cuba"] < 0.2  # sesgo: EE. UU. infraestima la fuerza
    assert "impose_quarantine" in usa.legal_actions and "impose_quarantine" not in ussr.legal_actions


def test_terminal_action_resolves_before_acceptance(bundle):
    # EE. UU. invade en el mismo turno en que la URSS acepta: la invasión ya está en marcha
    game = _deal_game(bundle, [DealTerm(term="withdraw_missiles")])
    game.agents["USSR"], game.agents["USA"] = (
        ReplayAgent({0: [Action(kind="propose_deal", channel="backchannel", terms=[DealTerm(term="no_invasion_pledge")])]}),
        ReplayAgent({1: [Action(kind="invasion"), Action(kind="accept_deal")]}),
    )
    state = game.run()
    assert state.outcome in (Outcome.CONVENTIONAL, Outcome.NUCLEAR)
    assert all(d.status != "accepted" for d in state.deals.values())


def test_lifting_quarantine_same_turn_prevents_naval_clash(bundle):
    def clash(usa_actions):
        usa = ReplayAgent({0: [Action(kind="impose_quarantine")], 1: usa_actions})
        ussr = ReplayAgent({1: [Action(kind="run_quarantine")]})
        override = bundle.rules.model_copy(deep=True)
        for rule in override.interactions:
            rule.prob = 1.0
        b = bundle.model_copy(update={"rules": override})
        game = Game(b, {"USA": usa, "USSR": ussr}, stochastic_events=False)
        game.step(); game.step()
        return any(e.kind == "incident" and "cuarentena" in e.description for e in game.state.events)
    assert clash([Action(kind="wait")])
    assert not clash([Action(kind="lift_quarantine")])


def test_tension_never_saturates(bundle):
    game = Game(bundle, _idle(bundle), stochastic_events=False)
    for _ in range(40):
        state_ops.add(game.state, "tension", 15.0)
    assert 99.0 < game.state.tension < 100.0
    before = game.state.tension
    state_ops.add(game.state, "tension", -15.0)
    assert game.state.tension < before


def test_terminal_action_ends_game(bundle):
    usa = ReplayAgent({0: [Action(kind="invasion")]})
    state = Game(bundle, {"USA": usa, "USSR": ReplayAgent({})}, seed=0).run()
    assert state.outcome in (Outcome.CONVENTIONAL, Outcome.NUCLEAR)
    assert state.turn == 1


def test_deal_that_does_not_remove_the_missiles_does_not_end_the_crisis(bundle):
    from cavalier.agents.scripted import ReplayAgent
    from cavalier.models import Action, DealTerm, Outcome

    ussr = ReplayAgent({0: [Action(kind="propose_deal", channel="backchannel",
                                   terms=[DealTerm(term="withdraw_jupiters", secret=True)])]})
    usa = ReplayAgent({1: [Action(kind="accept_deal")]})
    game = Game(bundle, {"USA": usa, "USSR": ussr}, stochastic_events=False)
    for _ in range(bundle.rules.implementation_turns + 3):
        game.step()
    assert game.state.deals["P1"].status == "fulfilled"
    assert game.state.outcome != Outcome.NEGOTIATED
    assert any("no resuelve la crisis" in e.description for e in game.state.events)
