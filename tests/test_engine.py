from cavalier.agents.scripted import ReplayAgent, ScriptedAgent
from cavalier.engine import Game
from cavalier.models import Action, DealTerm, Outcome, WorldState
from cavalier.montecarlo import play


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


def test_message_latency_by_channel(bundle):
    usa = ReplayAgent({0: [Action(kind="private_message", channel="official", text="lento"),
                           Action(kind="private_message", channel="backchannel", text="rápido")]})
    game = Game(bundle, {"USA": usa, "USSR": ReplayAgent({})}, stochastic_events=False)
    game.step()
    assert [m.text for m in game.observe("USSR").inbox] == ["rápido"]
    game.step()
    assert [m.text for m in game.observe("USSR").inbox] == ["lento"]


def test_deal_proposal_and_acceptance(bundle):
    terms = [DealTerm(term="withdraw_missiles"), DealTerm(term="withdraw_jupiters", secret=True)]
    usa = ReplayAgent({0: [Action(kind="propose_deal", channel="backchannel", terms=terms)]})
    ussr = ReplayAgent({1: [Action(kind="accept_deal")]})
    game = Game(bundle, {"USA": usa, "USSR": ussr}, stochastic_events=False)
    assert game.observe("USSR").open_deals == []
    game.step()
    assert len(game.observe("USSR").open_deals) == 1
    credibility = game.state.sides["USA"].credibility
    game.step()
    assert game.state.outcome == Outcome.NEGOTIATED
    assert game.state.vars["missiles_withdrawn"] == 1.0
    assert game.state.vars["jupiters_turkey"] == 0.0
    # término secreto: coste de credibilidad reducido (20 * 0.2)
    assert game.state.sides["USA"].credibility == credibility - 4.0


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


def test_terminal_action_ends_game(bundle):
    usa = ReplayAgent({0: [Action(kind="invasion")]})
    state = Game(bundle, {"USA": usa, "USSR": ReplayAgent({})}, seed=0).run()
    assert state.outcome in (Outcome.CONVENTIONAL, Outcome.NUCLEAR)
    assert state.turn == 1
