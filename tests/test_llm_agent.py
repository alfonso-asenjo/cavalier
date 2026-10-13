import json
import re

from cavalier.agents.llm import LLMAgent
from cavalier.agents.llm_client import FakeLLMClient
from cavalier.anonymize import Anonymizer
from cavalier.engine import Game
from cavalier.agents.scripted import ReplayAgent


def _obs(bundle, side="USA", turn_steps=0):
    game = Game(bundle, {s: ReplayAgent({}) for s in bundle.scenario.sides}, seed=0, anonymized=True)
    for _ in range(turn_steps):
        game.step()
    return game.observe(side)


def _answer(agent, kind, **extra):
    code = agent._action_to_code[kind]
    return json.dumps({"reasoning": "r", "actions": [{"code": code, **extra}], "confidence": 0.6})


def test_valid_decision_is_mapped(bundle):
    client = FakeLLMClient([])
    agent = LLMAgent(bundle, "USA", client)
    client._responder = [_answer(agent, "impose_quarantine")]
    decision = agent.decide(_obs(bundle))
    assert [a.kind for a in decision.actions] == ["impose_quarantine"]
    assert not decision.fallback


def test_retry_then_success(bundle):
    client = FakeLLMClient([])
    agent = LLMAgent(bundle, "USA", client)
    client._responder = ["no es json", _answer(agent, "gather_intel")]
    decision = agent.decide(_obs(bundle))
    assert [a.kind for a in decision.actions] == ["gather_intel"]
    assert len(client.calls) == 2
    assert "no era válida" in client.calls[1][1]


def test_illegal_action_falls_back_and_is_flagged(bundle):
    client = FakeLLMClient([])
    agent = LLMAgent(bundle, "USSR", client, max_attempts=2)
    client._responder = [_answer(agent, "impose_quarantine")]  # solo puede hacerlo USA
    decision = agent.decide(_obs(bundle, "USSR"))
    assert decision.fallback and decision.actions[0].kind == "wait"
    assert len(client.calls) == 2


def test_deal_terms_use_codes(bundle):
    client = FakeLLMClient([])
    agent = LLMAgent(bundle, "USA", client)
    t_missiles = agent._term_to_code["withdraw_missiles"]
    t_jup = agent._term_to_code["withdraw_jupiters"]
    client._responder = [_answer(agent, "propose_deal", channel="backchannel",
                                 terms=[{"code": t_missiles}, {"code": t_jup, "secret": True}])]
    action = agent.decide(_obs(bundle)).actions[0]
    assert [(t.term, t.secret) for t in action.terms] == [("withdraw_missiles", False), ("withdraw_jupiters", True)]


def test_anonymized_prompts_do_not_leak_names(bundle):
    client = FakeLLMClient([])
    agent = LLMAgent(bundle, "USA", client, anonymized=True)
    client._responder = lambda s, u: _answer(agent, "wait")
    for side in ("USA", "USSR"):
        agent = LLMAgent(bundle, side, client, anonymized=True)
        client._responder = lambda s, u, a=agent: _answer(a, "wait")
        agent.decide(_obs(bundle, side, turn_steps=2))
    anon = Anonymizer(bundle.aliases)
    side_ids = re.compile(r"\b(USA|USSR)\b")  # sensible a mayúsculas: "usa" es un verbo
    forbidden = re.compile(r"\b(1962|octubre|jupiters?|quarantine|cuba|soviet\w*)\b", re.IGNORECASE)
    for system, user in client.calls:
        for text in (system, user):
            assert anon.leaks(text) == []
            assert not side_ids.search(text), side_ids.search(text)
            assert not forbidden.search(text), forbidden.search(text)


def test_commitments_are_rendered_and_renege_is_validated(bundle):
    from cavalier.models import Action, DealTerm

    usa = ReplayAgent({0: [Action(kind="propose_deal", channel="backchannel", terms=[DealTerm(term="withdraw_missiles")])]})
    ussr = ReplayAgent({1: [Action(kind="accept_deal")]})
    game = Game(bundle, {"USA": usa, "USSR": ussr}, stochastic_events=False)
    game.step(); game.step()
    client = FakeLLMClient([])
    agent = LLMAgent(bundle, "USSR", client)
    obs = game.observe("USSR")
    assert "Compromisos de acuerdos aceptados" in agent.render(obs, [])
    client._responder = [_answer(agent, "renege_deal", deal_id="P1")]
    assert agent.decide(obs).actions[0].kind == "renege_deal"
    usa_agent = LLMAgent(bundle, "USA", FakeLLMClient([_answer(agent, "renege_deal", deal_id="P1")]))
    assert usa_agent.decide(game.observe("USA")).fallback  # EE. UU. no tiene compromisos pendientes en P1
