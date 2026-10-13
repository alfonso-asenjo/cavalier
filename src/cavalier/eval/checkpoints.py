"""Fidelidad histórica por forzado: en cada punto de control se reconstruye el estado
histórico y se piden K decisiones al agente. Métrica: frecuencia de la opción histórica
dentro de las opciones que se consideraron de verdad, con IC 95 % (Wilson)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from ..agents.base import Agent
from ..agents.scripted import ReplayAgent
from ..engine import Game
from ..history import Checkpoint, load_historical_script
from ..models import Decision, WorldState
from ..scenario import ScenarioBundle
from .stats import wilson_interval


def reconstruct_state(bundle: ScenarioBundle, scenario: str, checkpoint: Checkpoint, seed: int = 0) -> WorldState:
    """Reproduce la secuencia histórica hasta el turno del punto de control, sin incidentes aleatorios."""
    script = load_historical_script(scenario)
    agents = {s: ReplayAgent(script.get(s, {})) for s in bundle.scenario.sides}
    game = Game(bundle, agents, seed=seed, stochastic_events=False)
    while game.state.turn < checkpoint.turn and game.state.outcome is None:
        game.step()
    if game.state.outcome is not None:
        raise RuntimeError(f"La secuencia histórica terminó antes del punto {checkpoint.id}: {game.state.outcome_reason}")
    for incident in checkpoint.force_incidents:
        game.state.turn -= 1  # el incidente aparece como "desde el último turno"
        game.fire_incident(incident, allow_nuclear=False)
        game.state.turn += 1
    return game.state


def classify(decision: Decision, considered: list[str]) -> str:
    """Opción principal de una decisión. `considered` está ordenada por prioridad: si el agente
    toma varias de esas acciones, cuenta la que aparece antes en la lista (terminales primero)."""
    kinds = {a.kind for a in decision.actions}
    return next((option for option in considered if option in kinds), "otra")


def evaluate_checkpoint(
    bundle: ScenarioBundle,
    scenario: str,
    checkpoint: Checkpoint,
    agent_factory: Callable[[int], Agent],
    k: int = 30,
    seed: int = 0,
    anonymized: bool = False,
) -> dict:
    state = reconstruct_state(bundle, scenario, checkpoint, seed)
    choices: Counter = Counter()
    fallbacks = 0
    for i in range(k):
        agent = agent_factory(seed + i)
        game = Game(bundle, {s: agent for s in bundle.scenario.sides}, seed=seed + i, state=state, anonymized=anonymized)
        decision = agent.decide(game.observe(checkpoint.side))
        fallbacks += decision.fallback
        choices[classify(decision, checkpoint.considered)] += 1
    hits = choices[checkpoint.historical]
    lo, hi = wilson_interval(hits, k)
    return {
        "checkpoint": checkpoint.id,
        "historical": checkpoint.historical,
        "k": k,
        "p_historical": hits / k,
        "ci95": (lo, hi),
        "uniform_baseline": 1 / len(checkpoint.considered),
        "choices": dict(choices),
        "fallbacks": fallbacks,
    }
