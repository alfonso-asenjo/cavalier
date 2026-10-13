"""Análisis de sensibilidad de los supuestos del motor.

Primera versión: un factor cada vez (OAT), ±delta sobre cada parámetro.
Pendiente (plan, fase 1): cribado de Morris para interacciones entre parámetros.
"""

from __future__ import annotations

from .montecarlo import AgentFactory, run_batch
from .models import Outcome
from .scenario import load_scenario

DEFAULT_PARAMS = [
    "rules.tension_decay",
    "rules.deal_tension_relief",
    "rules.incident_tension_scale",
    "rules.incident_defcon_factor",
    "rules.incident_prob_multiplier",
    "rules.observation_noise",
    "rules.incidents.0.base_prob",
    "rules.incidents.1.base_prob",
    "rules.incidents.1.nuclear_risk",
    "rules.actions.impose_quarantine.effects.tension",
    "rules.actions.mobilize.effects.tension",
]


def _get(d, path: list[str]):
    for key in path:
        d = d[int(key)] if isinstance(d, list) else d[key]
    return d


def _nested(path: list[str], value) -> dict:
    out: dict = value
    for key in reversed(path):
        out = {key: out}
    return out


def one_at_a_time(scenario: str, factory: AgentFactory, n: int, delta: float = 0.2,
                  params: list[str] | None = None, seed: int = 0, variant: str | None = None) -> list[dict]:
    base_bundle = load_scenario(scenario, variant)
    base = run_batch(base_bundle, factory, n, seed)
    raw = {"rules": base_bundle.rules.model_dump(), "scenario": base_bundle.scenario.model_dump()}
    rows = []
    for param in params or DEFAULT_PARAMS:
        path = param.split(".")
        value = _get(raw, path)
        row = {"param": param, "base_value": value}
        for label, factor in (("low", 1 - delta), ("high", 1 + delta)):
            bundle = load_scenario(scenario, variant, overrides=_nested(path, value * factor))
            res = run_batch(bundle, factory, n, seed)  # mismas semillas: números aleatorios comunes
            row[label] = {o.value: res.proportion(o) - base.proportion(o) for o in Outcome}
        rows.append(row)
    return [{"param": "base", "proportions": {o.value: base.proportion(o) for o in Outcome}}] + rows
