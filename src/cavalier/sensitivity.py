"""Análisis de sensibilidad de los supuestos del motor.

- `one_at_a_time`: un factor cada vez, ±delta sobre el valor base (exploración rápida).
- `morris`: cribado global sobre los rangos plausibles de ranges.toml (criterio de salida de F1).
"""

from __future__ import annotations

import tomllib
from concurrent.futures import Executor
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .models import Outcome
from .montecarlo import AgentFactory, run_batch
from .scenario import SCENARIOS_DIR, deep_merge, load_scenario

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


# --------------------------------------------------------------------------- Morris

class ParamRange(BaseModel):
    low: float
    high: float
    basis: str
    why: str


def load_ranges(scenario: str) -> dict[str, ParamRange]:
    path = Path(scenario) if Path(scenario).is_dir() else SCENARIOS_DIR / scenario
    with (path / "ranges.toml").open("rb") as f:
        raw = tomllib.load(f)
    return {k: ParamRange.model_validate(v) for k, v in raw.get("params", {}).items()}


METRICS = {
    "nuclear": lambda r: r.proportion(Outcome.NUCLEAR),
    "guerra": lambda r: r.proportion(Outcome.NUCLEAR) + r.proportion(Outcome.CONVENTIONAL),
    "acuerdo": lambda r: r.proportion(Outcome.NEGOTIATED),
}


def morris(scenario: str, factory: AgentFactory, n: int, trajectories: int = 10, seed: int = 0,
           variant: str | None = None, ranges: dict[str, ParamRange] | None = None,
           executor: Executor | None = None) -> dict:
    """Cribado de Morris (efectos elementales, 4 niveles, Δ = 2/3) sobre los rangos de ranges.toml.

    Cada evaluación es un lote de `n` partidas con las mismas semillas (números aleatorios comunes),
    de modo que la diferencia entre dos puntos refleja el parámetro y no el azar.
    Devuelve, por métrica, μ* (importancia), σ (no linealidad / interacciones) y la cuota de μ*.
    """
    ranges = ranges or load_ranges(scenario)
    names = list(ranges)
    base_raw = {"rules": load_scenario(scenario, variant).rules.model_dump()}
    is_int = {p: isinstance(_get(base_raw, p.split(".")), int) for p in names}
    rng = np.random.default_rng(seed)
    delta = 2.0 / 3.0

    def evaluate(x: np.ndarray) -> dict[str, float]:
        overrides: dict = {}
        for name, xi in zip(names, x):
            rg = ranges[name]
            value = rg.low + xi * (rg.high - rg.low)
            value = round(value) if is_int[name] else value
            overrides = deep_merge(overrides, _nested(name.split("."), value))
        result = run_batch(load_scenario(scenario, variant, overrides=overrides), factory, n, seed, executor=executor)
        return {m: f(result) for m, f in METRICS.items()}

    effects: dict[str, dict[str, list[float]]] = {m: {p: [] for p in names} for m in METRICS}
    for _ in range(trajectories):
        x = rng.choice([0.0, 1.0 / 3.0], size=len(names))
        y = evaluate(x)
        for i in rng.permutation(len(names)):
            x_next = x.copy()
            x_next[i] += delta
            y_next = evaluate(x_next)
            for m in METRICS:
                effects[m][names[i]].append((y_next[m] - y[m]) / delta)
            x, y = x_next, y_next

    out: dict = {"n": n, "trajectories": trajectories, "metrics": {}}
    for m in METRICS:
        mu_star = {p: float(np.mean(np.abs(e))) for p, e in effects[m].items()}
        total = sum(mu_star.values()) or 1.0
        out["metrics"][m] = sorted(
            ({"param": p, "mu_star": mu_star[p], "sigma": float(np.std(effects[m][p])),
              "share": mu_star[p] / total, "basis": ranges[p].basis} for p in names),
            key=lambda row: row["mu_star"], reverse=True,
        )
    return out
