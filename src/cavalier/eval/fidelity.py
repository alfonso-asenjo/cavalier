"""Fidelidad histórica en los puntos de control (fase 3, protocolo 4.2).

Condiciones por punto de control, K muestras cada una:
  - LLM con persona histórica y LLM sin persona, en modo real y anonimizado;
  - agentes scripted (las cinco personalidades), como referencia sin lenguaje;
  - azar uniforme sobre las opciones consideradas de verdad (analítico).
Métrica: frecuencia de la opción histórica, con IC 95 % de Wilson, por punto y nunca agregada.
La diferencia real − anonimizado se contrasta con la prueba exacta de Fisher (fuga, 4.3).

Las muestras se intercalan (muestra i de todas las celdas antes de la i+1): si la ejecución se
corta, todas las celdas tienen un K parcial equilibrado y se puede reanudar con `done`.
"""

from __future__ import annotations

from collections import Counter

from ..agents.scripted import PERSONALITIES, ReplayAgent, ScriptedAgent
from ..engine import Game
from ..history import load_checkpoints
from ..scenario import load_scenario
from .bench import _decide
from .checkpoints import classify, reconstruct_state
from .stats import fisher_exact, wilson_interval

LLM_CONDITIONS = [("historico", False), ("historico", True), ("ninguna", False), ("ninguna", True)]


def condition_name(persona: str, anonymized: bool) -> str:
    return f"llm:{persona}:{'anon' if anonymized else 'real'}"


def _key(d: dict) -> tuple:
    return (d["checkpoint"], d["condition"], d["seed"])


def scripted_choices(scenario: str, k: int, seed: int = 0) -> list[dict]:
    bundle = load_scenario(scenario)
    out = []
    for cp in load_checkpoints(scenario):
        state = reconstruct_state(bundle, scenario, cp)
        for name in PERSONALITIES:
            for i in range(k):
                agent = ScriptedAgent(bundle, cp.side, name, seed=seed + i)
                idle = {s: ReplayAgent({}) for s in bundle.scenario.sides}
                game = Game(bundle, idle, seed=seed + i, state=state)
                decision = agent.decide(game.observe(cp.side))
                out.append({"checkpoint": cp.id, "condition": f"scripted:{name}", "seed": seed + i,
                            "choice": classify(decision, cp.considered), "historical": cp.historical,
                            "fallback": False})
    return out


def run_fidelity(client, scenario: str = "cuba1962", k: int = 30, seed: int = 0,
                 done: list[dict] | None = None, on_progress=None, log=print) -> list[dict]:
    bundle = load_scenario(scenario)
    checkpoints = load_checkpoints(scenario)
    previous = {_key(d): d for d in (done or [])}
    records: list[dict] = []
    for i in range(k):
        for cp in checkpoints:
            for persona, anonymized in LLM_CONDITIONS:
                cond = condition_name(persona, anonymized)
                key = (cp.id, cond, seed + i)
                d = previous.get(key)
                if d is None:
                    d = _decide(bundle, scenario, cp, client, anonymized, seed + i, persona=persona)
                    d["condition"] = cond
                    records.append(d)
                    if on_progress:
                        on_progress(records)
                else:
                    records.append(d)
        log(f"muestra {i + 1}/{k} completa ({len(records)} decisiones)")
    return records


def report(records: list[dict], scenario: str = "cuba1962") -> dict:
    """Tabla por punto de control y condición, más el contraste real/anonimizado por persona."""
    out = {}
    for cp in load_checkpoints(scenario):
        rows = {}
        for cond in sorted({d["condition"] for d in records if d["checkpoint"] == cp.id}):
            ds = [d for d in records if d["checkpoint"] == cp.id and d["condition"] == cond]
            hits = sum(d["choice"] == cp.historical for d in ds)
            lo, hi = wilson_interval(hits, len(ds))
            rows[cond] = {"n": len(ds), "hits": hits, "p": hits / len(ds), "ci95": (lo, hi),
                          "fallbacks": sum(d["fallback"] for d in ds),
                          "choices": dict(Counter(d["choice"] for d in ds).most_common())}
        leakage = {}
        for persona in {p for p, _ in LLM_CONDITIONS}:
            real, anon = rows.get(condition_name(persona, False)), rows.get(condition_name(persona, True))
            if real and anon:
                leakage[persona] = {"real_minus_anon": real["p"] - anon["p"],
                                    "fisher_p": fisher_exact(real["hits"], real["n"] - real["hits"],
                                                             anon["hits"], anon["n"] - anon["hits"])}
        out[cp.id] = {"side": cp.side, "historical": cp.historical, "uniform": 1 / len(cp.considered),
                      "conditions": rows, "real_vs_anon": leakage}
    return out
