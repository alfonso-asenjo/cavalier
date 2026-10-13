"""Experimentos contrafactuales por bifurcación (fase 5).

Se reconstruye el estado histórico en un punto de control (mismo pasado para todas las celdas) y, desde
ahí, cada celda juega con su variante del escenario. Así el contrafactual empieza en el momento elegido
("¿y si el Sábado Negro no hubiera existido el canal secreto?") en vez de cambiar toda la crisis.

Cada partida guarda su JSONL, su punto de control por turno y un resumen. Las semillas se intercalan
entre celdas (la semilla i de todas antes que la i+1): si la ejecución se corta, las celdas quedan
equilibradas, y al relanzar se saltan las partidas terminadas y se reanuda la que iba por la mitad.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from .eval.checkpoints import reconstruct_state
from .eval.stats import fisher_exact, wilson_interval
from .history import load_checkpoints
from .models import Outcome
from .montecarlo import GameSummary, play
from .scenario import ScenarioBundle, load_scenario

ESCALATION = {Outcome.CONVENTIONAL.value, Outcome.NUCLEAR.value}
BASE = "base"


def fork_state(scenario: str, checkpoint_id: str):
    """Estado histórico en el punto de control (con sus incidentes forzados), común a todas las celdas."""
    cp = next(c for c in load_checkpoints(scenario) if c.id == checkpoint_id)
    return reconstruct_state(load_scenario(scenario), scenario, cp)


def run_experiment(scenario: str, checkpoint_id: str, cells: list[str], n: int, out_dir: str | Path,
                   factory: Callable[[ScenarioBundle, int], dict], seed: int = 0, anonymized: bool = True,
                   log=print) -> dict[str, list[GameSummary]]:
    out_dir = Path(out_dir)
    state = fork_state(scenario, checkpoint_id)
    bundles = {c: load_scenario(scenario, None if c == BASE else c) for c in cells}
    results: dict[str, list[GameSummary]] = {c: [] for c in cells}
    for game_seed in range(seed, seed + n):
        for cell in cells:
            summary_path = out_dir / cell / f"s{game_seed:03d}.json"
            if summary_path.exists():
                results[cell].append(GameSummary(**json.loads(summary_path.read_text(encoding="utf-8"))))
                continue
            summary = play(bundles[cell], factory(bundles[cell], game_seed), game_seed, anonymized=anonymized,
                           log_path=out_dir / cell / f"s{game_seed:03d}.jsonl", resume=True, state=state)
            summary_path.write_text(json.dumps(asdict(summary), ensure_ascii=False), encoding="utf-8")
            results[cell].append(summary)
            log(f"{cell} s{game_seed}: {summary.outcome} en el turno {summary.turns}")
    return results


def load_results(out_dir: str | Path) -> dict[str, list[GameSummary]]:
    out_dir = Path(out_dir)
    return {d.name: [GameSummary(**json.loads(p.read_text(encoding="utf-8"))) for p in sorted(d.glob("s*.json"))]
            for d in sorted(out_dir.iterdir()) if d.is_dir()}


def _holm(pvalues: dict) -> dict:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, p * (len(ordered) - i)))
        adjusted[key] = running
    return adjusted


def report(results: dict[str, list[GameSummary]], metrics: dict[str, set[str]] | None = None) -> dict:
    """Proporciones por celda con IC de Wilson y contrastes de cada celda frente a la base
    (Fisher bilateral, Holm sobre todas las comparaciones celda × métrica)."""
    metrics = metrics or {"acuerdo": {Outcome.NEGOTIATED.value}, "escalada_militar": ESCALATION}
    cells = {}
    for cell, games in results.items():
        n = len(games)
        row = {"n": n, "outcomes": dict(Counter(g.outcome for g in games).most_common()),
               "fallback_games": sum(1 for g in games if g.fallback_decisions),
               "turns_mean": sum(g.turns for g in games) / n if n else 0.0}
        for name, outs in metrics.items():
            k = sum(g.outcome in outs for g in games)
            row[name] = {"k": k, "p": k / n if n else 0.0, "ci95": wilson_interval(k, n)}
        cells[cell] = row
    raw = {}
    base = cells.get(BASE)
    if base:
        for cell, row in cells.items():
            if cell == BASE:
                continue
            for name in metrics:
                a, b = row[name]["k"], base[name]["k"]
                raw[f"{cell}/{name}"] = fisher_exact(a, row["n"] - a, b, base["n"] - b)
    adjusted = _holm(raw)
    contrasts = {key: {"diff": cells[key.split("/")[0]][key.split("/")[1]]["p"] - base[key.split("/")[1]]["p"],
                       "fisher_p": p, "holm_p": adjusted[key]} for key, p in raw.items()}
    return {"cells": cells, "contrasts": contrasts}
