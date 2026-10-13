"""Visor de partidas: convierte el registro JSONL de una partida en un HTML autónomo
(estética DEFCON, sin servidor). Uso: `cavalier view runs/partida.jsonl`."""

from __future__ import annotations

import json
from pathlib import Path

from .scenario import load_scenario

TEMPLATE = Path(__file__).parent / "viewer_template.html"


def _catalog(scenario_id: str) -> dict:
    bundle = load_scenario(scenario_id)
    sc, rules = bundle.scenario, bundle.rules
    return {
        "title": sc.title,
        "hours_per_turn": sc.hours_per_turn,
        "sides": {s: {"name": c.name, "leader": c.leader} for s, c in sc.sides.items()},
        "actions": {k: {"label": a.label, "stance": a.stance} for k, a in rules.actions.items()},
        "terms": {k: t.label for k, t in sc.deal_terms.items()},
        "vars": {k: v.label for k, v in sc.variables.items()},
    }


def render(log_path: str | Path, out_path: str | Path | None = None) -> Path:
    log_path = Path(log_path)
    header: dict = {}
    turns: list[dict] = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "header":
                header = record
            else:
                turns.append(record)
    if not turns:
        raise ValueError(f"{log_path} no contiene turnos")
    data = {"header": header, "turns": turns, "catalog": _catalog(header.get("scenario", "cuba1962"))}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
    out = Path(out_path) if out_path else log_path.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    return out
