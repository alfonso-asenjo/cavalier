"""Secuencia histórica (historical.toml) y puntos de control (checkpoints.toml) de un escenario."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from .models import Action
from .scenario import SCENARIOS_DIR


class Checkpoint(BaseModel):
    id: str
    side: str
    turn: int
    description: str
    considered: list[str]
    historical: str
    force_incidents: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


def _dir(scenario: str | Path) -> Path:
    path = Path(scenario)
    return path if path.is_dir() else SCENARIOS_DIR / str(scenario)


def load_historical_script(scenario: str | Path) -> dict[str, dict[int, list[Action]]]:
    with (_dir(scenario) / "historical.toml").open("rb") as f:
        raw = tomllib.load(f)
    script: dict[str, dict[int, list[Action]]] = {}
    for entry in raw.get("actions", []):
        action = Action.model_validate({k: v for k, v in entry.items() if k not in ("turn", "side", "note")})
        script.setdefault(entry["side"], {}).setdefault(entry["turn"], []).append(action)
    return script


def load_checkpoints(scenario: str | Path) -> list[Checkpoint]:
    with (_dir(scenario) / "checkpoints.toml").open("rb") as f:
        raw = tomllib.load(f)
    return [Checkpoint.model_validate(c) for c in raw.get("checkpoints", [])]
