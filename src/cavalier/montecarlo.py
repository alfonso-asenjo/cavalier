"""Lotes de partidas y agregados con intervalos de confianza."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agents.base import Agent
from .agents.scripted import ReplayAgent, ScriptedAgent
from .engine import Game
from .eval.stats import wilson_interval
from .history import load_historical_script
from .models import Outcome, WorldState
from .recorder import JsonlRecorder
from .scenario import ScenarioBundle

AgentFactory = Callable[[ScenarioBundle, int], dict[str, Agent]]


@dataclass
class GameSummary:
    seed: int
    outcome: str
    reason: str
    turns: int
    max_tension: float
    incidents: list[str]
    invalid_actions: int
    fallback_decisions: int


@dataclass
class BatchResult:
    summaries: list[GameSummary] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.summaries)

    def counts(self) -> Counter:
        return Counter(s.outcome for s in self.summaries)

    def proportion(self, outcome: Outcome | str) -> float:
        return self.counts()[str(outcome)] / self.n if self.n else 0.0

    def table(self) -> list[dict]:
        rows = []
        counts = self.counts()
        for outcome in Outcome:
            k = counts[outcome.value]
            lo, hi = wilson_interval(k, self.n)
            rows.append({"outcome": outcome.value, "n": k, "p": k / self.n if self.n else 0.0, "ci95": (lo, hi)})
        return rows

    def clean(self) -> BatchResult:
        """Solo las partidas sin decisiones de respaldo: las otras se reportan aparte."""
        return BatchResult([s for s in self.summaries if s.fallback_decisions == 0])

    def to_dict(self) -> dict:
        return {"n": self.n, "table": self.table(), "games": [asdict(s) for s in self.summaries]}


def checkpoint_path(log_path: str | Path) -> Path:
    return Path(log_path).with_suffix(".ckpt.json")


def _save_checkpoint(path: Path, game: Game, max_tension: float) -> None:
    data = {
        "state": game.state.model_dump(mode="json"),
        "rng": game.rng.bit_generator.state,
        "agents": {s: a.snapshot() for s, a in game.agents.items() if hasattr(a, "snapshot")},
        "max_tension": max_tension,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # atómico: un corte a mitad de escritura no deja un punto de control roto


def _restore_checkpoint(path: Path, game: Game, log_path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    game.state = WorldState.model_validate(data["state"])
    game.rng.bit_generator.state = data["rng"]
    for side, snap in data["agents"].items():
        game.agents[side].restore(snap)
    # El registro puede llevar algún turno posterior al punto de control (corte entre ambas escrituras).
    records = [(ln, json.loads(ln)) for ln in log_path.read_text(encoding="utf-8").splitlines(keepends=True)]
    keep = [ln for ln, r in records if r.get("type") != "turn" or r["turn"] < game.state.turn]
    log_path.write_text("".join(keep), encoding="utf-8")
    return data["max_tension"]


def play(bundle: ScenarioBundle, agents: dict[str, Agent], seed: int, anonymized: bool = False,
         log_path: str | Path | None = None, resume: bool = False, log=None,
         state: WorldState | None = None) -> GameSummary:
    """Con `log_path`, guarda tras cada turno un punto de control (`.ckpt.json`) con el estado real,
    el generador aleatorio y la memoria de los agentes; con `resume=True` continúa desde él. Con
    agentes deterministas el resultado es idéntico al de una partida sin interrupción.
    `state`: estado inicial (bifurcación desde un punto crítico); por defecto, el inicio del escenario."""
    ckpt = checkpoint_path(log_path) if log_path else None
    resuming = bool(resume and ckpt and ckpt.exists() and Path(log_path).exists())
    header = {"scenario": bundle.scenario.id, "seed": seed, "agents": {s: a.name for s, a in agents.items()}}
    game = Game(bundle, agents, seed=seed, anonymized=anonymized, state=state)
    max_tension = game.state.tension
    if resuming:
        max_tension = _restore_checkpoint(ckpt, game, Path(log_path))
        if log:
            log(f"reanudando en el turno {game.state.turn}")
    recorder = JsonlRecorder(log_path, header, append=resuming) if log_path else None
    game.recorder = recorder
    while game.state.outcome is None:
        game.step()
        max_tension = max(max_tension, game.state.tension)
        if ckpt:
            _save_checkpoint(ckpt, game, max_tension)
        if log:
            log(f"turno {game.state.turn - 1} ({game.date_label(game.state.turn - 1)}) · tensión {game.state.tension:.0f}")
    if recorder:
        recorder.close()
    if ckpt and ckpt.exists():
        ckpt.unlink()
    st = game.state
    return GameSummary(
        seed=seed,
        outcome=st.outcome.value,
        reason=st.outcome_reason or "",
        turns=st.turn,
        max_tension=round(max_tension, 1),
        incidents=list(st.fired_incidents),
        invalid_actions=sum(st.invalid_actions.values()),
        fallback_decisions=sum(st.fallback_decisions.values()),
    )


def _play_seeds(bundle: ScenarioBundle, factory: AgentFactory, seeds: list[int], anonymized: bool) -> list[GameSummary]:
    return [play(bundle, factory(bundle, s), s, anonymized) for s in seeds]


def run_batch(bundle: ScenarioBundle, factory: AgentFactory, n: int, seed: int = 0,
              anonymized: bool = False, log_dir: str | Path | None = None,
              executor: Executor | None = None, chunks: int = 16) -> BatchResult:
    """Juega `n` partidas con semillas seed..seed+n-1. Con `executor` (p. ej. ProcessPoolExecutor)
    reparte las semillas en `chunks` bloques; el resultado es idéntico al secuencial.
    La fábrica debe ser serializable (ScriptedFactory, ReplayFactory) para usar procesos."""
    seeds = list(range(seed, seed + n))
    result = BatchResult()
    if executor is not None and log_dir is None and n > 1:
        size = max(1, -(-n // chunks))
        blocks = [seeds[i:i + size] for i in range(0, n, size)]
        for summaries in executor.map(_play_seeds, *zip(*[(bundle, factory, b, anonymized) for b in blocks])):
            result.summaries.extend(summaries)
        return result
    for game_seed in seeds:
        log_path = Path(log_dir) / f"game_{game_seed:05d}.jsonl" if log_dir else None
        result.summaries.append(play(bundle, factory(bundle, game_seed), game_seed, anonymized, log_path))
    return result


@dataclass
class ScriptedFactory:
    personalities: dict[str, str]

    def __call__(self, bundle: ScenarioBundle, seed: int) -> dict[str, Agent]:
        return {side: ScriptedAgent(bundle, side, p, seed=seed * 7919 + i)
                for i, (side, p) in enumerate(self.personalities.items())}


@dataclass
class ReplayFactory:
    script: dict[str, dict[int, list]]

    def __call__(self, bundle: ScenarioBundle, seed: int) -> dict[str, Agent]:
        return {side: ReplayAgent(self.script.get(side, {}), name=f"historical:{side}")
                for side in bundle.scenario.sides}


def scripted_factory(personalities: dict[str, str]) -> AgentFactory:
    return ScriptedFactory(dict(personalities))


def replay_factory(scenario: str | Path) -> AgentFactory:
    return ReplayFactory(load_historical_script(scenario))
