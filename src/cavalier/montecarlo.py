"""Lotes de partidas y agregados con intervalos de confianza."""

from __future__ import annotations

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
from .models import Outcome
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


def play(bundle: ScenarioBundle, agents: dict[str, Agent], seed: int, anonymized: bool = False,
         log_path: Path | None = None) -> GameSummary:
    recorder = JsonlRecorder(log_path, {"scenario": bundle.scenario.id, "seed": seed,
                                        "agents": {s: a.name for s, a in agents.items()}}) if log_path else None
    game = Game(bundle, agents, seed=seed, anonymized=anonymized, recorder=recorder)
    max_tension = game.state.tension
    while game.state.outcome is None:
        game.step()
        max_tension = max(max_tension, game.state.tension)
    if recorder:
        recorder.close()
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
