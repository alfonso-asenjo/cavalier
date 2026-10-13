import json

import pytest

from cavalier.agents.scripted import ScriptedAgent
from cavalier.montecarlo import checkpoint_path, play


class _Crash(Exception):
    pass


class _CrashingAgent(ScriptedAgent):
    """Se corta en un turno dado, como si el proceso muriera a mitad de partida."""

    crash_turn = 3

    def decide(self, obs):
        if obs.turn == self.crash_turn:
            raise _Crash
        return super().decide(obs)


def _agents(bundle, cls_usa=ScriptedAgent):
    return {"USA": cls_usa(bundle, "USA", "pragmatica", seed=11), "USSR": ScriptedAgent(bundle, "USSR", "defensiva", seed=12)}


def _turns(path):
    return [r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r["type"] == "turn"]


@pytest.mark.parametrize("seed", [1, 4])
def test_interrupted_game_resumes_to_the_same_result(bundle, tmp_path, seed):
    full_log = tmp_path / "full.jsonl"
    full = play(bundle, _agents(bundle), seed, log_path=full_log)
    assert full.turns > _CrashingAgent.crash_turn + 1

    log = tmp_path / "cut.jsonl"
    with pytest.raises(_Crash):
        play(bundle, _agents(bundle, _CrashingAgent), seed, log_path=log)
    assert checkpoint_path(log).exists()
    assert len(_turns(log)) == _CrashingAgent.crash_turn

    resumed = play(bundle, _agents(bundle), seed, log_path=log, resume=True)
    assert resumed.__dict__ == full.__dict__
    assert _turns(log) == _turns(full_log)
    assert not checkpoint_path(log).exists()
