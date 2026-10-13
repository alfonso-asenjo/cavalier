from __future__ import annotations

from typing import Protocol

from ..models import Decision, Observation


class Agent(Protocol):
    name: str

    def decide(self, obs: Observation) -> Decision: ...
