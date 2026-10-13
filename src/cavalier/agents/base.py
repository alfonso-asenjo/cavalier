from __future__ import annotations

from typing import Protocol

from ..models import Decision, Observation


class Agent(Protocol):
    name: str

    def decide(self, obs: Observation) -> Decision: ...

# Opcional para reanudar partidas: un agente con memoria propia puede exponer
#   snapshot() -> dict   y   restore(data: dict) -> None
# (estado serializable en JSON). Los agentes sin estado no necesitan implementarlos.
