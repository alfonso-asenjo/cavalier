"""Lectura y escritura de variables del estado mediante rutas: `tension`, `self.x`, `other.x`, `vars.x`."""

from __future__ import annotations

from ..models import SideState, WorldState

_SIDE_BOUNDS: dict[str, tuple[float, float]] = {
    "defcon": (1, 5),
    "mobilization": (0, 100),
    "control": (0, 1),
    "domestic_pressure": (0, 100),
    "credibility": (0, 100),
    "intel_quality": (0, 1),
}


def _resolve(state: WorldState, path: str, actor: str | None) -> tuple[object, str]:
    if path == "tension":
        return state, "tension"
    prefix, _, name = path.partition(".")
    if prefix == "vars":
        return state.vars, name
    if prefix in ("self", "other"):
        if actor is None:
            raise ValueError(f"La ruta {path!r} necesita un bando de referencia")
        side = actor if prefix == "self" else other_side(state, actor)
        return state.sides[side], name
    if prefix in state.sides:
        return state.sides[prefix], name
    raise ValueError(f"Ruta de estado desconocida: {path!r}")


def get(state: WorldState, path: str, actor: str | None = None) -> float:
    target, name = _resolve(state, path, actor)
    if isinstance(target, dict):
        return target.get(name, 0.0)
    return getattr(target, name)


def put(state: WorldState, path: str, value: float, actor: str | None = None) -> None:
    target, name = _resolve(state, path, actor)
    if isinstance(target, dict):
        target[name] = min(1.0, max(0.0, value))
    elif isinstance(target, SideState):
        lo, hi = _SIDE_BOUNDS[name]
        clamped = min(hi, max(lo, value))
        setattr(target, name, round(clamped) if name == "defcon" else clamped)
    else:
        setattr(target, name, min(100.0, max(0.0, value)))


def add(state: WorldState, path: str, delta: float, actor: str | None = None) -> None:
    put(state, path, get(state, path, actor) + delta, actor)


def matches(state: WorldState, eq: dict[str, float], gte: dict[str, float] | None = None, actor: str | None = None) -> bool:
    if any(abs(get(state, p, actor) - v) > 1e-9 for p, v in eq.items()):
        return False
    return all(get(state, p, actor) >= v for p, v in (gte or {}).items())


def other_side(state: WorldState, side: str) -> str:
    return next(s for s in state.sides if s != side)
