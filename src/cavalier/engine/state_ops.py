"""Lectura y escritura de variables del estado mediante rutas: `tension`, `self.x`, `other.x`, `<BANDO>.x`, `vars.x`.

Las magnitudes continuas acotadas (tensión, presión, credibilidad, control...) se actualizan en
escala logit: un delta de d puntos vale ~d en el centro del rango y cada vez menos cerca de los
extremos. Así no hay saturación: dos situaciones "muy tensas" siguen siendo distinguibles y los
parámetros siguen teniendo efecto (en la v1 la tensión se quedaba clavada en 100).
Las variables de escenario (`vars.*`) son progresos o banderas y se actualizan de forma lineal.
"""

from __future__ import annotations

import math

from ..models import SideState, WorldState

_SIDE_BOUNDS: dict[str, tuple[float, float]] = {
    "defcon": (1, 5),
    "mobilization": (0, 100),
    "control": (0, 1),
    "domestic_pressure": (0, 100),
    "credibility": (0, 100),
    "intel_quality": (0, 1),
}
_TENSION_BOUNDS = (0.0, 100.0)
_EPS = 1e-3  # fracción mínima del rango: los extremos nunca se alcanzan del todo


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


def _bounds(target: object, name: str) -> tuple[float, float] | None:
    if isinstance(target, dict):
        return None
    if isinstance(target, SideState):
        return None if name == "defcon" else _SIDE_BOUNDS[name]
    return _TENSION_BOUNDS


def bounded_add(value: float, delta: float, lo: float, hi: float) -> float:
    """Suma en escala logit con pendiente 1 en el centro del rango [lo, hi]."""
    span = hi - lo
    x = min(1 - _EPS, max(_EPS, (value - lo) / span))
    z = math.log(x / (1 - x)) + 4.0 * delta / span
    return lo + span / (1 + math.exp(-z))


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
    target, name = _resolve(state, path, actor)
    bounds = _bounds(target, name)
    current = get(state, path, actor)
    if bounds is None:
        put(state, path, current + delta, actor)
    else:
        put(state, path, bounded_add(current, delta, *bounds), actor)


def matches(state: WorldState, eq: dict[str, float], gte: dict[str, float] | None = None, actor: str | None = None) -> bool:
    if any(abs(get(state, p, actor) - v) > 1e-9 for p, v in eq.items()):
        return False
    return all(get(state, p, actor) >= v for p, v in (gte or {}).items())


def other_side(state: WorldState, side: str) -> str:
    return next(s for s in state.sides if s != side)
