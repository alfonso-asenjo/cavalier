from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporción (95 % por defecto)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def runs_for_margin(p: float, margin: float, z: float = 1.96) -> int:
    """Partidas necesarias para estimar una proporción p con ± margin."""
    return math.ceil(z * z * p * (1 - p) / (margin * margin))
