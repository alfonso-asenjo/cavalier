"""Carga de escenarios. Un escenario es una carpeta con tres ficheros TOML:

- scenario.toml  situación: bandos, estado inicial, variables (quién sabe qué), términos negociables
- rules.toml     supuestos del motor: efectos de acciones, dinámica, incidentes, probabilidades
- aliases.toml   renombrado para el modo anonimizado

Separar datos (scenario) de supuestos (rules) es deliberado: las reglas son lo que
se somete a análisis de sensibilidad.
"""

from __future__ import annotations

import tomllib
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .models import SideState, Visibility

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "scenarios"


class SideConfig(BaseModel):
    name: str
    leader: str
    briefing: str
    initial: SideState


class VariableConfig(BaseModel):
    label: str
    initial: float
    visible_to: list[str]  # bandos que conocen el valor real
    noise: float = 0.0  # desviación típica para quien lo observa sin conocerlo (sobre 0..1)
    bias: dict[str, float] = Field(default_factory=dict)  # error sistemático por observador
    hidden_from: list[str] = Field(default_factory=list)  # ni siquiera saben que existe


class DealTermConfig(BaseModel):
    label: str
    obligor: str  # bando que asume la obligación
    value: dict[str, float]  # utilidad para cada bando (la usan los agentes scripted)
    credibility_cost: float = 0.0  # coste para el obligado si el término es público
    secretable: bool = False  # puede pactarse en secreto (p. ej. los Jupiter de Turquía)
    enabled: bool = True  # las variantes pueden desactivar términos
    requires: dict[str, float] = Field(default_factory=dict)  # condiciones para poder pactarlo
    verification: float = 0.0  # prob. mínima de verificar ESTE compromiso (1.0 = observable, p. ej. un compromiso público)
    verifies_deal: bool = False  # si el acuerdo lo incluye, todos sus compromisos se verifican (inspección)
    verification_required: bool = True  # False: se pacta a confianza y no bloquea el cierre de la crisis
    sets: dict[str, float] = Field(default_factory=dict)  # variables que fija al cumplirse


class IncidentConfig(BaseModel):
    id: str
    label: str
    actor: str  # fuerzas de qué bando actúan sin orden
    min_tension: float = 0.0
    base_prob: float
    effects: dict[str, float] = Field(default_factory=dict)
    nuclear_risk: float = 0.0  # probabilidad de que el incidente acabe en uso nuclear
    requires: dict[str, float] = Field(default_factory=dict)
    once: bool = True


class FalseReportConfig(BaseModel):
    target: str  # bando que recibe el informe falso
    label: str
    prob: float


class Scenario(BaseModel):
    id: str
    title: str
    start: datetime
    hours_per_turn: int
    max_turns: int
    sides: dict[str, SideConfig]
    variables: dict[str, VariableConfig] = Field(default_factory=dict)
    deal_terms: dict[str, DealTermConfig] = Field(default_factory=dict)
    initial_tension: float = 50.0
    historical_data: dict = Field(default_factory=dict)  # referencia, con fuentes; el motor no lo usa


class ActionRule(BaseModel):
    label: str
    stance: str = "neutral"  # escalate | deescalate | neutral | terminal (lo usan los agentes scripted)
    visibility: Visibility = Visibility.PUBLIC
    actor: str | None = None  # None = cualquier bando
    requires: dict[str, float] = Field(default_factory=dict)  # igualdad sobre variables
    effects: dict[str, float] = Field(default_factory=dict)  # deltas: tension, self.x, other.x, vars.x
    sets: dict[str, float] = Field(default_factory=dict)  # asignaciones
    outcome: str | None = None  # acción que termina la partida (p. ej. invasión)
    nuclear_escalation: dict[str, float] = Field(default_factory=dict)  # base + contribución de variables


class InteractionRule(BaseModel):
    """Efecto de acciones que coinciden en el mismo turno (p. ej. buques que fuerzan la cuarentena).
    Se evalúa después de aplicar las acciones de postura y antes de las terminales."""

    id: str
    label: str
    actions: dict[str, str]  # bando -> acción; todas deben ocurrir en el turno
    requires: dict[str, float] = Field(default_factory=dict)
    prob: float = 1.0
    effects: dict[str, float] = Field(default_factory=dict)  # rutas explícitas: tension, USA.x, vars.x
    nuclear_risk: float = 0.0


class DriftRule(BaseModel):
    target: str
    delta: float
    when: dict[str, float] = Field(default_factory=dict)  # igualdad
    when_gte: dict[str, float] = Field(default_factory=dict)


class Rules(BaseModel):
    max_actions_per_turn: int = 3
    channels: dict[str, int]  # canal -> retraso adicional en turnos
    actions: dict[str, ActionRule]
    drifts: list[DriftRule] = Field(default_factory=list)
    tension_decay: float = 1.0
    deal_tension_relief: float = 15.0  # al aceptar
    fulfillment_tension_relief: float = 15.0  # al cumplirse y verificarse
    deal_expiry_turns: int = 4
    implementation_turns: int = 4  # turnos para cumplir los compromisos de un acuerdo
    verification_rate: float = 0.5  # prob. por turno de verificar = calidad de inteligencia * tasa
    secret_leak_prob: float = 0.03  # prob. por turno de que se filtre un término secreto
    # Ceder con poca tensión parece debilidad: coste de credibilidad *= 1 + penal * (1 - tensión/100)
    concession_low_tension_penalty: float = 1.0
    concession_pressure_factor: float = 0.5  # presión interna añadida por punto de credibilidad perdido
    deescalation_threshold: float = 20.0
    deescalation_turns: int = 3
    incident_tension_scale: float = 50.0  # p *= exp((tension-50)/scale)
    incident_defcon_factor: float = 0.5  # p *= 1 + factor * (5 - defcon)
    incident_prob_multiplier: float = 1.0  # escala global de incidentes (para sensibilidad)
    observation_noise: float = 12.0  # ruido base sobre variables 0..100 del rival
    incidents: list[IncidentConfig] = Field(default_factory=list)
    false_reports: list[FalseReportConfig] = Field(default_factory=list)
    interactions: list[InteractionRule] = Field(default_factory=list)


class Aliases(BaseModel):
    replacements: dict[str, str] = Field(default_factory=dict)
    date_mode: str = "relative"  # relative: "Día 3, 12:00" en lugar de fechas reales


class ScenarioBundle(BaseModel):
    scenario: Scenario
    rules: Rules
    aliases: Aliases


def _read(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        elif isinstance(value, dict) and isinstance(out.get(key), list):
            # {"incidents": {"1": {...}}} modifica el elemento 1 de la lista
            items = list(out[key])
            for idx, item in value.items():
                i = int(idx)
                items[i] = deep_merge(items[i], item) if isinstance(item, dict) else item
            out[key] = items
        else:
            out[key] = value
    return out


def load_scenario(
    name_or_path: str | Path,
    variant: str | None = None,
    overrides: dict | None = None,
) -> ScenarioBundle:
    """Carga un escenario. `variant` aplica `variants/<variant>.toml` (secciones [scenario] y [rules])
    sobre la base: así se definen escenarios perturbados para el control de fuga.
    `overrides` tiene la misma forma y se aplica después (análisis de sensibilidad)."""
    path = Path(name_or_path)
    if not path.is_dir():
        path = SCENARIOS_DIR / str(name_or_path)
    scenario_data = _read(path / "scenario.toml")
    rules_data = _read(path / "rules.toml")
    overlays = [_read(path / "variants" / f"{variant}.toml")] if variant else []
    overlays += [overrides] if overrides else []
    for overlay in overlays:
        scenario_data = deep_merge(scenario_data, overlay.get("scenario", {}))
        rules_data = deep_merge(rules_data, overlay.get("rules", {}))
    aliases_path = path / "aliases.toml"
    return ScenarioBundle(
        scenario=Scenario.model_validate(scenario_data),
        rules=Rules.model_validate(rules_data),
        aliases=Aliases.model_validate(_read(aliases_path)) if aliases_path.exists() else Aliases(),
    )
