"""Tipos del simulador: estado del mundo, acciones, decisiones y observaciones.

Todo lo militar es abstracto: niveles de alerta, movilización y disciplina de mando.
No hay armas, trayectorias ni objetivos.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Outcome(StrEnum):
    NEGOTIATED = "acuerdo_negociado"
    DEESCALATED = "desescalada"
    UNRESOLVED = "sin_resolver"
    CONVENTIONAL = "conflicto_convencional"
    NUCLEAR = "uso_nuclear"


class Visibility(StrEnum):
    PUBLIC = "public"  # el otro bando lo ve siempre
    DETECTABLE = "detectable"  # el otro bando lo ve con probabilidad = su calidad de inteligencia
    COVERT = "covert"  # el otro bando no lo ve


class SideState(BaseModel):
    defcon: int = 5  # 5 = paz, 1 = máxima alerta
    mobilization: float = 0.0  # 0..100
    control: float = 0.8  # 0..1, disciplina de la cadena de mando
    domestic_pressure: float = 30.0  # 0..100
    credibility: float = 50.0  # 0..100
    intel_quality: float = 0.5  # 0..1


class Message(BaseModel):
    sender: str
    recipient: str
    channel: str
    text: str
    sent_turn: int
    deliver_turn: int


class DealTerm(BaseModel):
    term: str
    secret: bool = False


class Deal(BaseModel):
    id: str
    proposer: str
    terms: list[DealTerm]
    proposed_turn: int
    deliver_turn: int  # las propuestas viajan por un canal y llegan con retraso
    channel: str
    status: str = "open"  # open | accepted | rejected | expired


class Action(BaseModel):
    kind: str
    text: str | None = None  # mensajes y declaraciones
    channel: str | None = None  # mensajes privados
    terms: list[DealTerm] | None = None  # propuestas de acuerdo
    deal_id: str | None = None  # aceptar / rechazar


class Decision(BaseModel):
    actions: list[Action] = Field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.5
    cited_docs: list[str] = Field(default_factory=list)
    # True si el agente no produjo una decisión válida y se usó una de respaldo.
    # Esas partidas se reportan aparte: tratar fallos como "esperar" sesga hacia la desescalada.
    fallback: bool = False


class Event(BaseModel):
    """Algo que ocurre en el mundo. `seen_by` decide qué bandos lo observan."""

    turn: int
    kind: str  # action | incident | intel | deal | message | outcome
    actor: str | None = None
    description: str
    seen_by: list[str] = Field(default_factory=list)
    is_false: bool = False  # inteligencia falsa: solo el log de verdad lo sabe


class WorldState(BaseModel):
    turn: int = 0
    tension: float = 50.0  # 0..100
    sides: dict[str, SideState]
    vars: dict[str, float] = Field(default_factory=dict)
    messages: list[Message] = Field(default_factory=list)
    deals: dict[str, Deal] = Field(default_factory=dict)
    events: list[Event] = Field(default_factory=list)
    low_tension_streak: int = 0
    fired_incidents: list[str] = Field(default_factory=list)
    invalid_actions: dict[str, int] = Field(default_factory=dict)
    fallback_decisions: dict[str, int] = Field(default_factory=dict)
    outcome: Outcome | None = None
    outcome_reason: str | None = None


class Observation(BaseModel):
    """Lo que un bando cree saber al comienzo de un turno."""

    side: str
    turn: int
    date: str
    own: SideState
    other_side: str
    other_estimate: dict[str, float]  # variables del rival con ruido
    tension_estimate: float
    vars: dict[str, float]  # solo las variables visibles para este bando (con ruido si procede)
    inbox: list[Message]
    events: list[Event]  # eventos vistos desde el turno anterior (incluye inteligencia falsa)
    open_deals: list[Deal]
    legal_actions: dict[str, str]  # kind -> descripción
    deal_terms: dict[str, str]  # term -> descripción
    channels: list[str]
    max_actions: int
