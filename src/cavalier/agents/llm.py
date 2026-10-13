"""Agente LLM orquestado con LangGraph.

Grafo:  retrieve → deliberate → validate ─┬→ END
                       ↑                   │ (error y quedan intentos)
                       └───────────────────┘
Si se agotan los intentos se devuelve "esperar" marcado como `fallback=True`
(se cuenta y se reporta aparte: no es una decisión del agente).

El agente ve códigos opacos (A01, T1...) en lugar de las claves internas de acciones y
términos, de modo que el modo real y el anonimizado solo difieren en los nombres.
"""

from __future__ import annotations

import copy
import json
from datetime import timedelta
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from ..anonymize import Anonymizer
from ..models import Action, DealTerm, Decision, Observation
from ..rag.corpus import Document
from ..rag.retriever import TemporalRetriever
from ..scenario import ScenarioBundle
from .llm_client import LLMClient

PROMPTS = Path(__file__).parent / "prompts"

PERSONAS = {
    "historico": "Actúa según la doctrina, los intereses y el estilo de decisión documentados de este dirigente en su época.",
    "halcon": "Tienes una alta disposición a escalar: crees que la firmeza es lo único que el rival respeta.",
    "paloma": "Buscas acuerdos y estás dispuesto a hacer concesiones para evitar una guerra.",
    "defensiva": "Priorizas la supervivencia y la disuasión; no das el primer paso militar.",
    "pragmatica": "Intentas maximizar tus objetivos políticos minimizando riesgos y pérdidas.",
    "impredecible": "Desconfías de la información que recibes y tiendes a interpretar las señales del rival como amenazas.",
}


_FIELD_HINTS = {
    "private_message": " (requiere `text` con el mensaje y `channel`)",
    "public_statement": " (requiere `text` con la declaración)",
    "propose_deal": " (requiere `terms`: lista de códigos de término; `secret` opcional)",
    "accept_deal": " (requiere `deal_id`)",
    "reject_deal": " (requiere `deal_id`)",
    "renege_deal": " (requiere `deal_id` del acuerdo que rompes)",
}

# ---------------------------------------------------------------- esquema de salida del LLM

class LLMTerm(BaseModel):
    code: str = Field(description="Código del término, p. ej. T1")
    secret: bool = False


class LLMAction(BaseModel):
    code: str = Field(description="Código de la acción, p. ej. A03")
    text: str | None = Field(None, description="Texto del mensaje o declaración")
    channel: str | None = Field(None, description="Canal para mensajes y propuestas")
    terms: list[LLMTerm] | None = Field(None, description="Términos de una propuesta de acuerdo")
    deal_id: str | None = Field(None, description="Identificador de la propuesta a aceptar o rechazar")


class LLMDecision(BaseModel):
    reasoning: str
    actions: list[LLMAction]
    confidence: float = Field(ge=0, le=1)
    cited_docs: list[str] = Field(default_factory=list)


class _GraphState(TypedDict, total=False):
    obs: Observation
    docs: list[Document]
    prompt: str
    raw: str
    error: str | None
    attempts: int
    decision: Decision


class LLMAgent:
    def __init__(
        self,
        bundle: ScenarioBundle,
        side: str,
        client: LLMClient,
        persona: str = "historico",
        anonymized: bool = False,
        retriever: TemporalRetriever | None = None,
        max_attempts: int = 2,
        history_turns: int = 6,
    ):
        self.bundle = bundle
        self.side = side
        self.client = client
        self.persona = persona
        self.anon = Anonymizer(bundle.aliases, enabled=anonymized)
        self.retriever = retriever
        self.max_attempts = max_attempts
        self.history_turns = history_turns
        self.history: list[str] = []
        self.errors: list[str] = []  # motivos de reintento de la última decisión (diagnóstico)
        self.name = f"llm:{client.model_id}:{persona}{':anon' if anonymized else ''}"

        actions = list(bundle.rules.actions)
        terms = list(bundle.scenario.deal_terms)
        self.action_codes = {f"A{i + 1:02d}": k for i, k in enumerate(actions)}
        self.term_codes = {f"T{i + 1}": k for i, k in enumerate(terms)}
        self._action_to_code = {k: c for c, k in self.action_codes.items()}
        self._term_to_code = {k: c for c, k in self.term_codes.items()}
        self.schema = LLMDecision.model_json_schema()
        # Razonamiento corto: en una GPU de 8 GB el coste está en los tokens generados (plan, 4.4).
        self.schema["properties"]["reasoning"]["maxLength"] = 600
        self.system_prompt = self._system_prompt()
        self.graph = self._build_graph()

    def snapshot(self) -> dict:
        return {"history": list(self.history)}

    def restore(self, data: dict) -> None:
        self.history = list(data["history"])

    # ---------------------------------------------------------------- prompts

    def _system_prompt(self) -> str:
        sc = self.bundle.scenario
        cfg = sc.sides[self.side]
        template = (PROMPTS / "system.md").read_text(encoding="utf-8")
        return self.anon(template.format(
            leader=cfg.leader, side_name=cfg.name, persona=PERSONAS[self.persona],
            briefing=cfg.briefing.strip(), hours=sc.hours_per_turn, max_actions=self.bundle.rules.max_actions_per_turn,
        ))

    def _side_name(self, side: str) -> str:
        return self.bundle.scenario.sides[side].name

    def render(self, obs: Observation, docs: list[Document]) -> str:
        sc, own = self.bundle.scenario, obs.own
        lines = [f"## Situación — {obs.date} (turno {obs.turn})"]
        lines += ["", "### Tu estado",
                  f"- Nivel de alerta (DEFCON, 5 = paz, 1 = máxima): {own.defcon}",
                  f"- Movilización: {own.mobilization:.0f}/100",
                  f"- Disciplina de tu cadena de mando: {own.control:.2f}",
                  f"- Presión política interna: {own.domestic_pressure:.0f}/100",
                  f"- Credibilidad internacional: {own.credibility:.0f}/100",
                  f"- Calidad de tu inteligencia: {own.intel_quality:.2f}"]
        est = obs.other_estimate
        lines += ["", f"### Estimación sobre {self._side_name(obs.other_side)} (incierta)",
                  f"- DEFCON estimado: {est.get('defcon', 5):.0f}",
                  f"- Movilización estimada: {est.get('mobilization', 0):.0f}/100",
                  f"- Presión interna estimada: {est.get('domestic_pressure', 0):.0f}/100",
                  "", f"### Tensión internacional estimada: {obs.tension_estimate:.0f}/100"]
        if obs.vars:
            lines += ["", "### Lo que sabes de la situación (0 = no, 1 = sí / completo; puede ser una estimación)"]
            lines += [f"- {sc.variables[k].label}: {v:.2f}" for k, v in obs.vars.items()]
        lines += ["", "### Desde el último turno"]
        lines += [f"- [{e.kind}] {e.description}" for e in obs.events] or ["- Nada relevante."]
        if obs.inbox:
            lines += ["", "### Mensajes recibidos"]
            lines += [f"- ({m.channel}) {m.text}" for m in obs.inbox]
        if obs.open_deals:
            lines += ["", "### Propuestas de acuerdo abiertas"]
            for d in obs.open_deals:
                who = "tuya" if d.proposer == self.side else f"de {self._side_name(d.proposer)}"
                terms = ", ".join(f"{self._term_to_code[t.term]}{' (secreto)' if t.secret else ''}" for t in d.terms)
                lines.append(f"- {d.id} ({who}): {terms}")
        if obs.commitments:
            lines += ["", "### Compromisos de acuerdos aceptados"]
            status = {"pending": "pendiente / sin verificar", "fulfilled": "cumplido", "broken": "roto"}
            for c in obs.commitments:
                who = "tú" if c.obligor == self.side else self._side_name(c.obligor)
                extra = (" (secreto)" if c.secret else "") + (" (filtrado)" if c.leaked else "")
                lines.append(f"- {c.deal_id} · {self._term_to_code[c.term]}{extra}: obligado {who}, "
                             f"plazo turno {c.due_turn}, {status.get(c.status, c.status)}")
        if self.history:
            lines += ["", "### Tus decisiones anteriores"] + self.history[-self.history_turns:]
        if docs:
            lines += ["", "### Documentos disponibles"]
            lines += [f"[{d.id}] {d.title} ({d.date:%Y-%m-%d}): {d.text[:700]}" for d in docs]
        lines += ["", f"### Acciones disponibles (máximo {obs.max_actions})"]
        lines += [f"- {self._action_to_code[k]}: {label}{_FIELD_HINTS.get(k, '')}"
                  for k, label in obs.legal_actions.items()]
        lines += ["", "### Términos negociables"]
        lines += [f"- {self._term_to_code[k]}: {label.replace(f'obliga a {self.side}', 'te obliga a ti').replace(f'obliga a {obs.other_side}', 'obliga al rival')}"
                  + (" (puede pactarse en secreto)" if self.bundle.scenario.deal_terms[k].secretable else "")
                  for k, label in obs.deal_terms.items()]
        channels = {c: d for c, d in self.bundle.rules.channels.items()}
        lines += ["", "### Canales: " + ", ".join(f"{c} (retraso extra {d} turnos)" for c, d in channels.items())]
        return self.anon("\n".join(lines))

    # ---------------------------------------------------------------- grafo

    def _build_graph(self):
        g = StateGraph(_GraphState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("deliberate", self._deliberate)
        g.add_node("validate", self._validate)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "deliberate")
        g.add_edge("deliberate", "validate")
        g.add_conditional_edges("validate", self._route, {"retry": "deliberate", "done": END})
        return g.compile()

    def _retrieve(self, st: _GraphState) -> _GraphState:
        obs = st["obs"]
        if not self.retriever:
            return {"docs": []}
        at = self.bundle.scenario.start + timedelta(hours=self.bundle.scenario.hours_per_turn * obs.turn)
        query = " ".join(e.description for e in obs.events) + " " + " ".join(m.text for m in obs.inbox)
        docs = self.retriever.search(query or self.bundle.scenario.title, self.side, at)
        return {"docs": [d.model_copy(update={"title": self.anon(d.title), "text": self.anon(d.text)}) for d in docs]}

    def _deliberate(self, st: _GraphState) -> _GraphState:
        prompt = self.render(st["obs"], st.get("docs", []))
        if st.get("error"):
            prompt += f"\n\n### Tu respuesta anterior no era válida\n{st['error']}\nCorrígela."
        raw = self.client.complete_json(self.system_prompt, prompt, self.schema_for(st["obs"]))
        return {"prompt": prompt, "raw": raw, "attempts": st.get("attempts", 0) + 1}

    def _validate(self, st: _GraphState) -> _GraphState:
        obs = st["obs"]
        try:
            data = json.loads(st["raw"])
            # Corrección tolerante de formato que no cambia la decisión: confianza expresada en 0-100.
            if isinstance(data, dict) and isinstance(data.get("confidence"), (int, float)) and 1 < data["confidence"] <= 100:
                data["confidence"] = data["confidence"] / 100
            out = LLMDecision.model_validate(data)
            actions = [self._to_action(a, obs) for a in out.actions[: obs.max_actions]]
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            self.errors.append(str(exc)[:300])
            if st["attempts"] < self.max_attempts:
                return {"error": str(exc)[:500]}
            return {"error": None, "decision": Decision(
                actions=[Action(kind="wait")], reasoning=f"[respaldo] {exc}"[:500], fallback=True)}
        valid_docs = {d.id for d in st.get("docs", [])}
        decision = Decision(
            actions=actions or [Action(kind="wait")],
            reasoning=out.reasoning,
            confidence=out.confidence,
            cited_docs=[d for d in out.cited_docs if d in valid_docs],
        )
        return {"error": None, "decision": decision}

    def schema_for(self, obs: Observation) -> dict:
        """Esquema con listas cerradas para este turno: con decodificación restringida, el modelo
        no puede inventar códigos de acción, términos, propuestas ni canales."""
        schema = copy.deepcopy(self.schema)
        action = schema["$defs"]["LLMAction"]["properties"]
        action["code"]["enum"] = [self._action_to_code[k] for k in obs.legal_actions]
        action["channel"]["anyOf"][0]["enum"] = list(obs.channels)
        deal_ids = sorted({d.id for d in obs.open_deals if d.proposer != self.side}
                          | {c.deal_id for c in obs.commitments if c.obligor == self.side and c.status == "pending"})
        if deal_ids:
            action["deal_id"]["anyOf"][0]["enum"] = deal_ids
        else:
            action["deal_id"] = {"type": "null", "default": None}
        terms = [self._term_to_code[k] for k in obs.deal_terms]
        if terms:
            schema["$defs"]["LLMTerm"]["properties"]["code"]["enum"] = terms
        else:
            action["terms"] = {"type": "null", "default": None}
        schema["properties"]["actions"]["minItems"] = 1
        schema["properties"]["actions"]["maxItems"] = obs.max_actions
        return schema

    def _route(self, st: _GraphState) -> str:
        return "done" if "decision" in st and st.get("error") is None else "retry"

    def _to_action(self, a: LLMAction, obs: Observation) -> Action:
        kind = self.action_codes.get(a.code)
        if kind is None or kind not in obs.legal_actions:
            raise ValueError(f"Código de acción no disponible: {a.code}")
        terms = None
        if a.terms:
            unknown = [t.code for t in a.terms
                       if t.code not in self.term_codes or self.term_codes[t.code] not in obs.deal_terms]
            if unknown:
                raise ValueError(f"Términos no disponibles ahora: {unknown}")
            catalog = self.bundle.scenario.deal_terms
            # Corrección tolerante: un término que no admite secreto (p. ej. una inspección) se pacta en
            # público; si no, el motor descartaría la propuesta entera por un detalle de forma.
            terms = [DealTerm(term=(k := self.term_codes[t.code]), secret=t.secret and catalog[k].secretable)
                     for t in a.terms]
        if kind in ("private_message", "propose_deal") and a.channel and a.channel not in obs.channels:
            raise ValueError(f"Canal desconocido: {a.channel}. Opciones: {obs.channels}")
        deal_id = a.deal_id
        if kind in ("accept_deal", "reject_deal"):
            rival = [d.id for d in obs.open_deals if d.proposer != self.side]
            if deal_id is None and len(rival) == 1:
                deal_id = rival[0]  # única propuesta posible: no hay ambigüedad
            if deal_id not in rival:
                raise ValueError(f"Propuesta desconocida: {deal_id}. Abiertas del rival: {rival}")
        if kind == "renege_deal":
            mine = sorted({c.deal_id for c in obs.commitments if c.obligor == self.side and c.status == "pending"})
            if deal_id is None and len(mine) == 1:
                deal_id = mine[0]
            if deal_id not in mine:
                raise ValueError(f"No tienes compromisos pendientes en el acuerdo {deal_id}")
        if kind == "propose_deal" and not terms:
            raise ValueError("Una propuesta de acuerdo necesita términos")
        channel = a.channel
        if kind in ("private_message", "propose_deal") and not channel:
            channel = obs.channels[0]  # dato rellenable: canal oficial por defecto, no se rechaza la decisión
        if kind == "private_message" and not a.text:
            raise ValueError("Un mensaje privado necesita texto")
        return Action(kind=kind, text=a.text, channel=channel, terms=terms, deal_id=deal_id)

    # ---------------------------------------------------------------- interfaz Agent

    def decide(self, obs: Observation) -> Decision:
        self.errors = []
        result = self.graph.invoke({"obs": obs, "attempts": 0, "error": None})
        decision: Decision = result["decision"]
        summary = ", ".join(self._action_to_code[a.kind] for a in decision.actions)
        self.history.append(self.anon(f"- Turno {obs.turn}: {summary}. Motivo: {decision.reasoning[:200]}"))
        return decision
