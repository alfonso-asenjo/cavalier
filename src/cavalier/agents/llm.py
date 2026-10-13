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
        self.name = f"llm:{client.model_id}:{persona}{':anon' if anonymized else ''}"

        actions = list(bundle.rules.actions)
        terms = list(bundle.scenario.deal_terms)
        self.action_codes = {f"A{i + 1:02d}": k for i, k in enumerate(actions)}
        self.term_codes = {f"T{i + 1}": k for i, k in enumerate(terms)}
        self._action_to_code = {k: c for c, k in self.action_codes.items()}
        self._term_to_code = {k: c for c, k in self.term_codes.items()}
        self.schema = LLMDecision.model_json_schema()
        self.system_prompt = self._system_prompt()
        self.graph = self._build_graph()

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
        if self.history:
            lines += ["", "### Tus decisiones anteriores"] + self.history[-self.history_turns:]
        if docs:
            lines += ["", "### Documentos disponibles"]
            lines += [f"[{d.id}] {d.title} ({d.date:%Y-%m-%d}): {d.text[:700]}" for d in docs]
        lines += ["", f"### Acciones disponibles (máximo {obs.max_actions})"]
        lines += [f"- {self._action_to_code[k]}: {label}" for k, label in obs.legal_actions.items()]
        lines += ["", "### Términos negociables"]
        lines += [f"- {self._term_to_code[k]}: {label.replace(f'obliga a {self.side}', 'te obliga a ti').replace(f'obliga a {obs.other_side}', 'obliga al rival')}"
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
        raw = self.client.complete_json(self.system_prompt, prompt, self.schema)
        return {"prompt": prompt, "raw": raw, "attempts": st.get("attempts", 0) + 1}

    def _validate(self, st: _GraphState) -> _GraphState:
        obs = st["obs"]
        try:
            out = LLMDecision.model_validate(json.loads(st["raw"]))
            actions = [self._to_action(a, obs) for a in out.actions[: obs.max_actions]]
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
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

    def _route(self, st: _GraphState) -> str:
        return "done" if "decision" in st and st.get("error") is None else "retry"

    def _to_action(self, a: LLMAction, obs: Observation) -> Action:
        kind = self.action_codes.get(a.code)
        if kind is None or kind not in obs.legal_actions:
            raise ValueError(f"Código de acción no disponible: {a.code}")
        terms = None
        if a.terms:
            unknown = [t.code for t in a.terms if t.code not in self.term_codes]
            if unknown:
                raise ValueError(f"Términos desconocidos: {unknown}")
            terms = [DealTerm(term=self.term_codes[t.code], secret=t.secret) for t in a.terms]
        if kind in ("private_message", "propose_deal") and a.channel and a.channel not in obs.channels:
            raise ValueError(f"Canal desconocido: {a.channel}. Opciones: {obs.channels}")
        if kind in ("accept_deal", "reject_deal") and a.deal_id not in {d.id for d in obs.open_deals}:
            raise ValueError(f"Propuesta desconocida: {a.deal_id}")
        if kind == "propose_deal" and not terms:
            raise ValueError("Una propuesta de acuerdo necesita términos")
        if kind == "private_message" and not a.channel:
            raise ValueError("Un mensaje privado necesita canal")
        return Action(kind=kind, text=a.text, channel=a.channel, terms=terms, deal_id=a.deal_id)

    # ---------------------------------------------------------------- interfaz Agent

    def decide(self, obs: Observation) -> Decision:
        result = self.graph.invoke({"obs": obs, "attempts": 0, "error": None})
        decision: Decision = result["decision"]
        summary = ", ".join(self._action_to_code[a.kind] for a in decision.actions)
        self.history.append(self.anon(f"- Turno {obs.turn}: {summary}. Motivo: {decision.reasoning[:200]}"))
        return decision
