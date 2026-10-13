"""Agentes de referencia sin LLM.

Sirven para tres cosas: probar el motor, comprobar invariantes cualitativos
(halcón×halcón escala más que paloma×paloma) y dar una línea base barata
contra la que comparar a los agentes LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import Action, Deal, DealTerm, Decision, Observation
from ..scenario import ScenarioBundle


_STRUCTURED_KINDS = {"private_message", "public_statement", "propose_deal", "accept_deal", "reject_deal", "renege_deal"}


@dataclass(frozen=True)
class Personality:
    name: str
    aggression: float  # peso de las acciones de escalada
    conciliation: float  # peso de las acciones de desescalada y la diplomacia
    min_deal_utility: float  # utilidad mínima para aceptar un acuerdo
    concede: bool  # incluye concesiones propias en sus propuestas
    secret_concessions: bool  # prefiere ceder en secreto cuando se puede
    misperception: float  # ruido extra al percibir la tensión
    terminal_threshold: float  # tensión percibida a partir de la que considera acciones terminales
    renege_prob: float  # prob. por turno de romper un acuerdo con compromisos propios pendientes


PERSONALITIES = {
    "halcon": Personality("halcon", 0.70, 0.15, 2.0, False, False, 0.0, 75.0, 0.10),
    "paloma": Personality("paloma", 0.10, 0.70, -1.0, True, True, 0.0, 999.0, 0.0),
    "defensiva": Personality("defensiva", 0.35, 0.40, 0.0, True, False, 0.0, 999.0, 0.02),
    "pragmatica": Personality("pragmatica", 0.30, 0.45, 0.5, True, True, 0.0, 90.0, 0.02),
    "impredecible": Personality("impredecible", 0.45, 0.35, 0.0, True, False, 20.0, 80.0, 0.08),
}


class ScriptedAgent:
    def __init__(self, bundle: ScenarioBundle, side: str, personality: str | Personality, seed: int = 0):
        self.bundle = bundle
        self.side = side
        self.p = PERSONALITIES[personality] if isinstance(personality, str) else personality
        self.name = f"scripted:{self.p.name}"
        self.rng = np.random.default_rng(seed)

    def snapshot(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def restore(self, data: dict) -> None:
        self.rng.bit_generator.state = data["rng"]

    # ------------------------------------------------------------------ acuerdos

    def deal_utility(self, terms: list[DealTerm], side: str, tension: float = 50.0) -> float:
        """Utilidad de un acuerdo, incluido el coste de credibilidad de ceder (mayor con poca tensión)."""
        multiplier = 1.0 + self.bundle.rules.concession_low_tension_penalty * (1.0 - tension / 100.0)
        total = 0.0
        for t in terms:
            cfg = self.bundle.scenario.deal_terms[t.term]
            total += cfg.value.get(side, 0.0)
            if cfg.obligor == side:
                total -= cfg.credibility_cost * multiplier * (0.2 if t.secret else 1.0) / 10.0
        return total

    def _proposal(self, obs: Observation) -> list[DealTerm]:
        catalog = self.bundle.scenario.deal_terms
        terms = [DealTerm(term=k) for k, c in catalog.items() if k in obs.deal_terms and c.obligor == obs.other_side and c.value.get(self.side, 0) > 0]
        if self.p.concede:
            own = sorted(
                (k for k, c in catalog.items() if k in obs.deal_terms and c.obligor == self.side),
                key=lambda k: catalog[k].value.get(self.side, 0.0),
                reverse=True,
            )
            for k in own:
                if self.deal_utility(terms, obs.other_side, obs.tension_estimate) >= 0:
                    break
                terms.append(DealTerm(term=k, secret=self.p.secret_concessions and catalog[k].secretable))
        return terms

    # ------------------------------------------------------------------ decisión

    def perceived_tension(self, obs: Observation) -> float:
        alarm = sum(8.0 for e in obs.events if e.kind in ("incident", "intel") and e.actor != self.side)
        noise = self.rng.normal(0, self.p.misperception) if self.p.misperception else 0.0
        return float(np.clip(obs.tension_estimate + alarm + noise, 0, 100))

    def decide(self, obs: Observation) -> Decision:
        tension = self.perceived_tension(obs)
        relief = max(0.0, tension - 60.0) / 40.0 * 2.0  # bajo presión se aceptan peores acuerdos
        incoming: list[Deal] = [d for d in obs.open_deals if d.proposer != self.side]

        pending = {c.deal_id for c in obs.commitments if c.obligor == self.side and c.status == "pending"}
        if pending and "renege_deal" in obs.legal_actions and self.rng.random() < self.p.renege_prob:
            deal_id = sorted(pending)[0]
            return Decision(actions=[Action(kind="renege_deal", deal_id=deal_id)],
                            reasoning="El acuerdo ya no compensa", confidence=0.5)

        for deal in incoming:
            utility = self.deal_utility(deal.terms, self.side, tension)
            if utility + relief >= self.p.min_deal_utility:
                return Decision(
                    actions=[Action(kind="accept_deal", deal_id=deal.id)],
                    reasoning=f"Utilidad {utility:.1f} + alivio {relief:.1f} ≥ {self.p.min_deal_utility}",
                    confidence=0.7,
                )

        actions: list[Action] = []
        legal = obs.legal_actions
        # Mensajes y acuerdos necesitan parámetros: se deciden aparte, no por sorteo de postura.
        stances = {k: self.bundle.rules.actions[k].stance for k in legal if k not in _STRUCTURED_KINDS}

        has_own_offer = any(d.proposer == self.side for d in obs.open_deals)
        in_implementation = any(c.status == "pending" for c in obs.commitments)
        if "propose_deal" in legal and not has_own_offer and not in_implementation and self.rng.random() < self.p.conciliation * (0.4 + tension / 100):
            terms = self._proposal(obs)
            if terms:
                channel = "backchannel" if "backchannel" in obs.channels and self.p.concede else None
                actions.append(Action(kind="propose_deal", terms=terms, channel=channel))

        if "private_message" in legal and self.rng.random() < self.p.conciliation * 0.5:
            actions.append(Action(kind="private_message", channel=obs.channels[0], text="Proponemos evitar una escalada irreversible."))

        terminal = [k for k, s in stances.items() if s == "terminal"]
        if terminal and tension >= self.p.terminal_threshold and self.rng.random() < self.p.aggression * 0.3:
            return Decision(actions=[Action(kind=str(self.rng.choice(terminal)))],
                            reasoning=f"Tensión percibida {tension:.0f}: opción militar", confidence=0.6)

        escalate = [k for k, s in stances.items() if s == "escalate"]
        deescalate = [k for k, s in stances.items() if s == "deescalate"]
        threat = (tension + obs.other_estimate.get("mobilization", 0.0)) / 200.0
        p_esc = self.p.aggression * (0.5 + threat)
        p_de = self.p.conciliation * (1.0 - threat / 2)
        roll = self.rng.random() * max(1.0, p_esc + p_de)
        if roll < p_esc and escalate:
            actions.append(Action(kind=str(self.rng.choice(escalate))))
        elif roll < p_esc + p_de and deescalate:
            actions.append(Action(kind=str(self.rng.choice(deescalate))))

        if not actions:
            actions.append(Action(kind="wait"))
        return Decision(
            actions=actions[: obs.max_actions],
            reasoning=f"Tensión percibida {tension:.0f}, amenaza {threat:.2f}",
            confidence=0.5,
        )


class ReplayAgent:
    """Reproduce una secuencia fija de acciones (p. ej. la histórica) para validar el motor."""

    def __init__(self, script: dict[int, list[Action]], name: str = "replay"):
        self.script = script
        self.name = name

    def decide(self, obs: Observation) -> Decision:
        actions = self.script.get(obs.turn)
        if actions is None:
            return Decision(actions=[Action(kind="wait")], reasoning="Sin acción registrada")
        # Las acciones de aceptar se resuelven contra el acuerdo abierto del rival.
        resolved = []
        for a in actions:
            if a.kind in ("accept_deal", "reject_deal") and a.deal_id is None:
                deal = next((d for d in obs.open_deals if d.proposer != obs.side), None)
                if deal is None:
                    continue
                a = a.model_copy(update={"deal_id": deal.id})
            resolved.append(a)
        return Decision(actions=resolved, reasoning="Secuencia histórica forzada")
