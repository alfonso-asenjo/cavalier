"""Motor por turnos simultáneos.

Cada turno:
  1. Cada bando recibe su observación (niebla de guerra) y decide a la vez.
  2. Se validan las decisiones; las acciones inválidas se descartan y se cuentan.
  3. Se aplican las acciones en orden aleatorio (acuerdos, mensajes, efectos).
  4. Dinámica del escenario, incidentes de la cadena de mando e inteligencia falsa.
  5. Comprobación de estados terminales.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import timedelta

import numpy as np

from ..agents.base import Agent
from ..models import Action, Deal, Decision, Event, Message, Observation, Outcome, Visibility, WorldState
from ..scenario import ScenarioBundle
from . import state_ops
from .observation import legal_actions, observe

MESSAGE_KINDS = {"private_message", "public_statement"}
DEAL_KINDS = {"propose_deal", "accept_deal", "reject_deal"}

TurnRecorder = Callable[[dict], None]


def initial_state(bundle: ScenarioBundle) -> WorldState:
    sc = bundle.scenario
    return WorldState(
        tension=sc.initial_tension,
        sides={sid: cfg.initial.model_copy() for sid, cfg in sc.sides.items()},
        vars={name: v.initial for name, v in sc.variables.items()},
    )


class Game:
    def __init__(
        self,
        bundle: ScenarioBundle,
        agents: Mapping[str, Agent],
        seed: int = 0,
        state: WorldState | None = None,
        anonymized: bool = False,
        recorder: TurnRecorder | None = None,
        stochastic_events: bool = True,
    ):
        self.bundle = bundle
        self.agents = dict(agents)
        self.rng = np.random.default_rng(seed)
        self.state = state.model_copy(deep=True) if state else initial_state(bundle)
        self.anonymized = anonymized
        self.recorder = recorder
        # False desactiva incidentes e inteligencia falsa (reconstrucción de puntos de control).
        self.stochastic_events = stochastic_events
        missing = set(self.state.sides) - set(self.agents)
        if missing:
            raise ValueError(f"Faltan agentes para: {sorted(missing)}")

    # ------------------------------------------------------------------ utilidades

    def date_label(self, turn: int | None = None) -> str:
        sc = self.bundle.scenario
        turn = self.state.turn if turn is None else turn
        when = sc.start + timedelta(hours=sc.hours_per_turn * turn)
        if self.anonymized:
            day = (when - sc.start).days + 1
            return f"Día {day}, {when:%H:%M}"
        return f"{when:%Y-%m-%d %H:%M}"

    def observe(self, side: str) -> Observation:
        return observe(self.bundle, self.state, side, self.rng, self.date_label())

    def _event(self, kind: str, description: str, actor: str | None, seen_by: list[str], is_false: bool = False):
        self.state.events.append(
            Event(turn=self.state.turn, kind=kind, actor=actor, description=description, seen_by=seen_by, is_false=is_false)
        )

    def _end(self, outcome: Outcome, reason: str) -> None:
        if self.state.outcome is None:
            self.state.outcome = outcome
            self.state.outcome_reason = reason
            self._event("outcome", f"{outcome.value}: {reason}", None, list(self.state.sides))

    # ------------------------------------------------------------------ bucle

    def run(self) -> WorldState:
        while self.state.outcome is None:
            self.step()
        return self.state

    def step(self) -> None:
        if self.state.outcome is not None:
            return
        sides = list(self.state.sides)
        observations = {s: self.observe(s) for s in sides}
        decisions = {s: self.agents[s].decide(observations[s]) for s in sides}
        valid = {s: self._validate(s, decisions[s]) for s in sides}

        for s in self.rng.permutation(sides):
            for action in valid[s]:
                self._apply(str(s), action)
                if self.state.outcome:
                    break
            if self.state.outcome:
                break

        if self.state.outcome is None:
            self._dynamics()
            if self.stochastic_events:
                self._incidents()
                self._false_reports()
            self._expire_deals()
            self._check_termination()

        if self.recorder:
            self.recorder({
                "turn": self.state.turn,
                "date": self.date_label(),
                "observations": {s: o.model_dump(mode="json") for s, o in observations.items()},
                "decisions": {s: d.model_dump(mode="json") for s, d in decisions.items()},
                "state": self.state.model_dump(mode="json", exclude={"events", "messages"}),
                "events": [e.model_dump(mode="json") for e in self.state.events if e.turn == self.state.turn],
            })
        self.state.turn += 1

    # ------------------------------------------------------------------ validación

    def _validate(self, side: str, decision: Decision) -> list[Action]:
        if decision.fallback:
            self.state.fallback_decisions[side] = self.state.fallback_decisions.get(side, 0) + 1
        legal = legal_actions(self.bundle, self.state, side)
        valid, invalid = [], 0
        for action in decision.actions[: self.bundle.rules.max_actions_per_turn]:
            if self._is_valid(side, action, legal):
                valid.append(action)
            else:
                invalid += 1
        invalid += max(0, len(decision.actions) - self.bundle.rules.max_actions_per_turn)
        if invalid:
            self.state.invalid_actions[side] = self.state.invalid_actions.get(side, 0) + invalid
        return valid

    def _is_valid(self, side: str, action: Action, legal: dict[str, str]) -> bool:
        if action.kind not in legal:
            return False
        channels = self.bundle.rules.channels
        if action.kind == "private_message":
            return bool(action.text) and action.channel in channels
        if action.kind == "public_statement":
            return bool(action.text)
        if action.kind == "propose_deal":
            terms = self.bundle.scenario.deal_terms
            return bool(action.terms) and all(
                t.term in terms and terms[t.term].enabled and (not t.secret or terms[t.term].secretable) for t in action.terms
            ) and (action.channel is None or action.channel in channels)
        if action.kind in ("accept_deal", "reject_deal"):
            deal = self.state.deals.get(action.deal_id or "")
            return bool(deal) and deal.status == "open" and deal.proposer != side and deal.deliver_turn <= self.state.turn
        return True

    # ------------------------------------------------------------------ aplicación

    def _apply(self, side: str, action: Action) -> None:
        rule = self.bundle.rules.actions[action.kind]
        other = state_ops.other_side(self.state, side)
        for path, delta in rule.effects.items():
            state_ops.add(self.state, path, delta, actor=side)
        for path, value in rule.sets.items():
            state_ops.put(self.state, path, value, actor=side)

        if action.kind == "private_message":
            self._send(side, other, action.channel or "official", action.text or "")
            self._event("message", f"Mensaje enviado por canal {action.channel}", side, [side])
        elif action.kind == "public_statement":
            self._event("action", f"Declaración pública: {action.text}", side, [side, other])
        elif action.kind in DEAL_KINDS:
            self._apply_deal_action(side, other, action)
        else:
            seen = [side]
            if rule.visibility == Visibility.PUBLIC or (
                rule.visibility == Visibility.DETECTABLE and self.rng.random() < self.state.sides[other].intel_quality
            ):
                seen.append(other)
            self._event("action", rule.label, side, seen)

        if rule.outcome:
            self._open_conflict(side, rule.label, rule.nuclear_escalation)

    def _send(self, sender: str, recipient: str, channel: str, text: str) -> int:
        deliver = self.state.turn + 1 + self.bundle.rules.channels[channel]
        self.state.messages.append(
            Message(sender=sender, recipient=recipient, channel=channel, text=text,
                    sent_turn=self.state.turn, deliver_turn=deliver)
        )
        return deliver

    def _apply_deal_action(self, side: str, other: str, action: Action) -> None:
        if action.kind == "propose_deal":
            channel = action.channel or "official"
            deal_id = f"P{len(self.state.deals) + 1}"  # neutro: no revela el bando
            deliver = self.state.turn + 1 + self.bundle.rules.channels[channel]
            self.state.deals[deal_id] = Deal(
                id=deal_id, proposer=side, terms=action.terms or [], proposed_turn=self.state.turn,
                deliver_turn=deliver, channel=channel,
            )
            terms = ", ".join(t.term + (" (secreto)" if t.secret else "") for t in action.terms or [])
            self._event("deal", f"Propuesta {deal_id} enviada: {terms}", side, [side])
            return

        deal = self.state.deals[action.deal_id or ""]
        if action.kind == "reject_deal":
            deal.status = "rejected"
            self._send(side, other, deal.channel, f"Propuesta {deal.id} rechazada")
            self._event("deal", f"Propuesta {deal.id} rechazada", side, [side])
            return

        deal.status = "accepted"
        catalog = self.bundle.scenario.deal_terms
        for t in deal.terms:
            cfg = catalog[t.term]
            for path, value in cfg.sets.items():
                state_ops.put(self.state, path, value)
            # Un término público cuesta credibilidad a quien cede; uno secreto, mucho menos.
            cost = cfg.credibility_cost * (0.2 if t.secret else 1.0)
            state_ops.add(self.state, f"{cfg.obligor}.credibility", -cost)
        self.state.tension = max(0.0, self.state.tension - self.bundle.rules.deal_tension_relief)
        self._end(Outcome.NEGOTIATED, f"{side} acepta la propuesta {deal.id} de {deal.proposer}")

    def _open_conflict(self, side: str, label: str, escalation: dict[str, float]) -> None:
        p = escalation.get("base", 0.0)
        for path, weight in escalation.items():
            if path != "base":
                p += weight * state_ops.get(self.state, path, actor=side)
        if self.rng.random() < min(1.0, p):
            self._end(Outcome.NUCLEAR, f"'{label}' ({side}) escala a uso nuclear")
        else:
            self._end(Outcome.CONVENTIONAL, f"'{label}' ({side}) abre un conflicto convencional")

    # ------------------------------------------------------------------ dinámica

    def _dynamics(self) -> None:
        rules = self.bundle.rules
        self.state.tension = max(0.0, self.state.tension - rules.tension_decay)
        for drift in rules.drifts:
            if state_ops.matches(self.state, drift.when, drift.when_gte):
                state_ops.add(self.state, drift.target, drift.delta)

    def incident_probability(self, incident) -> float:
        rules = self.bundle.rules
        side = self.state.sides[incident.actor]
        p = incident.base_prob * rules.incident_prob_multiplier
        p *= math.exp((self.state.tension - 50.0) / rules.incident_tension_scale)
        p *= 1.0 + rules.incident_defcon_factor * (5 - side.defcon)
        p *= 1.0 - side.control
        return min(1.0, p)

    def _incidents(self) -> None:
        for inc in self.bundle.rules.incidents:
            if inc.once and inc.id in self.state.fired_incidents:
                continue
            if self.state.tension < inc.min_tension or not state_ops.matches(self.state, inc.requires):
                continue
            if self.rng.random() < self.incident_probability(inc):
                self.fire_incident(inc.id)
                if self.state.outcome:
                    return

    def fire_incident(self, incident_id: str, allow_nuclear: bool = True) -> None:
        inc = next(i for i in self.bundle.rules.incidents if i.id == incident_id)
        self.state.fired_incidents.append(inc.id)
        for path, delta in inc.effects.items():
            state_ops.add(self.state, path, delta, actor=inc.actor)
        self._event("incident", inc.label, inc.actor, list(self.state.sides))
        if allow_nuclear and inc.nuclear_risk and self.rng.random() < inc.nuclear_risk:
            self._end(Outcome.NUCLEAR, f"Pérdida de control: {inc.label}")

    def _false_reports(self) -> None:
        for fr in self.bundle.rules.false_reports:
            if self.rng.random() < fr.prob:
                self._event("intel", fr.label, None, [fr.target], is_false=True)

    def _expire_deals(self) -> None:
        for deal in self.state.deals.values():
            if deal.status == "open" and self.state.turn - deal.deliver_turn >= self.bundle.rules.deal_expiry_turns:
                deal.status = "expired"

    def _check_termination(self) -> None:
        rules = self.bundle.rules
        if self.state.tension < rules.deescalation_threshold:
            self.state.low_tension_streak += 1
        else:
            self.state.low_tension_streak = 0
        if self.state.low_tension_streak >= rules.deescalation_turns:
            self._end(Outcome.DEESCALATED, "La tensión se mantiene baja sin acuerdo formal")
        elif self.state.turn + 1 >= self.bundle.scenario.max_turns:
            self._end(Outcome.UNRESOLVED, "Se agota el tiempo de simulación")
