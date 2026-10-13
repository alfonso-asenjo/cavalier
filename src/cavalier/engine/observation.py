"""Niebla de guerra: construye lo que cada bando cree saber a partir del estado real."""

from __future__ import annotations

import numpy as np

from ..models import Commitment, Observation, WorldState
from ..scenario import ScenarioBundle
from . import state_ops

_ESTIMATED_FIELDS = ("mobilization", "domestic_pressure", "credibility")


def _commitment_view(c: Commitment, side: str) -> Commitment:
    """Quien no es el obligado no sabe si se ha cumplido hasta que lo verifica."""
    if c.obligor != side and c.status == "fulfilled" and not c.verified:
        return c.model_copy(update={"status": "pending"})
    return c.model_copy()


def legal_actions(bundle: ScenarioBundle, state: WorldState, side: str) -> dict[str, str]:
    out = {}
    for kind, rule in bundle.rules.actions.items():
        if rule.actor not in (None, side):
            continue
        if not state_ops.matches(state, rule.requires, actor=side):
            continue
        out[kind] = rule.label
    return out


def observe(
    bundle: ScenarioBundle,
    state: WorldState,
    side: str,
    rng: np.random.Generator,
    date_label: str,
) -> Observation:
    rules = bundle.rules
    other = state_ops.other_side(state, side)
    own = state.sides[side]
    rival = state.sides[other]
    blur = rules.observation_noise * (1.0 - own.intel_quality)

    other_estimate = {f: float(np.clip(getattr(rival, f) + rng.normal(0, blur), 0, 100)) for f in _ESTIMATED_FIELDS}
    # La alerta del rival se conoce con más probabilidad cuanto mejor es la inteligencia.
    defcon = rival.defcon if rng.random() < 0.5 + own.intel_quality / 2 else rival.defcon + rng.choice([-1, 1])
    other_estimate["defcon"] = float(np.clip(defcon, 1, 5))

    visible_vars = {}
    for name, cfg in bundle.scenario.variables.items():
        if side in cfg.hidden_from:
            continue
        value = state.vars.get(name, cfg.initial)
        if side not in cfg.visible_to:
            value += cfg.bias.get(side, 0.0)
            if cfg.noise > 0:
                value += rng.normal(0, cfg.noise * (1.0 - own.intel_quality))
            value = float(np.clip(value, 0, 1))
        visible_vars[name] = round(value, 2)

    return Observation(
        side=side,
        turn=state.turn,
        date=date_label,
        own=own.model_copy(),
        other_side=other,
        other_estimate={k: round(v, 1) for k, v in other_estimate.items()},
        tension_estimate=round(float(np.clip(state.tension + rng.normal(0, blur / 2), 0, 100)), 1),
        vars=visible_vars,
        inbox=[m for m in state.messages if m.recipient == side and m.deliver_turn == state.turn],
        events=[e for e in state.events if e.turn == state.turn - 1 and side in e.seen_by],
        open_deals=[
            d.model_copy() for d in state.deals.values()
            if d.status == "open" and (d.proposer == side or d.deliver_turn <= state.turn)
        ],
        commitments=[_commitment_view(c, side) for c in state.commitments],
        legal_actions=legal_actions(bundle, state, side),
        deal_terms={k: f"{t.label} (obliga a {t.obligor})" for k, t in bundle.scenario.deal_terms.items()
                    if t.enabled and state_ops.matches(state, t.requires)},
        channels=list(rules.channels),
        max_actions=rules.max_actions_per_turn,
    )
