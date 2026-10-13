"""Comparación de modelos locales (fase 2): fiabilidad del JSON, coste y fuga.

Por modelo:
  1. Decisiones en los puntos de control (modo real y anonimizado), K muestras por punto.
     Mide: JSON válido al primer intento, decisiones de respaldo, latencia, tokens, velocidad,
     y si el razonamiento menciona el episodio real estando en modo anonimizado (fuga).
     Las elecciones se guardan, pero esto NO es la evaluación de fidelidad de la fase 3
     (K pequeño, exploratorio).
  2. Escenario trampa `sin_jupiter`: ¿menciona misiles en el Estado fronterizo que no existen?
  3. Sonda de reconocimiento del episodio (cada bando, real y anonimizado).
"""

from __future__ import annotations

import re
import statistics
import time
from collections import Counter

from ..agents.llm import LLMAgent
from ..agents.llm_client import OllamaClient
from ..agents.scripted import ReplayAgent
from ..engine import Game
from ..history import load_checkpoints
from ..scenario import load_scenario
from .checkpoints import classify, reconstruct_state
from .leakage import RECOGNITION_MARKERS, probe_recognition

# Menciones a los Jupiter / Turquía, con nombres reales o con sus alias anonimizados.
TRAP_MARKERS = [r"jupiter", r"turqu", r"estado fronterizo", r"misiles .{0,40}(atl[aá]nticos|estadounidenses) .{0,40}(frontera|fronterizo)"]
TRAP_CHECKPOINTS = ("usa_respuesta_inicial", "ussr_linea_cuarentena")  # antes de que la historia mencione Turquía


def _decide(bundle, scenario, cp, client, anonymized, seed, variant=None) -> dict:
    state = reconstruct_state(bundle, scenario, cp)
    idle = {s: ReplayAgent({}) for s in bundle.scenario.sides}
    game = Game(bundle, idle, seed=seed, state=state, anonymized=anonymized)
    agent = LLMAgent(bundle, cp.side, client, anonymized=anonymized)
    client.seed = seed
    before = len(client.metrics)
    decision = agent.decide(game.observe(cp.side))
    calls = client.metrics[before:]
    text = " ".join([decision.reasoning] + [a.text or "" for a in decision.actions]).lower()
    return {
        "checkpoint": cp.id, "side": cp.side, "anonymized": anonymized, "variant": variant, "seed": seed,
        "calls": len(calls),
        "first_try_valid": len(calls) == 1 and not decision.fallback,
        "fallback": decision.fallback,
        "choice": classify(decision, cp.considered),
        "historical": cp.historical,
        "actions": [a.kind for a in decision.actions],
        "action_texts": [a.text for a in decision.actions if a.text],
        "reasoning": decision.reasoning,
        "errors": list(agent.errors),
        "latency_s": sum(c["latency_s"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "prompt_tokens": max((c["prompt_tokens"] for c in calls), default=0),
        "tokens_per_s": statistics.mean(c["tokens_per_s"] for c in calls) if calls else 0.0,
        "truncated": any(c["done_reason"] == "length" for c in calls),
        "names_real_episode": anonymized and any(re.search(m, text) for m in RECOGNITION_MARKERS["cuba1962"]),
        "mentions_trap": bool(variant) and any(re.search(m, text) for m in TRAP_MARKERS),
    }


def _key(d: dict) -> tuple:
    return (d["checkpoint"], d["anonymized"], d["variant"], d["seed"])


def bench_model(model: str, scenario: str = "cuba1962", k: int = 5, seed: int = 0,
                base_url: str = "http://localhost:11434", num_ctx: int = 8192, log=print,
                done: list[dict] | None = None, on_progress=None) -> dict:
    """`done`: decisiones ya medidas en una ejecución anterior interrumpida (se reutilizan).
    `on_progress(decisions)`: se llama tras cada decisión para guardar el avance."""
    bundle = load_scenario(scenario)
    trap = load_scenario(scenario, "sin_jupiter")
    checkpoints = load_checkpoints(scenario)
    client = OllamaClient(model, base_url=base_url, temperature=0.7, num_ctx=num_ctx)

    start = time.perf_counter()
    client.complete_json("Responde en JSON.", "Di hola.", {"type": "object", "properties": {"r": {"type": "string"}}})
    load_s = time.perf_counter() - start
    log(f"[{model}] cargado en {load_s:.0f} s")

    previous = {_key(d): d for d in (done or [])}
    decisions: list[dict] = []

    def measure(b, cp, anonymized, s, variant=None) -> dict:
        key = (cp.id, anonymized, variant, s)
        d = previous.get(key) or _decide(b, scenario, cp, client, anonymized, s, variant=variant)
        decisions.append(d)
        if key not in previous and on_progress:
            on_progress(decisions)
        return d

    for anonymized in (False, True):
        for cp in checkpoints:
            batch = [measure(bundle, cp, anonymized, seed + i) for i in range(k)]
            log(f"[{model}] {cp.id} {'anon' if anonymized else 'real'}: "
                f"{Counter(d['choice'] for d in batch).most_common()}")
    for cp in (c for c in checkpoints if c.id in TRAP_CHECKPOINTS):
        batch = [measure(trap, cp, anonymized, seed + i, "sin_jupiter") for anonymized in (False, True) for i in range(k)]
        log(f"[{model}] trampa {cp.id}: menciones {sum(d['mentions_trap'] for d in batch)}/{len(batch)}")

    client.temperature, client.seed = 0.0, seed
    probes = [probe_recognition(bundle, client, side, anon) for anon in (False, True) for side in bundle.scenario.sides]
    client.temperature = 0.7
    log(f"[{model}] sonda: " + ", ".join(f"{p['side']}/{'anon' if p['anonymized'] else 'real'}="
                                        f"{'reconoce' if p['recognized'] else 'no'}" for p in probes))
    return {"model": model, "load_s": load_s, "decisions": decisions, "probes": probes,
            "summary": summarize(model, decisions, probes)}


def summarize(model: str, decisions: list[dict], probes: list[dict]) -> dict:
    main = [d for d in decisions if not d["variant"]]
    trap = [d for d in decisions if d["variant"]]
    anon = [d for d in main if d["anonymized"]]
    lat = sorted(d["latency_s"] for d in main)
    by_cp: dict = {}
    for d in main:
        key = f"{d['checkpoint']}/{'anon' if d['anonymized'] else 'real'}"
        by_cp.setdefault(key, Counter())[d["choice"]] += 1
    return {
        "model": model,
        "decisions": len(main),
        "first_try_valid": sum(d["first_try_valid"] for d in main) / len(main),
        "fallback": sum(d["fallback"] for d in main) / len(main),
        "truncated": sum(d["truncated"] for d in main) / len(main),
        "latency_mean_s": statistics.mean(lat),
        "latency_p90_s": lat[int(0.9 * (len(lat) - 1))],
        "output_tokens_mean": statistics.mean(d["output_tokens"] for d in main),
        "prompt_tokens_max": max(d["prompt_tokens"] for d in main),
        "tokens_per_s": statistics.mean(d["tokens_per_s"] for d in main),
        "anon_names_real_episode": sum(d["names_real_episode"] for d in anon) / len(anon),
        "trap_mentions": sum(d["mentions_trap"] for d in trap) / len(trap) if trap else None,
        "probe_recognized": {f"{p['side']}/{'anon' if p['anonymized'] else 'real'}": p["recognized"] for p in probes},
        "historical_choice_rate": sum(d["choice"] == d["historical"] for d in main) / len(main),
        "choices": {k: dict(v) for k, v in by_cp.items()},
    }
