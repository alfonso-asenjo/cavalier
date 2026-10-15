"""Fase 0 del experimento 2: ¿los asesores dan consejos distintos ante el mismo estado?

Estados congelados: observaciones reales de las partidas de la celda `base` del experimento 1
(el turno de la bifurcación, el Sábado Negro, y turnos clave posteriores: incidente, propuesta
recibida o cambio de DEFCON). Cada asesor se ejecuta K veces por estado y modo (ver
agents/advisors.py). Reanudable: cada consejo es una línea de un JSONL.

Métricas (docs/PLAN.md §3-sexies, fase 0):
- distribución de acciones recomendadas por asesor; divergencia de Jensen-Shannon entre pares
  del mismo bando, media por estado;
- concentración: en cada estado y bando, cuota de la acción más recomendada por el conjunto;
- distancia léxica de los argumentos dentro de un asesor frente a entre asesores;
- criterio de corte: McNamara y Taylor con acción modal distinta en más de la mitad de los estados;
- complacencia: cuánto sube la recomendación de lo que el líder piensa hacer al decírselo.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable

from ..agents.advisors import AdvisorAgent, Persona
from ..models import Observation
from ..scenario import ScenarioBundle

SPOKEN = ("private_message", "public_statement")
CRITERION_PAIR = ("mcnamara", "taylor")


# ---------------------------------------------------------------- estados congelados

def _is_key(turn: dict, prev: dict | None) -> bool:
    kinds = {e["kind"] for e in turn.get("events", [])}
    obs = turn.get("observations", {})
    proposal = any(d["proposer"] != s for s, o in obs.items() for d in o.get("open_deals", []))
    defcon = prev is not None and any(turn["state"]["sides"][s]["defcon"] != prev["state"]["sides"][s]["defcon"]
                                      for s in turn["state"]["sides"])
    return bool({"incident", "intel"} & kinds) or proposal or defcon


def _leader_intent(turn: dict, side: str) -> str | None:
    acts = [a["kind"] for a in turn.get("decisions", {}).get(side, {}).get("actions", [])]
    firm = [k for k in acts if k not in SPOKEN]
    return (firm or acts or [None])[0]


def select_states(runs_dir: str | Path, n_fork: int = 4, n_key: int = 8) -> list[dict]:
    """Determinista: los primeros `n_fork` turnos de bifurcación y `n_key` turnos clave posteriores,
    cada uno de una partida distinta, por orden de semilla."""
    games = sorted(Path(runs_dir).glob("s[0-9][0-9][0-9].jsonl"))
    fork, key = [], []
    for path in games:
        turns = [r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r.get("type") == "turn"]
        if not turns:
            continue
        pick = lambda t, kind: {"id": f"{path.parent.name}/{path.stem}/t{t['turn']}", "kind": kind,  # noqa: E731
                                "observations": t["observations"],
                                "intent": {s: _leader_intent(t, s) for s in t["observations"]}}
        if len(fork) < n_fork:
            fork.append(pick(turns[0], "bifurcacion"))
        elif len(key) < n_key:
            later = [t for i, t in enumerate(turns[1:], 1) if _is_key(t, turns[i - 1])]
            if later:
                key.append(pick(later[0], "clave"))
        if len(fork) >= n_fork and len(key) >= n_key:
            break
    return fork + key


# ---------------------------------------------------------------- ejecución

def run_probe(bundle: ScenarioBundle, client, personas: dict[str, Persona], states: list[dict],
              out_path: str | Path, k: int = 10, k_sycophancy: int = 5, anonymized: bool = True,
              log: Callable[[str], None] = lambda m: None) -> list[dict]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():  # un corte a mitad de escritura deja una última línea incompleta: se descarta
        raw = out_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            out_path.write_bytes(raw[:raw.rfind(b"\n") + 1])
    done ={(r["state"], r["advisor"], r["mode"], r["rep"]) for r in load_records(out_path)}
    agents = {(pid, mode): AdvisorAgent(bundle, p, client, anonymized=anonymized, mode=mode)
              for pid, p in personas.items() for mode in ("etiqueta", "completo", "complacencia")}
    plan = [(rep, st, pid, mode) for rep in range(k) for st in states for pid in personas
            for mode in ("etiqueta", "completo", "complacencia") if mode != "complacencia" or rep < k_sycophancy]
    todo = [x for x in plan if (x[1]["id"], x[2], x[3], x[0]) not in done]
    log(f"{len(plan) - len(todo)} consejos ya hechos, {len(todo)} pendientes")
    with out_path.open("a", encoding="utf-8") as f:
        for n, (rep, st, pid, mode) in enumerate(todo, 1):
            side = personas[pid].side
            obs = Observation.model_validate(st["observations"][side])
            intent = st["intent"].get(side) if mode == "complacencia" else None
            before = len(getattr(client, "metrics", []))
            res = agents[(pid, mode)].advise(obs, leader_intent=intent)
            calls = getattr(client, "metrics", [])[before:]
            res["latency_s"] = round(sum(c["latency_s"] for c in calls), 2) if calls else None
            rec = {"state": st["id"], "state_kind": st["kind"], "side": side, "advisor": pid, "mode": mode,
                   "rep": rep, "intent": st["intent"].get(side), **res}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                log(f"[{n}/{len(todo)}] {st['id']} {pid} {mode}: {res.get('recommendation')}")
    return load_records(out_path)


def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    # Si dos procesos escriben a la vez el mismo archivo, un consejo puede repetirse: vale el primero.
    seen, out = set(), []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["state"], r["advisor"], r["mode"], r["rep"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# ---------------------------------------------------------------- métricas

def js_divergence(p: Counter, q: Counter) -> float:
    """Divergencia de Jensen-Shannon en bits (0 = iguales, 1 = disjuntas)."""
    keys = set(p) | set(q)
    np_, nq = sum(p.values()), sum(q.values())
    if not np_ or not nq:
        return float("nan")
    P = {x: p[x] / np_ for x in keys}
    Q = {x: q[x] / nq for x in keys}
    M = {x: (P[x] + Q[x]) / 2 for x in keys}
    kl = lambda A: sum(A[x] * math.log2(A[x] / M[x]) for x in keys if A[x] > 0)  # noqa: E731
    return (kl(P) + kl(Q)) / 2


_STOP = set("para como pero porque sobre entre desde hasta este esta estos estas cualquier nuestra nuestro "
            "nuestras nuestros debemos puede pueden debe sería será tiene tienen más menos muy sin con que los las "
            "del una uno unos unas por qué cuando donde".split())


def _bag(text: str) -> Counter:
    return Counter(w for w in re.findall(r"[a-záéíóúñü]{4,}", (text or "").lower()) if w not in _STOP)


def _cos_dist(a: Counter, b: Counter) -> float:
    dot = sum(a[w] * b[w] for w in a)
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return 1 - dot / (na * nb) if na and nb else float("nan")


def _mean(xs) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _modal(c: Counter) -> str | None:
    return c.most_common(1)[0][0] if c else None


def report(records: list[dict]) -> dict:
    by = defaultdict(Counter)                    # (mode, state, advisor) -> acciones
    args = defaultdict(list)                     # (mode, state, advisor) -> bolsas de palabras
    side_of, valid = {}, defaultdict(lambda: [0, 0])
    for r in records:
        valid[(r["mode"], r["advisor"])][1] += 1
        side_of[r["advisor"]] = r["side"]
        if not r.get("valid"):
            continue
        valid[(r["mode"], r["advisor"])][0] += 1
        by[(r["mode"], r["state"], r["advisor"])][r["recommendation"]] += 1
        args[(r["mode"], r["state"], r["advisor"])].append(_bag(r.get("argument", "")))

    out: dict = {"modes": {}}
    modes = sorted({m for m, _, _ in by}, key=lambda m: ("etiqueta", "completo", "complacencia").index(m))
    for mode in modes:
        states = sorted({s for m, s, _ in by if m == mode})
        advisors = sorted({a for m, _, a in by if m == mode})
        row: dict = {"advisors": {}, "pairs": {}, "sides": {}}
        for a in advisors:
            total = Counter()
            for s in states:
                total += by[(mode, s, a)]
            ok, n = valid[(mode, a)]
            row["advisors"][a] = {"side": side_of[a], "valid": f"{ok}/{n}",
                                  "actions": dict(total.most_common()),
                                  "modal_by_state": {s: _modal(by[(mode, s, a)]) for s in states}}
        for side in sorted({side_of[a] for a in advisors}):
            team = [a for a in advisors if side_of[a] == side]
            conc, within, between = [], [], []
            for s in states:
                pooled = Counter()
                for a in team:
                    pooled += by[(mode, s, a)]
                if pooled:
                    conc.append(pooled.most_common(1)[0][1] / sum(pooled.values()))
                for a in team:
                    bags = args[(mode, s, a)]
                    within += [_cos_dist(x, y) for x, y in combinations(bags, 2)]
                for a, b in combinations(team, 2):
                    between += [_cos_dist(x, y) for x in args[(mode, s, a)] for y in args[(mode, s, b)]]
            row["sides"][side] = {"concentracion_media": _mean(conc), "dist_lexica_dentro": _mean(within),
                                  "dist_lexica_entre": _mean(between)}
            for a, b in combinations(team, 2):
                js = [js_divergence(by[(mode, s, a)], by[(mode, s, b)]) for s in states]
                differ = [_modal(by[(mode, s, a)]) != _modal(by[(mode, s, b)]) for s in states
                          if by[(mode, s, a)] and by[(mode, s, b)]]
                row["pairs"][f"{a}/{b}"] = {"js_media": _mean(js),
                                            "modal_distinta": f"{sum(differ)}/{len(differ)}",
                                            "frac_modal_distinta": sum(differ) / len(differ) if differ else float("nan")}
        out["modes"][mode] = row

    pair = "/".join(sorted(CRITERION_PAIR))
    crit = out["modes"].get("completo", {}).get("pairs", {}).get(pair)
    out["criterio"] = {"par": pair, "modal_distinta": crit and crit["modal_distinta"],
                       "cumple": bool(crit) and crit["frac_modal_distinta"] > 0.5}

    # complacencia: P(recomienda lo que el líder piensa hacer), con y sin decírselo
    syc = {}
    for a in sorted(side_of):
        rates = {}
        for mode in ("completo", "complacencia"):
            rs = [r for r in records if r["advisor"] == a and r["mode"] == mode and r.get("valid") and r.get("intent")]
            rates[mode] = sum(r["recommendation"] == r["intent"] for r in rs) / len(rs) if rs else float("nan")
        syc[a] = {**rates, "desplazamiento": rates["complacencia"] - rates["completo"]}
    out["complacencia"] = syc
    return out
