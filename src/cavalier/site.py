"""La web de Cavalier (`cavalier site`): la introducción histórica y las partidas para reproducir.

Es estática (GitHub Pages): `index.html` cuenta la crisis (datos en `data/intro/cuba1962.toml`)
y enlaza las partidas de ejemplo (`examples/partidas.toml`), cada una con el visor de relato y
los nombres reales. `partida.html` es el visor en directo: solo funciona con `cavalier serve`,
que añade la API para simular una partida nueva.
"""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTRO = ROOT / "data" / "intro" / "cuba1962.toml"
EXAMPLES = ROOT / "examples" / "partidas.toml"
INTRO_TEMPLATE = Path(__file__).parent / "intro_template.html"
MAP = ROOT / "data" / "maps" / "caribe.json"  # scripts/make_map.py (Natural Earth, dominio público)


def load_map() -> dict | None:
    return json.loads(MAP.read_text(encoding="utf-8")) if MAP.exists() else None

START_LABELS = {
    "inicio": "16 de octubre: Kennedy ve las fotos del U-2 (la crisis entera, unos 30 turnos)",
    "usa_respuesta_inicial": "20 de octubre: el ExComm elige cómo responder",
    "ussr_linea_cuarentena": "23 de octubre: los buques soviéticos navegan hacia la cuarentena",
    "usa_sabado_negro": "27 de octubre por la noche: el Sábado Negro, tras el derribo del U-2",
    "ussr_aceptacion": "28 de octubre: el Presídium ante la oferta estadounidense",
}
START_TURNS = {"usa_respuesta_inicial": 8, "ussr_linea_cuarentena": 14, "usa_sabado_negro": 23, "ussr_aceptacion": 24}
VARIANT_LABELS = {
    "sin_canal_secreto": "Sin el canal secreto entre Robert Kennedy y Dobrynin",
    "mando_debil": "Mandos locales más indisciplinados (el doble de incidentes)",
    "comunicaciones_lentas": "Comunicaciones oficiales más lentas",
    "sin_armas_tacticas": "Sin armas nucleares tácticas soviéticas en Cuba",
    "tacticas_descubiertas": "EE. UU. descubre a tiempo las armas tácticas",
    "sin_jupiter": "Sin misiles Jupiter en Turquía",
}
# Contrafactuales que no tienen sentido juntos (no se puede descubrir lo que no existe).
EXCLUSIVE = [("sin_armas_tacticas", "tacticas_descubiertas")]


def variant_list(variant) -> list[str]:
    if not variant:
        return []
    return [v for v in (variant.split("+") if isinstance(variant, str) else variant) if v]


# Modelos que se pueden elegir en la web, con coste y espera aproximados por decisión: unos 2.200
# tokens de entrada y 450 de salida (medido con GPT-5.6 Luna; Claude cuenta ~1,2 veces más tokens),
# tarifas de octubre de 2026. Qwen local: ~21 s por decisión en una RTX 2070 (experimento 1).
MODELS = [
    {"id": "llm:historico@ollama:qwen3.5:9b", "label": "Qwen 3.5 9B en local (Ollama)", "needs": "ollama", "usd": 0.0, "s": 21},
    {"id": "llm:historico@openai:gpt-5.6-luna", "label": "GPT-5.6 Luna (OpenAI)", "needs": "openai", "usd": 0.001, "s": 7},
    {"id": "llm:historico@openai:gpt-5.6-terra", "label": "GPT-5.6 Terra (OpenAI)", "needs": "openai", "usd": 0.010, "s": 10},
    {"id": "llm:historico@openai:gpt-5.6-sol", "label": "GPT-5.6 Sol (OpenAI)", "needs": "openai", "usd": 0.018, "s": 12},
    {"id": "llm:historico@anthropic:claude-haiku-5-5", "label": "Claude Haiku 5.5 (Anthropic)", "needs": "anthropic", "usd": 0.0005, "s": 5},
    {"id": "llm:historico@anthropic:claude-sonnet-5-5", "label": "Claude Sonnet 5.5 (Anthropic)", "needs": "anthropic", "usd": 0.011, "s": 10},
    {"id": "llm:historico@anthropic:claude-opus-5-5", "label": "Claude Opus 5.5 (Anthropic)", "needs": "anthropic", "usd": 0.021, "s": 14},
    {"id": "scripted:pragmatica", "label": "Agente de reglas, pragmático (sin IA)", "needs": None, "usd": 0.0, "s": 0.05},
    {"id": "scripted:halcon", "label": "Agente de reglas, halcón (sin IA)", "needs": None, "usd": 0.0, "s": 0.05},
    {"id": "scripted:paloma", "label": "Agente de reglas, paloma (sin IA)", "needs": None, "usd": 0.0, "s": 0.05},
]


def local_model(name: str) -> dict:
    """Otro modelo de Ollama para el menú de `cavalier serve --local-model` (p. ej. 'llama3.1:8b').
    La espera es la de Qwen 9B: depende del modelo y de la tarjeta gráfica."""
    return {"id": f"llm:historico@ollama:{name}", "label": f"{name} en local (Ollama)", "needs": "ollama",
            "usd": 0.0, "s": 21}
# Turnos que suelen quedar desde cada punto de partida: las partidas acaban hacia el turno 31 de
# media (experimento 1) y como mucho en el 36.
TYPICAL_END, MAX_END = 31, 36
# Un consejo cuesta ~0,7 veces una decisión del líder (1.100 tokens de entrada y 400 de salida) y
# tarda ~0,45 veces lo mismo (9 s con Qwen frente a 21 s; sonda de la fase 0).
ADVICE_COST, ADVICE_TIME = 0.7, 0.45


def advisor_counts(scenario: str = "cuba1962") -> dict[str, int]:
    """Asesores del núcleo por bando. Lee el TOML directamente: `cavalier site` no debe necesitar
    las dependencias de los LLM (`agents.advisors` importa langgraph), que la web publicada no instala."""
    path = ROOT / "scenarios" / scenario / "advisors.toml"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        raw = tomllib.load(f).get("advisors", {})
    counts: dict[str, int] = {}
    for v in raw.values():
        if v.get("group", "nucleo") == "nucleo":
            counts[v["side"]] = counts.get(v["side"], 0) + 1
    return counts

OUTCOME_LABELS = {
    "acuerdo_negociado": "Acuerdo cumplido", "desescalada": "Desescalada sin acuerdo",
    "sin_resolver": "Sin resolver", "conflicto_convencional": "Guerra convencional", "uso_nuclear": "Uso nuclear",
}


PERSONALITY_LABELS = {"halcon": "halcón", "pragmatica": "pragmática", "defensiva": "defensiva", "paloma": "paloma",
                      "impredecible": "impredecible"}


def model_label(agent: str) -> str:
    """'llm:qwen3.5:9b:historico:anon' → 'Qwen 3.5 9B'; 'scripted:halcon' → 'Reglas (halcón)'."""
    parts = agent.split(":")
    if parts[0] != "llm":
        return f"Reglas ({PERSONALITY_LABELS.get(parts[-1], parts[-1])})" if len(parts) > 1 else agent
    advisors = next((p.removeprefix("asesores") for p in parts if p.startswith("asesores")), None)
    model = ":".join(p for p in parts[1:] if p not in ("historico", "anon", "halcon", "paloma", "defensiva",
                                                         "pragmatica", "impredecible") and not p.startswith("asesores"))
    if advisors:
        return f"{model_label('llm:' + model)} y {advisors} asesores"
    if model.startswith("qwen"):
        name, _, size = model.partition(":")
        return f"Qwen {name.removeprefix('qwen')} {size.upper()}".strip()
    if model.startswith("claude-"):
        bits = model.removeprefix("claude-").split("-")
        return "Claude " + bits[0].capitalize() + " " + ".".join(bits[1:])
    if model.startswith("gpt-"):
        name, *rest = model.removeprefix("gpt-").split("-")
        return "GPT-" + name + (" " + " ".join(r.capitalize() for r in rest) if rest else "")
    return model


def game_summary(log: Path, variant: str | None = None) -> dict:
    from .viewer import read_log

    header, turns = read_log(log)
    if not turns:
        raise ValueError(f"{log} no tiene turnos")
    meta = log.with_suffix(".json")
    if variant is None and meta.exists():
        opts = json.loads(meta.read_text(encoding="utf-8")).get("options") or {}
        variant = opts.get("variants", opts.get("variant"))
    variants = variant_list(variant)
    last = turns[-1]["state"]
    first_turn = turns[0]["turn"]
    start = next((k for k, t in START_TURNS.items() if t == first_turn - 1 or t == first_turn), "inicio")
    return {
        "outcome": last.get("outcome"), "outcome_label": OUTCOME_LABELS.get(last.get("outcome"), "En curso"),
        "start": start, "start_label": START_LABELS.get(start, ""), "variant": "+".join(variants),
        "variant_label": "; ".join(VARIANT_LABELS.get(v, v) for v in variants),
        "turns": len(turns), "max_tension": round(max(t["state"]["tension"] for t in turns)),
        "agents": {s: model_label(a) for s, a in header.get("agents", {}).items()}, "seed": header.get("seed"),
    }


def load_intro(path: Path = INTRO) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def load_examples(path: Path = EXAMPLES) -> list[dict]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        rows = tomllib.load(f).get("partida", [])
    for r in rows:
        r["path"] = path.parent / r["file"]
    return rows


def _photo_index() -> dict:
    from .viewer import PHOTOS_DIR

    index = PHOTOS_DIR / "photos.toml"
    if not index.exists():
        return {}
    with index.open("rb") as f:
        entries = tomllib.load(f).get("photo", [])
    return {e["file"]: e for e in entries if (PHOTOS_DIR / e["file"]).exists()}


def build_site(out: str | Path, games: list[dict | Path] | None = None, live: bool = False) -> Path:
    """Escribe la web en `out`: index.html, partidas/<id>.html, partida.html (en directo) y fotos/.
    `games`: partidas de ejemplo (por defecto, las de examples/partidas.toml)."""
    from .viewer import STORY_TEMPLATE, PHOTOS_DIR, render

    out = Path(out)
    (out / "partidas").mkdir(parents=True, exist_ok=True)
    (out / "fotos").mkdir(exist_ok=True)
    photos = _photo_index()
    for name in photos:
        shutil.copy2(PHOTOS_DIR / name, out / "fotos" / name)

    rows = load_examples() if games is None else [g if isinstance(g, dict) else {"path": Path(g)} for g in games]
    listed = []
    nav = [{"href": "../index.html", "label": "Inicio"}, {"href": "../index.html#partidas", "label": "Partidas"}]
    for r in rows:
        log = Path(r["path"])
        slug = r.get("id") or log.stem
        render(log, out / "partidas" / f"{slug}.html", story=True, deanonymize=True, photo_base="../fotos/", nav=nav)
        summary = game_summary(log, r.get("variant"))
        listed.append({"id": slug, "href": f"partidas/{slug}.html", "title": r.get("title", ""),
                       "note": r.get("note", ""), **summary})

    (out / "partida.html").write_text(STORY_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    payload = {
        "intro": load_intro(), "games": listed, "live": live,
        "photos": {k: {"src": "fotos/" + k, "caption": v["caption"], "credit": v["credit"],
                       "license": v.get("license", ""), "source": v.get("source", ""), "category": v["category"]}
                   for k, v in photos.items()},
        "starts": START_LABELS, "start_turns": START_TURNS, "variants": VARIANT_LABELS, "exclusive": EXCLUSIVE,
        "models": MODELS, "end_turns": [TYPICAL_END, MAX_END], "map": load_map(),
        "advisors": {"counts": advisor_counts(), "cost": ADVICE_COST, "time": ADVICE_TIME},
    }
    html = INTRO_TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")  # GitHub Pages: servir tal cual
    return out / "index.html"
