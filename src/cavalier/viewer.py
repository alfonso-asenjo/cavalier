"""Visor de partidas: convierte el registro JSONL de una partida en un HTML autónomo
(estética DEFCON, sin servidor). Uso: `cavalier view runs/partida.jsonl`."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

from .scenario import load_scenario

TEMPLATE = Path(__file__).parent / "viewer_template.html"
STORY_TEMPLATE = Path(__file__).parent / "story_template.html"
PHOTOS_DIR = Path(__file__).resolve().parents[2] / "data" / "photos"


def _catalog(scenario_id: str) -> dict:
    bundle = load_scenario(scenario_id)
    sc, rules = bundle.scenario, bundle.rules
    return {
        "title": sc.title,
        "hours_per_turn": sc.hours_per_turn,
        "start": sc.start.isoformat(),
        "sides": {s: {"name": c.name, "leader": c.leader} for s, c in sc.sides.items()},
        "actions": {k: {"label": a.label, "stance": a.stance} for k, a in rules.actions.items()},
        "terms": {k: t.label for k, t in sc.deal_terms.items()},
        "vars": {k: v.label for k, v in sc.variables.items()},
        "advisors": _advisors(scenario_id),
    }


def _history() -> list[dict]:
    """La cronología real (data/intro), sin las citas, para contar los antecedentes de cada partida."""
    import re

    from .site import load_intro

    clean = lambda t: re.sub(r"\s*\[s\d+(?:\s*,\s*s\d+)*\]", "", t or "")  # noqa: E731
    return [{"date": e["date"], "time": e.get("time", ""), "title": e["title"], "text": clean(e["text"]),
             "danger": e.get("danger", 0)} for e in load_intro().get("timeline", [])]


def _advisors(scenario_id: str) -> dict:
    """Nombre y cargo de los asesores (se lee la ficha directamente: el visor no necesita el agente)."""
    import tomllib

    from .scenario import SCENARIOS_DIR

    path = SCENARIOS_DIR / scenario_id / "advisors.toml"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        raw = tomllib.load(f).get("advisors", {})
    return {k: {"name": v["name"], "office": v["office"], "side": v["side"]} for k, v in raw.items()}


def _mp3_data_uri(path: Path) -> str:
    mp3 = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-b:a", "48k", "-f", "mp3", "-"],
                         capture_output=True, check=True).stdout
    return "data:audio/mpeg;base64," + base64.b64encode(mp3).decode("ascii")


def _voices(log_path: Path, out: Path, embed: bool = False) -> dict | None:
    """Audio sintetizado con `cavalier voices`, si existe: rutas relativas al HTML o, con
    `embed`, incrustado como MP3 (HTML autónomo, ~6 KB por segundo de audio; requiere ffmpeg)."""
    manifest = log_path.with_name(log_path.stem + ".voces") / "manifest.json"
    if not manifest.exists():
        return None
    m = json.loads(manifest.read_text(encoding="utf-8"))
    if embed:
        files = {ln["key"]: _mp3_data_uri(manifest.parent / ln["file"]) for ln in m["lines"]}
    else:
        base = Path(os.path.relpath(manifest.parent, out.resolve().parent)).as_posix()
        files = {ln["key"]: f"{base}/{ln['file']}" for ln in m["lines"]}
    return {"notice": m["notice"], "files": files}


def _photos(folder: Path = PHOTOS_DIR, base: str | None = None) -> tuple[dict, list]:
    """Fotos históricas por categoría (data/photos/photos.toml) y sus créditos. Por defecto van
    incrustadas en base64 (HTML autónomo); con `base`, como URL relativa (`base` + fichero) para la web."""
    index = folder / "photos.toml"
    if not index.exists():
        return {}, []
    import tomllib

    with index.open("rb") as f:
        entries = tomllib.load(f).get("photo", [])
    by_cat: dict[str, list] = {}
    credits = []
    for e in entries:
        path = folder / e["file"]
        if not path.exists():
            continue
        if base is not None:
            src = base + e["file"]
        else:
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            src = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        by_cat.setdefault(e["category"], []).append({"src": src, "caption": e["caption"], "credit": e["credit"],
                                                     "file": e["file"]})
        credits.append({k: e.get(k, "") for k in ("caption", "credit", "license", "source")})
    return by_cat, credits


def read_log(log_path: str | Path) -> tuple[dict, list[dict]]:
    header: dict = {}
    turns: list[dict] = []
    with Path(log_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:  # partida en curso: la última línea puede estar a medias
                break
            if record.get("type") == "header":
                header = record
            else:
                turns.append(record)
    return header, turns


def game_data(log_path: str | Path, out: Path | None = None, embed_audio: bool = False, story: bool = False,
              deanonymize: bool = False, photo_base: str | None = None) -> dict:
    """Los datos que pinta el visor. `deanonymize`: nombres y fechas reales para el espectador si
    la partida se jugó anonimizada (los agentes nunca los vieron)."""
    log_path = Path(log_path)
    header, turns = read_log(log_path)
    scenario = header.get("scenario", "cuba1962")
    anon = any(":anon" in a for a in header.get("agents", {}).values()) or         bool(turns and str(turns[0].get("date", "")).startswith("Día "))
    if deanonymize and anon:
        from .display import Deanonymizer

        turns = Deanonymizer.for_bundle(load_scenario(scenario)).walk(turns)
    from .site import model_label

    header = {**header, "agent_labels": {s: model_label(a) for s, a in header.get("agents", {}).items()}}
    data = {"header": header, "turns": turns, "catalog": _catalog(scenario),
            "voices": _voices(log_path, out or log_path.with_suffix(".html"), embed_audio),
            "deanonymized": bool(deanonymize and anon)}
    if story:
        from .site import load_map

        data["photos"], data["photo_credits"] = _photos(base=photo_base)
        data["map"] = load_map()
        data["history"] = _history()
    return data


def render_data(data: dict, out: Path, story: bool = False) -> Path:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = (STORY_TEMPLATE if story else TEMPLATE).read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
    out.write_text(html, encoding="utf-8")
    return out


def render(log_path: str | Path, out_path: str | Path | None = None, embed_audio: bool = False,
           story: bool = False, deanonymize: bool = False, photo_base: str | None = None,
           nav: list[dict] | None = None) -> Path:
    """Visor HTML autónomo. `story`: relato por capítulos con fotos históricas (si no, el panel DEFCON).
    `nav`: enlaces de la barra superior (en la web), p. ej. [{"href": "../index.html", "label": "Inicio"}]."""
    log_path = Path(log_path)
    out = Path(out_path) if out_path else log_path.with_name(log_path.stem + ("_relato" if story else "") + ".html")
    data = game_data(log_path, out, embed_audio, story, deanonymize, photo_base)
    if not data["turns"]:
        raise ValueError(f"{log_path} no contiene turnos")
    if nav:
        data["nav"] = nav
    return render_data(data, out, story)
