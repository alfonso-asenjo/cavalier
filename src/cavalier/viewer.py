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
        "sides": {s: {"name": c.name, "leader": c.leader} for s, c in sc.sides.items()},
        "actions": {k: {"label": a.label, "stance": a.stance} for k, a in rules.actions.items()},
        "terms": {k: t.label for k, t in sc.deal_terms.items()},
        "vars": {k: v.label for k, v in sc.variables.items()},
    }


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


def _photos(folder: Path = PHOTOS_DIR) -> tuple[dict, list]:
    """Fotos históricas por categoría (data/photos/photos.toml), incrustadas en base64, y sus créditos."""
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
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        src = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        by_cat.setdefault(e["category"], []).append({"src": src, "caption": e["caption"], "credit": e["credit"]})
        credits.append({k: e.get(k, "") for k in ("caption", "credit", "license", "source")})
    return by_cat, credits


def render(log_path: str | Path, out_path: str | Path | None = None, embed_audio: bool = False,
           story: bool = False) -> Path:
    """Visor HTML autónomo. `story`: relato por capítulos con fotos históricas (si no, el panel DEFCON)."""
    log_path = Path(log_path)
    header: dict = {}
    turns: list[dict] = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "header":
                header = record
            else:
                turns.append(record)
    if not turns:
        raise ValueError(f"{log_path} no contiene turnos")
    out = Path(out_path) if out_path else log_path.with_name(log_path.stem + ("_relato" if story else "") + ".html")
    data = {"header": header, "turns": turns, "catalog": _catalog(header.get("scenario", "cuba1962")),
            "voices": _voices(log_path, out, embed_audio)}
    if story:
        data["photos"], data["photo_credits"] = _photos()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = (STORY_TEMPLATE if story else TEMPLATE).read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
    out.write_text(html, encoding="utf-8")
    return out
