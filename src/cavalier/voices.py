"""Voces sintéticas de los líderes (experimental, fase 6).

Lee el registro JSONL de una partida, extrae lo que dice cada bando (mensajes privados y
declaraciones públicas) y lo sintetiza con XTTS‑v2 clonando una voz de referencia por bando.
El audio va a `<partida>.voces/` junto a un `manifest.json`; `cavalier view` lo detecta y
añade un botón de reproducción a cada mensaje.

Las voces son SINTÉTICAS y los textos, inventados por los agentes: el manifiesto y el visor
lo dicen siempre. Ver `data/voices/README.md` (fuentes, preparación y licencias).

XTTS‑v2 no está entre las dependencias del proyecto: se instala aparte (`.venv-tts`, ver README)
porque arrastra PyTorch con CUDA. Este módulo solo lo importa al sintetizar.
"""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
VOICES_DIR = ROOT / "data" / "voices"
VOICED_KINDS = ("private_message", "public_statement")
AUDIO_EXTS = (".wav", ".flac", ".mp3")


NUM_WORDS = {"1": "uno", "2": "dos", "3": "tres", "4": "cuatro", "5": "cinco", "6": "seis", "7": "siete",
             "8": "ocho", "9": "nueve"}
KEEP_CAPS = {"ONU", "OTAN", "URSS", "EE"}
_CODE = r"T\d+"
_CODE_LIST = rf"{_CODE}(?:\s*(?:,|y|e)\s*{_CODE})*"


def term_labels(log_path: str | Path) -> dict[str, str]:
    """T1…Tn → etiqueta del término, como las numera el agente LLM (anonimizada si la partida lo era)."""
    from .anonymize import Anonymizer
    from .scenario import load_scenario

    with Path(log_path).open(encoding="utf-8") as f:
        header = json.loads(f.readline())
    bundle = load_scenario(header.get("scenario", "cuba1962"))
    anon = Anonymizer(bundle.aliases, enabled=any(":anon" in a for a in header.get("agents", {}).values()))
    labels = {}
    for i, t in enumerate(bundle.scenario.deal_terms.values()):
        label = anon(t.label)
        labels[f"T{i + 1}"] = label[0].lower() + label[1:]
    return labels


def speakable(text: str, terms: dict[str, str]) -> str:
    """Texto para el sintetizador: códigos del protocolo (T4, P1) en palabras, sin mayúsculas gritadas.
    El texto original se conserva en el manifiesto y en el visor."""
    term = lambda m: terms.get(m.group(0), m.group(0))  # noqa: E731
    # «retirada de misiles (T5)», «(T2, T3)»: el código entre paréntesis sobra
    text = re.sub(rf"\s*\(\s*{_CODE_LIST}\s*\)", "", text)
    # «el T5 (retirada de misiles)»: se queda la explicación del propio agente
    text = re.sub(rf"\bdel\s+{_CODE}\s*\(([^)]*)\)", r"de \1", text)
    text = re.sub(rf"\b(?:el|la|los)\s+{_CODE}\s*\(([^)]*)\)", r"\1", text)
    text = re.sub(rf"\b{_CODE}\s*\(([^)]*)\)", r"\1", text)
    text = re.sub(rf"\b{_CODE}\b", term, text)
    # propuestas: «la propuesta P1» → «la propuesta uno»; «P2» suelto → «la propuesta dos»
    text = re.sub(r"\b(propuestas?|acuerdos?)((?:\s+de\s+acuerdo)?)\s+P(\d)\b",
                  lambda m: f"{m.group(1)}{m.group(2)} {NUM_WORDS[m.group(3)]}", text)
    text = re.sub(r"\bP(\d)\b", lambda m: f"la propuesta {NUM_WORDS[m.group(1)]}", text)
    text = re.sub(r"\bDEFCON\s*(\d)", lambda m: f"Défcon {NUM_WORDS[m.group(1)]}", text)
    text = re.sub(r"\b[A-ZÁÉÍÓÚÑ]{4,}\b",
                  lambda m: m.group(0) if m.group(0) in KEEP_CAPS else m.group(0).capitalize(), text)
    return re.sub(r"\s{2,}", " ", text).strip()


@dataclass
class Line:
    turn: int
    side: str
    index: int  # posición de la acción dentro de la decisión (la clave que usa el visor)
    kind: str
    channel: str | None
    text: str

    @property
    def key(self) -> str:
        return f"{self.turn}:{self.side}:{self.index}"

    @property
    def filename(self) -> str:
        return f"t{self.turn:02d}_{self.side}_{self.index}.wav"


def extract_lines(log_path: str | Path, kinds: tuple[str, ...] = VOICED_KINDS) -> list[Line]:
    lines: list[Line] = []
    with Path(log_path).open(encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            record = json.loads(raw)
            if record.get("type") != "turn":
                continue
            for side, dec in sorted((record.get("decisions") or {}).items()):
                for i, a in enumerate(dec.get("actions", [])):
                    text = (a.get("text") or "").strip()
                    if a.get("kind") in kinds and text:
                        lines.append(Line(record["turn"], side, i, a["kind"], a.get("channel"), text))
    return lines


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else VOICES_DIR / "voices.toml"
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    base = path.parent
    for side, spec in cfg["sides"].items():
        folder = base / spec["speaker"]
        refs = sorted(p for p in folder.glob("*") if p.suffix.lower() in AUDIO_EXTS) if folder.is_dir() else []
        spec["refs"] = [str(p) for p in refs]
    return cfg


def voices_dir_for(log_path: str | Path) -> Path:
    log_path = Path(log_path)
    return log_path.with_name(log_path.stem + ".voces")


LANGUAGE_NAMES = {"en": "inglés", "ru": "ruso", "es": "español"}
TRANSLATE_SYSTEM = """Eres el intérprete de un jefe de gobierno durante una crisis de 1962.
Traduces al {lang} lo que dice, para leerlo en voz alta con su propia voz.
- Registro oral, formal y de la época, como en un discurso o una carta diplomática de 1962.
- Traduce el sentido, no palabra por palabra; frases que se puedan decir de un tirón.
- Los nombres de países, lugares y organizaciones se traducen de forma natural al {lang}
  (p. ej. «la Isla» → «the Island» / «Остров»), siempre igual en todo el texto.
- Números y siglas, escritos como se pronuncian.
- No añadas ni quites nada. Devuelve solo la traducción."""
TRANSLATION_SCHEMA = {"type": "object", "properties": {"translation": {"type": "string"}},
                      "required": ["translation"]}


def _real_names(log_path: str | Path) -> Callable[[str], str]:
    """Si la partida se jugó anonimizada, las voces dicen los nombres reales (como el visor):
    Kennedy no puede hablar de «la Federación Atlántica»."""
    from .display import Deanonymizer
    from .scenario import load_scenario

    with Path(log_path).open(encoding="utf-8") as f:
        header = json.loads(f.readline())
    bundle = load_scenario(header.get("scenario", "cuba1962"))
    if not any(":anon" in a for a in header.get("agents", {}).values()):
        return lambda text: text
    return Deanonymizer.for_bundle(bundle)


def side_language(config: dict, side: str) -> str:
    return config["sides"].get(side, {}).get("language", config.get("language", "es"))


def translate_lines(lines: list[Line], spoken: dict[str, str], config: dict, out: Path,
                    client=None, log: Callable[[str], None] = lambda m: None) -> dict[str, str]:
    """Traduce cada línea al idioma de su bando (si no es español) con un LLM; local por defecto
    (`translate_with` en voices.toml). Las traducciones se guardan en `translations.json`: no se
    repiten ni se pagan dos veces. Devuelve clave → texto que se lee."""
    path = out / "translations.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [ln for ln in lines if side_language(config, ln.side) != "es"
            and cache.get(ln.key, {}).get("source") != spoken[ln.key]]
    if todo and client is None:
        from .agents.llm_client import make_client

        backend, _, model = config.get("translate_with", "ollama:qwen3.5:9b").partition(":")
        client = make_client(backend, model or None, temperature=0.2)
    for n, ln in enumerate(todo, 1):
        lang = side_language(config, ln.side)
        log(f"[traducción {n}/{len(todo)}] T{ln.turn} {ln.side} → {lang}")
        for _ in range(3):  # un modelo pequeño a veces mezcla alfabetos («solemnно»): se repite
            raw = client.complete_json(TRANSLATE_SYSTEM.format(lang=LANGUAGE_NAMES.get(lang, lang)),
                                       spoken[ln.key], TRANSLATION_SCHEMA)
            text = json.loads(raw)["translation"].strip()
            if lang != "ru" or not re.search(r"[A-Za-z]{2,}", text):
                break
        cache[ln.key] = {"language": lang, "source": spoken[ln.key], "text": text,
                         "model": getattr(client, "model_id", None)}
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    if todo and hasattr(client, "unload"):
        client.unload()  # libera la GPU para XTTS
    return {ln.key: cache[ln.key]["text"] if side_language(config, ln.side) != "es" else spoken[ln.key]
            for ln in lines}


def xtts_synth(cfg: dict, device: str = "cuda") -> Callable[[str, list[str], str, str], None]:
    """Sintetizador XTTS‑v2. La primera vez descarga el modelo (~1,8 GB) y pide aceptar su
    licencia (CPML, solo uso no comercial); con COQUI_TOS_AGREED=1 se acepta sin preguntar.
    La tabla [xtts] de voices.toml pasa ajustes de inferencia (temperatura, velocidad…)."""
    from TTS.api import TTS  # noqa: PLC0415 (dependencia opcional y pesada)

    import torch  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415

    tts = TTS(cfg.get("model", "tts_models/multilingual/multi-dataset/xtts_v2")).to(device)
    settings = cfg.get("xtts", {})
    rate = 24000
    pause = torch.zeros(int(rate * 0.3))

    def synth(text: str, refs: list[str], out: str, language: str) -> None:
        # Frase a frase: con el mensaje entero de una vez, XTTS pierde la entonación en los largos.
        parts = []
        for sentence in split_sentences(for_xtts(text)):
            parts += [torch.tensor(tts.tts(text=sentence, speaker_wav=refs, language=language, **settings)), pause]
        torchaudio.save(out, torch.cat(parts[:-1]).unsqueeze(0), rate)

    return synth


def for_xtts(text: str) -> str:
    """Lo que XTTS lee mal: abreviaturas con punto («Mr.» → «Mr. Redd») y comillas angulares."""
    text = re.sub(r"\bMr\.", "Mister", text)
    text = re.sub(r"\bMrs\.", "Missus", text)
    return re.sub(r"[«»“”\"]", "", text)


def split_sentences(text: str, limit: int = 160) -> list[str]:
    """Frases para XTTS, que corta el audio por encima de ~180 caracteres (ruso) o ~250 (inglés):
    las más largas se parten por comas, dos puntos o rayas, agrupando trozos hasta el límite."""
    out = []
    for sentence in (s for s in re.split(r"(?<=[.!?;])\s+", text.strip()) if s):
        chunk = ""
        pieces = []
        for piece in re.split(r"(?<=[,:—])\s+", sentence):  # trozos sin comas aún largos: por palabras
            while len(piece) > limit:
                cut = piece.rfind(" ", 0, limit)
                cut = cut if cut > limit // 2 else limit
                pieces.append(piece[:cut].strip())
                piece = piece[cut:].strip()
            pieces.append(piece)
        for piece in pieces:
            if chunk and len(chunk) + 1 + len(piece) > limit:
                out.append(chunk)
                chunk = piece
            else:
                chunk = f"{chunk} {piece}".strip()
        out.append(chunk)
    return out


def synthesize(log_path: str | Path, config: dict, out_dir: str | Path | None = None,
               synth: Callable[[str, list[str], str, str], None] | None = None,
               limit: int | None = None, log: Callable[[str], None] = lambda m: None,
               translator=None) -> Path:
    """Traduce (si el bando habla otro idioma) y sintetiza las líneas de una partida.
    Reanudable: no rehace traducciones ni ficheros que ya existen. Devuelve la ruta del manifiesto."""
    out = Path(out_dir) if out_dir else voices_dir_for(log_path)
    out.mkdir(parents=True, exist_ok=True)
    lines = extract_lines(log_path)[:limit]
    terms = term_labels(log_path)
    missing = [s for s in {ln.side for ln in lines} if not config["sides"].get(s, {}).get("refs")]
    if missing:
        raise SystemExit(f"Sin audio de referencia para {', '.join(sorted(missing))}: ver data/voices/README.md")
    real = _real_names(log_path)
    spanish = {ln.key: speakable(real(ln.text), terms) for ln in lines}
    spoken = translate_lines(lines, spanish, config, out, client=translator, log=log)
    if synth is None:
        synth = xtts_synth(config)
    entries = []
    for n, ln in enumerate(lines, 1):
        spec = config["sides"][ln.side]
        lang = side_language(config, ln.side)
        target = out / ln.filename
        stamp = target.with_suffix(".txt")  # texto con el que se generó: si cambia, se rehace
        if not target.exists() or not stamp.exists() or stamp.read_text(encoding="utf-8") != spoken[ln.key]:
            log(f"[{n}/{len(lines)}] T{ln.turn} {ln.side} ({spec['speaker']}, {lang}): {spoken[ln.key][:60]}…")
            tmp = target.with_suffix(".part.wav")
            synth(spoken[ln.key], spec["refs"], str(tmp), lang)
            tmp.replace(target)
            stamp.write_text(spoken[ln.key], encoding="utf-8")
        entries.append({"key": ln.key, "turn": ln.turn, "side": ln.side, "kind": ln.kind,
                        "channel": ln.channel, "speaker": spec["speaker"], "file": ln.filename,
                        "language": lang, "text": ln.text, "spoken": spoken[ln.key]})
    manifest = {
        "synthetic": True,
        "notice": "Voces sintéticas (clonación con XTTS-v2), con traducción automática al idioma de cada líder. "
                  "Los textos los generaron agentes de IA en una simulación; no son grabaciones ni palabras reales.",
        "log": Path(log_path).name,
        "model": config.get("model"),
        "languages": {s: side_language(config, s) for s in config["sides"]},
        "translate_with": config.get("translate_with", "ollama:qwen3.5:9b"),
        "lines": entries,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def conversation_mp3(manifest_path: str | Path, gap: float = 0.8) -> Path:
    """Toda la partida en un solo MP3, en orden de turno, con una pausa entre mensajes
    (para escucharla donde el visor no pueda reproducir audio). Requiere ffmpeg."""
    manifest_path = Path(manifest_path)
    folder = manifest_path.parent
    lines = json.loads(manifest_path.read_text(encoding="utf-8"))["lines"]
    inputs, chains = [], []
    for i, ln in enumerate(lines):
        inputs += ["-i", str(folder / ln["file"])]
        chains.append(f"[{i}:a]aresample=24000,aformat=channel_layouts=mono,apad=pad_dur={gap}[a{i}]")
    graph = ";".join(chains) + ";" + "".join(f"[a{i}]" for i in range(len(lines))) +         f"concat=n={len(lines)}:v=0:a=1[out]"
    out = folder / "conversacion.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[out]",
                    "-b:a", "64k", str(out)], check=True)
    return out
