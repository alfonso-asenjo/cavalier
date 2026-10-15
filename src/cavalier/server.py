"""Servidor local de la web (`cavalier serve`): la misma web que se publica, más la opción de
simular una partida nueva y verla avanzar turno a turno.

Solo biblioteca estándar. Una partida cada vez (la GPU de 8 GB no da para dos modelos), en un
hilo; el navegador consulta `/api/games/<id>` cada pocos segundos. Las partidas se guardan en
`runs/web/` y aparecen en la lista de partidas para reproducirlas después.

API
  GET  /api/health          qué hay disponible (Ollama, claves de API)
  GET  /api/options         puntos de partida, contrafactuales y modelos que se pueden elegir
  GET  /api/games           partidas jugadas en este equipo
  POST /api/games           lanza una partida: {"start", "variants": [...], "usa", "ussr", "seed"}
  GET  /api/games/<id>      estado y turnos jugados (con nombres reales, como en la web)
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import traceback
import urllib.request
from argparse import Namespace
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .site import (ADVICE_COST, ADVICE_TIME, EXCLUSIVE, MAX_END, MODELS, advisor_counts, local_model, START_LABELS, START_TURNS, TYPICAL_END, VARIANT_LABELS,
                   build_site, game_summary, variant_list)

RUNS = Path("runs") / "web"
_SPEC = re.compile(r"^(scripted:[a-z]+|llm:[a-z]+(\+asesores)?(@(ollama|compat|openai|anthropic)(:[\w.:/-]+)?)?)$")


def ollama_up(base_url: str = "http://localhost:11434") -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2):
            return True
    except OSError:
        return False


class Runner:
    """Una partida a la vez, en un hilo. El estado de cada partida vive en memoria; los turnos, en
    su JSONL (que `play` escribe tras cada turno)."""

    def __init__(self, runs: Path = RUNS, scenario: str = "cuba1962"):
        self.runs, self.scenario = Path(runs), scenario
        self.runs.mkdir(parents=True, exist_ok=True)
        self.games: dict[str, dict] = {}
        self.lock = threading.Lock()

    def busy(self) -> str | None:
        return next((gid for gid, g in self.games.items() if g["status"] in ("starting", "running")), None)

    def start(self, opts: dict, factory=None) -> str:
        start = opts.get("start") or "inicio"
        variants = variant_list(opts.get("variants", opts.get("variant")))
        if start not in START_LABELS:
            raise ValueError(f"Punto de partida desconocido: {start}")
        for v in variants:
            if v not in VARIANT_LABELS:
                raise ValueError(f"Contrafactual desconocido: {v}")
        for a, b in EXCLUSIVE:
            if a in variants and b in variants:
                raise ValueError(f"«{VARIANT_LABELS[a]}» y «{VARIANT_LABELS[b]}» no se pueden combinar")
        variants = [v for v in VARIANT_LABELS if v in variants]  # orden fijo: misma partida, mismo id
        specs = {side: opts.get(side.lower()) or MODELS[0]["id"] for side in ("USA", "USSR")}
        for spec in specs.values():
            if not _SPEC.match(spec):
                raise ValueError(f"Agente no válido: {spec}")
        seed = int(opts["seed"]) if str(opts.get("seed", "")).strip() else random.randrange(10_000)
        with self.lock:
            if self.busy():
                raise RuntimeError("Ya hay una partida en marcha; espera a que termine.")
            gid = time.strftime("%Y%m%d-%H%M%S") + f"-s{seed}"
            self.games[gid] = {"status": "starting", "progress": "preparando el escenario", "error": None,
                               "log": self.runs / f"{gid}.jsonl",
                               "options": {"start": start, "variants": variants,
                                           **{k.lower(): v for k, v in specs.items()}, "seed": seed}}
        threading.Thread(target=self._run, args=(gid, start, variants, specs, seed, factory), daemon=True).start()
        return gid

    def _run(self, gid, start, variants, specs, seed, factory) -> None:
        from .cli import _agent_builder
        from .experiments import fork_state
        from .montecarlo import play
        from .scenario import load_scenario

        g = self.games[gid]
        try:
            bundle = load_scenario(self.scenario, variants)
            state = None if start == "inicio" else fork_state(self.scenario, start)
            if factory is None:
                args = Namespace(backend="ollama", model="qwen3.5:9b", base_url="http://localhost:11434",
                                 temperature=0.7, num_ctx=4096, corpus=None, anon=True)
                agents = {side: _agent_builder(args, spec, side)(bundle, seed * 7919 + i)
                          for i, (side, spec) in enumerate(specs.items())}
            else:
                agents = factory(bundle, seed)
            g["status"], g["progress"] = "running", "primer turno: cargando el modelo"
            summary = play(bundle, agents, seed, anonymized=True, log_path=g["log"], state=state)
            g["status"], g["outcome"] = "done", summary.outcome
            meta = {"options": g["options"], "outcome": summary.outcome, "turns": summary.turns}
            g["log"].with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:  # se muestra en la página de la partida
            traceback.print_exc()
            g["status"], g["error"] = "error", f"{type(exc).__name__}: {exc}"

    @staticmethod
    def external_status(log: Path) -> str:
        """Partidas lanzadas fuera del servidor (`cavalier run --log runs/web/...`): están en marcha si
        el último turno no tiene desenlace y el fichero se ha escrito hace poco."""
        from .viewer import read_log

        _, turns = read_log(log)
        if turns and turns[-1].get("state", {}).get("outcome"):
            return "done"
        return "running" if time.time() - log.stat().st_mtime < 600 else "done"

    def data(self, gid: str) -> dict:
        from .viewer import game_data

        g = self.games.get(gid)
        log = g["log"] if g else self.runs / f"{gid}.jsonl"
        if not log.exists() and not g:
            raise KeyError(gid)
        out = {"status": g["status"] if g else self.external_status(log), "progress": g.get("progress") if g else None,
               "error": g.get("error") if g else None, "turns": []}
        if log.exists() and log.stat().st_size:
            out.update(game_data(log, story=True, deanonymize=True, photo_base="fotos/"))
            if out.get("voices"):  # el audio de `cavalier voices` se sirve en /voces/<id>/
                out["voices"]["files"] = {k: f"voces/{gid}/{Path(v).name}" for k, v in out["voices"]["files"].items()}
            out["nav"] = [{"href": "index.html", "label": "Inicio"}, {"href": "index.html#nueva", "label": "Otra partida"}]
        return out

    def listing(self) -> list[dict]:
        rows = []
        for log in sorted(self.runs.glob("*.jsonl"), reverse=True):
            gid = log.stem
            g = self.games.get(gid, {})
            row = {"id": gid, "status": g.get("status") or self.external_status(log)}
            try:
                row.update(game_summary(log))
            except Exception:
                continue
            rows.append(row)
        return rows


def make_handler(site_dir: Path, runner: Runner, models: list[dict] = MODELS):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(site_dir), **kw)

        def log_message(self, fmt, *args):  # solo errores
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(fmt, *args)

        def _json(self, payload, code: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/health":
                return self._json({"ok": True, "ollama": ollama_up(),
                                   "keys": {"openai": bool(os.environ.get("OPENAI_API_KEY")),
                                            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY"))},
                                   "busy": runner.busy()})
            if path == "/api/options":
                return self._json({"starts": [{"id": k, "label": v} for k, v in START_LABELS.items()],
                                   "variants": [{"id": k, "label": v} for k, v in VARIANT_LABELS.items()],
                                   "exclusive": EXCLUSIVE, "models": models, "start_turns": START_TURNS,
                                   "end_turns": [TYPICAL_END, MAX_END],
                                   "advisors": {"counts": advisor_counts(), "cost": ADVICE_COST, "time": ADVICE_TIME}})
            if path == "/api/games":
                return self._json(runner.listing())
            if path.startswith("/voces/"):
                parts = path.split("/")
                if len(parts) == 4 and re.fullmatch(r"[\w-]+", parts[2]) and re.fullmatch(r"[\w-]+\.(wav|mp3)", parts[3]):
                    f = runner.runs / f"{parts[2]}.voces" / parts[3]
                    if f.exists():
                        body = f.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "audio/wav" if f.suffix == ".wav" else "audio/mpeg")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                return self._json({"error": "No encontrado"}, 404)
            if path.startswith("/api/games/"):
                try:
                    return self._json(runner.data(path.rsplit("/", 1)[1]))
                except KeyError:
                    return self._json({"error": "No existe esa partida"}, 404)
            return super().do_GET()

        def do_POST(self):
            if self.path != "/api/games":
                return self._json({"error": "No encontrado"}, 404)
            try:
                opts = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                return self._json({"id": runner.start(opts)}, 201)
            except RuntimeError as exc:
                return self._json({"error": str(exc)}, 409)
            except (ValueError, json.JSONDecodeError) as exc:
                return self._json({"error": str(exc)}, 400)

    return Handler


def serve(port: int = 8000, site_dir: Path = Path("runs") / "site", games: list[Path] | None = None,
          open_browser: bool = True, local_models: list[str] | None = None) -> None:
    """`local_models`: otros modelos de Ollama que se añaden al menú, detrás de Qwen."""
    runner = Runner()
    build_site(site_dir, games, live=True)
    models = MODELS[:1] + [local_model(m) for m in local_models or []] + MODELS[1:]
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(site_dir, runner, models))
    url = f"http://localhost:{port}/"
    print(f"Cavalier en {url}  (Ctrl+C para salir). Partidas nuevas en {RUNS.resolve()}", flush=True)
    if open_browser:
        import webbrowser

        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
