import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

from cavalier.montecarlo import play, scripted_factory
from cavalier.server import Runner, make_handler
from cavalier.site import build_site, game_summary, model_label


def _payload(html: str, var: str = "const DATA = ") -> dict:
    return json.loads(html.split(var, 1)[1].split(";\n", 1)[0].replace("<\\/", "</"))


def _game(bundle, tmp_path, name="p.jsonl", anonymized=True):
    log = tmp_path / name
    play(bundle, scripted_factory({"USA": "pragmatica", "USSR": "defensiva"})(bundle, 3), 3, log_path=log,
         anonymized=anonymized)
    return log


def test_model_labels():
    assert model_label("llm:qwen3.5:9b:historico:anon") == "Qwen 3.5 9B"
    assert model_label("llm:claude-sonnet-5-5:historico:anon") == "Claude Sonnet 5.5"
    assert model_label("llm:gpt-5.6-terra:historico") == "GPT-5.6 Terra"
    assert model_label("scripted:halcon") == "Reglas (halcón)"
    assert model_label("llm:qwen3.5:9b:historico:anon:asesores4") == "Qwen 3.5 9B y 4 asesores"


def test_site_has_intro_games_and_live_page(bundle, tmp_path):
    log = _game(bundle, tmp_path)
    index = build_site(tmp_path / "site", [{"path": log, "id": "demo", "title": "Demo"}])
    site = index.parent
    data = _payload(index.read_text(encoding="utf-8"))
    assert data["games"][0]["href"] == "partidas/demo.html" and data["games"][0]["title"] == "Demo"
    assert data["games"][0]["start"] == "inicio" and data["games"][0]["turns"] > 0
    assert data["photos"] and all((site / p["src"]).exists() for p in data["photos"].values())
    replay = (site / "partidas" / "demo.html").read_text(encoding="utf-8")
    game = _payload(replay, "let DATA = ")
    assert game["deanonymized"] and game["nav"][0]["href"] == "../index.html"
    assert all(p["src"].startswith("../fotos/") for ps in game["photos"].values() for p in ps)
    assert "data:image" not in replay  # fotos enlazadas, no incrustadas: la web pesa poco
    assert "Día " not in json.dumps(game["turns"], ensure_ascii=False)  # fechas reales para el espectador
    assert "/*__DATA__*/null" in (site / "partida.html").read_text(encoding="utf-8")  # visor en directo


def test_summary_detects_fork_point(bundle, tmp_path):
    from cavalier.experiments import fork_state

    log = tmp_path / "f.jsonl"
    play(bundle, scripted_factory({"USA": "pragmatica", "USSR": "pragmatica"})(bundle, 1), 1, log_path=log,
         anonymized=True, state=fork_state("cuba1962", "usa_sabado_negro"))
    s = game_summary(log, "sin_canal_secreto")
    assert s["start"] == "usa_sabado_negro" and "Sábado Negro" in s["start_label"]
    assert s["variant_label"].startswith("Sin el canal secreto")


def test_server_runs_a_game_and_serves_it(tmp_path):
    runner = Runner(tmp_path / "web")
    build_site(tmp_path / "site", [])
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path / "site", runner))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    get = lambda p: json.load(urllib.request.urlopen(base + p))  # noqa: E731
    try:
        assert get("/api/health")["ok"]
        assert any(s["id"] == "usa_sabado_negro" for s in get("/api/options")["starts"])
        body = json.dumps({"start": "usa_sabado_negro", "variants": ["sin_canal_secreto", "mando_debil"], "usa": "scripted:halcon",
                           "ussr": "scripted:paloma", "seed": "5"}).encode()
        req = urllib.request.Request(base + "/api/games", data=body, headers={"Content-Type": "application/json"})
        gid = json.load(urllib.request.urlopen(req))["id"]
        for _ in range(200):
            d = get(f"/api/games/{gid}")
            if d["status"] in ("done", "error"):
                break
            time.sleep(0.05)
        assert d["status"] == "done", d.get("error")
        assert d["turns"][0]["turn"] == 23 and d["deanonymized"]
        assert d["turns"][-1]["state"]["outcome"]
        listed = get("/api/games")
        assert listed[0]["id"] == gid and listed[0]["variant"] == "sin_canal_secreto+mando_debil"
        assert "; " in listed[0]["variant_label"]
        assert b"CAVALIER" in urllib.request.urlopen(base + "/index.html").read()
    finally:
        server.shutdown()


def test_server_rejects_bad_options(tmp_path):
    runner = Runner(tmp_path / "web")
    for bad in ({"start": "nunca"}, {"variants": ["inventada"]}, {"usa": "llm:x; rm -rf /"},
                {"variants": ["sin_armas_tacticas", "tacticas_descubiertas"]}):
        try:
            runner.start(bad)
        except ValueError:
            continue
        raise AssertionError(f"debía rechazar {bad}")


def test_local_models_join_the_menu(tmp_path):
    from cavalier.server import _SPEC
    from cavalier.site import MODELS, local_model

    m = local_model("hf.co/bartowski/Llama-3.1-8B-Instruct-GGUF:Q4_K_M")
    assert _SPEC.match(m["id"]) and m["needs"] == "ollama" and m["usd"] == 0
    models = MODELS[:1] + [local_model("llama3.1:8b")] + MODELS[1:]
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, Runner(tmp_path / "web"), models))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        opts = json.load(urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/options"))
        assert opts["models"][1]["id"] == "llm:historico@ollama:llama3.1:8b"
    finally:
        server.shutdown()
