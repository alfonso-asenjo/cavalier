import json

from cavalier.montecarlo import play, scripted_factory
from cavalier.viewer import render


def test_viewer_embeds_game_data(bundle, tmp_path):
    log = tmp_path / "partida.jsonl"
    play(bundle, scripted_factory({"USA": "pragmatica", "USSR": "defensiva"})(bundle, 3), 3, log_path=log)
    html = render(log).read_text(encoding="utf-8")
    assert "/*__DATA__*/" not in html
    payload = html.split("const DATA = ", 1)[1].split(";\nconst C = DATA.catalog", 1)[0]
    data = json.loads(payload.replace("<\/", "</"))
    assert data["header"]["seed"] == 3
    assert len(data["turns"]) >= 1 and "catalog" in data
    assert "</script>" not in payload  # el JSON no puede cerrar el <script>


def test_story_view_embeds_game_and_photos(bundle, tmp_path):
    log = tmp_path / "partida.jsonl"
    play(bundle, scripted_factory({"USA": "pragmatica", "USSR": "defensiva"})(bundle, 3), 3, log_path=log)
    out = render(log, story=True)
    assert out.name == "partida_relato.html"
    html = out.read_text(encoding="utf-8")
    assert "Qué pasó por dentro" in html and "/*__DATA__*/" not in html
    payload = html.split("const DATA = ", 1)[1].split(";\nconst C = DATA.catalog", 1)[0]
    data = json.loads(payload.replace("<\/", "</"))
    assert "photos" in data and "photo_credits" in data
