import json

from cavalier.voices import extract_lines, load_config, synthesize
from cavalier.viewer import render


def _log(tmp_path):
    turn = {
        "type": "turn", "turn": 23, "date": "Día 12, 20:00", "observations": {}, "events": [],
        "state": {"turn": 23, "tension": 90.0, "sides": {s: {"defcon": 3} for s in ("USA", "USSR")}, "vars": {}},
        "decisions": {
            "USA": {"actions": [
                {"kind": "gather_intel", "text": "justificación, no se lee"},
                {"kind": "private_message", "channel": "backchannel", "text": "Aceptamos P1."},
            ]},
            "USSR": {"actions": [{"kind": "public_statement", "text": "Retiraremos los medios."},
                                 {"kind": "public_statement", "text": "  "}]},
        },
    }
    log = tmp_path / "s001.jsonl"
    log.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in
                             [{"type": "header", "scenario": "cuba1962", "seed": 1}, turn]), encoding="utf-8")
    return log


def _config(tmp_path):
    (tmp_path / "voices.toml").write_text(
        'language = "es"\n[sides.USA]\nspeaker = "k"\n[sides.USSR]\nspeaker = "j"\n', encoding="utf-8")
    for sp in ("k", "j"):
        (tmp_path / sp).mkdir()
        (tmp_path / sp / "ref.wav").write_bytes(b"RIFF")
    return load_config(tmp_path / "voices.toml")


def test_extract_only_spoken_lines(tmp_path):
    lines = extract_lines(_log(tmp_path))
    assert [(ln.key, ln.text) for ln in lines] == [("23:USA:1", "Aceptamos P1."), ("23:USSR:0", "Retiraremos los medios.")]


def test_synthesize_is_resumable_and_feeds_viewer(tmp_path):
    log, cfg = _log(tmp_path), _config(tmp_path)
    calls = []

    def fake(text, refs, out):
        calls.append((text, refs))
        open(out, "wb").write(b"RIFF")

    manifest = json.loads(synthesize(log, cfg, synth=fake).read_text(encoding="utf-8"))
    assert manifest["synthetic"] and len(manifest["lines"]) == 2
    assert calls[0][1][0].endswith("ref.wav")
    synthesize(log, cfg, synth=fake)
    assert len(calls) == 2  # no rehace lo que ya existe

    html = render(log).read_text(encoding="utf-8")
    assert "s001.voces/t23_USA_1.wav" in html and "sintéticas" in html


def test_missing_reference_audio_fails_clearly(tmp_path):
    cfg = _config(tmp_path)
    cfg["sides"]["USSR"]["refs"] = []
    try:
        synthesize(_log(tmp_path), cfg, synth=lambda *a: None)
    except SystemExit as e:
        assert "USSR" in str(e)
    else:
        raise AssertionError("debía fallar sin audio de referencia")


def test_speakable_reads_protocol_codes():
    from cavalier.voices import speakable

    terms = {"T4": "compromiso público de no invadir", "T5": "retirada de los misiles fronterizos"}
    assert speakable("aceptar la propuesta P1 (T4) y el T5 (retirada de misiles)", terms) == \
        "aceptar la propuesta uno y retirada de misiles"
    assert speakable("los términos T4 y T5. URGENTE: DEFCON 2 en la ONU", terms) == \
        "los términos compromiso público de no invadir y retirada de los misiles fronterizos. Urgente: Défcon dos en la ONU"
