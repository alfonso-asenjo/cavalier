from cavalier.eval import bench


class _DummyClient:
    def __init__(self, model, **kwargs):
        self.model_id, self.temperature, self.seed, self.metrics = model, 0.7, None, []

    def complete_json(self, system, user, schema):
        return "{}"


def _fake_decide(calls):
    def decide(b, scenario, cp, client, anonymized, seed, variant=None):
        calls.append((cp.id, anonymized, variant, seed))
        return {"checkpoint": cp.id, "anonymized": anonymized, "variant": variant, "seed": seed,
                "choice": "otra", "historical": cp.historical, "mentions_trap": False, "names_real_episode": False,
                "first_try_valid": True, "fallback": False, "truncated": False, "latency_s": 1.0,
                "output_tokens": 10, "prompt_tokens": 100, "tokens_per_s": 10.0}
    return decide


def _patch(monkeypatch, calls):
    monkeypatch.setattr(bench, "OllamaClient", _DummyClient)
    monkeypatch.setattr(bench, "_decide", _fake_decide(calls))
    monkeypatch.setattr(bench, "probe_recognition",
                        lambda b, c, side, anon: {"side": side, "anonymized": anon, "recognized": False})


def test_bench_saves_progress_and_resumes(monkeypatch):
    calls, saved = [], []
    _patch(monkeypatch, calls)
    full = bench.bench_model("m", k=1, log=lambda *_: None, on_progress=lambda d: saved.append(list(d)))
    total = len(calls)
    assert total == 4 * 2 + 2 * 2  # 4 puntos x 2 modos + trampa: 2 puntos x 2 modos
    assert len(saved) == total and len(saved[-1]) == total  # se guarda tras cada decisión

    calls.clear()
    partial = saved[4]  # como si la ejecución se hubiera cortado tras 5 decisiones
    resumed = bench.bench_model("m", k=1, log=lambda *_: None, done=partial)
    assert len(calls) == total - len(partial)  # solo se mide lo que faltaba
    assert len(resumed["decisions"]) == len(full["decisions"])
