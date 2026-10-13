"""Línea de comandos.

  cavalier run cuba1962 --usa scripted:halcon --ussr scripted:pragmatica --seed 3 --log runs/p.jsonl
  cavalier run cuba1962 --usa llm:historico --ussr llm:historico --model qwen3:8b --anon
  cavalier montecarlo cuba1962 --usa scripted:halcon --ussr scripted:paloma -n 1000
  cavalier replay cuba1962 -n 1000
  cavalier sensitivity cuba1962 --usa scripted:pragmatica --ussr scripted:pragmatica -n 300
  cavalier probe cuba1962 --model qwen3:8b
  cavalier checkpoints cuba1962 --agent llm:historico --model qwen3:8b -k 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents.base import Agent
from .agents.scripted import ScriptedAgent
from .montecarlo import BatchResult, play, replay_factory, run_batch, scripted_factory
from .scenario import ScenarioBundle, load_scenario


def _agent_builder(args, spec: str, side: str):
    """Devuelve f(bundle, seed) -> Agent a partir de 'scripted:<personalidad>' o 'llm:<persona>'."""
    kind, _, persona = spec.partition(":")
    if kind == "scripted":
        return lambda bundle, seed: ScriptedAgent(bundle, side, persona or "pragmatica", seed=seed)
    if kind == "llm":
        from .agents.llm import LLMAgent
        from .agents.llm_client import OpenAICompatibleClient

        retriever = None
        if args.corpus:
            from .rag.corpus import load_corpus
            from .rag.retriever import TemporalRetriever

            retriever = TemporalRetriever(load_corpus(args.corpus))

        def build(bundle: ScenarioBundle, seed: int) -> Agent:
            client = OpenAICompatibleClient(args.model, base_url=args.base_url, temperature=args.temperature, seed=seed)
            return LLMAgent(bundle, side, client, persona=persona or "historico", anonymized=args.anon, retriever=retriever)
        return build
    raise SystemExit(f"Agente desconocido: {spec!r} (usa scripted:<personalidad> o llm:<persona>)")


def _factory(args):
    if args.usa.startswith("scripted") and args.ussr.startswith("scripted"):
        # serializable: permite repartir partidas entre procesos
        return scripted_factory({"USA": args.usa.partition(":")[2] or "pragmatica",
                                 "USSR": args.ussr.partition(":")[2] or "pragmatica"})
    builders = {"USA": _agent_builder(args, args.usa, "USA"), "USSR": _agent_builder(args, args.ussr, "USSR")}
    return lambda bundle, seed: {side: b(bundle, seed * 7919 + i) for i, (side, b) in enumerate(builders.items())}


def _executor(args):
    """Pool de procesos si se pide (--workers > 1) y los agentes son scripted o históricos."""
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import nullcontext

    llm = any(getattr(args, a, "").startswith("llm") for a in ("usa", "ussr"))
    if args.workers > 1 and not llm:
        return ProcessPoolExecutor(max_workers=args.workers)
    return nullcontext()


def _print_table(result: BatchResult, title: str) -> None:
    print(f"\n{title} — {result.n} partidas")
    for row in result.table():
        lo, hi = row["ci95"]
        print(f"  {row['outcome']:<24} {row['n']:>5}  {row['p']:6.1%}  IC95 [{lo:5.1%}, {hi:5.1%}]")
    fallback_games = sum(1 for s in result.summaries if s.fallback_decisions)
    invalid = sum(s.invalid_actions for s in result.summaries)
    print(f"  partidas con decisiones de respaldo: {fallback_games} · acciones inválidas: {invalid}")


def cmd_run(args) -> None:
    bundle = load_scenario(args.scenario, args.variant)
    agents = _factory(args)(bundle, args.seed)
    log = Path(args.log) if args.log else (Path("runs") / f"{args.scenario}_{args.seed}.jsonl" if args.view else None)
    summary = play(bundle, agents, args.seed, anonymized=args.anon, log_path=log)
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    if args.view:
        from .viewer import render

        print(f"Visor: {render(log).resolve()}")


def cmd_view(args) -> None:
    from .viewer import render

    print(f"Visor: {render(args.log, args.out).resolve()}")


def cmd_montecarlo(args) -> None:
    bundle = load_scenario(args.scenario, args.variant)
    with _executor(args) as ex:
        result = run_batch(bundle, _factory(args), args.n, args.seed, anonymized=args.anon, log_dir=args.log_dir,
                           executor=ex)
    _print_table(result, f"{args.usa} vs {args.ussr}")
    if any(s.fallback_decisions for s in result.summaries):
        _print_table(result.clean(), "Solo partidas sin respaldo")
    if args.out:
        Path(args.out).write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_replay(args) -> None:
    bundle = load_scenario(args.scenario, args.variant)
    with _executor(args) as ex:
        result = run_batch(bundle, replay_factory(args.scenario), args.n, args.seed, executor=ex)
    _print_table(result, "Secuencia histórica forzada")


def cmd_sensitivity(args) -> None:
    from .sensitivity import one_at_a_time

    rows = one_at_a_time(args.scenario, _factory(args), args.n, delta=args.delta, seed=args.seed, variant=args.variant)
    base = rows[0]["proportions"]
    print("Base: " + ", ".join(f"{k}={v:.1%}" for k, v in base.items()))
    print(f"\nCambio en puntos porcentuales con ±{args.delta:.0%}:")
    for row in rows[1:]:
        lo, hi = row["low"], row["high"]
        print(f"  {row['param']:<52} nuclear {lo['uso_nuclear']:+6.1%} / {hi['uso_nuclear']:+6.1%}"
              f"   acuerdo {lo['acuerdo_negociado']:+6.1%} / {hi['acuerdo_negociado']:+6.1%}")


def cmd_morris(args) -> None:
    from .sensitivity import morris

    factory = replay_factory(args.scenario) if args.historical else _factory(args)
    label = "secuencia histórica" if args.historical else f"{args.usa} vs {args.ussr}"
    with _executor(args) as ex:
        res = morris(args.scenario, factory, args.n, trajectories=args.r, seed=args.seed, variant=args.variant,
                     executor=ex)
    print(f"Morris ({label}): {args.r} trayectorias x {args.n} partidas por punto")
    for metric, rows in res["metrics"].items():
        print(f"\n  {metric}")
        for row in rows:
            flag = "  <-- domina" if row["share"] > 0.5 else ""
            print(f"    {row['param']:<52} mu*={row['mu_star']:.3f} sigma={row['sigma']:.3f} "
                  f"cuota={row['share']:5.1%} [{row['basis']}]{flag}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_probe(args) -> None:
    from .agents.llm_client import OpenAICompatibleClient
    from .eval.leakage import probe_recognition

    bundle = load_scenario(args.scenario, args.variant)
    client = OpenAICompatibleClient(args.model, base_url=args.base_url, temperature=0.0)
    for anonymized in (False, True):
        for side in bundle.scenario.sides:
            print(json.dumps(probe_recognition(bundle, client, side, anonymized), ensure_ascii=False))


def cmd_checkpoints(args) -> None:
    from .eval.checkpoints import evaluate_checkpoint
    from .history import load_checkpoints

    bundle = load_scenario(args.scenario, args.variant)
    for cp in load_checkpoints(args.scenario):
        build = _agent_builder(args, args.agent, cp.side)
        res = evaluate_checkpoint(bundle, args.scenario, cp, lambda seed, b=build: b(bundle, seed),
                                  k=args.k, seed=args.seed, anonymized=args.anon)
        lo, hi = res["ci95"]
        print(f"{cp.id:<26} p(histórica)={res['p_historical']:.0%} IC95 [{lo:.0%}, {hi:.0%}]"
              f"  azar={res['uniform_baseline']:.0%}  {res['choices']}  respaldo={res['fallbacks']}")


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="cavalier")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, agents: bool = True):
        p.add_argument("scenario")
        p.add_argument("--variant")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--anon", action="store_true", help="modo anonimizado")
        p.add_argument("--model", default="qwen3:8b")
        p.add_argument("--base-url", default="http://localhost:11434/v1")
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--corpus", help="carpeta con documentos .jsonl para el RAG")
        p.add_argument("--workers", type=int, default=1, help="procesos en paralelo (solo agentes scripted/históricos)")
        if agents:
            p.add_argument("--usa", default="scripted:pragmatica")
            p.add_argument("--ussr", default="scripted:pragmatica")

    p = sub.add_parser("run", help="una partida")
    common(p)
    p.add_argument("--log", help="fichero JSONL de la partida")
    p.add_argument("--view", action="store_true", help="generar el visor HTML al terminar")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("view", help="visor HTML de una partida registrada")
    p.add_argument("log", help="fichero JSONL de la partida")
    p.add_argument("-o", "--out", help="HTML de salida (por defecto, junto al JSONL)")
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("montecarlo", help="lote de partidas")
    common(p)
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--log-dir")
    p.add_argument("--out", help="JSON con resultados")
    p.set_defaults(func=cmd_montecarlo)

    p = sub.add_parser("replay", help="secuencia histórica forzada")
    common(p, agents=False)
    p.add_argument("-n", type=int, default=500)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("sensitivity", help="sensibilidad de los supuestos del motor")
    common(p)
    p.add_argument("-n", type=int, default=300)
    p.add_argument("--delta", type=float, default=0.2)
    p.set_defaults(func=cmd_sensitivity)

    p = sub.add_parser("morris", help="sensibilidad global sobre ranges.toml")
    common(p)
    p.add_argument("-n", type=int, default=300, help="partidas por punto")
    p.add_argument("-r", type=int, default=10, help="trayectorias")
    p.add_argument("--historical", action="store_true", help="usar la secuencia histórica forzada")
    p.add_argument("--out")
    p.set_defaults(func=cmd_morris)

    p = sub.add_parser("probe", help="sonda de fuga (requiere LLM)")
    common(p, agents=False)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("checkpoints", help="fidelidad en puntos de control")
    common(p, agents=False)
    p.add_argument("--agent", default="llm:historico")
    p.add_argument("-k", type=int, default=30)
    p.set_defaults(func=cmd_checkpoints)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
