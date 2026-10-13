# Cavalier

Simulador de toma de decisiones en crisis nucleares históricas con agentes de IA.
MVP: crisis de los misiles de Cuba (octubre de 1962).

No es un simulador de guerra: es un simulador de **decisiones bajo incertidumbre**. Lo militar
es abstracto (alerta, movilización, disciplina de mando, incidentes) y los resultados describen
el comportamiento de los agentes dentro de un motor con supuestos explícitos (`rules.toml`),
no probabilidades reales.

- Plan y evaluación punto por punto: [docs/PLAN.md](docs/PLAN.md)
- Idea original: [inicial.md](inicial.md)

## Uso

```bash
pip install -e ".[llm,dev]"
pytest

cavalier replay cuba1962 -n 1000                                    # secuencia histórica forzada
cavalier montecarlo cuba1962 --usa scripted:halcon --ussr scripted:paloma -n 1000 --workers 16
cavalier run cuba1962 --usa scripted:pragmatica --ussr scripted:impredecible --seed 112 --view   # + visor HTML
cavalier view runs/p.jsonl                                           # visor de una partida registrada
cavalier sensitivity cuba1962 -n 300                                # supuestos ±20 %
cavalier morris cuba1962 --historical -n 300 -r 10 --workers 16     # sensibilidad global (ranges.toml)
cavalier montecarlo cuba1962 --variant sin_armas_tacticas -n 1000   # contrafactual

# Con un modelo local (Ollama, llama.cpp, LM Studio o vLLM con API compatible OpenAI)
cavalier probe cuba1962 --model qwen3:8b                            # sonda de fuga
cavalier run cuba1962 --usa llm:historico --ussr llm:historico --model qwen3:8b --anon --log runs/llm.jsonl
cavalier checkpoints cuba1962 --agent llm:historico --model qwen3:8b -k 30
```

Agentes: `scripted:{halcon,paloma,defensiva,pragmatica,impredecible}` y
`llm:{historico,halcon,paloma,defensiva,pragmatica,impredecible}`.

## Estructura de un escenario

| Fichero | Contenido |
|---|---|
| `scenario.toml` | Situación: bandos, briefings, variables y quién conoce qué (ruido, sesgo, ocultas), términos negociables, datos históricos de referencia |
| `rules.toml` | **Supuestos** del motor: efectos de acciones, dinámica, incidentes, inteligencia falsa |
| `aliases.toml` | Renombrado para el modo anonimizado |
| `historical.toml` | Secuencia histórica traducida a acciones (validación del motor) |
| `checkpoints.toml` | Puntos de control para medir fidelidad (prerregistro) |
| `ranges.toml` | Rango plausible y justificación de cada supuesto (Morris) |
| `variants/*.toml` | Contrafactuales y escenarios trampa |
