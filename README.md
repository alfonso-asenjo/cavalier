# Cavalier

Simulador de toma de decisiones en crisis nucleares históricas con agentes de IA.
MVP: crisis de los misiles de Cuba (octubre de 1962).

No es un simulador de guerra: es un simulador de **decisiones bajo incertidumbre**. Lo militar
es abstracto (alerta, movilización, disciplina de mando, incidentes) y los resultados describen
el comportamiento de los agentes dentro de un motor con supuestos explícitos (`rules.toml`),
no probabilidades reales.

- Plan y evaluación punto por punto: [docs/PLAN.md](docs/PLAN.md)

## La web: la crisis contada y reescrita

```bash
cavalier serve      # abre http://localhost:8000
```

Cuenta la crisis para quien no sabe nada de ella: el contexto, los personajes y sus relaciones,
las fuerzas de cada bando, la cronología, los momentos en que casi se lía y por qué los Jupiter
importaban menos de lo que parecía. Cada dato lleva su fuente (`data/intro/cuba1962.toml`).
Después se puede **reproducir** una partida ya jugada o **simular una nueva**: se elige desde qué
momento empieza (el 16 de octubre, el Sábado Negro…), qué cambia respecto a la historia (sin canal
secreto, sin armas tácticas…) y qué modelo hace de cada líder, y se ve avanzar turno a turno.

Los agentes juegan con nombres ficticios (control del experimento); la web muestra los reales y
las fechas de 1962 (`[display]` en `aliases.toml`, `src/cavalier/display.py`).

`cavalier site --out site` genera la versión estática (introducción y partidas de ejemplo de
`examples/`), que se publica en GitHub Pages con `.github/workflows/pages.yml`. Simular una partida
nueva necesita el servidor local.

## Uso

```bash
pip install -e ".[llm,dev]"
pytest

cavalier replay cuba1962 -n 1000                                    # secuencia histórica forzada
cavalier montecarlo cuba1962 --usa scripted:halcon --ussr scripted:paloma -n 1000 --workers 16
cavalier run cuba1962 --usa scripted:pragmatica --ussr scripted:impredecible --seed 112 --view   # + visor HTML
cavalier view runs/p.jsonl                                           # visor de una partida registrada
cavalier view runs/p.jsonl --story                                   # relato por capítulos con fotos históricas (data/photos)
cavalier sensitivity cuba1962 -n 300                                # supuestos ±20 %
cavalier morris cuba1962 --historical -n 300 -r 10 --workers 16     # sensibilidad global (ranges.toml)
cavalier montecarlo cuba1962 --variant sin_armas_tacticas -n 1000   # contrafactual

# Con un modelo local (Ollama, llama.cpp, LM Studio o vLLM con API compatible OpenAI)
cavalier bench cuba1962 -k 5                                         # comparar modelos: JSON, coste, fuga
cavalier probe cuba1962 --model qwen3.5:9b                          # sonda de fuga
cavalier run cuba1962 --usa llm:historico --ussr llm:historico --model qwen3.5:9b --anon --log runs/llm.jsonl
cavalier run cuba1962 ... --log runs/llm.jsonl --resume              # continuar una partida cortada (guarda cada turno)
cavalier checkpoints cuba1962 --agent llm:historico --model qwen3.5:9b -k 30
cavalier fidelity cuba1962 -k 30                                     # fase 3: fidelidad con referencias (reanudable)
cavalier experiment cuba1962 --usa llm:historico --ussr llm:historico --anon -n 30   # fase 5: bifurcación desde el Sábado Negro
cavalier advisors-probe cuba1962 --anon                             # experimento 2, fase 0: ¿difieren los consejos de los asesores?

# Voces sintéticas de los líderes (experimental; entorno aparte, ver data/voices/README.md)
.venv-tts/Scripts/cavalier voices runs/exp1/base/s001.jsonl          # audio por mensaje + botones en el visor
```

Agentes: `scripted:{halcon,paloma,defensiva,pragmatica,impredecible}` y
`llm:{historico,halcon,paloma,defensiva,pragmatica,impredecible}`. A un agente LLM se le pueden
añadir los asesores de su bando: `llm:historico+asesores` (los 7 del experimento 2: McNamara,
Taylor, Thompson y Stevenson; Mikoyán, Malinovski y Dobrynin) o `+asesores_todos` (además, RFK,
Rusk, Bundy, LeMay, Gromyko y Castro). Con `--start usa_sabado_negro` (u otro punto de
`checkpoints.toml`) la partida arranca desde ese momento, y con `--effort {low,medium,high}` se
fija el razonamiento de los modelos comerciales.

### Una partida con modelos comerciales

Cada bando puede jugar con un modelo distinto: `llm:<persona>@<backend>[:<modelo>]`, con
backend `ollama`, `compat` (servidor local compatible con OpenAI), `openai` o `anthropic`.
Las claves se leen de `OPENAI_API_KEY` y `ANTHROPIC_API_KEY`.

```bash
pip install -e ".[llm]"
export OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...     # Windows (cmd): set OPENAI_API_KEY=...
cavalier run cuba1962 --anon --view \
  --usa llm:historico@openai:gpt-5.6-terra --ussr llm:historico@anthropic:claude-sonnet-5-5
```

Al terminar escribe el visor HTML (la ruta sale en pantalla); `cavalier view <log> --story` cuenta
la partida por capítulos.

**Coste aproximado de una partida completa** (unos 31 turnos; una llamada por bando y turno, de
unos 3.000 tokens de entrada y unos 850 de salida contando el razonamiento, con esfuerzo bajo).
Estimación conservadora con las tarifas de octubre de 2026 (una llamada real de prueba gastó
menos: 1.675 tokens de entrada y 452 de salida):

| EE. UU. (OpenAI) | URSS (Anthropic) | Una partida |
|---|---|---|
| `gpt-5.6-sol` | `claude-opus-5-5` | ~1,8 $ |
| `gpt-5.6-terra` | `claude-sonnet-5-5` | ~1 $ |
| `gpt-5.6-luna` | `claude-haiku-5-5` | <0,10 $ |
| `qwen3.5:9b` en local (Ollama, GPU de 8 GB) | | gratis |

Con asesores, cada consejo es una llamada más: con los 13 de `+asesores_todos` y esfuerzo medio,
una partida de 7 u 8 turnos con `gpt-5.6-luna` y `claude-haiku-5-5` gastó unos 200.000 tokens de
entrada y 45.000 de salida (entre 0,06 y 0,25 $, según la tarifa de Luna).

Los modelos comerciales razonan antes de responder y ese razonamiento se cobra como salida: el
cliente fija el esfuerzo en bajo. No se usan modelos de respaldo ante un rechazo, para que un bando
no cambie de modelo a mitad de partida; un rechazo cuenta como respuesta inválida (se reintenta y,
si vuelve a fallar, el bando espera ese turno).

### Usar otro modelo local

Por defecto, el modelo local es Qwen 3.5 9B, pero vale cualquier modelo de Ollama:

```bash
ollama pull llama3.1:8b
cavalier run cuba1962 --anon --view --usa llm:historico@ollama:llama3.1:8b --ussr llm:historico@ollama:gemma4:e4b
cavalier serve --local-model llama3.1:8b --local-model gemma4:e4b   # los añade al menú de la web
```

Con `@compat:<modelo>` y `--base-url` sirve también un servidor compatible con OpenAI (LM Studio,
llama.cpp, vLLM). Dos advertencias:

- El modelo tiene que responder con JSON que siga un esquema, y en español. Los modelos pequeños
  fallan más: las respuestas inválidas se reintentan y, si vuelven a fallar, el bando espera ese
  turno (lo cuenta `fallback_decisions` en el resumen).
- La fidelidad histórica y la fuga del modo anonimizado (`cavalier fidelity`, `cavalier probe`) se
  midieron con Qwen. Con otro modelo, conviene repetirlas antes de sacar conclusiones.

## Estructura de un escenario

| Fichero | Contenido |
|---|---|
| `scenario.toml` | Situación: bandos, briefings, variables y quién conoce qué (ruido, sesgo, ocultas), términos negociables, datos históricos de referencia |
| `rules.toml` | **Supuestos** del motor: efectos de acciones, dinámica, incidentes, inteligencia falsa |
| `aliases.toml` | Renombrado para el modo anonimizado |
| `advisors.toml` | Fichas de los asesores históricos (postura, acciones preferidas, línea roja, informe privado) |
| `historical.toml` | Secuencia histórica traducida a acciones (validación del motor) |
| `checkpoints.toml` | Puntos de control para medir fidelidad (prerregistro) |
| `ranges.toml` | Rango plausible y justificación de cada supuesto (Morris) |
| `variants/*.toml` | Contrafactuales y escenarios trampa |

## Licencia

El código es [MIT](LICENSE). Las fotografías de `data/photos` conservan su licencia original
(dominio público o Creative Commons, con autor y fuente en `photos.toml` y en el pie de la web).
