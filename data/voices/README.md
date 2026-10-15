# Voces sintéticas (experimental, fase 6)

`cavalier voices` lee una partida registrada, extrae lo que dice cada bando (mensajes privados
y declaraciones públicas) y lo lee con una voz clonada por bando: Kennedy para EE. UU. y
Jrushchov para la URSS. El visor (`cavalier view`) añade entonces un botón **▶ voz** a cada mensaje.

Es una capa de presentación: no interviene en la simulación ni en los experimentos.

## Uso responsable (obligatorio)

- **Las voces son sintéticas y los textos los inventan agentes de IA.** El manifiesto
  (`manifest.json`) y el visor lo dicen siempre; no hay que quitar ese aviso.
- **No publicar clips sueltos** que puedan pasar por grabaciones auténticas, y menos en una
  crisis nuclear con frases que nadie dijo. El audio se queda junto a su partida, en `runs/`
  (fuera de git).
- **XTTS‑v2 tiene licencia CPML (Coqui Public Model License)**: solo uso no comercial. La primera
  ejecución pide aceptarla. Para aceptarla sin la pregunta: `COQUI_TOS_AGREED=1`.

## 1. Entorno (aparte del proyecto)

XTTS necesita PyTorch con CUDA, y el Python global tiene PyTorch solo para CPU. Por eso va en
su propio entorno virtual, `.venv-tts` (fuera de git; en Linux y macOS, `.venv-tts/bin/` en lugar de
`.venv-tts/Scripts/`):

```bash
python -m venv .venv-tts
.venv-tts/Scripts/python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.venv-tts/Scripts/python -m pip install coqui-tts "transformers>=4.43,<5"   # coqui-tts 0.27 falla con transformers 5
.venv-tts/Scripts/python -m pip install -e ".[dev]"
```

**GPU:** XTTS‑v2 ocupa unos 2–3 GB de VRAM. No se puede usar mientras Ollama corre el modelo
de 9B (`num_ctx` 4096 llena los 8 GB), así que hay que sintetizar **entre tandas de
experimentos**, nunca durante una.

## 2. Audio de referencia

Hay que poner de 3 a 6 clips por líder en `data/voices/kennedy/` y `data/voices/jrushchov/`.

| | Fuente | Notas |
|---|---|---|
| **Kennedy** | JFK Presidential Library (jfklibrary.org, «Digital Collections»): discurso a la nación del 22‑oct‑1962 sobre Cuba, discurso de la American University (10‑jun‑1963), ruedas de prensa | Obras del gobierno de EE. UU.: dominio público. El discurso del 22 de octubre tiene el tono adecuado; las ruedas de prensa, una voz más conversacional |
| **Jrushchov** | Noticiarios soviéticos (net‑film.ru), Internet Archive, discurso en la ONU (1960) y entrevistas | En ruso, con más ruido. Conviene separar la voz (Demucs) y elegir tramos sin intérprete ni público |

Requisitos de cada clip: **6–20 s**, una sola persona hablando, sin música, aplausos ni
traductor simultáneo. Lo mejor es WAV mono a 22,05 o 24 kHz. Con más clips de tonos distintos,
la voz sale más estable.

```bash
# recortar un tramo (inicio 00:01:12, 15 s) y pasarlo a WAV mono 24 kHz
ffmpeg -i discurso_22oct.mp3 -ss 00:01:12 -t 15 -ac 1 -ar 24000 data/voices/kennedy/k01.wav

# grabaciones con ruido o música de fondo: separar la voz antes de recortar
pip install demucs   # en .venv-tts
.venv-tts/Scripts/python -m demucs --two-stems=vocals noticiario.mp4   # → separated/htdemucs/.../vocals.wav
```

Las grabaciones completas pueden quedarse en una subcarpeta (`kennedy/fuente/`): solo se usan los
ficheros sueltos de la carpeta del líder. Para comprobar que cada clip es del líder y no de un
locutor o un intérprete, se puede transcribir con Whisper (`faster-whisper` en `.venv-tts`).

**Idioma.** Cada líder habla en su idioma: Kennedy en inglés y Jrushchov en ruso (`language` de
cada bando en `voices.toml`). Los agentes escriben en español; antes de sintetizar, un LLM traduce
cada mensaje (por defecto Qwen en local, gratis: `translate_with`) y el español queda como
subtítulo en el visor. Al principio hablaban en español, pero así no se parecían: XTTS conserva
algo del timbre al cambiar de idioma, pero el acento y la entonación son los del idioma de salida.
Las traducciones se guardan en `translations.json` y no se repiten. El ruso de un modelo de 9B es
correcto, pero tiene algún error de vocabulario; para mejorarlo, `--translate-with anthropic:claude-sonnet-5-5`
(unos 10 céntimos por partida).

## 3. Sintetizar una partida

```bash
.venv-tts/Scripts/cavalier voices runs/exp1/base/s001.jsonl --dry-run      # ver qué se va a leer
cavalier voices runs/exp1/base/s001.jsonl --translate-only                 # solo traducir (sin XTTS; con Ollama)
.venv-tts/Scripts/cavalier voices runs/exp1/base/s001.jsonl --limit 4      # prueba corta
.venv-tts/Scripts/cavalier voices runs/exp1/base/s001.jsonl                # todo (reanudable)
```

El audio se guarda en `runs/exp1/base/s001.voces/` (`tNN_<BANDO>_<i>.wav` y `manifest.json`), y
`s001.html` se vuelve a generar con los botones de voz. Si el proceso se corta, al relanzarlo
no repite los ficheros que ya existen.
