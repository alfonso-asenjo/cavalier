# Cavalier — Plan de proyecto (v2)

> Simulador de toma de decisiones en crisis nucleares históricas con agentes de IA.
> MVP: crisis de los misiles de Cuba, octubre de 1962. **Estado: fase 1 completada (oct 2026).**

**Cómo se ha hecho este plan** (tres pasadas):
1. Borrador evaluando punto por punto el documento inicial (`inicial.md`).
2. Revisión adversarial independiente del borrador (validez científica, evaluación, coste, hoja de ruta).
3. Construcción del esqueleto (fase 0) y primeras ejecuciones reales: los hallazgos empíricos
   de la sección 3 han cambiado prioridades del plan.

---

## 0. Objetivo y encuadre

**Objetivo primario (decidido)**: estudiar cómo distintas *reglas de decisión* (agentes LLM con
perfiles distintos) y distintos *fallos* (inteligencia errónea, pérdida de control de la cadena de
mando, latencia de comunicaciones) cambian el desenlace de una crisis **dentro de este motor**.

**Objetivo secundario**: fidelidad histórica, como *validación* de que los agentes se comportan de
forma verosímil, no como fin.

Son dos diseños distintos y el primario manda: cuando choquen, se prioriza lo contrafactual.

**Encuadre**:
- Los resultados describen el comportamiento de modelos de lenguaje en un motor con supuestos
  explícitos. **No son probabilidades reales de guerra**. Todo número se publica junto a los
  supuestos (`rules.toml`) y su sensibilidad.
- Todo lo militar es abstracto: alerta, movilización, disciplina de mando e incidentes. No hay
  trayectorias, objetivos ni optimización de ataques. **Los líderes no tienen una acción de "lanzar"**:
  el uso nuclear solo aparece como estado terminal, por pérdida de control o por escalada tras un
  conflicto convencional.
- No se publican "estrategias óptimas".

---

## 1. Evaluación punto por punto del documento inicial

Veredictos: ✅ mantener · 🔧 mantener con cambios · ⏸ posponer · ❌ descartar

| # | Punto del documento inicial | Veredicto | Motivo y decisión |
|---|---|---|---|
| 1.1 | Arsenales SIPRI 2026 | 🔧 | Son de 2026 y el MVP es 1962. En 1962 la asimetría era enorme (EE. UU. ≈25.000 ojivas frente a ≈3.000 soviéticas), y esa inferioridad es lo que motivó el despliegue en Cuba. Los datos de 1962 van en `[historical_data]` con fuente y `verificado = false` hasta contrastarlos con Norris & Kristensen. El motor no los usa como inventario. |
| 1.2 | Tipos de misiles, contramedidas, objetivos críticos | ❌ | No responden a la pregunta de investigación y empujan hacia la planificación de ataques. Se sustituyen por variables abstractas. |
| 1.3 | Dos bandos con IA | 🔧 | Cuba no fue un juego de dos jugadores: el U‑2 derribado, el B‑59 y Castro. Por eso hay dos líderes más una **capa de cadena de mando** (incidentes estocásticos que dependen de la tensión, la alerta y el `control`). Los líderes pueden subir el control con órdenes restrictivas. ⏸ Castro y las facciones del ExComm como agentes, más adelante. |
| 1.4 | Información incompleta o falsa | ✅ | Pieza central. Niebla de guerra con ruido, **sesgo sistemático** (EE. UU. estimaba ≈10.000 soldados cuando eran ≈42.000), variables ocultas (armas tácticas en Cuba) e informes falsos. |
| 1.5 | Personalidades (halcón, paloma…) | 🔧 | Tienen dos usos: (a) agentes scripted de referencia, que son baratos y deterministas; (b) **personas para agentes LLM** en emparejamientos contrafactuales. Los perfiles históricos son otra persona LLM. |
| 1.6 | Stack (NumPy, LangGraph, FastAPI, React, MapLibre, SQLite, Pandas) | 🔧 | Orden de construcción: motor, CLI y JSONL primero (hecho); un visor mínimo de partidas en F2; el mapa estilo DEFCON en F6. SQLite cuando haya consultas entre partidas. |
| 1.7 | Fine‑tuning LoRA | ⏸ | 13 días producen decenas de decisiones por líder, no 500–2.000. Llegar a esa cifra exige datos sintéticos, y el LoRA aprendería la interpretación de otro LLM. Solo tiene sentido para **destilar**, es decir, para abaratar el Monte Carlo (F7, condicional). |
| 1.8 | RAG histórico con corte temporal | 🔧 | El filtro no es la fecha sino **quién lo sabía y desde cuándo** (`available_to`, `known_from`). Los testimonios retrospectivos **nunca** entran en el RAG de los agentes: filtran el desenlace. Ya implementado. |
| 1.9 | "El agente no debe conocer el desenlace" | ❌ → rediseño | **El problema principal.** El modelo base ya sabe cómo acabó Cuba, y renombrar no oculta la *estructura* del episodio. Se sustituye por el control de fuga de la sección 4.3. |
| 1.10 | Métricas (fidelidad, consistencia, escalada…) | 🔧 | Se redefinen en la sección 4. Se añaden fuga, sensibilidad a parámetros, tasa de respaldo y coste. Se elimina "calidad diplomática" mientras no tenga definición operativa. |
| 1.11 | "Modelos probabilísticos ficticios" para resolver consecuencias | 🔧 | Es lo más débil del documento inicial y lo que más pesa en los resultados (ver 3.1). Todos los supuestos están en `rules.toml`, con análisis de sensibilidad. |
| 1.12 | Turnos | 🔧 | Simultáneos, de 12 h. **La latencia de comunicación es una mecánica**: el canal oficial es lento, el extraoficial rápido, y las declaraciones públicas las ve todo el mundo. Implementado. |
| 1.13 | Épocas (Berlín 48, Cuba 62, Able Archer 83) | ✅ / ⏸ | El motor es genérico y cada escenario es una carpeta de datos. Able Archer será el segundo escenario. |
| 1.14 | Botón "Cambiar la historia" | ✅ | El estado es serializable y se puede bifurcar desde cualquier turno (`Game(state=...)`). También sirve para el Monte Carlo por división (4.4). |
| 1.15 | Modo espectador y registro de razonamientos | ✅ | JSONL por turno con la observación de cada bando, su decisión, el razonamiento, los documentos citados y el estado real. |
| 1.16 | Fuentes | 🔧 | Se añaden las grabaciones del ExComm (May & Zelikow, Miller Center), el National Security Archive, las notas de Malin del Presidium y Fursenko & Naftali. |
| 1.17 | **Potencias inventadas** (del documento inicial) | ✅ recuperado | La primera versión del plan lo había perdido. Es el mejor control de fuga: escenarios con estructura de crisis pero sin correlato histórico (F5). |

---

## 2. Lo que ya está construido (fase 0)

```
scenarios/cuba1962/
  scenario.toml        situación: bandos, briefings, variables (quién sabe qué, con ruido y sesgo), términos
  rules.toml           SUPUESTOS: efectos de acciones, dinámica, incidentes, interacciones, acuerdos
  ranges.toml          rango plausible y justificación de cada supuesto (para Morris)
  aliases.toml         modo anonimizado
  historical.toml      secuencia histórica traducida a acciones abstractas (borrador)
  checkpoints.toml     4 puntos de control prerregistrables (borrador)
  variants/            sin_armas_tacticas · tacticas_descubiertas · sin_jupiter (trampa) · comunicaciones_lentas
src/cavalier/
  models.py            estado, acciones, decisiones, observaciones, eventos (Pydantic)
  scenario.py          carga TOML, variantes y overrides (para sensibilidad)
  engine/              turnos simultáneos, niebla de guerra, mensajes con latencia, acuerdos, incidentes
  agents/scripted.py   5 personalidades + ReplayAgent (secuencia histórica forzada)
  agents/llm.py        LangGraph: retrieve → deliberate → validate (reintento) → respaldo marcado
  agents/llm_client.py API compatible OpenAI (Ollama, llama.cpp, LM Studio, vLLM) con JSON Schema
  rag/                 corpus con metadatos temporales y recuperación filtrada por bando y fecha
  anonymize.py         renombrado; el agente ve códigos opacos (A01, T1) en lugar de claves internas
  eval/                sonda de fuga, puntos de control forzados, IC de Wilson
  montecarlo.py        lotes, agregados con IC, partidas con respaldo separadas
  sensitivity.py       OAT y cribado de Morris, con números aleatorios comunes
  cli.py               run · montecarlo · replay · sensitivity · morris · probe · checkpoints (--workers)
tests/                 26 tests: motor, invariantes cualitativos, agente LLM con cliente falso, RAG, checkpoints
```

---

## 3. Fase 1: motor v2 (octubre de 2026)

### 3.0 Qué se ha cambiado
| Problema (fase 0) | Solución |
|---|---|
| La tensión se saturaba en 100 | Actualizaciones en **escala logit** para todas las magnitudes acotadas: un +10 vale ~10 puntos en el centro y cada vez menos cerca del extremo, que nunca se alcanza |
| Aceptar un acuerdo terminaba la partida | **Fase de cumplimiento**: compromisos con plazo (`implementation_turns`), **verificación** por el otro bando (inteligencia o inspección), **ruptura** (`renege_deal`) y **filtración** de términos secretos. La crisis termina cuando lo exigible está cumplido y verificado. La retirada de los Jupiter se pacta a confianza (`verification_required = false`), como en la historia |
| Ceder pronto salía gratis | El coste de credibilidad de una concesión se multiplica por `1 + penal·(1 − tensión/100)` y genera presión interna |
| Orden aleatorio de acciones simultáneas | **Resolución por fases**: comunicación → postura → **interacciones** → terminales → aceptaciones. Una invasión ordenada no la para una carta que llega el mismo turno; levantar la cuarentena el mismo turno en que los buques la fuerzan evita el choque |
| Incidentes como datos | Incidentes e inteligencia falsa movidos a `rules.toml`: son supuestos |
| Sensibilidad solo local | **Cribado de Morris** sobre `ranges.toml` (17 parámetros con rango y justificación, `basis = fuente/autor`) y ejecución en paralelo (`--workers`) |

**Verificación histórica** (con fuentes; detalles en `historical.toml` y `checkpoints.toml`):
- La orden soviética del 22 de octubre solo retenía *implícitamente* el uso nuclear; la prohibición explícita es del 27.
- La retirada de los buques la decidió Moscú el 23 (turno 14) y Washington la observó el 24.
- Solo el SAC pasó a DEFCON 2.
- La CIA estimaba 8.000 → 10.000 → 22.000 soldados soviéticos; eran 41.902.
- Había ~98 ojivas tácticas en Cuba.
- **La primera carta de Jruschov tardó ~10 h**: a la resolución de 12 h, el canal oficial y el extraoficial colapsan en un turno. La latencia se estudia como contrafactual (`variants/comunicaciones_lentas.toml`; el "teléfono rojo" se creó en 1963 precisamente por esto).
- El punto de control inicial pasa al **20 de octubre** (turno 8), cuando JFK decidió la cuarentena.

### 3.1 Resultados con el motor v2 (2.000 partidas por fila)

> **Corrección (13‑oct‑2026)**: la primera versión de esta tabla tenía un fallo. Las observaciones
> ofrecían el término "levantar la cuarentena" aunque no hubiera cuarentena, y el motor descartaba
> esas propuestas enteras. Las cifras de abajo ya están corregidas, y un test impide que los agentes
> scripted generen acciones inválidas. La secuencia histórica no estaba afectada.

| Emparejamiento | Acuerdo | Desescalada | Sin resolver | Convencional | Nuclear | Turnos |
|---|---|---|---|---|---|---|
| Histórico (forzado) | 83,2 % | 0 % | 1,2 % | 0 % | 15,6 % | 28,4 |
| halcón × halcón | 0 % | 0 % | 0 % | 59,6 % | 40,4 % | 7,4 |
| paloma × paloma | 86,3 % | 13,7 % | 0 % | 0 % | 0,1 % | 6,4 |
| pragmática × pragmática | 74,6 % | 12,0 % | 5,0 % | 2,9 % | 5,5 % | 15,6 |
| pragmática × defensiva | 77,8 % | 7,2 % | 3,4 % | 4,2 % | 7,3 % | 14,7 |
| halcón × paloma | 26,9 % | 2,2 % | 3,9 % | 38,9 % | 28,1 % | 13,9 |
| impredecible × pragmática | 77,5 % | 4,0 % | 0,3 % | 10,4 % | 7,8 % | 9,3 |
| prag × def, sin armas tácticas | 77,8 % | 7,2 % | 3,4 % | 5,2 % | 6,3 % | 14,7 |
| prag × def, comunicaciones lentas | 77,3 % | 7,1 % | 2,8 % | 4,3 % | 8,4 % | 14,6 |

Las cifras son de agentes scripted: validan el motor, no dicen nada sobre 1962. Dos palomas
cierran un acuerdo en ~3 días: aceptan casi cualquier propuesta, y la duración la marca solo el
plazo de cumplimiento.

### 3.2 Sensibilidad global (Morris, 10 trayectorias × 300 partidas por punto)

| Métrica | Secuencia histórica | pragmática × defensiva |
|---|---|---|
| Uso nuclear | `incidents.1.nuclear_risk` (submarino) **49,3 %** · Maultsby 10,5 % · resto < 9 % | submarino 20,3 % · `implementation_turns` 14,9 % · penalización por ceder 12,9 % |
| Guerra (convencional + nuclear) | igual que nuclear (las acciones fijas no incluyen ataques) | `implementation_turns` 21,5 % · **penalización por ceder 15,8 %** · submarino 12,8 % |
| Acuerdo | submarino 36,2 % · `verification_rate` 13,7 % | `implementation_turns` 21,1 % · `tension_decay` 18,7 % · penalización por ceder 18,5 % |

Lectura:
- **El riesgo nuclear del submarino es el supuesto crítico del modelo.** En la secuencia histórica,
  recorrer su rango plausible (2 %–30 %) mueve el uso nuclear en ~24 puntos. Coincide con la
  historiografía (el B‑59 se considera el momento más peligroso), pero significa que **ningún
  porcentaje de uso nuclear se puede dar como un número**. Se presentará siempre como una curva
  en función de ese parámetro.
- **Con agentes que deciden, lo que más pesa es la mecánica de los acuerdos.** `implementation_turns`
  tiene base en fuentes; `concession_low_tension_penalty` es un supuesto mío y es el siguiente
  candidato a justificar o a presentar como condicionante. (Una versión anterior señalaba
  `tension_decay`: era un artefacto del fallo corregido arriba.)
- Con agentes, ningún parámetro pasa del 22 %: los resultados contrafactuales son más robustos
  que la repetición histórica.

### 3.3 Criterios de salida de la fase 1
| Criterio | Estado |
|---|---|
| Invariantes cualitativos en verde | ✅ 38 tests en verde |
| Secuencia histórica con P(acuerdo) ∈ [50 %, 95 %] | ✅ 83,2 % |
| Duración media ≥ 8 turnos | ✅ histórico 28,4; pragmáticos 15–16 (antes 2–4) · ⚠️ paloma × paloma 6,4 (aceptan casi todo; la duración es solo el plazo de cumplimiento) |
| Parámetros dominantes identificados y tratados (criterio revisado; el original pedía "ningún parámetro > 50 %") | ✅ Submarino (49,3 % en la secuencia histórica, rango con base en fuentes): todo resultado de uso nuclear se presenta como curva sobre `incidents.1.nuclear_risk`. `concession_low_tension_penalty` (15,8 % de la guerra con agentes, supuesto del autor): pendiente de justificar; mientras tanto, condicionante declarado |
| Secuencia histórica y puntos de control verificados y congelados | ✅ verificados con fuentes y congelados con la etiqueta `prereg-v1` |

### 3.4 Pendiente, conocido
- Clasificación del punto de control del 27 de octubre: no distingue una propuesta con término secreto de una pública.
- La granularidad de 12 h aplana las latencias reales (horas). Si la latencia resulta importante en los experimentos, habrá que bajar a turnos de 6 h.
- Solo hay dos interacciones simultáneas modeladas (choque naval y espiral de movilización).

---

## 3‑ter. Fase 2: comparación de modelos locales (13‑oct‑2026)

`cavalier bench`: 4 puntos de control × 2 modos (real y anonimizado) × K = 5, más la trampa
`sin_jupiter` (2 puntos × 2 modos × 5) y la sonda de reconocimiento. Todo con Ollama en la RTX 2070,
contexto de 4.096 tokens y sin modo "thinking". Resultados en `runs/bench_v2.json`.

| | qwen3.5:9b | qwen3.5:4b |
|---|---|---|
| JSON válido al primer intento | **98 %** ✅ | 82 % ❌ |
| Decisiones de respaldo | **0 %** ✅ | **0 %** ✅ |
| Segundos por decisión (media / p90) | 10,7 / 13,2 | 8,1 / 12,6 |
| Velocidad de generación | 48,7 tok/s | 70,5 tok/s |
| Tokens de salida por decisión | 340 | 314 |
| Nombra el episodio real en modo anonimizado | 0 % | 0 % |
| Trampa (menciona misiles inexistentes en Turquía) | 1 de 40 (en el texto de una acción, sin verificar) | 0 de 40 |
| Sonda: reconoce el episodio con nombres reales | sí (ambos bandos) | sí (ambos bandos) |
| Sonda: reconoce el episodio anonimizado | **sí, desde el briefing de EE. UU.** | no |

**Qué cambió respecto a la primera medición** (55 % de JSON válido y 17,5 % de respaldo con el 9B):
- un **esquema JSON cerrado por turno**, que solo admite acciones legales, términos disponibles, propuestas abiertas y canales;
- **pistas de campos requeridos** en el prompt;
- **correcciones de formato que no cambian la decisión** (confianza 0–100, `deal_id` único).

También se corrigió la trampa: la variable de los Jupiter seguía visible con valor 0 y contaminaba la medición.

**Lectura:**
1. **Modelo principal: `qwen3.5:9b`.** Es el único que cumple el criterio de la fase 2 (JSON ≥ 98 %,
   respaldo ≤ 1 %).
2. **El plan de "dos modelos" pierde sentido.** El 4B solo es 1,3 veces más rápido: genera casi los
   mismos tokens, y el coste está en generar. No compensa su peor formato. El Monte Carlo con LLM
   se hará con el 9B: unos 9–10 minutos por partida completa, es decir, lotes nocturnos de 50–80
   partidas, más bifurcación desde estados críticos. Queda descartado salvo que un 4B posterior iguale
   el formato.
3. **Señal de fuga (exploratoria, a confirmar en la fase 3).** En la respuesta inicial de EE. UU. (20 de
   octubre), los modelos eligen la cuarentena histórica en **7 de 10 casos con nombres reales y 0 de 10
   anonimizados** (Fisher, una cola, p = 0,0015; ambos modelos sumados). Con nombres reales, el modelo
   tiende a repetir la historia. **La fidelidad medida con nombres reales no se puede interpretar sin el
   control anonimizado**, que es justo lo que el plan preveía (sección 4.3).
4. **El 9B reconoce la crisis anonimizada** a partir del briefing de EE. UU.: la estructura delata el
   episodio. Refuerza la necesidad de variantes perturbadas y de potencias inventadas (fase 5).
5. **Memoria.** En Windows, `llama-server` reserva ~12 GB de memoria del sistema aunque el modelo vaya
   al 100 % en la GPU. Por eso la comparación es reanudable y guarda tras cada decisión. Gemma 4 e4b no
   cabe en la GPU (deja casi todo en RAM): descartada.

**Criterios de salida de la fase 2:** JSON ✅ (9B) · respaldo ✅ · coste medido ✅ · fuga reportada en
modo real, anonimizado y trampa ✅ · visor mínimo ✅ (adelantado, con estética DEFCON) ·
partidas reanudables ✅ · primeras partidas completas entre dos LLM ✅. **Fase 2 cerrada.**

### Primeras partidas completas LLM contra LLM (9B, anonimizado, `llm:historico`)

Las partidas guardan tras cada turno un punto de control (`runs/<partida>.ckpt.json`) con el estado,
el generador aleatorio y la memoria de los agentes. `cavalier run ... --log X --resume` continúa desde
ahí, y un test comprueba que una partida cortada y reanudada termina igual que una sin cortes. Cada
turno cuesta unos 25 s (dos decisiones), así que una partida de 10–20 turnos dura 4–8 minutos.

| Semilla | Resultado | Turnos | Tensión máx. | Respaldos | Inválidas |
|---|---|---|---|---|---|
| 0 | conflicto convencional (ataque aéreo de EE. UU., día 5) | 10 | 70 | 0 | 1 |
| 1 | "acuerdo negociado" **falso** (ver abajo) | 19 | 97 | 0 | 0 |

Son dos partidas: sirven para encontrar fallos, no para medir nada. Encontraron tres.

1. **Propuestas descartadas en silencio.** El modelo marcó como secreta la inspección de la ONU, que
   no admite secreto, y el motor descartó la propuesta entera. Ahora el prompt indica qué términos
   pueden pactarse en secreto, y el agente pacta en público un término no secretable (corrección
   tolerante, con test).
2. **Un acuerdo que no resuelve la crisis la cerraba.** En la semilla 1, la URSS propuso la retirada
   secreta de los Jupiter sin dar nada a cambio, EE. UU. aceptó en DEFCON 1 y el motor declaró
   "acuerdo negociado" con los misiles todavía en la isla. Ahora los términos llevan `resolves_crisis`
   (en Cuba, solo la retirada de los misiles). Otro acuerdo cumplido baja la tensión, pero la crisis
   sigue. La tabla 3.1 no cambia, porque los acuerdos de los agentes scripted siempre incluyen la retirada.
3. **El registro guardaba propuestas ya caducadas como abiertas.** La observación compartía objetos
   con el estado real y el JSONL las grababa ya modificadas. No afectaba a las decisiones; ahora se copian.

Comportamiento a vigilar en la fase 3, sin sacar conclusiones con n = 2:
- La URSS no responde a las propuestas de EE. UU.: ni las acepta ni las rechaza, solo contrapropone.
- EE. UU. sube a DEFCON 1 en el día 3 (en 1962 nunca pasó de DEFCON 2) y se queda ahí una semana.
- En modo anonimizado, la URSS llega sola a la estructura del pacto histórico: retirada secreta de los
  Jupiter más compromiso de no invasión. No es necesariamente fuga, porque los Jupiter están en su briefing.

---

## 3‑quater. Fase 3: plan de análisis prerregistrado (13‑oct‑2026, etiqueta `prereg-v2`)

Se fija **antes** de ejecutar ningún LLM en la fase 3. Los puntos de control y su codificación
siguen congelados desde `prereg-v1`.

**Configuración.** Modelo `qwen3.5:9b` (Ollama, id `56671c2ab938`), temperatura 0,7, `num_ctx` 4096,
sin "thinking". K = 30 por celda, semillas 0–29. Comando:
`cavalier fidelity cuba1962 -k 30 --out runs/fidelity_v1.json`.

**Condiciones** (en los 4 puntos de control):
- `llm:historico` y `llm:ninguna` (mismo briefing y nombre del dirigente, sin la instrucción de
  imitarlo), cada una en modo real y anonimizado: 4 × 4 × 30 = 480 decisiones;
- referencias: las cinco personalidades scripted (K = 30) y el azar uniforme sobre las opciones consideradas.

**Métrica.** P(opción histórica) por punto y condición, con IC 95 % de Wilson. **Nunca se agrega**
en una sola cifra.

**Contrastes** (Fisher exacta bilateral, corrección de Holm dentro de cada familia):
1. *Fuga*: real frente a anonimizado, por punto y persona (8 pruebas). Confirmatoria para
   `usa_respuesta_inicial` con `historico`, donde la fase 2 dio 7/10 frente a 0/10. El resto es descriptivo.
2. *Aporte de la persona*: `historico` frente a `ninguna` **en modo anonimizado**, por punto (4
   pruebas). Se usa el modo anonimizado porque en modo real el efecto se mezcla con la fuga.

**Reglas de lectura fijadas de antemano:**
- Un punto "se reproduce" en una condición si el límite inferior del IC supera al azar uniforme.
- Si la fuga es significativa (Holm) en un punto, la fidelidad en modo real de ese punto se
  reporta como **no interpretable**.
- La persona histórica se mantiene en la fase 5 si en modo anonimizado no queda por debajo de
  `ninguna` en ningún punto con diferencia significativa. Si queda por debajo, la fase 5 usa `ninguna`.
- Las decisiones de respaldo cuentan como "otra" y se reportan aparte. Si superan el 2 % en una
  celda, la celda se marca.

### Resultados (13‑oct‑2026, `runs/fidelity_v1.json`)

480 decisiones LLM: 0 respaldos, 97,9 % de JSON válido al primer intento, 10,5 s por decisión.
Ninguna celda marcada. Columnas: P(opción histórica) [IC 95 %], n = 30 por celda.

| Punto (opción histórica, azar) | histórico real | histórico anon | sin persona real | sin persona anon | scripted (rango) |
|---|---|---|---|---|---|
| EE. UU. respuesta inicial (cuarentena, 17 %) | 50 % [33, 67] | 7 % [2, 21] | 53 % [36, 70] | 10 % [3, 26] | 0–17 % |
| URSS línea de cuarentena (dar la vuelta, 20 %) | 0 % [0, 11] | 0 % [0, 11] | 0 % [0, 11] | 3 % [1, 17] | 0–10 % |
| EE. UU. Sábado Negro (proponer acuerdo, 20 %) | 20 % [10, 37] | 10 % [3, 26] | 17 % [7, 34] | 17 % [7, 34] | 0 % |
| URSS aceptación (aceptar, 33 %) | 43 % [27, 61] | 3 % [1, 17] | 37 % [22, 54] | 17 % [7, 34] | 100 % |

**Contraste 1, fuga (Holm, 8 pruebas).** Es significativa en la respuesta inicial con ambas personas
(+43 puntos; p ajustado 0,003 y 0,004) y en la aceptación con persona histórica (+40 puntos;
p ajustado 0,003). **La hipótesis confirmatoria se cumple**: la señal 7/10 frente a 0/10 de la fase 2
se replica con n = 30. En el resto de puntos no hay diferencia (p ajustado ≥ 0,7).

**Contraste 2, aporte de la persona (modo anonimizado, Holm, 4 pruebas).** No hay ninguna diferencia
significativa (la menor es la aceptación, 1/30 frente a 5/30, p ajustado 0,78).

**Lectura según las reglas fijadas:**
1. **Ningún punto se reproduce en una condición interpretable.** En modo anonimizado, todos los
   límites inferiores quedan por debajo del azar. En modo real, los dos únicos puntos que se acercan
   (respuesta inicial y aceptación) tienen fuga significativa, así que son **no interpretables**.
2. **La persona histórica se mantiene en la fase 5**: no queda por debajo de `ninguna` en ningún
   punto. Tampoco aporta nada medible.
3. Con nombres reales, el modelo "acierta" justo donde el desenlace es famoso (la cuarentena y la
   aceptación de Jruschov) y no en los puntos menos conocidos (los buques, el Sábado Negro).
   **Recuerda, no razona como los dirigentes de 1962.**

**Exploratorio** (no prerregistrado; sirve para diseñar la siguiente evaluación, no como resultado):
- **"Otra" domina** (63–87 % de muchas celdas). El agente toma hasta 3 acciones por turno y suele
  combinar acciones "blandas" (declaración, inteligencia, contención) sin ninguna de las opciones
  consideradas. En la línea de cuarentena, la URSS envía mensajes y contiene a sus fuerzas, pero
  nunca da la vuelta a los buques (0/120). Todas las opciones consideradas son legales en los cuatro puntos.
- **El Sábado Negro está mal codificado**: el modelo acepta a menudo la propuesta soviética abierta
  (la primera carta: retirada a cambio de no invasión). Es lo más parecido a lo que hizo Kennedy, pero
  `accept_deal` no está entre las opciones y cuenta como "otra". Contando proponer o aceptar sin
  acción militar, la vía diplomática sube al 43–60 % en las cuatro condiciones.
- **El modelo es muy poco escalador**: solo 1 de 480 decisiones incluye ataque o invasión. Contrasta
  con la literatura (Rivera 2024) y con la partida libre de la semilla 0, donde EE. UU. bombardeó tras
  varios días sin respuesta. En los puntos de control falta esa frustración acumulada.
- Para una `prereg-v3`: pedir una "decisión principal" explícita en los puntos de control, en vez de
  inferirla de hasta 3 acciones, y añadir `accept_deal` a las opciones del Sábado Negro.

**Consecuencia para la fase 5.** Los contrafactuales no pueden presentarse como "qué habrían hecho
Kennedy y Jruschov". Son experimentos sobre agentes LLM dentro de un modelo de crisis con la
estructura de 1962, siempre en modo anonimizado y condicionados a `rules.toml`. Era el objetivo
primario (sección 0); ahora está justificado con datos y no solo como precaución.

---

## 3‑quinquies. Fase 5, experimento 1: bifurcación desde el Sábado Negro (prerregistro `prereg-v3`)

Fijado el 13‑oct‑2026 **antes** de ejecutar ninguna partida LLM del experimento. Solo se jugó una
partida piloto (semilla 1000, fuera del rango) para medir la duración: 13 turnos, 334 s, sin respaldos.

**Diseño.** Se reconstruye el estado histórico del 27 de octubre por la noche (punto de control
`usa_sabado_negro`, con el derribo del U‑2 forzado). Todas las celdas parten de ese mismo pasado y
juegan hasta el final (como mucho 13 turnos) con su variante:

| Celda | Cambio desde el Sábado Negro | Mecánica en el motor |
|---|---|---|
| `base` | ninguno | — |
| `sin_canal_secreto` | sin el canal R. Kennedy–Dobrynin; los Jupiter no pueden pactarse en secreto | ceder los Jupiter cuesta 5 veces más credibilidad; el canal solo cambia lo que ve el agente (en el motor ambos canales tienen la misma latencia) |
| `mando_debil` | el doble de incidentes de la cadena de mando | `incident_prob_multiplier` = 2,0 (extremo superior de `ranges.toml`) |

**Agentes.** `llm:historico` en ambos bandos (la regla de la fase 3 la mantiene), `qwen3.5:9b`
(`56671c2ab938`), temperatura 0,7, `num_ctx` 4096, **modo anonimizado**. Los agentes empiezan la
bifurcación sin memoria de turnos anteriores; el pasado les llega por el estado y los eventos.
n = 30 partidas por celda, semillas 0–29 intercaladas. Comando:
`cavalier experiment cuba1962 --usa llm:historico --ussr llm:historico --anon -n 30 --out runs/exp1`.

**Métricas.** P(acuerdo negociado) y P(escalada militar = conflicto convencional o uso nuclear), con
IC 95 % de Wilson. El uso nuclear se reporta solo como descripción, porque depende sobre todo del
parámetro del submarino (Morris, 3.2).

**Contrastes.** Cada variante frente a `base`, en ambas métricas: 4 pruebas exactas de Fisher
bilaterales con corrección de Holm.
- H1: `sin_canal_secreto` cambia P(acuerdo). La historiografía considera decisivo el pacto secreto.
- H2: `mando_debil` cambia P(escalada).

**Referencia del motor** (agentes scripted, 2.000 partidas por celda desde el mismo estado):

| Emparejamiento | base: acuerdo / escalada | sin canal secreto | mando débil |
|---|---|---|---|
| pragmática × defensiva | 56 % / 43 % | +0,1 / −0,1 pp | −3,1 / +3,3 pp |
| pragmática × pragmática | 57 % / 43 % | +0,2 / −0,2 pp | −3,1 / +3,1 pp |
| halcón × paloma | 26 % / 73 % | 0 / 0 pp | −1,6 / +1,6 pp |

Por tanto, el motor no produce por sí mismo un efecto del canal secreto: **cualquier efecto en H1
vendrá del razonamiento de los agentes LLM**. El efecto mecánico de `mando_debil` es de unos 3 puntos,
indetectable con n = 30. Si H2 sale significativa, el efecto viene de cómo reaccionan los agentes a
los incidentes.

**Potencia.** Con n = 30 por celda solo se detectan diferencias de ~35 puntos (80 % de potencia,
α = 0,05). Un resultado nulo **no** demuestra ausencia de efecto y se reporta con su IC.

**Reglas de lectura:**
- Las partidas con decisiones de respaldo se reportan aparte. Si hay más del 10 % en una celda, se marca.
- Las conclusiones se formulan como "agentes LLM en este modelo", nunca como "qué habría pasado en 1962".
- Si la ejecución se corta antes de n = 30, se reporta el n alcanzado (las celdas quedan equilibradas
  porque las semillas se intercalan) y se marca como incompleto.

### Resultados (14‑oct‑2026, `runs/exp1/report.json`)

Ejecución completa: n = 30 por celda, semillas 0–29, el mismo modelo fijado (`56671c2ab938`). La
ejecución se cortó tres veces (dos por falta de memoria en el equipo) y se reanudó desde los
puntos de control. Las semillas intercaladas mantuvieron las celdas equilibradas, y ninguna
partida se repitió ni se perdió.

| Celda | P(acuerdo) [IC 95 %] | P(escalada militar) [IC 95 %] | Desenlaces | Partidas con respaldo |
|---|---|---|---|---|
| `base` | 70 % [52, 83] | 10 % [3, 26] | 21 acuerdos, 6 sin resolver, 2 conflictos convencionales, 1 uso nuclear | 1 |
| `sin_canal_secreto` | 63 % [46, 78] | 20 % [10, 37] | 19 / 5 / 4 / 2 | 1 |
| `mando_debil` | 57 % [39, 73] | 10 % [3, 26] | 17 / 10 / 1 / 2 | 3 (10 %, en el límite de la regla) |

| Contraste frente a `base` | Diferencia | Fisher p | Holm p |
|---|---|---|---|
| H1: `sin_canal_secreto`, P(acuerdo) | −7 pp | 0,78 | 1 |
| `sin_canal_secreto`, P(escalada) | +10 pp | 0,47 | 1 |
| `mando_debil`, P(acuerdo) | −13 pp | 0,42 | 1 |
| H2: `mando_debil`, P(escalada) | 0 pp | 1 | 1 |

**Lectura (según las reglas fijadas):**
- **H1 y H2: sin efecto detectable.** Ningún contraste se acerca a la significación, ni antes ni
  después de la corrección de Holm. Con n = 30, la potencia solo alcanza diferencias de unos
  35 puntos, así que **esto no demuestra que no haya efecto**: los IC son anchos y compatibles con
  efectos moderados en cualquier dirección.
- **Partidas con decisiones de respaldo, aparte.** Son 5 en total. Si se excluyen, `mando_debil`
  pasa del −13 al −6 pp en P(acuerdo), porque sus tres partidas con respaldo acabaron sin resolver.
  Parte de la caída aparente de `mando_debil` viene, por tanto, de fallos del agente y no de su
  razonamiento. Los demás contrastes apenas cambian: −7 y +10 pp en `sin_canal_secreto`, +1 pp en
  la escalada de `mando_debil`. Todos los p ≥ 0,47.
- **Patrón descriptivo, no concluyente:** sin el canal secreto se duplican las escaladas militares
  (6 frente a 3). La dirección coincide con la historiografía, pero el experimento no tiene potencia
  para afirmarlo. Si se quiere estudiar, hace falta un experimento con más n solo para esa celda.
- **El uso nuclear se da solo como descripción** (depende sobre todo del parámetro del submarino):
  1, 2 y 2 partidas.
- Las conclusiones se refieren a agentes LLM (`qwen3.5:9b`, anonimizados) dentro de este motor, no
  a lo que habría pasado en 1962.

---

## 3‑sexies. Fase 5, experimento 2 (propuesta, sin prerregistrar): asesores históricos

Propuesto el 14‑oct‑2026 y revisado el mismo día con el informe
[`reports/Asesores en la crisis de Cuba.md`](../reports/Asesores%20en%20la%20crisis%20de%20Cuba.md)
(notas de investigación en `research_notes/`). **No es un prerregistro.** Antes de ejecutar nada hay
que fijar en un `prereg-v4` las métricas, las celdas, n y el criterio de corte de la fase 0.

**Motivación.** Hoy cada bando es un solo agente que decide a solas. La hipótesis es que dar voz a
asesores con posturas documentadas **aumenta la variedad de las partidas**: trayectorias distintas,
no solo desenlaces distintos.

**Lo que dice la historia.**
- El ExComm fue un debate abierto y dividido. La línea divisoria se movió tres veces: el 16 de octubre casi
  todos querían atacar; entre el 17 y el 20 se formaron un bando del bloqueo y otro del ataque; el 27 se
  discutía si canjear los Jupiter o no.
- Jrushchov ideó y cerró la crisis prácticamente solo, y el Presídium lo ratificaba por unanimidad.
  La exigencia de Turquía del 27 fue iniciativa suya.
- La asimetría es en sí una predicción comprobable: con la misma estructura, los asesores deberían
  dar más variedad al bando estadounidense. Si no lo hacen, el modelo está aplanando la asimetría histórica.

**Lo que dice la literatura sobre LLM.** La predicción nula es fuerte:
- En equipos simulados por LLM se observa una «armonía farsesca», incluso con rasgos extremos
  (Lamparth et al., 2024).
- Los rasgos de persona explican menos del 10 % de la varianza (Hu y Collier, 2024).
- Las personas derivan en menos de 8 rondas (Li et al., 2024).
- El debate entre agentes colapsa en consenso.

Lo que sí ha bajado la concentración de opiniones en consejos de modelos de 7–9B es la **estructura**:
información distinta por asesor y, sobre todo, **un modelo distinto por perspectiva** (Sela, 2026, preprint).

**Asesores** (fichas completas con su nivel de evidencia en el informe; posturas corregidas respecto
a la primera versión de esta propuesta):

| Bando | Asesor | Acción ancla | Línea roja | Notas |
|---|---|---|---|---|
| EE. UU. | McNamara | Cuarentena, presión gradual | Ataque sin aviso | Aceptaba el canje de los Jupiter; el 27 también preparaba lo militar |
| EE. UU. | Taylor (LeMay como variante dura) | Ataque aéreo, represalia contra el SAM | Ataque limitado | LeMay no estaba en el ExComm |
| EE. UU. | Thompson | Responder a la oferta más blanda | Canje **público** con la OTAN | Lee a Jrushchov; se opuso al canje de Turquía el 27 |
| EE. UU. | Stevenson (opcional) | Negociar, retirar los Jupiter | Ataque por sorpresa | Aporta la opción de acomodo que los modelos no generan solos |
| URSS | Mikoyán | Desmantelar a cambio de garantías, la ONU | Dar pretextos para un ataque | El único escéptico documentado del despliegue |
| URSS | Malinovski | Alerta sin uso nuclear | Uso sin orden de Moscú | Leal, subestimaba a EE. UU.; **no** es «el halcón» |
| URSS | Dobrynin (opcional) | Canal secreto | Ultimátum público | Lee Washington; no habla en el Presídium |

**Qué no inventar.**
- Cada rasgo va etiquetado como cinta o documento, memoria o interpretación de historiador. El modelo puede responder «no consta».
- No atribuir a RFK el pacifismo de *Thirteen Days*, ni el «Trollope ploy» a nadie en concreto.
- No dar a Kózlov ni a Biriuzov posturas de octubre.
- No tratar como hecho la regla de los tres oficiales del B‑59.
- **En el modo anonimizado, los asesores llevan nombre de cargo** («Secretario de Defensa») y argumentos
  sin referencias históricas concretas («Pearl Harbor» rompería el anonimato).

**Mecánica (contra el eco).**
- El asesor habla solo en los turnos clave: incidente, propuesta recibida, cambio de DEFCON y primer turno.
- Cada asesor recibe un **informe privado distinto** derivado del estado del juego:
  - Taylor: la disponibilidad militar;
  - McNamara: la cuarentena y la postura de fuerzas;
  - Thompson y Stevenson: la última propuesta y la reacción de la ONU;
  - Mikoyán: el coste de escalar;
  - Malinovski: los informes de alerta;
  - Dobrynin: las señales del canal secreto.
- Los consejos se generan **de forma independiente y en una sola ronda**. Ningún asesor ve a los demás ni la intención del líder.
- La salida es estructurada: acción del enum, confianza, un riesgo y qué le haría cambiar de opinión.
- **El prompt se reconstruye en cada turno clave**, sin historial: evita la deriva y cabe en 4096 tokens.
- Si todos coinciden, el último asesor (Stevenson o Mikoyán) defiende la mejor alternativa.
- Los consejos se registran y el visor los muestra en «Qué pasó por dentro».

**Fase 0: validar antes de jugar** (pocas horas de GPU).
- Se congelan 10–20 estados reales (el Sábado Negro y otros turnos clave) y se ejecuta cada asesor K = 10 veces por estado.
- Medidas:
  - la distribución de acciones recomendadas por asesor (divergencia de Jensen–Shannon entre pares y concentración en la primera opción);
  - la distancia semántica entre asesores frente a la distancia dentro de un mismo asesor;
  - una **atribución ciega**: un juez debe acertar quién escribió cada consejo por encima del azar;
  - una **prueba de complacencia**: se inyecta la intención del líder y se mide cuánto se desplazan los consejos;
  - la deriva: adherencia a la acción ancla en los primeros y en los últimos turnos clave.
- Criterio de corte (a fijar en `prereg-v4`): si McNamara y Taylor no recomiendan acciones distintas
  en la mayoría de los estados, no se lanza el experimento.

**Diseño tentativo de las partidas.** Bifurcación desde `usa_sabado_negro`, con el mismo modelo
fijado para los líderes y en modo anonimizado. Celdas:
- **líder solo**: la `base` del experimento 1, si el código no cambia en nada más;
- **asesores solo con etiqueta**: separa el efecto del nombre;
- **asesores con informe privado y acción ancla**;
- opcionalmente, **asesores con otro modelo** (4B u otro de 7–9B, cargados uno tras otro en Ollama).

**Cómo medir la "variedad".**
- La entropía de los desenlaces.
- La distancia de edición entre las secuencias de acciones.
- La dispersión de la duración y de la tensión máxima.
- P(acuerdo) y P(escalada) como control.
- La **influencia**: la tasa a la que el líder sigue a la mayoría de los consejos y la tasa a la que adopta la recomendación minoritaria.
- El porcentaje de acciones rechazadas por el nodo `validate`, para no confundir variedad con incoherencia.

**Coste** (simulacro del 14‑oct, calibrado con el ritmo real del experimento 1, ~43 s por turno
con 2 decisiones):

| Configuración | Llamadas al LLM por turno | Por partida | 30 partidas |
|---|---|---|---|
| Sin asesores (hoy) | 2 | ~6 min | ~3 h |
| 5 asesores en todos los turnos (9B) | 7 | ~15 min | ~7,5 h |
| 5 asesores solo en los turnos clave (9B) | ~3,7 de media | ~9 min | ~4,5 h |
| Ídem con los asesores en el modelo 4B | ~3,7 | ~8 min | ~4 h |

**Riesgos.**
- El eco: los asesores repiten al líder o entre sí. Es lo que mide la fase 0.
- El sesgo de escalada de los LLM: todos los estudios lo encuentran. El prompt de cada asesor actúa
  como una intervención sobre la escalada, así que la mecánica del motor no debe cambiar entre celdas.
- Que la «variedad» venga de ruido y no de deliberación. Hay que leer una muestra de partidas a mano.

---

## 3‑bis. Hallazgos de la fase 0 (histórico)

Son resultados reales del esqueleto, con agentes scripted y la secuencia histórica forzada.

| Experimento | Resultado (n = 1000) |
|---|---|
| Secuencia histórica forzada | 84 % acuerdo · **16 % uso nuclear** |
| halcón × halcón | 59 % conflicto convencional · 41 % uso nuclear |
| pragmática × defensiva | 99 % acuerdo, normalmente **en 2–4 turnos** |

### 3.1 Un solo supuesto domina el resultado
Con la sensibilidad OAT (±20 %) sobre la secuencia histórica, `incidents.1.nuclear_risk` (el
submarino acosado) mueve el uso nuclear ±3 puntos y el resto de parámetros ≤1 punto. El "16 %"
es, casi literalmente, ese parámetro multiplicado por la probabilidad de que el incidente ocurra.
**Así funciona el motor; no es un hallazgo sobre 1962.**

### 3.2 La tensión se satura
Con las acciones históricas, la tensión llega a 100/100 el 26 de octubre (antes si hay incidentes)
y se queda ahí hasta el acuerdo.
Por encima del techo, cualquier acción "más tensa" no cambia nada, y eso explica por qué la
sensibilidad del resto de parámetros sale casi nula. **Hay que rediseñar la escala** (incrementos
decrecientes cerca del techo, o tensión en logit) antes de cualquier experimento.

### 3.3 Los acuerdos son demasiado fáciles
Los agentes cooperativos cierran un acuerdo en 1–2 días; la crisis real duró 13. Faltan:
- coste político de ceder pronto;
- condiciones previas (no se puede negociar lo que el otro no sabe que sabes);
- **fase de cumplimiento y verificación** (reconocimiento aéreo, inspección);
- posibilidad de romper un compromiso.

Hoy aceptar un acuerdo termina la partida al instante.

### 3.4 Los agentes scripted no sirven para medir estabilidad
Como predijo la revisión, con agentes deterministas y cooperativos la sensibilidad sale
trivialmente estable. La sensibilidad que importa es la que se mide con **la secuencia
histórica** y, después, **con agentes LLM**.

---

## 4. Protocolo de evaluación

### 4.1 Prerregistro
`checkpoints.toml`, las variantes trampa y la codificación de decisiones se verifican contra
las fuentes y **se congelan con una etiqueta de git antes de ejecutar ningún LLM**. Cambiarlos
después de ver resultados invalida la evaluación.

### 4.2 Fidelidad histórica (objetivo secundario)
- **Forzado histórico**: en cada punto de control se reconstruye el estado reproduciendo la
  secuencia histórica (sin incidentes aleatorios) y se pide la decisión al agente. No se espera a
  que una partida libre "llegue" allí, porque diverge.
- Por punto, **K ≥ 30 muestras** a temperatura fija. Métrica: frecuencia de la opción histórica
  **dentro de las opciones que se consideraron de verdad** (las grabaciones del ExComm las
  documentan), con IC 95 % de Wilson.
- Referencias obligatorias: azar uniforme sobre las opciones, agente scripted y LLM sin persona.
- Con 4–12 puntos no hay potencia para afirmar "reproduce a Kennedy". Se reporta por punto y nunca
  se agrega en una sola cifra.

### 4.3 Fuga (memoria del desenlace)
Tres niveles, de más débil a más fuerte:
1. **Reconocimiento**: ¿identifica el episodio a partir del briefing? Se mide en modo real y en
   modo anonimizado (`cavalier probe`).
2. **Predicción del siguiente evento** con contexto parcial (pendiente).
3. **Escenarios trampa**: variantes donde la acción histórica deja de tener sentido (`sin_jupiter`:
   no hay misiles en el Estado fronterizo). Si el agente los pide o los menciona igualmente, está
   recordando, no razonando.

4. **Modelos con corte de conocimiento histórico** (control "por construcción"). Son modelos
   entrenados desde cero solo con textos anteriores a una fecha, así que el desenlace no está en
   sus pesos. Proyectos conocidos (verificar detalles en sus fuentes primarias):
   - Ranke-4B / *history-llms*: ~4B parámetros, varios cortes; el más reciente que he encontrado es 1946.
   - TypewriterLM: 7B, corte en 1913.
   - TimeCapsuleLLM: textos de Londres entre 1800 y 1875.

   Para Cuba 1962 no hay ninguno: haría falta un corte en octubre de 1962, y la mayor parte del
   texto posterior a 1929 tiene derechos de autor. **Entrenarlo es inviable en una RTX 2070**
   (preentrenar un 4B con decenas de miles de millones de tokens requiere un clúster). Usos realistas:
   - **Berlín 1948 con un modelo con corte en 1946**: el escenario queda libre de fuga por
     construcción. Es un buen argumento para que Berlín sea el segundo escenario, antes que Able Archer.
   - **Comparar un modelo con corte antiguo y uno moderno de tamaño parecido en el mismo escenario
     inventado**. Aísla el efecto "sabe cómo acaban las crisis de la Guerra Fría", aunque se mezcla
     con la diferencia de capacidad: hay que reportarlo así.
   - Riesgos: el ajuste de instrucciones puede reintroducir conocimiento moderno, hay fechas mal
     catalogadas y reediciones con prólogos posteriores. Un 4B histórico razona peor que uno
     moderno, así que solo vale como control, no como agente principal.

La fidelidad solo se publica junto a estos indicadores. Además, la diferencia entre modo real
y anonimizado mezcla dos efectos: recuerdo del desenlace y pérdida de conocimiento doctrinal
legítimo. Por eso no se usa sola.

### 4.4 Monte Carlo: presupuesto
- Cada partida completa tiene unas 26 rondas × 2 agentes = **52 decisiones**. El grafo hace
  **1 llamada al LLM por decisión** (más los reintentos), así que salen ~55–70 llamadas por partida.
- Modelo local de 8B, ~1.500 tokens de entrada y ~300 de salida: ~5–8 s por llamada en secuencia,
  es decir, **~6–8 min por partida**, y 1.000 partidas son unos 5 días. Con vLLM o llama.cpp
  sirviendo en paralelo y con caché de prefijo (el prompt de sistema es fijo), estimo un orden de
  magnitud menos. **Hay que medirlo en F2.**
- Eventos raros: estimar un 2 % con ±1 punto exige unas 750 partidas (`runs_for_margin`). Se fija
  n por el intervalo deseado, no por costumbre.
- Para eventos raros conviene **bifurcar desde estados críticos** (muestreo por división): se
  juega hasta un estado de tensión alta, se guarda y se ramifica K veces.
- Las partidas con decisiones de respaldo se reportan aparte (`BatchResult.clean()`).

#### Hardware disponible: RTX 2070 (8 GB de VRAM)
- **Cabe**: un modelo de 7–8B en Q4_K_M (~5 GB de pesos) con contexto de 4–8k, o uno de 3–4B con
  margen para 3–4 peticiones en paralelo. vLLM queda justo en 8 GB; se usa **llama.cpp server**
  (`--parallel N`, caché de prompt) u Ollama.
- **Estimación** (hay que medirla en F2): 8B Q4 a ~35–45 tok/s de generación. Con ~300 tokens de
  salida salen ~8 s por llamada, ~8 min por partida, y **1.000 partidas son casi una semana**. No es viable.
- **Consecuencias para el diseño**:
  1. **Limitar la salida**: razonamiento de ≤120 tokens en el esquema y desactivar el modo
     "thinking" en los modelos que lo tengan. La salida es lo que cuesta en esta GPU; el prompt se
     procesa rápido.
  2. **Dos modelos**: un 8B para el desarrollo, la sonda de fuga y los puntos de control (pocas
     llamadas), y un 3–4B para el Monte Carlo masivo, **siempre** que en los puntos de control
     reproduzca la distribución de decisiones del 8B. Si no la reproduce, los experimentos se hacen
     con el 8B y menos partidas.
  3. **Experimentos pequeños y bien diseñados**: pocas celdas (perfil × variante × control), unas
     200–400 partidas por celda en lotes nocturnos, y bifurcación desde estados críticos para los
     eventos raros en lugar de partidas completas.
  4. La API remota queda como opción puntual para validar que las conclusiones no dependen del
     modelo local, no como motor del Monte Carlo.

### 4.5 Sensibilidad del motor
- **Invariantes cualitativos** (en tests): halcón×halcón escala más que paloma×paloma; menos
  control produce más incidentes; sin armas tácticas, menor proporción nuclear entre las guerras.
- **Secuencia histórica forzada**: P(acuerdo) entre 50 % y 95 %. Es la única comprobación posible
  con n = 1.
- **Sensibilidad global** (cribado de Morris, F1) para saber qué parámetros dominan. Ningún
  resultado se publica si depende de un único parámetro sin justificar.
- **Tabla de rangos**: cada parámetro de `rules.toml` lleva su rango y su justificación
  (literatura o criterio del autor declarado).

---

## 5. Hoja de ruta con criterios de salida

| Fase | Contenido | Criterio de salida (medible) |
|---|---|---|
| **0** ✅ | Esqueleto, agentes scripted, CLI, tests | Partida reproducible con semilla; 26 tests en verde |
| **1** ✅ | Motor v2: tensión sin saturación, cumplimiento y verificación de acuerdos, resolución de choques simultáneos (bloqueo frente a buques), coste de ceder pronto. Cribado de Morris. Tabla de rangos. Verificar y **congelar** `historical.toml` y `checkpoints.toml` | Invariantes en verde · secuencia histórica con P(acuerdo) ∈ [50 %, 95 %] y duración media ≥ 8 turnos · parámetros dominantes (Morris) identificados, con rango justificado, y resultados que dependan de ellos presentados como curva sobre su rango · etiqueta de prerregistro `prereg-v1` creada |
| **2** ✅ | Agente LLM sin RAG con modelo local **fijado** (nombre, cuantización y hash). Visor mínimo de partidas (JSONL → HTML). Sonda de fuga niveles 1 y 3 | JSON válido al primer intento ≥ 98 % · decisiones de respaldo ≤ 1 % · coste medido (s/decisión, tokens/partida) · fuga reportada en real, anonimizado y trampa |
| **3** ✅ | Fidelidad en los puntos de control (K = 30) frente a las referencias | Informe por punto con IC. Decisión justificada sobre si la persona histórica aporta algo frente a un LLM sin persona |
| **4** | Corpus mínimo: ~30 documentos **ligados a los puntos de control**, pre‑anotados por LLM y validados por una persona. RAG temporal | Acuerdo entre la anotación del LLM y la humana ≥ 90 % en `available_to` y `known_from` · el RAG no aumenta la fuga · cambio en fidelidad reportado con IC |
| **5** | **Experimentos contrafactuales** (objetivo primario): perfiles × variantes × disciplina de mando; potencias inventadas; bifurcación desde estados críticos | Resultados con IC y análisis de sensibilidad con agentes LLM, condicionados a `rules.toml` versionado |
| **6** | Interfaz espectador estilo DEFCON (FastAPI + MapLibre) y "cambiar la historia" en la interfaz. Voces sintéticas de los líderes (adelantado como experimento: `cavalier voices`, ver 8.4) | Reproducir una partida registrada y bifurcarla desde la interfaz |
| **7** | (Condicional) Destilación o LoRA si F5 está limitado por coste | El modelo pequeño reproduce la distribución de decisiones del grande en los puntos de control (divergencia por debajo de un umbral fijado antes) |

---

## 6. Riesgos

| Riesgo | Mitigación |
|---|---|
| Resultados dominados por supuestos del motor (ya observado, 3.1) | Sensibilidad global, tabla de rangos y resultados siempre condicionados a `rules.toml` |
| Fuga del desenlace | Protocolo 4.3, variantes trampa y potencias inventadas |
| Salida estructurada poco fiable con modelos pequeños | Decodificación con JSON Schema, reintento con el error, respaldo marcado y reportado aparte |
| Coste del Monte Carlo | Presupuesto medido en F2, servidor en paralelo, bifurcación desde estados críticos y destilación (F7) |
| Esfuerzo de anotar el corpus | Partir de cronologías existentes (NSA), corpus pequeño ligado a los puntos de control, pre‑anotación por LLM con validación muestral |
| Sesgo documental entre EE. UU. y la URSS | Anotar idioma, traducción y procedencia; reportar la cobertura por bando |
| Crecimiento del alcance (épocas, interfaz, fine‑tuning) | Fases con criterio de salida; nada de F6 o F7 hasta cerrar F5 |
| No determinismo (temperatura 0 con batching en llama.cpp) | Medir la varianza por semilla y la sensibilidad a paráfrasis del prompt por separado |

---

## 7. Trabajo relacionado (a revisar antes de F3)

- Rivera et al. (2024), *Escalation Risks from Language Models in Military and Diplomatic
  Decision-Making*: agentes LLM en simulaciones de conflicto y su tendencia a escalar.
- Lamparth et al. (2024), *Human vs. Machine: Language Models and Wargames*: comparación de LLMs
  con expertos humanos en un wargame.
- Bibliografía histórica: Allison & Zelikow, *Essence of Decision* (modelos de decisión aplicados
  precisamente a Cuba 1962). Es un marco natural para definir las personas.

---

## 8. Decisiones tomadas

1. **Objetivo primario**: experimentos contrafactuales. La fidelidad histórica queda como validación.
2. **Hardware**: RTX 2070 (8 GB). Un 8B Q4 para desarrollo y evaluación, y un 3–4B para el Monte
   Carlo si supera la comparación en los puntos de control (ver 4.4).
3. **Versionado**: repositorio git creado. El prerregistro será una etiqueta (`prereg-v1`) cuando
   `historical.toml` y `checkpoints.toml` estén verificados contra las fuentes (F1).
4. **Voces sintéticas** (14‑oct‑2026, experimental, adelanto de F6). No existe un TTS listo con
   las voces de Kennedy y Jrushchov (el proyecto «JFK Unsilenced», de CereProc en 2018, fue un
   encargo cerrado, y los servicios comerciales bloquean la clonación de políticos). Se clonan en
   local con **XTTS‑v2** (Coqui) a partir de clips de archivo: Kennedy, de la JFK Library (dominio
   público); Jrushchov, de noticiarios en ruso, ya que XTTS clona el timbre en otro idioma. Los
   agentes escriben en español y las voces hablan español.
   - Es solo presentación: no toca el motor, los agentes ni los experimentos.
   - Va en un entorno aparte (`.venv-tts`, PyTorch con CUDA) y solo entre tandas, porque XTTS
     (2–3 GB) no cabe en la GPU junto al modelo de 9B.
   - Siempre se marca como **sintético** (manifiesto y visor). No se publican clips sueltos. La
     licencia de XTTS‑v2 (CPML) es solo para uso no comercial.
   - Instrucciones: `data/voices/README.md`.
