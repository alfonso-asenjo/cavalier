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

| Emparejamiento | Acuerdo | Desescalada | Sin resolver | Convencional | Nuclear | Turnos |
|---|---|---|---|---|---|---|
| Histórico (forzado) | 83,2 % | 0 % | 1,2 % | 0 % | 15,6 % | 28,4 |
| halcón × halcón | 0 % | 0 % | 0 % | 59,0 % | 41,0 % | 7,3 |
| paloma × paloma | 19,6 % | 76,6 % | 3,6 % | 0 % | 0,1 % | 14,7 |
| pragmática × pragmática | 59,2 % | 8,6 % | 6,5 % | 10,7 % | 15,0 % | 17,8 |
| pragmática × defensiva | 58,5 % | 4,9 % | 5,0 % | 12,8 % | 18,9 % | 16,9 |
| halcón × paloma | 14,8 % | 1,6 % | 3,1 % | 48,6 % | 32,0 % | 12,7 |
| impredecible × pragmática | 48,2 % | 1,8 % | 1,3 % | 26,6 % | 22,1 % | 12,8 |
| prag × def, sin armas tácticas | 58,5 % | 4,9 % | 5,0 % | 17,3 % | 14,4 % | 16,9 |
| prag × def, comunicaciones lentas | 55,4 % | 5,0 % | 4,5 % | 15,0 % | 20,2 % | 17,2 |

Las cifras son de agentes scripted: validan el motor, no dicen nada sobre 1962.

### 3.2 Sensibilidad global (Morris, 10 trayectorias × 300 partidas por punto)

| Métrica | Secuencia histórica | pragmática × defensiva |
|---|---|---|
| Uso nuclear | `incidents.1.nuclear_risk` (submarino) **49,3 %** · Maultsby 10,5 % · resto < 9 % | submarino 21,0 % · multiplicador de incidentes 13,4 % · `tension_decay` 11,6 % |
| Guerra (convencional + nuclear) | igual que nuclear (las acciones fijas no incluyen ataques) | **`tension_decay` 20,5 %** · `implementation_turns` 16,9 % · incidentes 14,2 % |
| Acuerdo | submarino 36,2 % · `verification_rate` 13,7 % | `implementation_turns` 23,3 % · incidentes 14,6 % |

Lectura:
- **El riesgo nuclear del submarino es el supuesto crítico del modelo.** En la secuencia histórica, recorrer su rango plausible (2 %–30 %) mueve el uso nuclear en ~24 puntos. Coincide con la historiografía (el B‑59 se considera el momento más peligroso), pero significa que **ningún porcentaje de uso nuclear se puede dar como un número**. Se presentará siempre como una **curva en función de ese parámetro**.
- **`tension_decay` domina la probabilidad de guerra con agentes que deciden, y es un supuesto mío sin base empírica.** Es el siguiente candidato a justificar o a presentar como condicionante.
- Con agentes que deciden, la influencia se reparte (ningún parámetro pasa del 23 %): los resultados contrafactuales son más robustos que la repetición histórica.

### 3.3 Criterios de salida de la fase 1
| Criterio | Estado |
|---|---|
| Invariantes cualitativos en verde | ✅ 38 tests en verde |
| Secuencia histórica con P(acuerdo) ∈ [50 %, 95 %] | ✅ 83,2 % |
| Duración media ≥ 8 turnos | ✅ histórico 28,4; cooperativos 15–18 (antes 2–4) |
| Parámetros dominantes identificados y tratados (criterio revisado; el original pedía "ningún parámetro > 50 %") | ✅ Submarino (49,3 % en la secuencia histórica, rango con base en fuentes): todo resultado de uso nuclear se presenta como curva sobre `incidents.1.nuclear_risk`. `tension_decay` (20,5 % de la guerra con agentes, supuesto del autor): pendiente de justificar; mientras tanto, condicionante declarado |
| Secuencia histórica y puntos de control verificados y congelados | ✅ verificados con fuentes y congelados con la etiqueta `prereg-v1` |

### 3.4 Pendiente, conocido
- Clasificación del punto de control del 27 de octubre: no distingue una propuesta con término secreto de una pública.
- La granularidad de 12 h aplana las latencias reales (horas). Si la latencia resulta importante en los experimentos, habrá que bajar a turnos de 6 h.
- Solo hay dos interacciones simultáneas modeladas (choque naval y espiral de movilización).

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
| **2** | Agente LLM sin RAG con modelo local **fijado** (nombre, cuantización y hash). Visor mínimo de partidas (JSONL → HTML). Sonda de fuga niveles 1 y 3 | JSON válido al primer intento ≥ 98 % · decisiones de respaldo ≤ 1 % · coste medido (s/decisión, tokens/partida) · fuga reportada en real, anonimizado y trampa |
| **3** | Fidelidad en los puntos de control (K = 30) frente a las referencias | Informe por punto con IC. Decisión justificada sobre si la persona histórica aporta algo frente a un LLM sin persona |
| **4** | Corpus mínimo: ~30 documentos **ligados a los puntos de control**, pre‑anotados por LLM y validados por una persona. RAG temporal | Acuerdo entre la anotación del LLM y la humana ≥ 90 % en `available_to` y `known_from` · el RAG no aumenta la fuga · cambio en fidelidad reportado con IC |
| **5** | **Experimentos contrafactuales** (objetivo primario): perfiles × variantes × disciplina de mando; potencias inventadas; bifurcación desde estados críticos | Resultados con IC y análisis de sensibilidad con agentes LLM, condicionados a `rules.toml` versionado |
| **6** | Interfaz espectador estilo DEFCON (FastAPI + MapLibre) y "cambiar la historia" en la interfaz | Reproducir una partida registrada y bifurcarla desde la interfaz |
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
