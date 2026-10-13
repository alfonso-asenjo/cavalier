Era broma, pero hacer un simulador de ataque nuclear en python con los datos públicos que hay, arsenal disponible por países, diferentes tipos de misiles, contramedidas, objetivos críticos, puede estar interesante y simular 2 bandos con ia, pues oye, puede estar bien

Ha pensado durante 38s
Pues esto sí que me parece un proyecto interesante. 😄 Y, además, Python + simulación geopolítica + dos agentes de IA enfrentados + datos reales tiene bastante potencial.

Yo lo plantearía como una mezcla entre DEFCON y un simulador de teoría de juegos, donde dos potencias nucleares toman decisiones autónomas, reaccionan a información incompleta y pueden terminar negociando, escalando un conflicto o provocando una catástrofe.

Lo interesante sería que no tuviéramos un resultado preprogramado. Que cada simulación pudiera terminar de una forma completamente diferente.

1. Tenemos datos públicos para construirlo
Por ejemplo, el SIPRI publicó en junio de 2026 su estimación de los arsenales nucleares mundiales. 

SIPRI
+1

Estimación de ojivas nucleares totales, enero de 2026

0
1.500
3.000
4.500
6.000
Rusia
EE. UU.
China
Francia
Reino Unido
India
Pakistán
Israel
Corea del Norte
Rusia: Ojivas estimadas 5420
EE. UU.: Ojivas estimadas 5042
China: Ojivas estimadas 620
Francia: Ojivas estimadas 370
Reino Unido: Ojivas estimadas 225
India: Ojivas estimadas 190
Pakistán: Ojivas estimadas 170
Israel: Ojivas estimadas 90
Corea del Norte: Ojivas estimadas 60
Fuente: SIPRI 2026. Incluye ojivas retiradas pendientes de desmantelamiento.

Importante: el inventario total no equivale al número de armas operativas. De las aproximadamente 12.187 ojivas mundiales, SIPRI estima que 9.745 pertenecían a arsenales militares disponibles para un eventual uso.
Podríamos incluir diferentes categorías de sistemas: ICBM, SLBM lanzados desde submarinos, bombarderos estratégicos, misiles de crucero y sistemas hipersónicos, pero modelados con atributos simplificados en lugar de reproducir prestaciones operativas de armas reales.

2. Cómo sería el simulador
IA - Bando A

Objetivos políticos, recursos, inteligencia

IA - Bando B

Objetivos políticos, recursos, inteligencia

Motor de simulación Python

Turnos · Eventos · Incertidumbre · Escalada · Defensa abstracta

Estado del mundo

Tensión diplomática · Riesgo humanitario · Recursos · Estabilidad · Resultado

Cada IA podría recibir información distinta y decidir entre opciones como negociar, movilizar fuerzas, incrementar la alerta, mejorar defensas, solicitar mediación o reducir la tensión.

Las defensas y los intercambios nucleares se resolverían mediante modelos probabilísticos ficticios, sin calcular trayectorias reales, seleccionar instalaciones críticas ni optimizar ataques.

3. La parte más divertida: las IA y sus personalidades
Podríamos utilizar modelos locales mediante Ollama o llama.cpp, o APIs externas con LangGraph para orquestar los turnos.

Cada bando tendría una doctrina distinta:

Personalidad

Comportamiento

Halcón

Alta disposición a escalar

Defensiva

Prioriza la supervivencia y la disuasión

Diplomática

Busca acuerdos y concesiones

Impredecible

Puede interpretar mal las señales

Pragmática

Intenta maximizar objetivos políticos y minimizar pérdidas

Y lo mejor: la información de cada IA no tiene por qué ser correcta. Una alerta falsa podría desencadenar una crisis, o una amenaza mal interpretada podría provocar una escalada no deseada.

Eso permitiría estudiar emergencias, errores de inteligencia y fallos en la comunicación entre potencias.

4. La interfaz que me imagino
Xsolla Mall - DEFCON

"C:\Users\alfonso\Downloads\Q0GsDJmsEPCtWhQUBjFYpwOkxzGFq9KeEKgzID5_5vgHh8juzXsyFJxSsJdgMB3rJAOd4YKXMOBGOUM4RYoy8pRlEswwLAQZlD18yFyehlHe5iEjz_OlbQX-LehIk-MQuDH0MPzoRCSqgpocwbdh66-_Tbc75p7nsv6cDPThsqKo63SAPcOdTbKcW4ffKmYX.jpg"
DEFCON como referencia estética: mapa mundial oscuro, estados de alerta y seguimiento visual de la crisis.
Utilizaría estas tecnologías:

Componente

Tecnología

Motor de simulación

Python + NumPy

Agentes IA

LangGraph + LLM

Datos geográficos

GeoJSON + Natural Earth

Interfaz

FastAPI + React/Vite

Mapa interactivo

MapLibre GL

Persistencia de partidas

SQLite

Análisis de escenarios

Pandas + Monte Carlo

Para los objetivos usaría regiones ficticias con variables agregadas de población, economía e infraestructura, en lugar de coordenadas de objetivos estratégicos reales. Las consecuencias humanitarias serían una parte central del modelo.

5. El MVP que construiría
Empezaría con Estados Unidos y Rusia como perfiles de recursos basados en estadísticas públicas, pero en un escenario ficticio y con mecanismos militares abstractos.

Un primer prototipo tendría dos agentes, cinco niveles de alerta, decisiones por turnos, defensas probabilísticas, negociación y un registro de decisiones de cada IA.

Después añadiría un modo espectador, donde puedes ver cómo razonan los agentes, qué información reciben, cuándo se equivocan y cómo evoluciona la crisis.

Lo realmente potente sería ejecutar, por ejemplo, 1.000 simulaciones con las mismas condiciones iniciales y comparar cuántas terminan en acuerdo, conflicto convencional, escalada nuclear o desescalada.

Hay una decisión de diseño que me parece especialmente interesante: ¿queremos que las dos IA representen países reales con doctrinas históricas y estadísticas públicas, o que podamos inventar potencias y modificar sus políticas, recursos y personalidades?

Yo permitiría ambas cosas, diferenciando claramente los datos históricos de las reglas ficticias del simulador.

Sobre la decisión de diseño, tenemos la guerra fría que fue el periodo más tenso y con más probabilidad de aniquilación entre humanos, hasta que llegaron las IAs, 
Adicional, hay suficientes datos para hacer finetunning de modelos sencillos y simular diferentes perfiles de juego, de momento EEUU y cccp

Ha pensado durante 41s
EE. UU. contra la URSS durante la Guerra Fría es un punto de partida excelente, especialmente si queremos entrenar modelos pequeños para que representen diferentes doctrinas políticas y estilos de toma de decisiones.

Y tenemos una ventaja enorme: no necesitamos inventarnos las personalidades. Disponemos de memorandos, conversaciones diplomáticas, informes de inteligencia, correspondencia entre líderes y decisiones históricas documentadas.

Pero haría una distinción importante: no entrenaría dos IA para ganar una guerra nuclear. Las entrenaría para representar cómo razonaban los distintos gobiernos durante una crisis. Después, el motor de simulación resolvería las consecuencias de sus decisiones.

1. Tenemos documentación suficiente para intentarlo
He localizado cuatro fuentes especialmente interesantes.

Fuente

Qué contiene

Utilidad

FRUS – Departamento de Estado

Decisiones diplomáticas, memorandos y comunicaciones oficiales estadounidenses

Entrenamiento de perfiles de EE. UU.

Wilson Center Digital Archive

Documentos soviéticos, estadounidenses y de otros países

Perfiles soviéticos y contraste histórico

Archivo de la crisis de Cuba

Más de 700 documentos de más de 20 países

Reconstrucción de crisis y evaluaciones

National Archives

Correspondencia Kennedy-Jruschov, inteligencia, cronologías

Validación histórica

La colección de Maryland ofrece un punto de partida muy concreto: más de 700 fuentes primarias sobre octubre de 1962. 

digitalarchive.umd.edu
+2

También hay fondos sobre el Pacto de Varsovia, la crisis de los euromisiles y la proliferación nuclear. 

Wilson Center
+1

Eso sí: disponer de muchos documentos no equivale a disponer automáticamente de buenos ejemplos para fine-tuning. Hay que extraer decisiones, contexto y resultados, y separar lo que realmente se sabía entonces de lo que descubrimos décadas después.

2. No existirían solamente dos IA
Yo tendría dos bandos, pero múltiples perfiles de liderazgo seleccionables según el periodo.

John F. Kennedy at 100: Personal Remembrance of a President
John F. Kennedy — EE. UU. (1961–1963)

Crisis de Cuba, presión de asesores, gestión de alianzas y búsqueda de salidas diplomáticas sin perder credibilidad.

Abstract art that Khrushchev condemned to get home in Moscow as well as US and China shows - The Art Newspaper - International art news and events
Nikita Jrushchov — URSS (1953–1964)

Competencia estratégica, percepción de vulnerabilidad soviética y negociación desde posiciones de presión.

File:Official Portrait of President Reagan 1981 (cropped)(c).jpg - Wikimedia Commons
Ronald Reagan — EE. UU. (1981–1989)

Confrontación ideológica, presión económica, rearme y evolución hacia acuerdos de control de armamentos.

File:Mikhail Gorbachev 1987.jpg - Wikimedia Commons
Mijaíl Gorbachov — URSS (1985–1991)

Reformas internas, reducción de la confrontación y negociación del desarme.

Estos serían perfiles históricos aproximados y auditables, no copias exactas de personas reales.

Una IA entrenada con documentos de Kennedy podría comportarse de manera diferente a otra entrenada con documentación de Reagan, aunque ambas representasen a Estados Unidos.

3. Cómo haría el fine-tuning
No entrenaría un modelo desde cero. Utilizaría un modelo pequeño instruct, con LoRA o QLoRA, y combinaría el ajuste fino con RAG.

Documentos históricos desclasificados

RAG histórico

Conocimiento verificable por fecha y país

Fine-tuning LoRA

Estilo de razonamiento y política de decisiones

Agente de decisión histórica

Contexto limitado · Decisión estructurada · Explicación · Confianza

El conjunto de entrenamiento tendría ejemplos estructurados como este:

{
  "periodo": "1962",
  "bando": "USA",
  "contexto": {
    "crisis_diplomatica": "alta",
    "presion_interna": "alta",
    "informacion": "incompleta"
  },
  "decision": "abrir_canal_diplomatico",
  "justificacion": "Evitar una escalada irreversible",
  "resultado_historico": null
}

Este es un ejemplo sintético, no un documento histórico real.

La clave es que el agente no conozca de antemano cómo acabó la crisis. Además, los documentos posteriores a cada decisión deben excluirse de su contexto.

¿Cuántos documentos necesitaríamos?
Para experimentar:

100–300 ejemplos curados por perfil: suficientes para probar un primer LoRA, aunque no para garantizar una personalidad histórica convincente.

500–2.000 ejemplos de calidad: un objetivo más interesante para experimentar con decisiones históricas consistentes.

Miles de documentos adicionales: para recuperación documental mediante RAG, sin necesidad de convertirlos todos en ejemplos de entrenamiento.

No son umbrales universales: habría que evaluar si LoRA supera realmente a un buen prompt con RAG.

También vigilaría un problema: el sesgo documental. La documentación soviética disponible no tiene necesariamente la misma cobertura que la estadounidense; además, muchos documentos son traducciones o reflejan interpretaciones posteriores.

4. Tres épocas jugables
Considering the Utility of Modern Blockade in a Protracted Conflict With China > National Defense University Press > News Article View
1948–1949 · Bloqueo de Berlín

Ideal para probar presión política, logística, alianzas y negociación, en una fase en la que la URSS todavía no disponía de armas nucleares al comienzo de la crisis.

One Step from Nuclear War | National Archives
1962 · Crisis de los misiles de Cuba

Mi elección para el MVP. Tenemos decisiones documentadas día a día, comunicaciones entre líderes y una resolución histórica que permite contrastar el simulador.

Świat na krawędzi zagłady. Manewry Able Archer 83: początek trzeciej wojny światowej
1983 · Crisis de Able Archer

Perfecta para experimentar con falsas alarmas, percepciones erróneas, incertidumbre y el riesgo de interpretar un ejercicio como una amenaza real.

5. Cómo comprobaríamos si las IA son buenas
Aquí estaría el componente científico del proyecto.

En lugar de medir si una IA derrota a otra, mediríamos:

Métrica

Qué queremos evaluar

Fidelidad histórica

Coherencia con la doctrina del periodo

Consistencia

Decisiones similares en circunstancias similares

Sensibilidad al contexto

Reacción ante información nueva

Calidad diplomática

Capacidad de negociar acuerdos

Escalada

Frecuencia de crisis graves

Robustez

Efecto de los errores de inteligencia

Podríamos reproducir situaciones históricas hasta un determinado día, ocultar el desenlace y comprobar qué decisiones toman nuestros modelos. Después ejecutar miles de escenarios contrafactuales mediante Monte Carlo.

Eso no demostraría qué habría pasado realmente, pero sí permitiría estudiar cómo distintas reglas de decisión afectan a los resultados del simulador.

6. Mi propuesta de primera versión
Proyecto: COLD WAR AI — 1962

Comenzaría con dos agentes representando a Kennedy y Jrushchov, un único motor de simulación y un conjunto de documentos históricos versionados.

El primer desarrollo sería:

Ingesta documental, extracción de decisiones y metadatos temporales.

RAG separado para EE. UU. y la URSS.

Dos agentes sobre el mismo modelo base, inicialmente con prompts diferentes.

Simulador por turnos con decisiones diplomáticas, económicas y estados de alerta abstractos.

Registro de cada decisión, argumentos y documentos utilizados.

Comparativa entre agentes sin ajustar y agentes con LoRA.

Así podríamos comprobar si el fine-tuning aporta algo medible antes de gastar recursos entrenando múltiples modelos.

Con tu experiencia en LangGraph, Python y modelos locales, lo más complejo no sería montar la infraestructura: sería preparar un corpus histórico bien etiquetado y diseñar una evaluación que no premie simplemente repetir lo que ocurrió.

Y añadiría algo muy interesante: un botón de «Cambiar la historia». En cualquier turno podrías intervenir, cambiar una decisión diplomática y observar cómo reaccionan ambas IA desde ese momento.

Mi elección sería arrancar con la crisis de Cuba de octubre de 1962, porque contamos con documentación especialmente rica de ambos lados y múltiples perspectivas internacionales.