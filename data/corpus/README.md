# Corpus documental

Un fichero `.jsonl` por colección, un documento por línea (esquema: `cavalier.rag.corpus.Document`).

Campos clave:
- `date`: cuándo se produjo el documento.
- `available_to`: bandos que lo conocían **en su momento** (un memorando de la CIA: solo `USA`).
- `known_from`: si un bando lo conoció más tarde, desde cuándo.
- `kind`: `contemporaneo` | `retrospectivo`. Los retrospectivos (memorias, entrevistas,
  historiografía) **no** entran en el RAG de los agentes; solo sirven para evaluar.
- `language`, `translated`: para medir el sesgo documental entre bandos.
- `annotated_by` (`human` | `llm`) y `validated`: la pre‑anotación por LLM se valida por muestreo.

Prioridad (plan, F4): ~30 documentos ligados a los puntos de control de `checkpoints.toml`.

Fuentes: FRUS 1961‑1963 vol. XI · Wilson Center Digital Archive · National Security Archive (GWU) ·
UMD Cuban Missile Crisis collection · Miller Center (grabaciones del ExComm) · National Archives.
