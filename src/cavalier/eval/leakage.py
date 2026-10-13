"""Sonda de fuga: ¿reconoce el modelo el episodio histórico detrás del escenario?

Tres niveles (de más débil a más fuerte):
  1. Reconocimiento: se le muestra el briefing y se le pregunta qué episodio es.
  2. Predicción del siguiente evento: con contexto parcial, ¿anticipa lo que ocurrió?
     (pendiente: necesita la cronología del escenario)
  3. Escenarios trampa (variants/): ¿elige la acción histórica cuando ya no tiene sentido?
     (se mide con eval.checkpoints sobre la variante)
"""

from __future__ import annotations

import json
import re

from ..agents.llm_client import LLMClient
from ..anonymize import Anonymizer
from ..scenario import ScenarioBundle

PROBE_SYSTEM = "Eres un historiador experto. Responde solo con JSON."
PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "episode": {"type": "string"},
        "year": {"type": "string"},
        "confidence": {"type": "number"},
        "explanation": {"type": "string"},
    },
    "required": ["episode", "year", "confidence"],
}

# Palabras que indican reconocimiento del episodio real. Específicas de cuba1962;
# otros escenarios deben definir las suyas.
RECOGNITION_MARKERS = {
    "cuba1962": [r"cuba", r"1962", r"misiles de octubre", r"missile crisis", r"crisis de los misiles", r"kennedy",
                 r"jrushch", r"khrushch", r"caribe"],
}


def probe_recognition(bundle: ScenarioBundle, client: LLMClient, side: str, anonymized: bool) -> dict:
    anon = Anonymizer(bundle.aliases, enabled=anonymized)
    briefing = anon(bundle.scenario.sides[side].briefing)
    user = (
        "El siguiente texto describe una situación. Si se corresponde con un episodio histórico real, "
        "indica cuál y en qué año. Si no lo reconoces, responde 'desconocido'.\n\n" + briefing
    )
    raw = client.complete_json(PROBE_SYSTEM, user, PROBE_SCHEMA)
    try:
        answer = json.loads(raw)
    except json.JSONDecodeError:
        answer = {"episode": raw, "year": "", "confidence": 0.0}
    text = f"{answer.get('episode', '')} {answer.get('year', '')}".lower()
    markers = RECOGNITION_MARKERS.get(bundle.scenario.id, [])
    hits = [m for m in markers if re.search(m, text)]
    return {
        "side": side,
        "anonymized": anonymized,
        "recognized": bool(hits),
        "markers": hits,
        "answer": answer,
        "residual_names": anon.leaks(briefing) if anonymized else [],
    }
