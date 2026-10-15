"""Lo que ve el espectador: deshace el modo anonimizado al mostrar una partida.

Los agentes juegan con nombres ficticios («la Federación Atlántica», «el Primer Secretario»,
«Día 12, 20:00») para no reconocer el episodio. Quien mira la partida, en cambio, necesita
conectarla con la historia real: `Deanonymizer` devuelve Kennedy, Cuba, Turquía y las fechas de
1962. Nunca se aplica a lo que ve un agente.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from .anonymize import Anonymizer
from .scenario import Aliases, ScenarioBundle

WEEKDAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
          "octubre", "noviembre", "diciembre"]

# Gentilicios del alias → real, por género y número. El género se deduce del sustantivo anterior.
DEMONYMS = {
    "oriental": ("soviético", "soviética"), "orientales": ("soviéticos", "soviéticas"),
    "atlántico": ("estadounidense", "estadounidense"), "atlántica": ("estadounidense", "estadounidense"),
    "atlánticos": ("estadounidenses", "estadounidenses"), "atlánticas": ("estadounidenses", "estadounidenses"),
    "insular": ("cubano", "cubana"), "insulares": ("cubanos", "cubanas"),
}
_FEMININE = re.compile(r"(?:a|as|ión|iones|dad|dades|tud|tudes)$")
_DEMONYM = re.compile(r"(?:(\w+)\s+)?\b(" + "|".join(sorted(DEMONYMS, key=len, reverse=True)) + r")\b", re.IGNORECASE)
_DAY = re.compile(r"Día (\d+), (\d\d):(\d\d)")


def real_date(start: datetime, day: int, hour: int, minute: int) -> str:
    when = start.replace(hour=0, minute=0) + timedelta(days=day - 1, hours=hour, minutes=minute)
    return f"{WEEKDAYS[when.weekday()]} {when.day} de {MONTHS[when.month - 1]}, {when:%H:%M}"


class Deanonymizer:
    def __init__(self, aliases: Aliases, start: datetime | None = None):
        # Mismo motor de sustitución (límites de palabra, más largas primero) que el anonimizador.
        self._names = Anonymizer(Aliases(replacements=aliases.display))
        self.start = start

    @classmethod
    def for_bundle(cls, bundle: ScenarioBundle) -> "Deanonymizer":
        return cls(bundle.aliases, bundle.scenario.start)

    def _demonym(self, m: re.Match) -> str:
        noun, word = m.group(1), m.group(2)
        masc, fem = DEMONYMS[word.lower()]
        real = fem if noun and _FEMININE.search(noun.lower()) else masc
        return f"{noun} {real}" if noun else real

    def __call__(self, text: str) -> str:
        text = self._names(text)
        text = _DEMONYM.sub(self._demonym, text)
        if self.start:
            text = _DAY.sub(lambda m: real_date(self.start, int(m.group(1)), int(m.group(2)), int(m.group(3))), text)
        return text

    def walk(self, obj):
        """Aplica la sustitución a todas las cadenas de una estructura JSON (no a las claves)."""
        if isinstance(obj, str):
            return self(obj)
        if isinstance(obj, list):
            return [self.walk(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self.walk(v) for k, v in obj.items()}
        return obj
