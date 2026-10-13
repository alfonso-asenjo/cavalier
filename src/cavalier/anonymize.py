"""Modo anonimizado: renombra países, personas y lugares en todo lo que ve un agente."""

from __future__ import annotations

import re

from .scenario import Aliases


class Anonymizer:
    def __init__(self, aliases: Aliases, enabled: bool = True):
        self.enabled = enabled
        pairs = sorted(aliases.replacements.items(), key=lambda kv: len(kv[0]), reverse=True)
        self._patterns = [(re.compile(self._pattern(src)), dst) for src, dst in pairs]

    @staticmethod
    def _pattern(src: str) -> str:
        # Límites de palabra solo donde la cadena empieza o termina en carácter de palabra.
        head = r"(?<!\w)" if re.match(r"\w", src) else ""
        tail = r"(?!\w)" if re.search(r"\w$", src) else ""
        return head + re.escape(src) + tail

    def __call__(self, text: str) -> str:
        if not self.enabled:
            return text
        for pattern, dst in self._patterns:
            text = pattern.sub(dst, text)
        return text

    def leaks(self, text: str) -> list[str]:
        """Nombres reales que siguen apareciendo en un texto ya anonimizado (para tests y auditoría)."""
        return [p.pattern for p, _ in self._patterns if p.search(text)]
