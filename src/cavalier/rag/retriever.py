"""Recuperación con filtro temporal y por bando. TF-IDF sin dependencias como punto de partida;
se puede sustituir por embeddings sin tocar al agente (misma interfaz `search`)."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime

from .corpus import Document

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 2]


class TemporalRetriever:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self._tf = [Counter(_tokens(d.title + " " + d.text)) for d in docs]
        df = Counter(t for tf in self._tf for t in tf)
        n = max(1, len(docs))
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}

    def search(self, query: str, side: str, at: datetime, k: int = 4) -> list[Document]:
        q = Counter(_tokens(query))
        scored = []
        for doc, tf in zip(self.docs, self._tf):
            if not doc.visible(side, at):
                continue
            score = sum(qc * tf[t] * self._idf.get(t, 0.0) ** 2 for t, qc in q.items() if t in tf)
            norm = math.sqrt(sum(c * c for c in tf.values())) or 1.0
            if score > 0:
                scored.append((score / norm, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]
