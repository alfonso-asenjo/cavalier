"""Corpus documental con metadatos temporales.

El filtro relevante no es "fecha del documento" sino "quién lo sabía y desde cuándo":
un memorando de la CIA nunca está disponible para el liderazgo soviético.
Los testimonios retrospectivos (memorias, entrevistas posteriores) filtran el desenlace:
NO entran en el RAG de los agentes; solo sirven como referencia de evaluación.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class DocKind(StrEnum):
    CONTEMPORARY = "contemporaneo"  # producido durante la crisis
    RETROSPECTIVE = "retrospectivo"  # memorias, entrevistas, historiografía


class Document(BaseModel):
    id: str
    title: str
    text: str
    date: datetime  # cuándo se produjo
    available_to: list[str]  # bandos que lo conocían en su momento
    known_from: dict[str, datetime] = Field(default_factory=dict)  # si un bando lo conoció más tarde
    kind: DocKind = DocKind.CONTEMPORARY
    source: str
    url: str | None = None
    language: str = "en"
    translated: bool = False
    annotated_by: str = "human"  # human | llm (pendiente de validar)
    validated: bool = False

    def visible(self, side: str, at: datetime) -> bool:
        if self.kind != DocKind.CONTEMPORARY or side not in self.available_to:
            return False
        return self.known_from.get(side, self.date) <= at


def load_corpus(path: str | Path) -> list[Document]:
    """Lee todos los .jsonl de una carpeta (un documento por línea)."""
    docs = []
    for file in sorted(Path(path).glob("*.jsonl")):
        with file.open(encoding="utf-8") as f:
            docs.extend(Document.model_validate(json.loads(line)) for line in f if line.strip())
    return docs
