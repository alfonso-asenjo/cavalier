"""Registro de partidas en JSONL: una línea por turno con observaciones, decisiones y estado real."""

from __future__ import annotations

import json
from pathlib import Path


class JsonlRecorder:
    def __init__(self, path: str | Path, header: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        if header:
            self._write({"type": "header", **header})

    def _write(self, record: dict) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def __call__(self, record: dict) -> None:
        self._write({"type": "turn", **record})

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
