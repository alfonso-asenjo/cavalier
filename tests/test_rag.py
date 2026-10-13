from datetime import datetime

from cavalier.rag.corpus import DocKind, Document
from cavalier.rag.retriever import TemporalRetriever

# Documentos SINTÉTICOS de prueba, no históricos.
DOCS = [
    Document(id="a", title="Memo inteligencia", text="misiles emplazamientos isla fotografias",
             date=datetime(1962, 10, 16), available_to=["USA"], source="test"),
    Document(id="b", title="Telegrama embajada", text="misiles negociacion isla",
             date=datetime(1962, 10, 25), available_to=["USA", "USSR"], source="test"),
    Document(id="c", title="Memorias", text="misiles isla desenlace", date=datetime(1970, 1, 1),
             available_to=["USA", "USSR"], kind=DocKind.RETROSPECTIVE, source="test"),
    Document(id="d", title="Informe filtrado", text="misiles isla", date=datetime(1962, 10, 18),
             available_to=["USA", "USSR"], known_from={"USSR": datetime(1962, 10, 27)}, source="test"),
]


def test_filters_by_side_date_and_kind():
    r = TemporalRetriever(DOCS)
    at = datetime(1962, 10, 20)
    assert {d.id for d in r.search("misiles isla", "USA", at)} == {"a", "d"}
    assert {d.id for d in r.search("misiles isla", "USSR", at)} == set()
    late = datetime(1962, 10, 28)
    assert {d.id for d in r.search("misiles isla", "USSR", late)} == {"b", "d"}  # nunca "c" (retrospectivo)
