from cavalier.anonymize import Anonymizer
from cavalier.display import Deanonymizer


def test_spectator_sees_real_names_with_agreement(bundle):
    d = Deanonymizer.for_bundle(bundle)
    assert d("Mensaje del Primer Secretario al Presidente sobre la Isla") == "Mensaje de Jrushchov a Kennedy sobre Cuba"
    assert d("las fuerzas orientales y los buques orientales") == "las fuerzas soviéticas y los buques soviéticos"
    assert d("la delegación atlántica y el gobierno insular") == "la delegación estadounidense y el gobierno cubano"
    assert d("misiles en el Estado fronterizo") == "misiles en Turquía"
    assert d("El Presidente de la Federación Atlántica y la Federación Atlántica") == "El Presidente de Estados Unidos y Estados Unidos"
    assert d("El Presidente Atlántico ha solicitado inspección") == "El Presidente estadounidense ha solicitado inspección"


def test_relative_dates_become_1962(bundle):
    d = Deanonymizer.for_bundle(bundle)
    assert d("Día 1, 08:00") == "martes 16 de octubre, 08:00"
    assert d("Día 12, 20:00") == "sábado 27 de octubre, 20:00"  # el Sábado Negro


def test_round_trip_restores_the_briefing_names(bundle):
    anon, d = Anonymizer(bundle.aliases), Deanonymizer.for_bundle(bundle)
    text = "Kennedy exige a la Unión Soviética que retire los misiles de Cuba; Jrushchov pide lo mismo en Turquía."
    assert d(anon(text)) == text


def test_walk_changes_values_not_keys(bundle):
    d = Deanonymizer.for_bundle(bundle)
    data = {"la Isla": ["la Isla", {"text": "el Presidente", "turn": 3}]}
    assert d.walk(data) == {"la Isla": ["Cuba", {"text": "Kennedy", "turn": 3}]}
