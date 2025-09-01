import math
import pytest


def _import():
    # Import inside test to allow repository without sklearn to still be imported.
    from scorers.tfidf_char_cosine import score_tfidf_char_cosine
    return score_tfidf_char_cosine


def test_identical_strings_near_one():
    score = _import()("Canción número 1", "canción número 1")
    assert score >= 0.99


def test_minor_typo_and_accent_robustness():
    # Missing accent and a common typo (z->s)
    a = "programación avanzada"
    b = "programacion avansada"
    score = _import()(a, b)
    assert score > 0.7


def test_different_topics_low_similarity():
    a = "economía y finanzas"
    b = "fútbol y baloncesto"
    score = _import()(a, b)
    assert score < 0.2

