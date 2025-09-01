try:
    import sentence_transformers  # noqa: F401
    ST_OK = True
except Exception:  # pragma: no cover - environment without sentence-transformers
    ST_OK = False

import pytest


@pytest.mark.skipif(not ST_OK, reason="sentence-transformers not installed")
def test_semantic_paraphrase_multilingual_high_score():
    # Import module to register the scorer into SCORER_REGISTRY
    from scorers import SCORER_REGISTRY
    import scorers.embed_cosine  # noqa: F401

    f = SCORER_REGISTRY["embed_cosine"]
    # Cross-lingual paraphrase-like pair
    s = f("comprar válvula de bola", "purchase a ball valve")
    assert s > 0.7


@pytest.mark.skipif(not ST_OK, reason="sentence-transformers not installed")
def test_unrelated_low_score():
    from scorers import SCORER_REGISTRY
    import scorers.embed_cosine  # noqa: F401

    f = SCORER_REGISTRY["embed_cosine"]
    s = f("receta de ensalada de tomate", "ethernet cable cat6 specification")
    assert s < 0.3

