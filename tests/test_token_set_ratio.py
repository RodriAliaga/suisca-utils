try:
    from rapidfuzz import fuzz  # noqa: F401
    RAPIDFUZZ_OK = True
except Exception:  # pragma: no cover - environment without rapidfuzz
    RAPIDFUZZ_OK = False

import pytest
from scorers import SCORER_REGISTRY


@pytest.mark.skipif(not RAPIDFUZZ_OK, reason="rapidfuzz not installed")
def test_same_tokens_different_order():
    f = SCORER_REGISTRY["token_set_ratio"]
    s = f("valvula acero inoxidable", "inoxidable acero valvula")
    assert s > 0.9


@pytest.mark.skipif(not RAPIDFUZZ_OK, reason="rapidfuzz not installed")
def test_duplicates_still_high():
    f = SCORER_REGISTRY["token_set_ratio"]
    s = f("manguera manguera 1/2", "manguera 1/2")
    assert s > 0.85


@pytest.mark.skipif(not RAPIDFUZZ_OK, reason="rapidfuzz not installed")
def test_unrelated_low():
    f = SCORER_REGISTRY["token_set_ratio"]
    s = f("válvula esfera 1\"", "cable ethernet cat6")
    assert s < 0.2

