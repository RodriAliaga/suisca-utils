"""
Token Set Ratio scorer (RapidFuzz).

Summary:
- Order-insensitive token matching with duplicate handling. Uses
  `rapidfuzz.fuzz.token_set_ratio` and normalizes the percentage to [0, 1].

When to use:
- Useful when token order differs (e.g., "acero inoxidable" vs. "inoxidable acero")
  and where repeated tokens should not inflate scores.

Limitations:
- Purely lexical; no semantic understanding beyond token overlap.

Score range:
- Returns a float in [0.0, 1.0]. Both empty -> 1.0; exactly one empty -> 0.0.
"""

from .registry import SCORER_REGISTRY


def score_token_set_ratio(text_a: str, text_b: str) -> float:
    from rapidfuzz import fuzz

    a = (text_a or "").strip()
    b = (text_b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


# Register in global registry
SCORER_REGISTRY["token_set_ratio"] = score_token_set_ratio

