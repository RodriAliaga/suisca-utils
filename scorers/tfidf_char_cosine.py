"""
TF‑IDF character n‑gram cosine similarity.

Summary:
- Builds TF‑IDF vectors over character n‑grams (3–5) with L2 normalization and
  returns the cosine similarity between the two input strings.

Pros:
- Robust to small typos, insertions/deletions, casing changes, and diacritics.
- Works well on short texts; language‑agnostic.

Cons:
- Surface‑form only; does not capture semantics.
- Requires scikit‑learn and incurs CPU for vectorization.

Score range:
- Returns a float in [0.0, 1.0]. Both empty -> 1.0; exactly one empty -> 0.0.
"""

from __future__ import annotations

from typing import cast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .registry import SCORER_REGISTRY


def score_tfidf_char_cosine(
    text_a: str,
    text_b: str,
    ngram_low: int = 3,
    ngram_high: int = 5,
) -> float:
    """Compute cosine similarity over TF‑IDF character 3–5 n‑grams.

    Method: Vectorize `text_a` and `text_b` using character n‑gram TF‑IDF
    (3–5) with L2 normalization, then compute cosine similarity between the
    two vectors.

    Pros:
    - Tolerant to typos, accents, and small edits.
    - Effective for short texts; language‑agnostic.

    Cons:
    - Surface‑level; no semantic understanding.
    - Depends on scikit‑learn; adds runtime/import cost.

    Expected I/O:
    - Input: two strings; internally lowercased.
    - Output: float in [0.0, 1.0]. If both strings normalize to empty, returns
      1.0; if exactly one is empty, returns 0.0.

    Returns:
    - Cosine similarity as a Python float in [0.0, 1.0].
    """
    a = (text_a or "").strip().lower()
    b = (text_b or "").strip().lower()

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    vec = TfidfVectorizer(
        analyzer="char", ngram_range=(ngram_low, ngram_high), lowercase=True, norm="l2"
    )
    X = vec.fit_transform([a, b])
    # With L2 norm, cosine equals dot product between normalized vectors.
    sim = cosine_similarity(X[0], X[1])[0, 0]
    # Ensure a plain Python float in [0, 1]
    sim_f = float(max(0.0, min(1.0, cast(float, sim))))
    return sim_f


# Register in global registry without altering existing code paths.
SCORER_REGISTRY["tfidf_char_cosine"] = score_tfidf_char_cosine
