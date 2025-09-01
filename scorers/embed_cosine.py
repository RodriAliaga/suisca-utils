"""
Semantic similarity via Sentence-Transformers cosine similarity.

Summary:
- Encodes both texts with a multilingual Sentence-Transformer model and returns
  the cosine similarity mapped to [0, 1]. Lazy-loads the model on first use.

Multilingual:
- Default model "paraphrase-multilingual-MiniLM-L12-v2" supports many languages
  (ES/EN), making cross-lingual paraphrase detection feasible.

Pros:
- Captures semantic similarity beyond surface token overlap; robust to wording.

Cons:
- Heavier dependency; downloads model on first run; CPU/GPU cost > lexical.

Performance:
- First call incurs model load and potential download; cache persists in memory.

Score range:
- Returns float in [0.0, 1.0]. Both empty -> 1.0; exactly one empty -> 0.0.
"""

from __future__ import annotations

from typing import Dict

from .registry import SCORER_REGISTRY

_MODEL_CACHE: Dict[str, object] = {}


def _get_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:  # pragma: no cover - environment without dependency
        raise ImportError(
            "sentence-transformers is required for 'embed_cosine'. "
            "Install with: pip install sentence-transformers"
        ) from e

    model = _MODEL_CACHE.get(model_name)
    if model is None:
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            raise ImportError(
                "Could not load Sentence-Transformer model. Ensure network access "
                "or that the model is available locally. Try: pip install sentence-transformers"
            ) from e
        _MODEL_CACHE[model_name] = model
    return model


def score_embed_cosine(
    text_a: str,
    text_b: str,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> float:
    """Compute semantic similarity with a Sentence-Transformer model.

    Method: Encode both inputs using a multilingual model; L2-normalize
    embeddings and compute cosine similarity; map from [-1, 1] to [0, 1].

    Returns a Python float in [0.0, 1.0]. If both normalize to empty
    strings, returns 1.0; if exactly one is empty, returns 0.0.
    """
    a = (text_a or "").strip()
    b = (text_b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    model = _get_model(model_name)
    try:
        # Import here to avoid import-time dependency if never used.
        from sentence_transformers import util  # type: ignore
        embeddings = model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    except Exception as e:  # pragma: no cover - runtime issues
        raise ImportError(
            "Failed during embedding. Ensure sentence-transformers is installed and the model is available."
        ) from e

    v1, v2 = embeddings[0], embeddings[1]
    # With normalized embeddings, cosine is simply dot product in [-1, 1]
    cos = float((v1 * v2).sum())
    sim01 = max(0.0, min(1.0, (cos + 1.0) / 2.0))
    return sim01


# Register in global registry (module must be imported to register)
SCORER_REGISTRY["embed_cosine"] = score_embed_cosine

