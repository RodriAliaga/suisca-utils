TF‑IDF Char Cosine Scorer
-------------------------

- Approach: Builds TF‑IDF vectors over character n‑grams (3–5) with L2
  normalization, then computes cosine similarity between two input strings.
- Strengths: Resilient to small typos, insertions/deletions, casing changes,
  and diacritics; effective for short texts; language‑agnostic.
- Limitations: Purely surface‑form; does not capture semantic similarity.
- Input/Output: Accepts two strings, lowercased internally; returns a float in
  [0.0, 1.0]. If both normalized strings are empty → 1.0; if one is empty → 0.0.
- Dependencies: Requires `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`).
- Name: Registered under key `tfidf_char_cosine` in `SCORER_REGISTRY`.

Token Set Ratio (RapidFuzz)
---------------------------

- Approach: Order-insensitive token comparison using `rapidfuzz.fuzz.token_set_ratio`,
  dividing the 0–100 score by 100 to return [0, 1].
- Use cases: Strings where token order varies and duplicates should not inflate
  similarity (e.g., product attributes re-ordered or repeated).
- Limitations: Lexical-only; no semantic understanding; depends on tokenization quality.
- Input/Output: Two strings; returns float in [0.0, 1.0]; both empty → 1.0; one empty → 0.0.
- Name: Registered under key `token_set_ratio` in `SCORER_REGISTRY`.

Embed Cosine (Sentence-Transformers)
------------------------------------

- Approach: Encodes texts with a multilingual Sentence-Transformer and returns
  cosine similarity mapped to [0, 1] (semantic similarity).
- Multilingual: Default model `paraphrase-multilingual-MiniLM-L12-v2` works well
  across ES/EN and other languages for paraphrase-like similarity.
- Pros/Cons: Captures meaning beyond tokens; heavier dependency and initial
  model download; higher CPU/GPU cost than lexical scorers.
- Performance: Lazy-loads the model on first use and caches in memory.
- I/O & Range: Two strings; [0, 1]; both empty → 1.0; one empty → 0.0.
- Install: `pip install sentence-transformers`. Registered as `embed_cosine`.
