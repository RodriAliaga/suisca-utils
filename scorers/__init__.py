"""
Scorer package export and registration.

Exposes `SCORER_REGISTRY` and imports available scorers for side‑effect
registration into the registry.
"""

from .registry import SCORER_REGISTRY  # noqa: F401

# Import modules that register themselves in the registry on import.
# If a dependency is missing (e.g., scikit-learn), the import will raise.
from . import tfidf_char_cosine  # noqa: F401
from . import token_set_ratio  # noqa: F401  # side-effect: registers 'token_set_ratio'
