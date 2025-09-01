"""
Global scorer registry.

Exposes `SCORER_REGISTRY`: a mapping from a string key to a callable of the
form `(text_a: str, text_b: str) -> float` that returns a similarity score in
the range [0.0, 1.0].
"""

from typing import Callable, Dict

SCORER_REGISTRY: Dict[str, Callable[[str, str], float]] = {}

