"""Scoring and hashing utilities."""
from __future__ import annotations

import hashlib


def compute_score(results: list) -> tuple[float, float]:
    """Compute hard and soft accuracy from a list of episode results.

    Accepts both plain dicts and :class:    instances.  hard may be continuous (0.0-1.0) when using smoothed reward.
    """
    if not results:
        return 0.0, 0.0

    def _hard(r: object) -> float:
        return float(r.hard if hasattr(r, "hard") else r.get("hard", 0))

    def _soft(r: object) -> float:
        return float(r.soft if hasattr(r, "soft") else r.get("soft", 0.0))

    hard = sum(_hard(r) for r in results) / len(results)
    soft = sum(_soft(r) for r in results) / len(results)
    return hard, soft


def skill_hash(content: str) -> str:
    """Return a short deterministic hash of skill content (for caching)."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]
