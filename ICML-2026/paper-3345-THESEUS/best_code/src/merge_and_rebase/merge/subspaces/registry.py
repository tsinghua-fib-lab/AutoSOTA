from __future__ import annotations

from .base import Subspace

_SPACES: dict[str, Subspace] = {}


def register(space: Subspace) -> None:
    if space.name in _SPACES:
        raise KeyError(f"Subspace '{space.name}' already registered")
    _SPACES[space.name] = space


def get_subspace(name: str) -> Subspace:
    if name not in _SPACES:
        raise KeyError(f"Unknown subspace '{name}'. Available: {sorted(_SPACES)}")
    return _SPACES[name]


def list_subspaces() -> list[str]:
    return sorted(_SPACES.keys())
