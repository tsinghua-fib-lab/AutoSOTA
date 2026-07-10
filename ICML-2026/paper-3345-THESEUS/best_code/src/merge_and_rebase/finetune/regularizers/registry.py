from __future__ import annotations

from .base import Regularizer

_REGULARIZERS: dict[str, Regularizer] = {}


def register(regularizer: Regularizer) -> None:
    if regularizer.name in _REGULARIZERS:
        raise KeyError(f"Regularizer '{regularizer.name}' already registered")
    _REGULARIZERS[regularizer.name] = regularizer


def get_regularizer(name: str) -> Regularizer:
    if name not in _REGULARIZERS:
        raise KeyError(f"Unknown regularizer '{name}'. Available: {sorted(_REGULARIZERS)}")
    return _REGULARIZERS[name]


def list_regularizers() -> list[str]:
    return sorted(_REGULARIZERS.keys())
