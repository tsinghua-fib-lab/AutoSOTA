from __future__ import annotations

from .base import Strategy

_STRATS: dict[str, Strategy] = {}


def register(strategy: Strategy) -> None:
    if strategy.name in _STRATS:
        raise KeyError(f"Strategy '{strategy.name}' already registered")
    _STRATS[strategy.name] = strategy


def get_strategy(name: str) -> Strategy:
    if name not in _STRATS:
        raise KeyError(f"Unknown strategy '{name}'. Available: {sorted(_STRATS)}")
    return _STRATS[name]


def list_strategies() -> list[str]:
    return sorted(_STRATS.keys())
