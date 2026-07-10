from __future__ import annotations

from .base import RebaseMethod

_METHODS: dict[str, RebaseMethod] = {}


def register(method: RebaseMethod) -> None:
    if method.name in _METHODS:
        raise KeyError(f"Rebase method '{method.name}' already registered")
    _METHODS[method.name] = method


def get_method(name: str) -> RebaseMethod:
    if name not in _METHODS:
        raise KeyError(f"Unknown rebase method '{name}'. Available: {sorted(_METHODS)}")
    return _METHODS[name]


def list_methods() -> list[str]:
    return sorted(_METHODS.keys())


# Import built-in method modules for side-effect registration.
from . import methods as _methods  # noqa: F401,E402
