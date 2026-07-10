from __future__ import annotations

from .base import MergeMethod

_METHODS: dict[str, MergeMethod] = {}


def register(method: MergeMethod) -> None:
    if method.name in _METHODS:
        raise KeyError(f"Merge method '{method.name}' already registered")
    _METHODS[method.name] = method


def get_method(name: str) -> MergeMethod:
    if name not in _METHODS:
        raise KeyError(f"Unknown merge method '{name}'. Available: {sorted(_METHODS)}")
    return _METHODS[name]


def list_methods() -> list[str]:
    return sorted(_METHODS.keys())


# Import built-in method modules for side-effect registration.
# Must happen after `register` is defined because method modules import it.
from . import methods as _methods  # noqa: F401,E402
