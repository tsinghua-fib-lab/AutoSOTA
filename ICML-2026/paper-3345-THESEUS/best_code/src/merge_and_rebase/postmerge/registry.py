from __future__ import annotations

from .base import PostMergeMethod

_METHODS: dict[str, PostMergeMethod] = {}


def register(method: PostMergeMethod) -> None:
    if method.name in _METHODS:
        raise KeyError(f"Postmerge method '{method.name}' already registered")
    _METHODS[method.name] = method


def get_postmerge_method(name: str) -> PostMergeMethod:
    if name not in _METHODS:
        raise KeyError(f"Unknown postmerge method '{name}'. Available: {sorted(_METHODS)}")
    return _METHODS[name]


def list_postmerge_methods() -> list[str]:
    return sorted(_METHODS.keys())


from . import methods as _methods  # noqa: E402,F401
