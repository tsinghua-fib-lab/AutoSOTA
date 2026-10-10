from __future__ import annotations

import importlib
import inspect
import pkgutil

import meisenmeister.architectures as architecture_package
from meisenmeister.architectures.base_architecture import BaseArchitecture


def _iter_architecture_modules():
    for module_info in pkgutil.iter_modules(
        architecture_package.__path__,
        prefix=f"{architecture_package.__name__}.",
    ):
        yield importlib.import_module(module_info.name)


def get_architecture_registry() -> dict[str, type[BaseArchitecture]]:
    registry: dict[str, type[BaseArchitecture]] = {}
    for module in _iter_architecture_modules():
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj.__module__ == module.__name__
                and issubclass(obj, BaseArchitecture)
                and obj is not BaseArchitecture
            ):
                registry[obj.__name__] = obj
    return dict(sorted(registry.items()))


def get_available_architecture_names() -> list[str]:
    return sorted(get_architecture_registry())


def get_architecture_class(name: str) -> type[BaseArchitecture]:
    registry = get_architecture_registry()
    try:
        return registry[name]
    except KeyError as error:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(
            f"Unknown architecture '{name}'. Available architectures: {available}"
        ) from error
