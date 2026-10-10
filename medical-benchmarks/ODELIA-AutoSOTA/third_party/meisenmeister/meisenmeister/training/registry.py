from __future__ import annotations

import importlib
import inspect
import pkgutil

import meisenmeister.training.trainers as trainer_package
from meisenmeister.training.base_trainer import BaseTrainer


def _iter_trainer_modules():
    for module_info in pkgutil.walk_packages(
        trainer_package.__path__,
        prefix=f"{trainer_package.__name__}.",
    ):
        yield importlib.import_module(module_info.name)


def get_trainer_registry() -> dict[str, type[BaseTrainer]]:
    registry: dict[str, type[BaseTrainer]] = {}
    for module in _iter_trainer_modules():
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj.__module__ == module.__name__
                and issubclass(obj, BaseTrainer)
                and obj is not BaseTrainer
                and obj.__name__.startswith("mmTrainer")
            ):
                registry[obj.__name__] = obj
    return dict(sorted(registry.items()))


def get_available_trainer_names() -> list[str]:
    return sorted(get_trainer_registry())


def get_trainer_class(name: str) -> type[BaseTrainer]:
    registry = get_trainer_registry()
    try:
        return registry[name]
    except KeyError as error:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(
            f"Unknown trainer '{name}'. Available trainers: {available}"
        ) from error
