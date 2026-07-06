# probes/registry.py

from typing import Dict, Type
from probes.base import ProbeBase

_PROBE_REGISTRY: Dict[str, Type[ProbeBase]] = {}


def register_probe(name: str):
    """
    Decorator to register a probe class by name.
    """
    def decorator(cls):
        if name in _PROBE_REGISTRY:
            raise ValueError(f"Probe '{name}' already registered.")
        _PROBE_REGISTRY[name] = cls
        return cls
    return decorator


def get_probe(name: str):
    """
    Get a probe class by name.
    """
    if name not in _PROBE_REGISTRY:
        raise KeyError(f"Probe '{name}' not found.")
    return _PROBE_REGISTRY[name]


def list_probes():
    """
    List all registered probes.
    """
    return list(_PROBE_REGISTRY.keys())
