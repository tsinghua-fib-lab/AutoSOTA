"""DVM-AD: Discriminant Vector Machine for Anomaly Detection (ICML 2026).

`DVMADCore` is the NumPy-only implementation. `DVMAD` is the PyOD-compatible
wrapper; it requires `pyod` to be installed and is imported lazily so the
core class is usable without PyOD.
"""

from .core import DVMADCore

__version__ = "0.1.0"
__all__ = ["DVMAD", "DVMADCore"]


def __getattr__(name):
    if name == "DVMAD":
        from .dvmad import DVMAD as _DVMAD

        return _DVMAD
    raise AttributeError(f"module 'dvmad' has no attribute {name!r}")
