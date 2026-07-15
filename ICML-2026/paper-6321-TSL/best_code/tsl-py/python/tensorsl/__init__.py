from importlib.metadata import PackageNotFoundError, version as _version

from tensorsl._tensorsl import TSL, GridTensor, StagePredictor, FitResult
from tensorsl.sklearn import TSLRegressor

try:
    __version__ = _version("tensorsl")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["TSL", "GridTensor", "StagePredictor", "FitResult", "TSLRegressor", "__version__"]


def __getattr__(name):
    """Lazy import of the optional plot subpackage so importing tensorsl
    does not require matplotlib."""
    if name == "plot":
        import importlib
        mod = importlib.import_module("tensorsl.plot")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'tensorsl' has no attribute {name!r}")
