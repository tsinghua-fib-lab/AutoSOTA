"""
`jaxcld` package public API.

The goal is to support:

from jaxcld import ASRModel, CVXNNLangDetectHead
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "ASRModel",
    "CVXNNLangDetectHead",
    "NNLangDetectHead",
    "SklearnLangDetectHead",
    "SVMLangDetectHead",
]


def __getattr__(name: str):
    # Lazy imports so `import jaxcld` works even if optional heavy deps (torch, transformers)
    # are not installed, while still supporting `from jaxcld import ASRModel, ...` when they are.
    try:
        if name == "ASRModel":
            from .models.asr_model import ASRModel

            return ASRModel
        if name == "CVXNNLangDetectHead":
            from .models.lang_detect_head import CVXNNLangDetectHead

            return CVXNNLangDetectHead
        if name == "NNLangDetectHead":
            from .models.lang_detect_head import NNLangDetectHead

            return NNLangDetectHead
        if name == "SklearnLangDetectHead":
            from .models.lang_detect_head import SklearnLangDetectHead

            return SklearnLangDetectHead
        if name == "SVMLangDetectHead":
            from .models.lang_detect_head import SVMLangDetectHead

            return SVMLangDetectHead
    except ModuleNotFoundError as e:
        raise ImportError(
            "Missing optional dependency. Install CLD with its runtime dependencies, e.g. "
            "`pip install -e .` (or `pip install .`) and ensure `torch`, `torchaudio`, and "
            "`transformers` are available."
        ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")