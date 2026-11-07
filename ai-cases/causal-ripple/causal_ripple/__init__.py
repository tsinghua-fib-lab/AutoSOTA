"""Causal-Ripple inference-time sampling for masked diffusion language models."""

from .config import MODEL_ID, MODEL_REVISION, SamplerConfig, get_config

__all__ = ["generate", "SamplerConfig", "get_config", "MODEL_ID", "MODEL_REVISION"]
__version__ = "0.1.0"


def __getattr__(name):
    if name == "generate":
        from .sampling import generate

        return generate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
