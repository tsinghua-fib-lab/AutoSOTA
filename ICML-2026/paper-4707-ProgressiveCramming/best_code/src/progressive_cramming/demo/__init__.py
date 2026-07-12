"""Demo utilities for the interactive notebook and standalone scripts.

Public API is unchanged from when ``demo`` was a single module file -- import
sites like ``from progressive_cramming.demo import load_frozen_model`` keep working
because everything is re-exported here. Notebook-only helpers (HTML diff,
embedding-row loaders) live in :mod:`progressive_cramming.demo.notebook` and are
loaded only when the user asks for them.
"""

from ._core import (
    CrammingResult,
    ProgressiveResult,
    ProgressiveStage,
    cram_text,
    load_frozen_model,
    pick_device,
    progressive_cram_text,
    reconstruct_text,
    resolve_dtype,
)

__all__ = [
    "CrammingResult",
    "ProgressiveResult",
    "ProgressiveStage",
    "cram_text",
    "load_frozen_model",
    "pick_device",
    "progressive_cram_text",
    "reconstruct_text",
    "resolve_dtype",
]
