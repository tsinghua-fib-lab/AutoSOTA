"""
`sdi` - Step-Decomposed Influence (SDI) + TracIn utilities.

This package provides two primary estimators:

- `ProjectedTracInSDI`: hook-based "project-during-backprop" estimator using
  TensorSketch for matrix weights and CountSketch for vector parameters.
- `FullGradientTracInSDI`: full per-sample-gradient baseline (Opacus grad samplers)
  that materializes per-sample gradients (expensive, but exact).

Both estimators are **parameter-set explicit**: you must pass the exact submodule(s)
whose parameters you want to analyze (e.g. `model.looped_body`).
No parameters are silently excluded. If a selected module contains a trainable
parameter without Opacus grad-sampler coverage (or unsupported shapes), we raise.
"""

from .estimators import FullGradientTracInSDI, ProjectedTracInSDI
from .runner import CheckpointSpec, InfluenceOutputs

__all__ = [
    "ProjectedTracInSDI",
    "FullGradientTracInSDI",
    "CheckpointSpec",
    "InfluenceOutputs",
]
