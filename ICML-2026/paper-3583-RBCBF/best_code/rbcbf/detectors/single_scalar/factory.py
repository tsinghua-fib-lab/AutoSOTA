"""Factory for single scalar builders."""

from __future__ import annotations

from typing import Dict, Sequence

from .base_scalar import ScalarBuilder
from .delta_actset import DeltaActSetBuilder
from .delta_max import DeltaMaxBuilder
from .delta_min import DeltaMinBuilder
from .delta_soft_uncert import DeltaSoftUncertaintyBuilder
from .delta_softact import DeltaSoftActBuilder
from .delta_sum import DeltaSumBuilder
from .h_deficit import HDeficitBuilder


SCALAR_BUILDERS = {
    "delta_sum": DeltaSumBuilder,
    "delta_min": DeltaMinBuilder,
    "delta_max": DeltaMaxBuilder,
    "delta_softact": DeltaSoftActBuilder,
    "delta_actset": DeltaActSetBuilder,
    "delta_soft_uncert": DeltaSoftUncertaintyBuilder,
    "h_deficit": HDeficitBuilder,
}


def create_scalar_builder(
    scalar_type: str,
    labels: Sequence[str],
    config: Dict[str, object],
    beta_weights: Sequence[float] | None = None,
) -> ScalarBuilder:
    if scalar_type not in SCALAR_BUILDERS:
        raise ValueError(f"Unknown scalar_type: {scalar_type}")
    builder_cls = SCALAR_BUILDERS[scalar_type]
    if scalar_type == "delta_sum":
        return builder_cls(labels, config, beta_weights=beta_weights)
    return builder_cls(labels, config)


__all__ = ["create_scalar_builder", "SCALAR_BUILDERS"]
