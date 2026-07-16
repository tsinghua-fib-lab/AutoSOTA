"""Backend-dispatched Wasserstein barycenter helpers."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from active_wasserstein.measures.base import EmpiricalMeasure, ProbabilityMeasure

from .pot import wasserstein_barycenter as pot_wasserstein_barycenter


def wasserstein_barycenter(
    measures: Sequence[ProbabilityMeasure],
    weights: Optional[Sequence[float]] = None,
    barycenter_size: int = 128,
    num_iter: int = 100,
    reg: float = 0.0,
    rng: Any = None,
    *,
    backend: str = "pot",
) -> EmpiricalMeasure:
    """Compute a discrete Wasserstein barycenter with the POT backend."""

    backend_norm = str(backend).strip().lower()
    if backend_norm != "pot":
        raise ValueError(f"Unknown barycenter backend '{backend}'. Expected 'pot'.")
    return pot_wasserstein_barycenter(
        measures=measures,
        weights=weights,
        barycenter_size=barycenter_size,
        num_iter=num_iter,
        reg=reg,
        rng=rng,
    )
