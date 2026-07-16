"""Geometry helpers for Wasserstein tangent projections."""

from .barycenter import wasserstein_barycenter
from .pot import POTTransportSolver
from .tangent import (
    TangentBasis,
    TangentVector,
    TransportResult,
    TransportSolver,
    build_pca_tangent_basis,
    pca_vector_fields,
)

__all__ = [
    "POTTransportSolver",
    "TangentBasis",
    "TangentVector",
    "TransportResult",
    "TransportSolver",
    "build_pca_tangent_basis",
    "pca_vector_fields",
    "wasserstein_barycenter",
]
