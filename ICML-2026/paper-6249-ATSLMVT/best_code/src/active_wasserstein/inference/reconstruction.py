"""Reconstruct distributions from GP posterior predictions."""

import numpy as np
from typing import Union

from ..measures.base import EmpiricalMeasure, ProbabilityMeasure
from ..measures.weighted import WeightedEmpiricalMeasure
from ..geometry.tangent import TangentBasis
from .predictive import PredictiveProcess


def reconstruct_distribution_at_time(
    t: float,
    gp_posterior: PredictiveProcess,
    basis: TangentBasis,
    reference: ProbabilityMeasure,
) -> ProbabilityMeasure:
    # Get predicted tangent coefficients from GP mean
    coeffs = gp_posterior.mean(t)  # Shape: (n_basis,)

    # Reconstruct vector field at reference points
    displacement = basis.evaluate(reference.support, coeffs)

    # Push forward: new_points = reference_points + displacement
    pushed_support = reference.support + displacement

    reference_weights = getattr(reference, "weights", None)
    if reference_weights is not None:
        weights = np.asarray(reference_weights, dtype=float).reshape(-1)
        if weights.shape[0] != pushed_support.shape[0]:
            raise ValueError(
                "reference weights must match pushed support length "
                f"(got {weights.shape[0]} and {pushed_support.shape[0]})"
            )
        predicted_measure = WeightedEmpiricalMeasure(
            support=pushed_support,
            weights=weights,
        )
    else:
        # EmpiricalMeasure uses uniform weights implicitly.
        predicted_measure = EmpiricalMeasure(support=pushed_support)

    return predicted_measure


def reconstruct_distributions(
    times: Union[np.ndarray, list],
    gp_posterior: PredictiveProcess,
    basis: TangentBasis,
    reference: ProbabilityMeasure,
) -> list[ProbabilityMeasure]:
    times = np.asarray(times)
    return [
        reconstruct_distribution_at_time(t, gp_posterior, basis, reference)
        for t in times
    ]
