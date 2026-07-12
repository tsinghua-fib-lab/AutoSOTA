"""Analysis utilities used by the cramming trainers (convergence + information gain)."""

from progressive_cramming.analysis.convergence import (
    ConvergedSamplesGuard,
    ConvergenceTracker,
    ProgressiveSampleStateMachine,
)
from progressive_cramming.analysis.information_gain import (
    compute_information_gain,
    compute_prefix_surprisal_bits_per_token,
)

__all__ = [
    "ConvergedSamplesGuard",
    "ConvergenceTracker",
    "ProgressiveSampleStateMachine",
    "compute_information_gain",
    "compute_prefix_surprisal_bits_per_token",
]
