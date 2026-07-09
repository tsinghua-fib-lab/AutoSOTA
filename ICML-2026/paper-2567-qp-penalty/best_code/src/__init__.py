"""dXPP: Differentiable QP layer using penalty smoothing."""

from .dXPP import dXPPLayer
from .penalty_smooth_qp import PenaltySmoothQP
from .sparse_utils import _SPARSE_SOLVER, is_sparse_tensor, torch_sparse_to_scipy, sparse_solve_spd
from .qp_utils import (
    set_solver_tolerance,
    _DUAL_AVAILABLE_SOLVERS,
    _select_default_qp_solver,
    _compute_multipliers_from_kkt,
)

__all__ = [
    "dXPPLayer",
    "PenaltySmoothQP",
    "set_solver_tolerance",
    "is_sparse_tensor",
    "torch_sparse_to_scipy",
    "sparse_solve_spd",
]
