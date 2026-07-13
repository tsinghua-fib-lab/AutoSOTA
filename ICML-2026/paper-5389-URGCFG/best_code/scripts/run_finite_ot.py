"""Script to run FCOT-Separable on DGPS data and plot running statistics."""

import logging
import os

import numpy as np
import torch

from optimal_transport.ot_fc_sep_map import FCOTSeparable
from tools.dgps import generate_gaussian_pairs, generate_grid_XY
from tools.feedback import set_log_level
from tools.utils import (
    L22_1d,
    L33_1d,
    L44_1d,
    L55_1d,
    inverse_L22x,
    inverse_L33x,
    inverse_L44x,
    inverse_L55x,
    nL22_1d,
    inverse_nL22x,
)


SEED = 42
LOG_LEVEL = "DEBUG"

# Data distribution settings (1D Gaussian → Gaussian)
N_SAMPLES = 10000
MU_X_1D = torch.tensor([0.0])
SIGMA_X_1D = torch.tensor([[1.0]])
MU_Y_1D = torch.tensor([-1.0])
SIGMA_Y_1D = torch.tensor([[2.0]])

# Domain / grid resolution (matches target grid L in data generation)
RADIUS = 4.0
X_ACCURACY = 2e-3
Y_ACCURACY = 2e-3

# Optimizer (intercept optimizer)
OUTER_LR = 1e-2

# Softmax temperature schedule
TEMP_MIN = 1.0
TEMP_MAX = 60.0
TEMP_WARMUP_ITERS = 2_500

# Reactivation and full refresh rules
REACTIVATE_EVERY = 50
REACTIVATE_EPS = 1e-3
FULL_REFRESH_EVERY = 300

# Coarse-to-fine transform speedup
COARSE_X_FACTOR = 100
COARSE_TOP_K = 4
COARSE_WINDOW = 1

# Training / evaluation parameters
TRAIN_ITERS = 7_000
PRINT_EVERY = 10
LOG_EVERY = 20
CONVERGENCE_TOL = 1e-6
CONVERGENCE_PATIENCE = 300
FORCE_RETRAIN = False

# 1D evaluation settings
KERNELS_1D = [L22_1d, nL22_1d]
INV_KERNELS = [inverse_L22x, inverse_nL22x]


def _setup_randomness_and_threads():
    """Seed random generators and limit thread usage for reproducible runs."""
    torch.manual_seed(SEED)
    set_log_level(LOG_LEVEL)

    cpu = max(1, (os.cpu_count() or 1) // 2)
    try:
        torch.set_num_threads(cpu)
        torch.set_num_interop_threads(max(1, cpu // 2))
    except RuntimeError:
        pass


def _build_solver(
    dim: int,
    kernel_1d,
    inverse_kx,
    *,
    radius_override: float | None = None,
    x_accuracy_override: float | None = None,
    y_accuracy_override: float | None = None,
    ny_override: int | None = None,
) -> FCOTSeparable:
    radius = RADIUS if radius_override is None else radius_override
    x_accuracy = X_ACCURACY if x_accuracy_override is None else x_accuracy_override
    y_accuracy = Y_ACCURACY if y_accuracy_override is None else y_accuracy_override
    if ny_override is None:
        ny = int((2 * radius) / y_accuracy) + 1
    else:
        if ny_override < 2:
            raise ValueError("ny_override must be at least 2 grid points.")
        ny = ny_override
    n_params = ny * dim

    """Instantiate FCOTSeparable solver with the requested kernel/grid resolution."""
    fcot = FCOTSeparable.initialize_right_architecture(
        dim=dim,
        radius=radius,
        n_params=n_params,
        x_accuracy=x_accuracy,
        kernel_1d=kernel_1d,
        inverse_kx=inverse_kx,
        outer_lr=OUTER_LR,
        temp_min=TEMP_MIN,
        temp_max=TEMP_MAX,
        temp_warmup_iters=TEMP_WARMUP_ITERS,
        reactivate_every=REACTIVATE_EVERY,
        reactivate_eps=REACTIVATE_EPS,
        full_refresh_every=FULL_REFRESH_EVERY,
        cache_gradients=True,
        coarse_x_factor=COARSE_X_FACTOR,
        coarse_top_k=COARSE_TOP_K,
        coarse_window=COARSE_WINDOW,
    )
    return fcot


def _compute_transport(solver: FCOTSeparable, X: torch.Tensor) -> torch.Tensor:
    """Helper to evaluate the learned Monge map (c-gradient) on new points."""
    X_req = X.to(solver.device).requires_grad_(True)
    _, u_X = solver.model.forward(X_req, selection_mode="hard")
    grad_u = torch.autograd.grad(u_X.sum(), X_req, create_graph=False)[0]
    Y_pred = solver.inverse_kx(X_req.detach(), grad_u.detach())
    return Y_pred


def run_gaussian_1d():
    """Run 1D Gaussian→Gaussian OT experiments for multiple kernels."""
    _setup_randomness_and_threads()

    params = {
        "n": N_SAMPLES,
        "μ_x": MU_X_1D,
        "Σ_x": SIGMA_X_1D,
        "μ_y": MU_Y_1D,
        "Σ_y": SIGMA_Y_1D,
    }
    X, Y, _ = generate_gaussian_pairs(**params)

    sx = SIGMA_X_1D[0, 0].item()
    sy = SIGMA_Y_1D[0, 0].item()
    sigma_ratio = np.sqrt(sy / sx)

    def T_theory(x):
        return sigma_ratio * x - 1.0

    for kernel_1d, inverse_kx in zip(KERNELS_1D, INV_KERNELS):
        name = getattr(kernel_1d, "__name__", str(kernel_1d))
        print(f"\n[RUN 1D] Using kernel: {name}")

        fcot = _build_solver(dim=1, kernel_1d=kernel_1d, inverse_kx=inverse_kx)
        fcot.fit(
            X, Y,
            iters=TRAIN_ITERS,
            print_every=PRINT_EVERY,
            log_every=LOG_EVERY,
            convergence_tol=CONVERGENCE_TOL,
            convergence_patience=CONVERGENCE_PATIENCE,
            force_retrain=FORCE_RETRAIN,
        )

        X_test = torch.linspace(-3, 3, TEST_POINTS_1D).reshape(-1, 1)
        Y_pred = _compute_transport(fcot, X_test)
        Y_true = T_theory(X_test)

        mae = (Y_pred - Y_true).abs().mean().item()
        print(f"[RUN 1D] kernel={name} MAE vs theory = {mae:.4f}")


def run_gaussian_2d_grid():
    """Run 2D grid-like OT experiments for multiple kernels."""
    _setup_randomness_and_threads()

    centers = [-3.0, 0.0, 3.0]
    X, Y = generate_grid_XY(
        n=N_SAMPLES,
        L=RADIUS,
        std=0.45,
        centers=centers,
        force=True,
    )
    X = X.clamp(-RADIUS + 1e-3, RADIUS - 1e-3)
    Y = Y.clamp(-RADIUS + 1e-3, RADIUS - 1e-3)

    for kernel_1d, inverse_kx in zip(KERNELS_1D, INV_KERNELS):
        name = getattr(kernel_1d, "__name__", str(kernel_1d))
        print(f"\n[RUN 2D] Using kernel: {name}")

        fcot = _build_solver(dim=2, kernel_1d=kernel_1d, inverse_kx=inverse_kx)
        fcot.fit(
            X, Y,
            iters=TRAIN_ITERS,
            print_every=PRINT_EVERY,
            log_every=LOG_EVERY,
            convergence_tol=CONVERGENCE_TOL,
            convergence_patience=CONVERGENCE_PATIENCE,
            force_retrain=FORCE_RETRAIN,
        )

        Y_pred = _compute_transport(fcot, X)

        mean_X = X.mean(dim=0)
        mean_Y = Y.mean(dim=0)
        mean_Ypred = Y_pred.mean(dim=0)

        dist_initial = (mean_X - mean_Y).norm().item()
        dist_mapped = (mean_Ypred - mean_Y).norm().item()
        print(
            f"[RUN 2D] kernel={name} "
            f"mean dist initial={dist_initial:.4f}, mapped={dist_mapped:.4f}"
        )


if __name__ == "__main__":
    # run_gaussian_1d()
    run_gaussian_2d_grid()
