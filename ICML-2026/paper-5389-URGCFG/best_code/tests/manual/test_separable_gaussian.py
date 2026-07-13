"""
Manual end-to-end test for FCOTSeparable on 1D Gaussian → Gaussian optimal transport.

This test is meant for **manual debugging, regression checks, and performance tuning**.
All tuning knobs are grouped in a CONFIG BLOCK at the top, each parameter is explained.

Run manually with:

    pytest tests/manual/test_separable_gaussian.py -v -s
"""

import unittest
import os
import logging
import torch
import numpy as np
from optimal_transport.ot_fc_sep_map import FCOTSeparable
from tools.dgps import generate_gaussian_pairs, generate_grid_XY
from tools.utils import (
    L22_1d,
    L33_1d,
    L44_1d,
    L55_1d,
    inverse_L22x,
    inverse_L33x,
    inverse_L44x,
    inverse_L55x,
)
from tools.feedback import set_log_level

class TestGaussianEndToEnd(unittest.TestCase):
    # =====================================================================
    #                         CLASS-WIDE CONFIG
    # =====================================================================
    # Randomness / Logging
    SEED = 42
    LOG_LEVEL = "DEBUG"
    ENABLE_LOG_FILE = True
    LOG_FILE_1D = "test_gaussian_sep_1d.log"
    LOG_FILE_2D = "test_gaussian_sep_2d.log"

    # Data distribution settings
    N_SAMPLES = 2000
    MU_X_1D = torch.tensor([0.0])
    SIGMA_X_1D = torch.tensor([[1.0]])
    MU_Y_1D = torch.tensor([-1.0])
    SIGMA_Y_1D = torch.tensor([[2.0]])

    # Domain / grid resolution (should match generated grids)
    RADIUS = 4.0
    X_ACCURACY = 0.01
    Y_ACCURACY = 0.01

    # Optimizer (intercept optimizer)
    OUTER_LR = 1e-2

    # Softmax temperature schedule
    TEMP_MIN = 1.0
    TEMP_MAX = 60.0
    TEMP_WARMUP_ITERS = 2_500

    # Reactivation and full refresh rules
    REACTIVATE_EVERY = 50
    REACTIVATE_EPS = 1e-3
    FULL_REFRESH_EVERY = 250

    # Coarse-to-fine transform speedup
    COARSE_X_FACTOR = 40
    COARSE_TOP_K = 4
    COARSE_WINDOW = 1

    # Training / evaluation parameters
    TRAIN_ITERS = 4_000
    PRINT_EVERY = 10
    LOG_EVERY = 10
    CONVERGENCE_TOL = 1e-4
    CONVERGENCE_PATIENCE = 100
    FORCE_RETRAIN = False

    # 1D evaluation settings
    TEST_POINTS_1D = 25
    MONOTONIC_TOL = -0.5
    MAE_THR = 1.5

    # Kernel families for different p
    KERNELS_1D = [L22_1d, L33_1d, L44_1d, L55_1d]
    INV_KERNELS = [inverse_L22x, inverse_L33x, inverse_L44x, inverse_L55x]

    # =====================================================================
    #                          Helper methods
    # =====================================================================
    def _setup_randomness_and_threads(self):
        torch.manual_seed(self.SEED)
        set_log_level(self.LOG_LEVEL)

        cpu = max(1, (os.cpu_count() or 1) // 2)
        try:
            torch.set_num_threads(cpu)
            torch.set_num_interop_threads(max(1, cpu // 2))
        except RuntimeError:
            pass

    def _setup_log_file(self, log_file: str):
        if not self.ENABLE_LOG_FILE:
            return
        root_logger = logging.getLogger()
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
        ))
        root_logger.addHandler(fh)
        self.addCleanup(lambda: (root_logger.removeHandler(fh), fh.close()))

    def _build_solver(self, dim: int, kernel_1d, inverse_kx) -> FCOTSeparable:
        ny = int((2 * self.RADIUS) / self.Y_ACCURACY) + 1
        n_params = ny * dim

        fcot = FCOTSeparable.initialize_right_architecture(
            dim=dim,
            radius=self.RADIUS,
            n_params=n_params,
            x_accuracy=self.X_ACCURACY,
            kernel_1d=kernel_1d,
            inverse_kx=inverse_kx,
            outer_lr=self.OUTER_LR,
            temp_min=self.TEMP_MIN,
            temp_max=self.TEMP_MAX,
            temp_warmup_iters=self.TEMP_WARMUP_ITERS,
            reactivate_every=self.REACTIVATE_EVERY,
            reactivate_eps=self.REACTIVATE_EPS,
            full_refresh_every=self.FULL_REFRESH_EVERY,
            cache_gradients=True,
            coarse_x_factor=self.COARSE_X_FACTOR,
            coarse_top_k=self.COARSE_TOP_K,
            coarse_window=self.COARSE_WINDOW,
        )
        return fcot

    def _compute_transport(self, solver: FCOTSeparable, X: torch.Tensor) -> torch.Tensor:
        X_req = X.to(solver.device).requires_grad_(True)
        _, u_X = solver.model.forward(X_req, selection_mode="hard")
        grad_u = torch.autograd.grad(u_X.sum(), X_req, create_graph=False)[0]
        Y_pred = solver.inverse_kx(X_req.detach(), grad_u.detach())
        return Y_pred

    # =====================================================================
    #                         1D Gaussian test
    # =====================================================================
    def test_gaussian_transport_1d(self):
        self._setup_randomness_and_threads()
        self._setup_log_file(self.LOG_FILE_1D)

        # Generate 1D Gaussian samples
        params = {
            "n": self.N_SAMPLES,
            "μ_x": self.MU_X_1D,
            "Σ_x": self.SIGMA_X_1D,
            "μ_y": self.MU_Y_1D,
            "Σ_y": self.SIGMA_Y_1D,
        }
        X, Y, _ = generate_gaussian_pairs(**params)

        # Closed-form transport map for 1D Gaussian → Gaussian
        sx = self.SIGMA_X_1D[0, 0].item()
        sy = self.SIGMA_Y_1D[0, 0].item()
        sigma_ratio = np.sqrt(sy / sx)

        def T_theory(x):
            return sigma_ratio * x - 1.0

        # Loop over different kernels / inverse maps
        for kernel_1d, inverse_kx in zip(self.KERNELS_1D, self.INV_KERNELS):
            name = getattr(kernel_1d, "__name__", str(kernel_1d))
            print(f"\n[GAUSSIAN 1D TEST] Using kernel: {name}")

            fcot = self._build_solver(dim=1, kernel_1d=kernel_1d, inverse_kx=inverse_kx)
            fcot.fit(
                X, Y,
                iters=self.TRAIN_ITERS,
                print_every=self.PRINT_EVERY,
                log_every=self.LOG_EVERY,
                convergence_tol=self.CONVERGENCE_TOL,
                convergence_patience=self.CONVERGENCE_PATIENCE,
                force_retrain=self.FORCE_RETRAIN,
            )

            # Evaluate transport quality on a grid
            X_test = torch.linspace(-3, 3, self.TEST_POINTS_1D).reshape(-1, 1)
            Y_pred = self._compute_transport(fcot, X_test)
            Y_true = T_theory(X_test)

            # Basic sanity checks
            self.assertFalse(torch.isnan(Y_pred).any())
            self.assertFalse(torch.isinf(Y_pred).any())
            self.assertEqual(Y_pred.shape, X_test.shape)

            # Mean absolute error vs theory
            mae = (Y_pred - Y_true).abs().mean().item()
            print(f"[GAUSSIAN 1D TEST] kernel={name} MAE = {mae:.4f}")
            self.assertLess(mae, self.MAE_THR, f"Transport MAE too high for {name}: {mae:.4f}")

            # Monotonicity (OT in 1D is monotone)
            diffs = Y_pred[1:] - Y_pred[:-1]
            monotone = (diffs > self.MONOTONIC_TOL).float().mean().item()
            print(f"[GAUSSIAN 1D TEST] kernel={name} Monotonicity = {monotone:.2%}")
            self.assertGreater(monotone, 0.8)

        print("✓ Gaussian 1D end-to-end tests PASSED for all kernels.")

    # =====================================================================
    #                         2D mixture test
    # =====================================================================
    def test_gaussian_transport_2d(self):
        self._setup_randomness_and_threads()
        self._setup_log_file(self.LOG_FILE_2D)

        centers = [-3.0, 0.0, 3.0]
        X, Y = generate_grid_XY(
            n=self.N_SAMPLES,
            L=self.RADIUS,
            std=0.45,
            centers=centers,
            force=True,
        )
        X = X.clamp(-self.RADIUS + 1e-3, self.RADIUS - 1e-3)
        Y = Y.clamp(-self.RADIUS + 1e-3, self.RADIUS - 1e-3)

        # Loop over different kernels / inverse maps
        for kernel_1d, inverse_kx in zip(self.KERNELS_1D, self.INV_KERNELS):
            name = getattr(kernel_1d, "__name__", str(kernel_1d))
            print(f"\n[GAUSSIAN 2D TEST] Using kernel: {name}")

            fcot = self._build_solver(dim=2, kernel_1d=kernel_1d, inverse_kx=inverse_kx)
            fcot.fit(
                X, Y,
                iters=self.TRAIN_ITERS,
                print_every=self.PRINT_EVERY,
                log_every=self.LOG_EVERY,
                convergence_tol=self.CONVERGENCE_TOL,
                convergence_patience=self.CONVERGENCE_PATIENCE,
                force_retrain=self.FORCE_RETRAIN,
            )

            # Evaluate transport on training X
            Y_pred = self._compute_transport(fcot, X)

            # Basic sanity checks
            self.assertFalse(torch.isnan(Y_pred).any())
            self.assertFalse(torch.isinf(Y_pred).any())
            self.assertEqual(Y_pred.shape, X.shape)

            # Check that transport moves X closer (in mean) to Y than identity
            mean_X = X.mean(dim=0)
            mean_Y = Y.mean(dim=0)
            mean_Ypred = Y_pred.mean(dim=0)

            dist_initial = (mean_X - mean_Y).norm().item()
            dist_mapped = (mean_Ypred - mean_Y).norm().item()
            print(
                f"[GAUSSIAN 2D TEST] kernel={name} "
                f"mean dist initial={dist_initial:.4f}, mapped={dist_mapped:.4f}"
            )
            self.assertLess(
                dist_mapped,
                dist_initial,
                f"Mapped distribution mean is not closer to target mean than source for {name}.",
            )

        print("✓ Gaussian 2D mixture tests PASSED for all kernels.")


if __name__ == "__main__":
    unittest.main()
