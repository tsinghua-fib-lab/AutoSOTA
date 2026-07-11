"""ADM baseline for nonnegative matrix completion with missing entries.

Based on the alternating direction method in Xu et al. (2012).
"""

import logging
import time

import numpy as np
import numpy.linalg as LA

from nmf_algos.NMF_base import NMFBase
from nmf_algos.utils.linalg_utils import project_error

logger = logging.getLogger(__name__)


class NMFC_ADM(NMFBase):
    def __init__(self, params, method_name="ADM"):
        super().__init__(method_name, params)
        self.method_default_init()
        self.method_config_init(params)
        self.factor_init(params)

    def method_default_init(self):
        self.run_mode = ""
        self.rerun_times = 1
        self.dataset_name = "exp"
        self.target_error = 0
        self.eps = 1e-16
        self.max_iter = 1000
        self.target_run_time = None
        self.tol = 1e-3

    def factor_init(self, params):
        """Initialize factors and ADM-specific parameters."""
        if "known_mask" not in params:
            raise ValueError(
                "NMFC_ADM requires `known_mask` to indicate observed entries in X."
            )

        self.known_mask = params["known_mask"]
        m, n = self.X.shape

        if "U" in params and "V" in params:
            self.U = params["U"].copy()
            self.V = params["V"].copy()
        else:
            np.random.seed(self.cur_run_id)
            self.U = np.abs(np.random.rand(m, self.r))
            self.V = np.abs(np.random.rand(n, self.r))

        self.gamma = 1.618
        self.alpha = (50 * max(m, n)) / self.r
        self.beta = (self.alpha * n) / m

        # Stabilization term for the linear solves.
        self.lamda = 10

    def _save_run_result(self, file_name, time_list, error_list, save_time_error):
        if save_time_error:
            self.save_factors(
                file_name,
                {"iter_time": time_list, "iter_error": error_list},
            )
        else:
            self.save_factors(file_name)

    def basic_run(self, save_time_error=True):
        """Run ADM with the default fixed-iteration setting."""

        def f_continue_cond(n_iter, obj, cur_time):
            return n_iter < self.max_iter

        for _ in range(self.rerun_times):
            self.reset_status(self.params)
            self.cur_run_id += 1

            self.U, self.V, time_list, error_list = self.core_run(
                f_continue_cond,
                verbose=True,
                save_time_error=save_time_error,
            )

            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_default.npy"
            self._save_run_result(file_name, time_list, error_list, save_time_error)

    def one_iter(self, A, X, Y, Z, U, V, Delta, Kappa):
        """Run one ADM iteration.

        Notation follows the original ADM matrix-completion formulation:
        A is the observed data matrix, and X/Y are factor variables.
        """
        f_prev = self._masked_relative_residual(A, X, Y)

        # Update X.
        x_rhs = Z @ Y.T + self.alpha * U - Delta
        x_system = Y @ Y.T + (self.alpha + self.lamda) * np.eye(Y.shape[0])
        X = LA.solve(x_system.T, x_rhs.T).T

        # Update Y.
        y_system = X.T @ X + (self.beta + self.lamda) * np.eye(X.shape[1])
        y_rhs = X.T @ Z + self.beta * V - Kappa
        Y = LA.solve(y_system, y_rhs)

        # Update Z by preserving observed entries.
        XY = X @ Y
        Z = XY + self.known_mask * (A - XY)

        # Project auxiliary variables onto the nonnegative orthant.
        U = np.maximum(X + Delta / self.alpha, 0)
        V = np.maximum(Y + Kappa / self.beta, 0)

        # Dual updates.
        Delta = Delta + self.gamma * self.alpha * (X - U)
        Kappa = Kappa + self.gamma * self.beta * (Y - V)

        f_cur = self._masked_relative_residual(A, X, Y)

        error1 = abs(f_cur - f_prev) / max(1, abs(f_prev))
        error2 = f_prev
        exit_condition = (error1 <= self.tol) and (error2 <= self.tol)

        obj, _ = project_error(A, X, Y.T, self.known_mask)

        return X, Y, Z, U, V, Delta, Kappa, obj, exit_condition

    def core_run(self, f_continue_cond, verbose=True, save_time_error=True):
        start_t = time.time()
        time_list = []
        error_list = []

        # Rename variables to match the original ADM algorithm.
        A = self.X
        X = self.U.copy()
        Y = self.V.T.copy()
        Z = self.X.copy()

        m, n = self.X.shape
        U = np.zeros((m, self.r))
        V = np.zeros((self.r, n))
        Delta = np.zeros((m, self.r))
        Kappa = np.zeros((self.r, n))

        n_iter = 0
        continue_cond = True
        exit_cond = False
        obj = np.inf

        while continue_cond and not exit_cond:
            X, Y, Z, U, V, Delta, Kappa, obj, exit_cond = self.one_iter(
                A,
                X,
                Y,
                Z,
                U,
                V,
                Delta,
                Kappa,
            )

            cur_time = time.time() - start_t
            continue_cond = f_continue_cond(n_iter, obj, cur_time)
            exit_cond = exit_cond or not continue_cond
            n_iter += 1

            cur_left_factor = X
            cur_right_factor = Y.T

            if n_iter % 50 == 0 or exit_cond:
                tracker_payload = (
                    {"iter_time": cur_time, "iter_error": obj}
                    if save_time_error
                    else None
                )

                if tracker_payload is None:
                    self.tracker(
                        cur_left_factor,
                        cur_right_factor,
                        self.iter_save_dir,
                        n_iter,
                    )
                else:
                    self.tracker(
                        cur_left_factor,
                        cur_right_factor,
                        self.iter_save_dir,
                        n_iter,
                        tracker_payload,
                    )

            if verbose and n_iter % 200 == 0:
                logger.info("ADM iteration %d, loss %.6e.", n_iter, obj)

            if save_time_error:
                time_list.append(cur_time)
                error_list.append(obj)

        logger.info("ADM finished after %d iterations with loss %.6e.", n_iter, obj)

        return X, Y.T, time_list, error_list

    def _masked_relative_residual(self, A, X, Y):
        """Compute relative reconstruction residual on observed entries."""
        numerator = LA.norm(self.known_mask * (X @ Y - A))
        denominator = max(LA.norm(A), 1e-12)
        return numerator / denominator
