import logging
import time

import numpy as np

from nmf_algos.NMF_base import NMFBase
from nmf_algos.utils.algo_utils import calculate_obj_NMF

logger = logging.getLogger(__name__)


def findbar(X, Xgrad, sigma):
    """Return the safeguarded factor used in the Grad-Mult update."""
    assert X.shape == Xgrad.shape

    negative_mask = (Xgrad < 0).astype(float)
    return negative_mask * np.maximum(X, sigma) + (1 - negative_mask) * X


def gradient_factors(X, U, V):
    """Compute gradients with respect to both NMF factors."""
    # X [m,n], U [m, r], V[n, r]
    estimate_error = U @ V.T - X
    Ugrad = estimate_error @ V
    Vgrad = estimate_error.T @ U
    return Ugrad, Vgrad


def gradient_left_factor(X, U, V):
    """Compute the gradient with respect to the left factor U."""
    # X [m,n], U [m, r], V[n, r]
    estimate_error = U @ V.T - X
    return estimate_error @ V


class NMF_GRADMUL(NMFBase):
    def __init__(self, params, method_name="GRADMUL"):
        super().__init__(method_name, params)
        self.method_default_init()
        self.method_config_init(params)
        self.factor_init(params)

    def method_default_init(self):
        self.run_mode = ""
        self.rerun_times = 1
        self.target_error = 0
        self.max_iter = 300000 * 10**20
        self.sigma = 0.01
        self.delta = 1e-20
        self.tol = 1e-30
        self.target_run_time = None

    def factor_init(self, params):
        """Initialize NMF factors."""
        if "U" in params and "V" in params:
            self.U = params["U"].copy()
            self.V = params["V"].copy()
            return

        m, n = self.X.shape
        self.U = np.abs(np.random.rand(m, self.r))
        self.V = np.abs(np.random.rand(n, self.r))

    def _save_run_result(self, file_name, time_list, error_list, save_time_error):
        if save_time_error:
            self.save_factors(
                file_name,
                {"iter_time": time_list, "iter_error": error_list},
            )
        else:
            self.save_factors(file_name)

    def run_within_fixed_time(self, target_run_time, save_time_error=False):
        self.set_params({"target_run_time": target_run_time})

        def f_continue_cond(n_iter, obj, cur_time):
            return (
                n_iter < self.max_iter
                and obj > self.target_error
                and cur_time < self.target_run_time
            )

        for _ in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"target_run_time": target_run_time})
            self.cur_run_id += 1

            self.U, self.V, time_list, error_list = self.core_run(
                f_continue_cond,
                verbose=True,
                save_time_error=save_time_error,
            )

            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_tc.npy"
            self._save_run_result(file_name, time_list, error_list, save_time_error)

    def run_to_target_error(self, target_error, save_time_error=False):
        def f_continue_cond(n_iter, obj, cur_time):
            return n_iter < self.max_iter and obj > target_error

        for _ in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"target_error": target_error})
            self.cur_run_id += 1

            self.U, self.V, time_list, error_list = self.core_run(
                f_continue_cond,
                verbose=True,
                save_time_error=save_time_error,
            )

            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_ec.npy"
            self._save_run_result(file_name, time_list, error_list, save_time_error)

    def basic_run(self, save_time_error=True):
        """Run Grad-Mult with the default fixed-iteration setting."""

        def f_continue_cond(n_iter, obj, cur_time):
            return n_iter < self.max_iter

        for _ in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"max_iter": 1000})
            self.cur_run_id += 1

            self.U, self.V, time_list, error_list = self.core_run(
                f_continue_cond,
                verbose=True,
                save_time_error=save_time_error,
            )

            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_default.npy"
            self._save_run_result(file_name, time_list, error_list, save_time_error)

    def update_one_factor(self, X, U, V):
        """Update the left factor U while fixing V."""
        Ugrad = gradient_left_factor(X, U, V)
        Ubar = findbar(U, Ugrad, self.sigma)
        UbarVTV = Ubar @ (V.T @ V)

        return U - np.multiply(Ugrad, Ubar / np.maximum(UbarVTV, self.delta))

    def one_iter(self, X, U, V, trace_XTX, previous_obj):
        """Run one Grad-Mult iteration."""
        U = self.update_one_factor(X, U, V)
        V = self.update_one_factor(X.T, V, U)

        obj = calculate_obj_NMF(X, U, V, trace_XTX)
        rel = (previous_obj - obj) / previous_obj

        if abs(rel) < self.tol:
            logger.info("Grad-Mult converged.")
            return U, V, obj, True

        return U, V, obj, False

    def core_run(self, f_continue_cond, verbose=True, save_time_error=True):
        start_t = time.time()
        time_list = []
        error_list = []

        X = self.X
        U = self.U.copy()
        V = self.V.copy()

        trace_XTX = np.trace(X.T @ X)
        previous_obj = calculate_obj_NMF(X, U, V, trace_XTX)

        n_iter = 0
        continue_cond = True
        exit_cond = False

        while continue_cond and not exit_cond:
            U, V, obj, exit_cond = self.one_iter(
                X,
                U,
                V,
                trace_XTX,
                previous_obj,
            )

            cur_time = time.time() - start_t
            continue_cond = f_continue_cond(n_iter, obj, cur_time)
            exit_cond = exit_cond or not continue_cond
            n_iter += 1

            if n_iter % 50 == 0 or exit_cond:
                tracker_payload = (
                    {"iter_time": cur_time, "iter_error": obj}
                    if save_time_error
                    else None
                )

                if tracker_payload is None:
                    self.tracker(U, V, self.iter_save_dir, n_iter)
                else:
                    self.tracker(U, V, self.iter_save_dir, n_iter, tracker_payload)

            if verbose and n_iter % 200 == 0:
                logger.info("Grad-Mult iteration %d, loss %.6e.", n_iter, obj)

            if save_time_error:
                time_list.append(cur_time)
                error_list.append(obj)

            previous_obj = obj

        logger.info(
            "Grad-Mult finished after %d iterations with loss %.6e.",
            n_iter,
            obj,
        )

        return U, V, time_list, error_list
