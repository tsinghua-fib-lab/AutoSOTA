"""Our refined version of ADMM for NMF problem from Hajinezhad 2016 based on Boyd 2011."""

import logging
import time

import numpy as np
from numpy import linalg as LA

from nmf_algos.NMF_base import NMFBase
from nmf_algos.utils.algo_utils import calculate_obj_NMF

logger = logging.getLogger(__name__)


class NMF_ADMM(NMFBase):
    def __init__(self, params, method_name="ADMM"):
        super().__init__(method_name, params)
        self.method_default_init()
        self.method_config_init(params)

    def reset_status(self, params):
        """Reset algorithm status before each rerun."""
        self.method_config_init(params)
        self.factor_init(params)

    def method_default_init(self):
        self.run_mode = ""
        self.rerun_times = 1
        self.dataset_name = "exp"
        self.target_error = 0
        self.max_iter = 20000000
        self.mul_tol = 1e-30
        self.epsilon = 1e-10
        self.delta_Y = np.inf
        self.delta_D = np.inf
        self.rho = 5
        self.target_run_time = None

    def factor_init(self, params):
        """Initialize primal, auxiliary, and dual variables."""
        m, n = self.X.shape
        np.random.seed(self.cur_run_id)

        if "U" in params and "V" in params:
            self.U = params["U"]
            self.V = params["V"]
        else:
            self.U = np.random.rand(m, self.r)
            self.V = np.random.rand(n, self.r)

        self.Ainit = np.random.rand(m, self.r)
        self.Binit = np.random.rand(n, self.r)
        self.Yinit = np.random.rand(m, self.r)
        self.Dinit = np.random.rand(n, self.r)

    def run_within_fixed_time(self, target_run_time, save_time_error=False):
        def f_continue_cond(n_iter, obj, cur_time):
            iter_cond = n_iter < self.max_iter
            error_cond = obj > self.target_error
            time_cond = cur_time < self.target_run_time
            return iter_cond and error_cond and time_cond

        for _ in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"target_run_time": target_run_time})
            self.cur_run_id += 1

            U, V, time_list, error_list = self.core_run(
                f_continue_cond,
                verbose=True,
                save_time_error=save_time_error,
            )
            self.U, self.V = U, V

            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_tc.npy"
            if save_time_error:
                self.save_factors(
                    file_name,
                    {"iter_time": time_list, "iter_error": error_list},
                )
            else:
                self.save_factors(file_name)

    def run_to_target_error(self, target_error, save_time_error=False):
        def f_continue_cond(n_iter, obj, cur_time):
            iter_cond = n_iter < self.max_iter
            error_cond = obj > target_error
            return iter_cond and error_cond

        for _ in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"target_error": target_error})
            self.cur_run_id += 1

            U, V, time_list, error_list = self.core_run(
                f_continue_cond,
                verbose=True,
                save_time_error=save_time_error,
            )
            self.U, self.V = U, V

            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_ec.npy"
            if save_time_error:
                self.save_factors(
                    file_name,
                    {"iter_time": time_list, "iter_error": error_list},
                )
            else:
                self.save_factors(file_name)

    def update_one_factor(self, X, V, A, rho, Y, D):
        """Update one primal factor and its nonnegative auxiliary variable."""
        new_U = A - (1.0 / rho) * Y

        gram = A.T @ A + rho * np.eye(A.shape[1])
        numerator = rho * V + D + X.T @ A

        # Equivalent to numerator @ inv(gram), but numerically preferable.
        new_B = LA.solve(gram.T, numerator.T).T
        new_B = np.maximum(new_B, 0)

        return new_U, new_B

    def one_iter(self, X, U, V, A, B, Y, D, trace_XTX):
        """Run one ADMM iteration."""
        pre_B = B.copy()

        U, B = self.update_one_factor(X, V, A, self.rho, Y, D)
        V, A = self.update_one_factor(X.T, U, B, self.rho, D, Y)

        update_Y = self.rho * (U - A)
        update_D = self.rho * (V - pre_B)

        Y = Y + update_Y
        D = D + update_D

        delta_Y = LA.norm(update_Y)
        delta_D = LA.norm(update_D)
        exit_condition = (delta_Y <= self.epsilon) or (delta_D <= self.epsilon)

        obj = calculate_obj_NMF(X, U, V, trace_XTX)

        return U, V, A, B, Y, D, obj, exit_condition

    def core_run(self, f_continue_cond, verbose=True, save_time_error=True):
        start_t = time.time()
        time_list = []
        error_list = []

        X = self.X

        U = self.U.copy()
        V = self.V.copy()
        A = self.Ainit.copy()
        B = self.Binit.copy()
        Y = self.Yinit.copy()
        D = self.Dinit.copy()

        trace_XTX = np.trace(X.T @ X)

        n_iter = 0
        continue_cond = True
        exit_cond = False

        while continue_cond and not exit_cond:
            U, V, A, B, Y, D, obj, exit_cond = self.one_iter(
                X,
                U,
                V,
                A,
                B,
                Y,
                D,
                trace_XTX,
            )
            cur_time = time.time() - start_t

            continue_cond = f_continue_cond(n_iter, obj, cur_time)
            exit_cond = (not continue_cond) or exit_cond
            n_iter += 1

            if n_iter % 50 == 0 or exit_cond:
                if save_time_error:
                    self.tracker(
                        U,
                        V,
                        self.iter_save_dir,
                        n_iter,
                        {"iter_time": cur_time, "iter_error": obj},
                    )
                else:
                    self.tracker(U, V, self.iter_save_dir, n_iter)

            if verbose and n_iter % 200 == 0:
                logger.info("ADMM iteration %d, loss %.6e.", n_iter, obj)

            if save_time_error:
                time_list.append(cur_time)
                error_list.append(obj)

        if exit_cond:
            logger.info(
                "ADMM finished after %d iterations with loss %.6e.", n_iter, obj
            )

        return U, V, time_list, error_list

    def basic_run(self, save_time_error=True):
        """Run ADMM with the default fixed-iteration setting."""

        def f_continue_cond(n_iter, obj, cur_time):
            return n_iter < self.max_iter

        for _ in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"max_iter": 1000})
            self.cur_run_id += 1

            U, V, time_list, error_list = self.core_run(
                f_continue_cond,
                verbose=True,
                save_time_error=save_time_error,
            )
            self.U, self.V = U, V

            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_default.npy"
            if save_time_error:
                self.save_factors(
                    file_name,
                    {"iter_time": time_list, "iter_error": error_list},
                )
            else:
                self.save_factors(file_name)
