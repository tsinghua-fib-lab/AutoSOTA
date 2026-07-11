import logging
import time

import numpy as np
import numpy.linalg as LA

from nmf_algos.NMF_base import NMFBase
from nmf_algos.utils.algo_utils import NLS, compute_grad
from nmf_algos.utils.linalg_utils import get_l2_error

logger = logging.getLogger(__name__)


class NMF_ALS(NMFBase):
    def __init__(self, params, method_name="ALS"):
        super().__init__(method_name, params)
        self.method_default_init()
        self.method_config_init(params)
        self.factor_init(params)

    def method_default_init(self):
        self.run_mode = ""
        self.rerun_times = 1
        self.dataset_name = "exp"
        self.target_error = 0
        self.max_iter = 300000 * 10**20
        self.als_tol = 1e-30
        self.als_ratio = 1
        self.target_run_time = None

    def factor_init(self, params):
        """Initialize NMF factors."""
        if "U" in params and "V" in params:
            self.U = params["U"]
            self.V = params["V"]
            return

        m, n = self.X.shape
        self.U = np.abs(np.random.rand(m, self.r))
        self.V = np.abs(np.random.rand(n, self.r))

    def run_within_fixed_time(self, target_run_time, save_time_error=False):
        def f_continue_cond(n_iter, obj, cur_time):
            iter_cond = n_iter < self.max_iter
            error_cond = self.als_ratio * obj > self.target_error
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
            error_cond = self.als_ratio * obj > target_error
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

    def one_iter(self, X, U, V, gradU, gradV, tolU, tolV, min_projnorm):
        """Run one ALS iteration using projected-gradient stopping."""
        sel_gradU = gradU[(gradU < 0) | (U > 0)]
        sel_gradV = gradV[(gradV < 0) | (V > 0)]

        projnorm = LA.norm(
            np.vstack(
                (
                    sel_gradU.reshape(-1, 1),
                    sel_gradV.reshape(-1, 1),
                )
            )
        )

        if projnorm < min_projnorm:
            obj = get_l2_error(X, U, V.T)
            logger.info("ALS stopped because the projected-gradient norm is satisfied.")
            return U, V, gradU, gradV, tolU, tolV, obj, True

        U, gradU, iterU = NLS(X.T, V.T, U.T, tolU, 1000)
        U = U.T
        gradU = gradU.T

        if iterU == 0:
            tolU *= 0.1

        V, gradV, iterV = NLS(X, U, V, tolV, 1000)
        obj = get_l2_error(X, U, V.T)

        if iterV == 0:
            tolV *= 0.1

        return U, V, gradU, gradV, tolU, tolV, obj, False

    def core_run(self, f_continue_cond, verbose=True, save_time_error=True):
        start_t = time.time()
        time_list = []
        error_list = []

        X = self.X
        U = self.U.copy()
        V = self.V.T.copy()

        gradU, gradV, initgrad = compute_grad(X, U, V)
        obj = get_l2_error(X, U, V.T)

        n_iter = 0
        tolU = max(0.001, self.als_tol) * initgrad
        tolV = tolU
        min_projnorm = self.als_tol * initgrad

        continue_cond = True
        exit_cond = False

        while continue_cond and not exit_cond:
            U, V, gradU, gradV, tolU, tolV, obj, exit_cond = self.one_iter(
                X,
                U,
                V,
                gradU,
                gradV,
                tolU,
                tolV,
                min_projnorm,
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
                logger.info("ALS iteration %d, loss %.6e.", n_iter, obj)

            if save_time_error:
                time_list.append(cur_time)
                error_list.append(obj)

        logger.info("ALS finished after %d iterations with loss %.6e.", n_iter, obj)

        return U, V.T, time_list, error_list
