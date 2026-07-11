import logging
import time

import numpy as np

from nmf_algos.NMF_base import NMFBase
from nmf_algos.utils.algo_utils import calculate_obj_NMF, HALS_iter_solver
from nmf_algos.utils.linalg_utils import normalize_column_pair

logger = logging.getLogger(__name__)


class NMF_HALS(NMFBase):
    def __init__(self, params, method_name="HALS"):
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
        self.max_iter = 300000 * 10**20
        self.target_run_time = None

    def factor_init(self, params):
        """Initialize NMF factors."""
        if "U" in params and "V" in params:
            self.U = params["U"].copy()
            self.V = params["V"].copy()
            return

        m, n = self.X.shape
        np.random.seed(self.cur_run_id)

        self.U = np.abs(np.random.rand(m, self.r))
        self.V = np.abs(np.random.rand(n, self.r))
        self.U, self.V, _ = normalize_column_pair(self.U, self.V)

    def _save_run_result(self, file_name, time_list, error_list, save_time_error):
        if save_time_error:
            self.save_factors(
                file_name,
                {"iter_time": time_list, "iter_error": error_list},
            )
        else:
            self.save_factors(file_name)

    def run_within_fixed_time(self, target_run_time, save_time_error=False):
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
        """Run HALS with the default fixed-iteration setting."""

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

    def one_iter(self, X, U, V, trace_XTX, previous_obj=None):
        """Run one HALS iteration."""
        U, V = HALS_iter_solver(X, U, V, self.r, self.eps)
        obj = calculate_obj_NMF(X, U, V, trace_XTX)
        return U, V, obj, False

    def core_run(self, f_continue_cond, verbose=True, save_time_error=True):
        start_t = time.time()
        time_list = []
        error_list = []

        X = self.X
        U = self.U.copy()
        V = self.V.copy()

        trace_XTX = np.trace(X.T @ X)
        obj = calculate_obj_NMF(X, U, V, trace_XTX)

        n_iter = 0
        continue_cond = True
        exit_cond = False

        while continue_cond and not exit_cond:
            U, V, obj, exit_cond = self.one_iter(X, U, V, trace_XTX)

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
                logger.info("HALS iteration %d, loss %.6e.", n_iter, obj)

            if save_time_error:
                time_list.append(cur_time)
                error_list.append(obj)

        logger.info("HALS finished after %d iterations with loss %.6e.", n_iter, obj)

        return U, V, time_list, error_list
