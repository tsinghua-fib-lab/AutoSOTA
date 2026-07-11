import time
import sys
import logging

import numpy as np
from numpy import linalg as LA

from nmf_algos.NMF_base import NMFBase
from nmf_algos.utils.ENMF_utils import (
    gen_svd_sol,
    admm_rotation,
    move_to_positive_orthant,
    HALS_pos,
)
from nmf_algos.utils.algo_utils import calculate_obj_NMF

logger = logging.getLogger(__name__)


class NMF_ENMF(NMFBase):
    def __init__(self, params, method_name="ENMF"):
        super().__init__(method_name, params)
        self.method_default_init()
        self.method_config_init(params)

    def method_default_init(self):
        self.run_mode = ""
        self.rerun_times = 1
        self.dataset_name = "exp"
        self.target_error = 0
        self.target_run_time = 3600
        self.mu = 2
        self.rho_mode = 1
        self.normalize_data = False
        self.scale = 1.0

        self.enmf_config_init()
        self.intermediate_result_dict = {}

    def data_normalization(self):
        gap = float(np.max(self.X) - np.min(self.X))
        if gap <= 0:
            raise ValueError("Cannot normalize data because max(X) equals min(X).")

        self.scaled_X = self.X / gap
        self.scale = gap

        logger.info(
            "Normalized data with min=%s, max=%s, scale=%s.",
            np.min(self.X),
            np.max(self.X),
            self.scale,
        )

    def enmf_config_init(self):
        self.admm_config = {
            "rho": 10,
            "epsilon": 1e-4,
            "max_iter": 1000,
            "tau_inc": 1.1,
            "tau_dec": 1.1,
            "alpha": 1.5,
        }
        self.ascent_config = {
            "tol_asc": 0.2,
            "inner_iter_asc": 2,
            "num_steps": 100,
        }
        self.descent_config = {
            "hals_rounds": 10**2,
        }

        self.combined_config_dict = {}
        self.combined_config_dict.update(self.admm_config)
        self.combined_config_dict.update(self.ascent_config)
        self.combined_config_dict.update(self.descent_config)

        for key, value in self.combined_config_dict.items():
            setattr(self, key, value)

    def factor_init(self, params):
        """
        Initialize eNMF factors by either loading cached SVD/rotation results
        or computing them from the input matrix.
        """
        if self.normalize_data:
            self.data_normalization()
            self.X = self.scaled_X
        else:
            self.scaled_X = self.X

        if "U_eig" in params and "V_eig" in params:
            self.U_svd = params["U_eig"]
            self.V_svd = params["V_eig"]
            self.t_svd = params.get("t_svd", 0)
            logger.info("Step 1: loaded SVD factors.")
        else:
            self.get_svd()
            logger.info("Step 1: generated SVD factors in %.4f seconds.", self.t_svd)

        # Use LA.norm instead of calculate_obj_NMF to avoid numerical issues
        # when the SVD reconstruction error is close to zero.
        self.svd_error = LA.norm(self.X - self.U_svd @ self.V_svd.T)

        if "U_rotate" in params and "V_rotate" in params:
            self.U_rotation = params["U_rotate"]
            self.V_rotation = params["V_rotate"]
            self.t_rotate = params.get("t_rotate", 0)
            self.dist_po = params.get("distance_po", None)
            logger.info("Step 2: loaded rotated factors.")
        else:
            self.get_rotation()
            logger.info(
                "Step 2: generated rotated factors in %.4f seconds.",
                self.t_rotate,
            )

    def store_intermedia_results(self):
        svd_dict = {
            "U_eig": self.U_svd,
            "V_eig": self.V_svd,
            "t_svd": self.t_svd,
            "svd_error": self.svd_error,
        }
        rotated_dict = {
            "U_rotate": self.U_rotation,
            "V_rotate": self.V_rotation,
            "t_rotate": self.t_rotate,
            "distance_po": self.dist_po,
        }
        mp_dict = {
            "U_mp": self.U_mp,
            "V_mp": self.V_mp,
            "t_mp": self.t_mp,
            "hitmp_error": self.hitmp_error,
        }
        descent_dict = {
            "U_nmf": self.U_nmf,
            "V_nmf": self.V_nmf,
            "t_nmf": self.t_descent,
            "enmf_error": self.enmf_error,
            "total_time": self.total_runtime,
            "data_scale": self.scale,
        }

        self.intermediate_result_dict.update(svd_dict)
        self.intermediate_result_dict.update(rotated_dict)
        self.intermediate_result_dict.update(mp_dict)
        self.intermediate_result_dict.update(descent_dict)

    def get_svd(self):
        """
        Step 1: Obtain the SVD solution as the initial low-rank factors.
        """
        start_t = time.time()
        self.U_svd, self.V_svd = gen_svd_sol(self.X, self.r)
        self.t_svd = time.time() - start_t

    def get_rotation(self):
        """
        Step 2: Compute the rotated SVD solution closest to the positive orthant.
        """
        W = np.vstack((self.U_svd, self.V_svd))

        start_t = time.time()
        res_R, obj_f1 = admm_rotation(
            W,
            self.rho,
            self.epsilon,
            self.max_iter,
            self.tau_inc,
            self.tau_dec,
            self.mu,
            self.rho_mode,
            self.alpha,
        )
        self.t_rotate = time.time() - start_t

        self.U_rotation = self.U_svd @ res_R
        self.V_rotation = self.V_svd @ res_R
        self.dist_po = obj_f1

        logger.info(
            "Distance to positive orthant after rotation: %.6e.",
            self.dist_po,
        )

    def move_to_PO(self):
        """
        Step 3: Attain feasibility of the rotated factors using PBCD.
        """
        start_t = time.time()
        self.U_mp, self.V_mp = move_to_positive_orthant(
            self.X,
            self.U_rotation,
            self.V_rotation,
            self.tol_asc,
            self.inner_iter_asc,
            self.num_steps,
            self.dist_po,
        )
        self.t_mp = time.time() - start_t

        self.hitmp_error = calculate_obj_NMF(
            self.X,
            self.U_mp,
            self.V_mp,
            self.trace_XTX,
        )

        logger.info(
            "Step 3: moved factors to positive orthant in %.4f seconds.",
            self.t_mp,
        )

    def descend_to_enmf(self):
        """
        Step 4: Refine the feasible factors using HALS.
        """
        start_t = time.time()

        if abs(self.hitmp_error - self.svd_error) > 1e-4:
            hals_target_run_time = (
                self.target_run_time - self.t_svd - self.t_mp - self.t_rotate
            )
            self.U_nmf, self.V_nmf = HALS_pos(
                self.X,
                self.trace_XTX,
                self.U_mp,
                self.V_mp,
                self.r,
                self.hals_rounds,
                hals_target_run_time,
                self.target_error,
            )
        else:
            self.U_nmf, self.V_nmf = self.U_mp, self.V_mp
            logger.info(
                "Step 4: skipped HALS because the positive-orthant error is close to the SVD error."
            )

        self.t_descent = time.time() - start_t
        self.total_runtime = self.t_descent + self.t_mp + self.t_svd + self.t_rotate

        self.enmf_error = calculate_obj_NMF(
            self.X,
            self.U_nmf,
            self.V_nmf,
            self.trace_XTX,
        )

        logger.info(
            "Step 4: completed HALS refinement in %.4f seconds. Final error: %.6e.",
            self.t_descent,
            self.enmf_error,
        )

    def core_run(self):
        self.trace_XTX = np.trace(self.X.T @ self.X)
        self.move_to_PO()
        self.descend_to_enmf()

        self.U = self.U_nmf
        self.V = self.V_nmf

        # PROFILING: print per-phase timing to stderr (stdout is redirected by eval)
        print(f'[PROFILE] svd={self.t_svd:.3f}s rotate={self.t_rotate:.3f}s ascent={self.t_mp:.3f}s descent={self.t_descent:.3f}s total={self.total_runtime:.3f}s', file=sys.stderr)

        if self.normalize_data:
            self.rescale_result()

    def rescale_result(self):
        self.X = self.scaled_X * self.scale
        self.U = self.U * self.scale
        self.U_nmf = self.U_nmf * self.scale
        self.trace_XTX = np.trace(self.X.T @ self.X)

        self.enmf_error = calculate_obj_NMF(
            self.X,
            self.U_nmf,
            self.V_nmf,
            self.trace_XTX,
        )

    def basic_run(self):
        for _ in range(self.rerun_times):
            self.reset_status(self.params)
            self.cur_run_id += 1
            self.core_run()

            file_name = f"{self.method_name}_{self.dataset_name}_r_{self.r}_default.npy"
            self.store_intermedia_results()
            self.save_factors(file_name, self.intermediate_result_dict)

    def run_to_target_error(self, target_error, save_time_error=False):
        for _ in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"target_error": target_error, "hals_rounds": 10**10})
            self.cur_run_id += 1
            self.core_run()

            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_ec.npy"
            self.store_intermedia_results()
            self.save_factors(file_name, self.intermediate_result_dict)

    def run_within_fixed_time(self, target_run_time, save_time_error=False):
        for _ in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params(
                {
                    "target_run_time": target_run_time,
                    "hals_rounds": 10**10,
                }
            )
            self.cur_run_id += 1
            self.core_run()

            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_tc.npy"
            self.store_intermedia_results()
            self.save_factors(file_name, self.intermediate_result_dict)
