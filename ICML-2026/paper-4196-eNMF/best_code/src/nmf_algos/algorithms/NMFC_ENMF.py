"""eNMC: extension of eNMF for matrix completion with missing entries."""

import logging
import time

import numpy as np
import numpy.linalg as LA

from nmf_algos.utils.linalg_utils import project_error
from nmf_algos.utils.ENMF_utils import move_to_positive_orthant
from .NMF_ENMF import NMF_ENMF

logger = logging.getLogger(__name__)


# softImpute_ALS [Hastie, 2015]
# Computes local minimum of the unconstrained problem. ||M_E(X-UV^T)||
def obj_softimpute_UV(X, U, V, known_mask, lamda):
    """Compute the SoftImpute-ALS objective."""
    residual = known_mask * (X - U @ V.T)
    reconstruction_term = 0.5 * LA.norm(residual) ** 2
    regularization_term = 0.5 * lamda * (LA.norm(U) ** 2 + LA.norm(V) ** 2)

    return (
        reconstruction_term + regularization_term,
        reconstruction_term,
        regularization_term,
    )


def softImpute_ALS(X, U, V, known_mask, lamda=10, max_iter=1000, tol=1e-3):
    """Compute a local minimum of ||M_E(X - UV^T)|| using SoftImpute-ALS."""
    error_u = np.inf
    error_v = np.inf
    n_iter = 0

    while (error_u > tol or error_v > tol) and n_iter < max_iter:
        uv_t = U @ V.T
        x_star = known_mask * (X - uv_t) + uv_t

        v_system = V.T @ V + lamda * np.eye(V.shape[1])
        prev_U = U.copy()
        U = LA.solve(v_system.T, (x_star @ V).T).T

        uv_t = U @ V.T
        x_star = known_mask * (X - uv_t) + uv_t

        u_system = U.T @ U + lamda * np.eye(U.shape[1])
        prev_V = V.copy()
        V = LA.solve(u_system.T, (x_star.T @ U).T).T

        error_u = LA.norm(U - prev_U)
        error_v = LA.norm(V - prev_V)
        n_iter += 1

        if n_iter % 100 == 0:
            obj, _ = project_error(X, U, V, known_mask)
            logger.debug("SoftImpute-ALS iteration %d, error %.6e.", n_iter, obj)

    logger.info("SoftImpute-ALS finished after %d iterations.", n_iter)
    return U, V


class NMFC_ENMF(NMF_ENMF):
    def __init__(self, params, method_name="ENMFC"):
        # Initialize directly from NMFBase through the parent hierarchy without
        # running NMF_ENMF.__init__, because NMFC_ENMF uses a different setup.
        super(NMF_ENMF, self).__init__(method_name, params)
        self.method_default_init()
        self.method_config_init(params)

    def method_default_init(self):
        self.run_mode = ""
        self.rerun_times = 1
        self.dataset_name = "exp"
        self.target_error = 0
        self.eps = 1e-16
        self.target_run_time = None

        self.enmfc_config_init()
        self.intermediate_result_dict = {}

    def enmfc_config_init(self):
        self.admm_config = {
            "rho": 5,
            "epsilon": 1e-4,
            "max_iter": 4000,
            "tau_inc": 1.1,
            "tau_dec": 1.1,
            "mu": 2,
            "rho_mode": 0,
        }
        self.ascent_config = {
            "tol_asc": 0.2,
            "inner_iter_asc": 2,
            "num_steps": 1000,
        }

        self.combined_config_dict = {}
        self.combined_config_dict.update(self.admm_config)
        self.combined_config_dict.update(self.ascent_config)

        for key, value in self.combined_config_dict.items():
            setattr(self, key, value)

    def factor_init(self, params):
        """Initialize factors using SoftImpute-ALS and optional cached rotation."""
        if "known_mask" not in params:
            raise ValueError(
                "NMFC_ENMF requires `known_mask` to indicate observed entries in X."
            )

        self.known_mask = params["known_mask"]

        if "softimpute_U" in params and "softimpute_V" in params:
            self.U_svd = params["softimpute_U"].copy()
            self.V_svd = params["softimpute_V"].copy()
            self.t_svd = params.get("t_svd", 0)
            logger.info("Step 1: loaded SoftImpute factors.")
        else:
            m, n = self.X.shape
            np.random.seed(self.cur_run_id)

            Uinit = np.random.rand(m, self.r)
            Vinit = np.random.rand(n, self.r)

            start_t = time.time()
            self.U_svd, self.V_svd = softImpute_ALS(
                self.X,
                Uinit,
                Vinit,
                self.known_mask,
            )
            self.t_svd = time.time() - start_t

            logger.info(
                "Step 1: initialized factors using SoftImpute-ALS in %.4f seconds.",
                self.t_svd,
            )

        self.trace_XTX = np.trace(self.X.T @ self.X)
        self.svd_error, _ = project_error(
            self.X,
            self.U_svd,
            self.V_svd,
            self.known_mask,
        )

        if "U_rotate" in params and "V_rotate" in params:
            self.U_rotation = params["U_rotate"].copy()
            self.V_rotation = params["V_rotate"].copy()
            self.t_rotate = params.get("t_rotate", 0)
            self.dist_po = params.get("distance_po", None)
            logger.info("Step 2: loaded rotated factors.")
        else:
            self.get_rotation()
            logger.info(
                "Step 2: generated rotated factors in %.4f seconds.",
                self.t_rotate,
            )

    def move_to_PO(self):
        """Step 3: attain feasibility of the rotated factors using PBCD."""
        start_t = time.time()

        self.U_mp, self.V_mp = move_to_positive_orthant(
            self.X,
            self.U_rotation,
            self.V_rotation,
            self.tol_asc,
            self.inner_iter_asc,
            self.num_steps,
            self.dist_po,
            self.known_mask,
        )

        self.t_mp = time.time() - start_t
        self.hitmp_error, _ = project_error(
            self.X,
            self.U_mp,
            self.V_mp,
            self.known_mask,
        )

        logger.info(
            "Step 3: moved factors to positive orthant in %.4f seconds.",
            self.t_mp,
        )

    def store_intermedia_results(self):
        softimpute_result = {
            "U_softimpute": self.U_svd,
            "V_softimpute": self.V_svd,
            "softimpute_error": self.svd_error,
        }
        rotated_result = {
            "U_rotate": self.U_rotation,
            "V_rotate": self.V_rotation,
            "t_rotate": self.t_rotate,
            "distance_po": self.dist_po,
        }
        mp_result = {
            "U_mp": self.U_mp,
            "V_mp": self.V_mp,
            "t_mp": self.t_mp,
            "hitmp_error": self.hitmp_error,
            "enmf_error": self.enmf_error,
            "total_time": self.total_runtime,
        }

        self.intermediate_result_dict.update(softimpute_result)
        self.intermediate_result_dict.update(rotated_result)
        self.intermediate_result_dict.update(mp_result)

    def core_run(self):
        self.move_to_PO()

        self.U = self.U_mp
        self.V = self.V_mp
        self.total_runtime = self.t_mp + self.t_svd + self.t_rotate

        self.enmf_error, _ = project_error(
            self.X,
            self.U_mp,
            self.V_mp,
            self.known_mask,
        )

        self.store_intermedia_results()

        file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_default.npy"
        self.save_factors(file_name, self.intermediate_result_dict)

    def basic_run(self):
        """Run ENMFC with SoftImpute initialization and positive-orthant projection."""
        for _ in range(self.rerun_times):
            self.reset_status(self.params)
            self.cur_run_id += 1
            self.core_run()
