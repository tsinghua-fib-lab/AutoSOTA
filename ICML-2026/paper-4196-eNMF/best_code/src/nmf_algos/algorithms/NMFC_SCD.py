"""SCD baseline for NMF with missing entries.

This implementation follows a sequential coordinate descent update structure
for matrix completion and is similar in spirit to the NNLM R package.
"""

import logging
import time
from multiprocessing import Pool

import numpy as np

from nmf_algos.NMF_base import NMFBase
from nmf_algos.utils.linalg_utils import project_error

logger = logging.getLogger(__name__)

TINY_NUM = 1e-16


# Keep the original notation to make the implementation easier to compare
# with the algorithm described in the reference paper.
def scd_ls_update_column(Hj, mu, WtW, inner_rel_tol, inner_max_iter):
    """Update one column of H using sequential coordinate descent.

    Args:
        Hj: Current column of H with shape [r].
        mu: Residual-related vector with shape [r].
        WtW: Gram matrix W.T @ W with shape [r, r].
        inner_rel_tol: Relative tolerance for the inner coordinate updates.
        inner_max_iter: Maximum number of inner coordinate-descent iterations.

    Returns:
        Updated Hj, updated mu, and final relative update error.
    """
    rel_err = 1.0 + inner_rel_tol
    inner_iter = 0
    r = Hj.shape[0]

    while inner_iter < inner_max_iter and rel_err > inner_rel_tol:
        rel_err = 0.0

        for k in range(r):
            tmp = Hj[k] - mu[k] / WtW[k, k]
            tmp = max(tmp, 0.0)

            if tmp == Hj[k]:
                continue

            mu += WtW[:, k] * (tmp - Hj[k])

            etmp = 2.0 * abs(Hj[k] - tmp) / (tmp + Hj[k] + TINY_NUM)
            rel_err = max(rel_err, etmp)

            Hj[k] = tmp

        inner_iter += 1

    return Hj, mu, rel_err


# Non-parallel version.
# def scd_ls_update_factor(A, W, H, known_mask, inner_rel_tol, inner_max_iter, beta):
#     # ||A - WH||, solve H. beta(1)=0, beta(2) = 0, beta refers to beta(0).
#     # A [m, n], W[m, r], H[r,n], known_mask [m. n]
#     r, n = H.shape
#     for j in range(n):
#         # Describe which row to be masked.
#         #[m,]
#         row_mask = known_mask[:, j]
#         num_known_rows = np.sum(row_mask)
#         if num_known_rows == 0:
#             continue
#         # [g, r]
#         W_known = W[row_mask, :]
#         # [r, r]
#         WtW = W_known.T@W_known
#         WtW.flat[::r+1]+= beta
#         # [g, n]
#         Aj_known = A[:, j][row_mask]
#         # [r, n]
#         mu = W_known.T@Aj_known
#         mu = WtW @ H[:, j] - mu
#         H[:, j], mu, rel_err = scd_ls_update_column(H[:, j], mu, WtW, inner_rel_tol, inner_max_iter)
#     return H, rel_err


def scd_ls_update_column_parallel(args):
    """Parallel wrapper for updating one column of H."""
    Aj, W, Hj, row_mask, beta, inner_rel_tol, inner_max_iter = args

    if not np.any(row_mask):
        return Hj

    r = Hj.shape[0]
    W_known = W[row_mask, :]
    Aj_known = Aj[row_mask]

    WtW = W_known.T @ W_known
    WtW.flat[:: r + 1] += beta

    mu = W_known.T @ Aj_known
    mu = WtW @ Hj - mu

    Hj, _, _ = scd_ls_update_column(
        Hj,
        mu,
        WtW,
        inner_rel_tol,
        inner_max_iter,
    )

    return Hj


def scd_ls_update_factor(A, W, H, known_mask, inner_rel_tol, inner_max_iter, beta):
    """Update H in min ||M_E(A - WH)|| using column-wise SCD updates.

    Args:
        A: Data matrix with shape [m, n].
        W: Left factor with shape [m, r].
        H: Right factor transposed with shape [r, n].
        known_mask: Boolean observation mask with shape [m, n].
    """
    _, n = H.shape

    update_args = [
        (
            A[:, j],
            W,
            H[:, j].copy(),
            known_mask[:, j],
            beta,
            inner_rel_tol,
            inner_max_iter,
        )
        for j in range(n)
    ]

    with Pool() as pool:
        updated_columns = pool.map(scd_ls_update_column_parallel, update_args)

    return np.column_stack(updated_columns), 0


class NMFC_SCD(NMFBase):
    def __init__(self, params, method_name="SCD"):
        super().__init__(method_name, params)
        self.method_default_init()
        self.method_config_init(params)
        self.factor_init(params)

    def method_default_init(self):
        self.run_mode = ""
        self.rerun_times = 1
        self.dataset_name = "exp"
        self.target_error = 0
        self.max_iter = 500
        self.target_run_time = None

        self.beta = 10
        self.rel_tol = 1e-3
        self.inner_max_iter = 10
        self.inner_rel_tol = 1e-8

    def factor_init(self, params):
        """Initialize factors and the known-entry mask."""
        if "known_mask" not in params:
            raise ValueError(
                "NMFC_SCD requires `known_mask` to indicate observed entries in X."
            )

        self.known_mask = params["known_mask"]
        self.bool_known_mask = self.known_mask.astype(bool)

        if "U" in params and "V" in params:
            self.U = params["U"].copy()
            self.V = params["V"].copy()
            return

        np.random.seed(self.cur_run_id)
        m, n = self.X.shape
        self.U = 0.01 * np.random.uniform(size=(m, self.r))
        self.V = 0.01 * np.random.uniform(size=(n, self.r))

    def _save_run_result(self, file_name, time_list, error_list, save_time_error):
        if save_time_error:
            self.save_factors(
                file_name,
                {"iter_time": time_list, "iter_error": error_list},
            )
        else:
            self.save_factors(file_name)

    def _track_iteration(self, U, V, n_iter, cur_time, obj, save_time_error):
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

    def one_iter(self, A, W, H, previous_err):
        """Run one matrix-completion SCD iteration."""
        H, _ = scd_ls_update_factor(
            A,
            W,
            H,
            self.bool_known_mask,
            self.inner_rel_tol,
            self.inner_max_iter,
            self.beta,
        )

        Wt, _ = scd_ls_update_factor(
            A.T,
            H.T,
            W.T,
            self.bool_known_mask.T,
            self.inner_rel_tol,
            self.inner_max_iter,
            self.beta,
        )
        W = Wt.T

        obj, _ = project_error(A, W, H.T, self.known_mask)

        current_err = 0.5 * obj
        rel_err = (
            2.0 * (previous_err - current_err) / (previous_err + current_err + TINY_NUM)
        )
        exit_condition = abs(rel_err) <= self.rel_tol

        return W, H, obj, exit_condition

    def basic_run(self, save_time_error=True):
        """Run NMFC_SCD with the default fixed-iteration setting."""

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

    def core_run(self, f_continue_cond, verbose=True, save_time_error=True):
        start_t = time.time()
        time_list = []
        error_list = []

        previous_obj, _ = project_error(self.X, self.U, self.V, self.known_mask)

        # Rename variables to be consistent with the original algorithm.
        W = self.U.copy()
        H = self.V.T.copy()

        n_iter = 0
        continue_cond = True
        exit_cond = False
        obj = previous_obj

        while continue_cond and not exit_cond:
            W, H, obj, exit_cond = self.one_iter(self.X, W, H, previous_obj)

            cur_time = time.time() - start_t
            continue_cond = f_continue_cond(n_iter, obj, cur_time)
            exit_cond = exit_cond or not continue_cond
            n_iter += 1

            if n_iter % 20 == 0 or exit_cond:
                self._track_iteration(
                    W,
                    H.T,
                    n_iter,
                    cur_time,
                    obj,
                    save_time_error,
                )

            if verbose and n_iter % 200 == 0:
                logger.info("NMFC-SCD iteration %d, loss %.6e.", n_iter, obj)

            if save_time_error:
                time_list.append(cur_time)
                error_list.append(obj)

            previous_obj = obj

        logger.info(
            "NMFC-SCD finished after %d iterations with loss %.6e.",
            n_iter,
            obj,
        )

        return W, H.T, time_list, error_list
