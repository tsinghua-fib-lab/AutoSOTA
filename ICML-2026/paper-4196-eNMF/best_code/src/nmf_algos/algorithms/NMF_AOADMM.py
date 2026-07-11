import os
import time
import copy
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from nmf_algos.utils.utils import load_data_basedon_proto, load_data_matrix
from nmf_algos.utils.algo_utils import calculate_obj_NMF
from nmf_algos.NMF_base import NMFBase


def flip_svd_based_on_norm(U, V, s):
    """
    Given svd solution, decide whether flip the U, V column sign to minimize norm
    """

    def divide_with_mask(a, b, mask):
        return np.divide(a, b, out=np.zeros_like(a), where=mask)

    U_pos = (U + np.abs(U)) / 2
    U_neg = (np.abs(U) - U) / 2
    V_pos = (V + np.abs(V)) / 2
    V_neg = (np.abs(V) - V) / 2
    U_pos_norm = LA.norm(U_pos, axis=0)
    V_pos_norm = LA.norm(V_pos, axis=0)
    U_neg_norm = LA.norm(U_neg, axis=0)
    V_neg_norm = LA.norm(V_neg, axis=0)
    pos_norm = U_pos_norm * V_pos_norm
    neg_norm = U_neg_norm * V_neg_norm
    print("pos_norm shape", pos_norm.shape, U_pos_norm.shape, V_pos_norm.shape)
    print("neg_norm shpae", neg_norm.shape, U_neg_norm.shape, V_neg_norm.shape)

    selection_mask = pos_norm >= neg_norm
    updated_U = np.where(
        selection_mask,
        U_pos * np.sqrt(divide_with_mask(V_pos_norm, U_pos_norm, selection_mask)),
        U_neg * np.sqrt(divide_with_mask(V_neg_norm, U_neg_norm, ~selection_mask)),
    )
    updated_V = np.where(
        selection_mask,
        V_pos * np.sqrt(divide_with_mask(U_pos_norm, V_pos_norm, selection_mask)),
        V_neg * np.sqrt(divide_with_mask(U_neg_norm, V_neg_norm, ~selection_mask)),
    )
    return updated_U * np.sqrt(s), updated_V * np.sqrt(s)


# (TODO) Check whether sparse matrix faster?
def prox_operator(prox_type, mat_aux, dual, *, rho=None, lambda_=None, upper_bound=1):
    if prox_type == "l2n":
        n = mat_aux.shape[0]
        k = -np.array([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], dtype=object)
        offset = [-1, 0, 1]
        tikh = sp.diags(k, offset)
        a = 1 / rho * (lambda_ * tikh.T @ tikh + rho * sp.eye(n))
        b = mat_aux - dual
        mat = spla.spsolve(a, b)
        mat = np.where(mat < 0, 0, mat)
        return mat
    elif prox_type == "nn":
        # proximal operators for nn : non-negativity
        diff = mat_aux - dual
        mat = np.where(diff < 0, 0, diff)
        return mat
    else:
        raise NotImplementedError


class NMF_AOADMM(NMFBase):
    def __init__(self, params, method_name="AOADMM"):
        super().__init__(method_name, params)
        self.method_default_init()
        self.method_config_init(params)
        print("self.save_dir", self.save_dir)
        print("self.iter_save_dir", self.iter_save_dir)
        # set initial factors
        self.factor_init(params)

    def method_default_init(self):
        self.run_mode = ""
        # (TODO) rerun times
        self.rerun_times = 1
        self.dataset_name = "exp"
        self.target_error = 0
        self.max_iter = 300000 * 10 ** (20)
        self.aoadmm_tol1 = 1e-3  # use their default
        self.aoadmm_tol2 = 1e-3
        self.target_run_time = None
        self.reg_u = 0
        self.reg_v = 0
        self.aoadmm_inner_iter = 10  # inside each update

    def factor_init(self, params):
        if "U" not in params:
            """svd based nmf initialization:
            Paper:
                Boutsidis, Gallopoulos: SVD based initialization: A head start for
                nonnegative matrix factorization
            Adapted from https://github.com/raleng/nmf.git
            """

            u, s, v = LA.svd(self.X, full_matrices=False)
            # print("SVD shape u, s, v", u.shape, s.shape, v.shape)
            v = v.T
            updated_U, updated_V = flip_svd_based_on_norm(
                u[:, : self.r], v[:, : self.r], s[: self.r]
            )
            updated_U[:, 0] = np.sqrt(s[0]) * np.abs(u[:, 0])
            updated_V[:, 0] = np.sqrt(s[0]) * np.abs(v[:, 0])
            self.U = updated_U
            self.V = updated_V
        else:
            self.U = params["U"]
            self.V = params["V"]

    def run_within_fixed_time(self, target_run_time, save_time_error=False):
        def f_continue_cond(n_iter, obj, cur_time):
            iter_cond = n_iter < self.max_iter
            error_cond = obj > self.target_error
            time_cond = cur_time < self.target_run_time
            return iter_cond and error_cond and time_cond

        for run_i in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"target_run_time": target_run_time})
            self.cur_run_id += 1

            U, V, time_list, error_list = self.core_run(f_continue_cond, verbose=True)
            self.U, self.V = U, V
            # model_name: MUL_verb:
            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_tc.npy"
            # save time and error for given iters.
            if save_time_error:
                self.save_factors(
                    file_name, {"iter_time": time_list, "iter_error": error_list}
                )
            else:
                self.save_factors(file_name)

    def run_to_target_error(self, target_error, save_time_error=False):
        def f_continue_cond(n_iter, obj, cur_time):
            iter_cond = n_iter < self.max_iter
            error_cond = obj > target_error
            return iter_cond and error_cond

        for run_i in range(self.rerun_times):
            self.reset_status(self.params)
            self.set_params({"target_error": target_error})
            self.cur_run_id += 1

            U, V, time_list, error_list = self.core_run(f_continue_cond, verbose=True)
            self.U, self.V = U, V
            # model_name: MUL_verb:
            file_name = f"{self.model_name}_{self.dataset_name}_r_{self.r}_ec.npy"
            if save_time_error:
                self.save_factors(
                    file_name, {"iter_time": time_list, "iter_error": error_list}
                )
            else:
                self.save_factors(file_name)

    def update_one_factor(self, X, U, V, dual, lambda_, prox_type, admm_iter):
        # def admm_ls_update(y, w, h, dual, k, prox_type='nn', *, admm_iter=10, lambda_=0):
        """ADMM update for NMF subproblem, when one of the factors is fixed
        using least-squares loss
        Here input V[r, n], return V[n,r]
        """

        def terminate(mat, mat_prev, aux, dual, tol=1e-2):
            # relative primal residual
            r = LA.norm(mat - aux) / LA.norm(mat)
            # relative dual residual
            s = LA.norm(mat - mat_prev) / LA.norm(dual)
            if r < tol and s < tol:
                return True
            else:
                return False

        # precompute all the things
        G = U.T @ U
        rho = np.trace(G) / self.r
        cho = sla.cholesky(G + rho * np.eye(G.shape[0]), lower=True)
        UTX = U.T @ X

        for i in range(admm_iter):
            V_aux = sla.cho_solve((cho, True), UTX + rho * (V + dual))
            V_prev = V.copy()
            V = prox_operator(prox_type, V_aux, dual, rho=rho, lambda_=lambda_)
            dual = dual + V - V_aux
            if terminate(V, V_prev, V_aux, dual):
                # print('ADMM break after {} iterations.'.format(i))
                break

        return V.T, dual.T

    def one_iter(self, X, U, V, trace_XTX, dual_U, dual_V, previous_obj):
        ## Update V : X[m, n], U[m,r], V[n,r]
        V, dual_V = self.update_one_factor(
            self.X,
            U,
            V.T,
            dual_V.T,
            lambda_=self.reg_v,
            prox_type="l2n",
            admm_iter=self.aoadmm_inner_iter,
        )
        ## Update U : X[n, m], U[n,r], V[m,r]
        U, dual_U = self.update_one_factor(
            self.X.T,
            V,
            U.T,
            dual_U.T,
            lambda_=self.reg_u,
            prox_type="nn",
            admm_iter=self.aoadmm_inner_iter,
        )

        cur_obj = calculate_obj_NMF(X, U, V, trace_XTX)

        def convergence_check(new, old):
            """Checks the convergence criteria"""
            convergence_break = True
            if new < self.aoadmm_tol1:
                print("Algorithm converged (1).")
            elif new >= old - self.aoadmm_tol2:
                print("Algorithm converged (2).")
            else:
                convergence_break = False
            return convergence_break

        exit_condition = convergence_check(cur_obj, previous_obj)

        return U, V, dual_U, dual_V, cur_obj, exit_condition

    def core_run(self, f_continue_cond, verbose=True, save_time_error=True):
        # default run with tracking of time
        start_t = time.time()
        time_list = []
        error_list = []
        # copy data for further computation
        X = self.X
        trace_XTX = np.trace(X.T @ X)
        U = copy.deepcopy(self.U)
        V = copy.deepcopy(self.V)
        # dual intialization:
        dual_U = np.zeros_like(U)
        dual_V = np.zeros_like(V)

        previous_obj = calculate_obj_NMF(X, U, V, trace_XTX)
        n_iter = 0
        continue_cond = True
        exit_cond = False

        while continue_cond and (not exit_cond):
            U, V, dual_U, dual_V, obj, exit_cond = self.one_iter(
                X, U, V, trace_XTX, dual_U, dual_V, previous_obj
            )
            cur_time = time.time() - start_t
            continue_cond = f_continue_cond(n_iter, obj, cur_time)

            exit_cond = (not continue_cond) or exit_cond
            n_iter += 1
            if (n_iter % 50 == 0) or exit_cond:
                print("Saving, checking exit_conition", exit_cond)
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
            if verbose and (n_iter % 200 == 0):
                print(f"-----AOADMM comes into {n_iter} iter with loss {obj}")
            if save_time_error:
                time_list.append(cur_time)
                error_list.append(obj)

        return U, V, time_list, error_list


def test():
    method_name = "AOADMM"
    data_dir = os.getcwd()
    print(os.getcwd())
    proto_path = os.path.join(data_dir, "ENMF/Data", "real_data_algo_exp1.json")
    dataset_config = load_data_basedon_proto(proto_path, mode="realDataset")
    f_path = os.path.join(dataset_config.data_dir, dataset_config.data_path)
    org_data_mat = load_data_matrix(f_path)
    print(org_data_mat.shape)
    # print(real_dataset.name)
    # params = {"X": org_data_mat, "r": 20, "save_dir": "", "iter_save_dir": }
    latent_dims = [10, 20, 40, 80, 100]
    target_time = [3600, 3600, 3600, 3600, 3600]
    for latent_dim, target_t in zip(latent_dims, target_time):
        params = {"X": org_data_mat, "dataset_name": "Verb", "r": latent_dim}
        nmf_aoadmm = NMF_AOADMM(method_name=method_name, params=params)
        nmf_aoadmm.run_within_fixed_time(target_run_time=target_t)
        # nmf_aoadmm.run_to_target_error(target_error=13.6)


if __name__ == "__main__":
    test()
