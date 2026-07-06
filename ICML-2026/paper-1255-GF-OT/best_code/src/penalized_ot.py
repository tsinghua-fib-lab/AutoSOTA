import torch
import numpy as np
import pandas as pd

import ot

from joblib import Parallel, delayed, Memory

from src.utils import kl_divergence
from src.solvers import penalized_ot_solver
from src.loss_funcs import quota_loss, weighted_quota_loss

mem = Memory(location="__cache__")


class PenalizedOT:
    def __init__(
        self, penalty_grid, entropic_grid, fairness_loss="quota_loss"
    ):
        """
        Initializes the penalized OT solver.

        Parameters:
        ----------
        penalty_grid : list or array-like
            A grid of penalty values for the rate constraint.
        entropic_grid : list or array-like
            A grid of entropic regularization values.
        fairness_loss : str, optional
            The type of fairness loss to use. Options are "quota_loss" or
            "weighted_quota_loss".
        """

        self.penalty_grid = penalty_grid
        self.entropic_grid = entropic_grid
        if fairness_loss not in [
            "quota_loss",
            "weighted_quota_loss",
        ]:
            raise ValueError(
                "fairness_loss must be either 'quota_loss', "
                + "'weighted_quota_loss'"
            )

        self.fairness_loss_name = fairness_loss

        if fairness_loss == "quota_loss":
            self.fairness_loss = quota_loss
        elif fairness_loss == "weighted_quota_loss":
            self.fairness_loss = weighted_quota_loss

    def _compute_cost_matrix(self, X, Y, metric="sqeuclidean", p=2):
        """
        Computes the cost matrix between two distributions X and Y using POT.

        Parameters:
        ----------
        X : array-like
            The first distribution, shape (n_samples_X, n_features).
        Y : array-like
            The second distribution, shape (n_samples_Y, n_features).
        metric : str, optional
            The metric to use for distance computation. Default is
            "sqeuclidean".
        p : int, optional
            The power parameter for the metric. Default is 2.

        Returns:
        -------
        cost_matrix : array-like
            The cost matrix of shape (n_samples_X, n_samples_Y).
        """
        # Convert CUDA tensors to CPU for POT compatibility
        X_cpu = X.cpu() if isinstance(X, torch.Tensor) else X
        Y_cpu = Y.cpu() if isinstance(Y, torch.Tensor) else Y
        return ot.dist(X_cpu, Y_cpu, metric=metric, p=p)

    def _compute_cost(self, cost_matrix, transport_plan):
        return torch.sum(transport_plan * cost_matrix)

    def _solve_ot(self, cost_matrix, a, b, eps):

        return ot.sinkhorn(
            a,
            b,
            cost_matrix,
            eps,
            method="sinkhorn_log" if eps < 1.0 else "sinkhorn",
            numItermax=50000,
            warn=False,
        )

    def _check_args_fairness_loss(self, T, C, S_X, S_Y, F):
        if self.fairness_loss_name == "weighted_quota_loss":
            return T, C, S_X, S_Y, F
        elif self.fairness_loss_name == "quota_loss":
            return T, S_X, S_Y, F

    def _single_solve(self, X, Y, a, b, S_X, S_Y, F, eps, pen, cost_matrix):
        """
        Solves the rate-constrained optimal transport problem for a single pair
        of entropic regularization and penalty.

        Parameters:
        ----------
        X : array-like
            The first distribution, shape (n_samples_X, n_features).
        Y : array-like
            The second distribution, shape (n_samples_Y, n_features).
        a : array-like
            The weights for the first distribution, shape (n_samples_X,).
        b : array-like
            The weights for the second distribution, shape (n_samples_Y,).
        S_X : array-like
            Sensitive attributes for the first distribution, shape
            (n_samples_X,).
        S_Y : array-like
            Sensitive attributes for the second distribution, shape
            (n_samples_Y,).
        F: array-like
            Target fairness matrix, shape (n_sensitive_groups_X,
            n_sensitive_groups_Y).
        eps : float
            Entropic regularization parameter.
        pen : float
            Penalty for the rate constraint.
        cost_matrix : array-like, optional
            Precomputed cost matrix between X and Y. If None, it will be
            computed.
            Shape should be (n_samples_X, n_samples_Y).
        """
        if cost_matrix is None:
            cost_matrix = self._compute_cost_matrix(X, Y)

        # Remember original device for restoring after solve
        orig_device = None
        if isinstance(cost_matrix, torch.Tensor):
            orig_device = cost_matrix.device

        # Move all tensors to CPU for consistent dtype handling in the solver
        def to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            return x

        cost_matrix_cpu = to_cpu(cost_matrix)
        a_cpu = to_cpu(a)
        b_cpu = to_cpu(b)
        S_X_cpu = to_cpu(S_X)
        S_Y_cpu = to_cpu(S_Y)
        F_cpu = to_cpu(F)

        ot_plan, log = penalized_ot_solver(
            cost_matrix_cpu,
            a_cpu,
            b_cpu,
            lambda plan: self.fairness_loss(
                *self._check_args_fairness_loss(plan, cost_matrix_cpu, S_X_cpu, S_Y_cpu, F_cpu)
            ),
            eps=eps,
            reg_constraints=pen,
            log=True,
        )

        # Restore ot_plan to original device (CUDA) if needed
        if orig_device is not None and orig_device.type == "cuda":
            if isinstance(ot_plan, torch.Tensor):
                ot_plan = ot_plan.to(orig_device)
            else:
                ot_plan = torch.as_tensor(ot_plan, device=orig_device)

        return ot_plan, log

    def solve(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        S_X: torch.Tensor,
        S_Y: torch.Tensor,
        F: torch.Tensor,
        a: torch.Tensor = None,
        b: torch.Tensor = None,
        cost_matrix=None,
        cost_metric="sqeuclidean",
        p=2,
        n_jobs=1,
        use_cache=True,
    ):
        """
        Solves the rate-constrained optimal transport problem for a grid of
        entropic regularization parameters and penalties.

        Parameters:
        ----------
        X : array-like
            The first distribution, shape (n_samples_X, n_features).
        Y : array-like
            The second distribution, shape (n_samples_Y, n_features).
        a : array-like
            The weights for the first distribution, shape (n_samples_X,).
        b : array-like
            The weights for the second distribution, shape (n_samples_Y,).
        S_X : array-like
            Sensitive attributes for the first distribution,
            shape (n_samples_X,).
        S_Y : array-like
            Sensitive attributes for the second distribution,
            shape (n_samples_Y,).
        F: array-like
            Target fairness matrix, shape (n_sensitive_groups_X,
            n_sensitive_groups_Y).
        cost_matrix : array-like, optional
            Precomputed cost matrix between X and Y. If None, it will be
            computed.
            Shape should be (n_samples_X, n_samples_Y).
        cost_metric : str, optional
            The metric to use for distance computation. Default is
            "sqeuclidean".
        p : int, optional
            The power parameter for the metric. Default is 2.
        n_jobs : int, optional
            The number of jobs to run in parallel. If -1, use all available
            cores. Default is 1.
        use_cache : bool, optional
            If True, uses caching to speed up repeated computations. Default is
            True.

        Returns:
        -------
        results : pandas.DataFrame
            A DataFrame containing the results of the optimal transport problem
            for each combination of entropic regularization and penalty. The
            DataFrame contains the following columns:
            - entropic_reg: The entropic regularization parameter used.
            - penalty: The penalty for the rate constraint used.
            - fair_cost: The cost of the fair optimal transport plan.
            - true_cost: The cost of the true optimal transport plan.
            - cost_diff: The absolute difference between fair and true costs.
            - kl_div: The KL divergence between the fair and true optimal
              transport plans.
            - fairness_loss_value: The value of the fairness loss for the fair
              optimal transport plan.
            - fair_ot_plan: The fair optimal transport plan.
            - true_ot_plan: The true optimal transport plan.
        """

        results = pd.DataFrame(
            columns=[
                "entropic_reg",
                "penalty",
                "fair_cost",
                "true_cost",
                "cost_diff",
                "kl_div",
                "fairness_loss_value",
                "fair_ot_plan",
                "true_ot_plan",
            ]
        )

        if torch.cuda.is_available():
            if cost_matrix is None:
                X = X.cuda()
                Y = Y.cuda()
            S_X = S_X.cuda()
            S_Y = S_Y.cuda()
            F = F.cuda()

        if cost_matrix is None:
            cost_matrix = self._compute_cost_matrix(
                X, Y, metric=cost_metric, p=p
            )

        if a is None:
            a = torch.ones(X.shape[0]) / X.shape[0]
        if b is None:
            b = torch.ones(Y.shape[0]) / Y.shape[0]

        if torch.cuda.is_available():
            a = a.cuda()
            b = b.cuda()
            cost_matrix = cost_matrix.cuda()

        def run_one(eps, reg):
            true_ot_plan = self._solve_ot(cost_matrix, a, b, eps=eps)
            true_cost = torch.sum(true_ot_plan * cost_matrix).cpu()
            ot_plan, log = self._single_solve(
                X, Y, a, b, S_X, S_Y, F, eps, reg, cost_matrix
            )
            cost = self._compute_cost(cost_matrix, ot_plan).cpu()
            fairness_loss_value = self.fairness_loss(
                *self._check_args_fairness_loss(
                    ot_plan, cost_matrix, S_X, S_Y, F
                )
            )
            results = pd.DataFrame(
                {
                    "entropic_reg": [eps],
                    "penalty": [reg],
                    "fair_cost": [cost.detach().cpu().numpy()],
                    "true_cost": [true_cost.detach().cpu().numpy()],
                    "cost_diff": [np.abs(cost - true_cost)],
                    "kl_div": [
                        kl_divergence(
                            ot_plan.detach().cpu().numpy(),
                            true_ot_plan.detach().cpu().numpy(),
                        )
                    ],
                    "loss": [np.array(log["loss"])],
                    "n_iters": [int(log["niter"])],
                    "inner_err": [np.array(log["err"])],
                    "fairness_loss_value": [fairness_loss_value],
                    "fair_ot_plan": [ot_plan.detach().cpu().numpy()],
                    "true_ot_plan": [true_ot_plan.detach().cpu().numpy()],
                    "fairness_loss_name": [self.fairness_loss_name],
                }
            )
            return results

        if use_cache:
            run_one = mem.cache(run_one)

        if n_jobs == 1:
            results = pd.concat(
                [
                    run_one(eps, reg)
                    for eps in self.entropic_grid
                    for reg in self.penalty_grid
                ],
                ignore_index=True,
            )
        else:
            all_results = Parallel(n_jobs=n_jobs)(
                delayed(run_one)(eps, reg)
                for eps in self.entropic_grid
                for reg in self.penalty_grid
            )
            results = pd.concat(all_results, ignore_index=True)

        return self._numpyfy(results)

    def _numpyfy(self, df):
        """Convert all jax arrays in the dataframe to numpy arrays.

        Parameters:
        ----------
        df : pandas.DataFrame
            The DataFrame to convert.

        Returns:
        -------
        pandas.DataFrame
            The DataFrame with all jax arrays converted to numpy arrays.
        """
        return df.map(
            lambda x: (
                x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            )
        )
