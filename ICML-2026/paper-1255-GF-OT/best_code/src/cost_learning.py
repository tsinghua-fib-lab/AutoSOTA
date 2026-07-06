import ot
import torch
import numpy as np
import pandas as pd

from src.utils import kl_divergence
from src.cost_models import (
    MLPCostModel,
    ResNetCostModel,
    MahalanobisCostModel,
    WeightedDistanceFunction,
)
from src.loss_funcs import quota_loss, weighted_quota_loss

from joblib import Memory, Parallel, delayed

mem = Memory("__cache__")


class TorchFairCostOT:
    """A class for solving the fair cost optimal transport problem using
    PyTorch.

    Parameters
    ----------
    penalty_grid : list of float
        A grid of penalty values to be used in the optimization.
    entropic_grid : list of float
        A grid of entropic regularization values to be used in the
        optimization.
    lr_grid : list of float
        A grid of learning rates to be used in the optimization.
    fairness_loss : str, optional
        The type of fairness loss to be used. Can be either 'homogamy' or
        'cost_per_group'. Default is 'homogamy'.
    verbose : bool, optional
        If True, prints detailed information during the optimization process.
        Default is False.

    Attributes
    ----------
    penalty_grid : list of float
        The grid of penalty values used in the optimization.
    entropic_grid : list of float
        The grid of entropic regularization values used in the optimization.
    fairness_loss : function
        The fairness loss function to be used in the optimization.
    lr_grid : list of float
        The grid of learning rates used in the optimization.
    verbose : bool
        If True, prints detailed information during the optimization process.

    Methods
    -------
    _homogamy_loss(ot_plan, S_X, S_Y) -> float:
        Computes the homogamy loss based on the optimal transport plan and
        sensitive attributes.
    _single_solve(X, Y, S_X, S_Y, lr, pen, n_iter, entropic_reg, add_encoders,
    latent_dim, M_init, X_encoder_init, Y_encoder_init, log, optimizer)
    -> dict:
        Solves a single instance of the fair cost optimal transport problem.
    _log_iteration(cost_model, iter, loss, ot_plan, S_X, S_Y) -> pd.DataFrame:
        Logs the results of a single iteration during the optimization process.
    solve(X, Y, S_X, S_Y, a, b, n_iter, add_encoders, latent_dim, optimizer)
    -> tuple:
        Solves the fair cost optimal transport problem for the given data and
        sensitive attributes.
        Returns a tuple containing the results DataFrame and a dictionary of
        logs.
    _numpyfy(df) -> pd.DataFrame:
        Converts all PyTorch tensors in the DataFrame to NumPy arrays.
    """

    def __init__(
        self,
        penalty_grid: list,
        entropic_grid: list,
        lr_grid: list,
        fairness_loss: str = "quota_loss",
        cost_model_name: str = "mahalanobis",
        return_cost_model: bool = False,
        verbose: bool = False,
        optimizer: str = "SGD",
        **cost_model_kwargs,
    ):

        self.penalty_grid = penalty_grid
        self.entropic_grid = entropic_grid
        self.return_cost_model = return_cost_model

        if fairness_loss not in ["quota_loss", "weighted_quota_loss"]:
            raise ValueError(
                "fairness_loss must be either 'quota_loss' or"
                + " 'weighted_quota_loss'"
            )
        self.fairness_loss_name = fairness_loss
        self.lr_grid = lr_grid
        self.verbose = verbose
        if fairness_loss == "quota_loss":
            self.fairness_loss = quota_loss
        elif fairness_loss == "weighted_quota_loss":
            self.fairness_loss = weighted_quota_loss

        self.cost_model_name = cost_model_name
        self.cost_model_kwargs = cost_model_kwargs
        self.optimizer = optimizer

    def _check_args_fairness_loss(self, T, C, S_X, S_Y, F):
        if self.fairness_loss_name == "weighted_quota_loss":
            return T, C, S_X, S_Y, F
        else:
            return T, S_X, S_Y, F

    def _target_size_check(self, F, S_X, S_Y):
        n_s_x, n_s_y = F.shape
        if n_s_x != len(torch.unique(S_X)):
            raise ValueError(
                f"F has {n_s_x} rows, but S_X has "
                + f"{len(torch.unique(S_X))} unique values."
            )
        if n_s_y != len(torch.unique(S_Y)):
            raise ValueError(
                f"F has {n_s_y} columns, but S_Y has "
                + f"{len(torch.unique(S_Y))} unique values."
            )

    def _check_cost_model_args(self, cost_model_args: dict):

        if self.cost_model_name == "mahalanobis":
            required_args = []
        elif self.cost_model_name == "mlp":
            required_args = ["d_hidden", "d_out", "n_layers"]
        elif self.cost_model_name == "resnet":
            required_args = ["d_hidden", "d_out", "n_layers"]
        elif self.cost_model_name == "attention":
            required_args = ["d_value", "d_query"]
        elif self.cost_model_name == "weighted_distance":
            required_args = ["n_features"]

        for arg in required_args:
            if arg not in cost_model_args:
                raise ValueError(
                    f"Missing required argument '{arg}' for cost model "
                    + f"'{self.cost_model_name}'."
                )

    def _single_solve(
        self,
        X,
        Y,
        a,
        b,
        S_X,
        S_Y,
        F,
        id_X=None,
        id_Y=None,
        lr=None,
        pen=None,
        n_iter=None,
        entropic_reg=None,
        log=False,
        log_freq=1,
        optimizer="SGD",
        true_cost_matrix=None,
        pretrained_weights=None,
    ):
        """Solves a single instance of the fair cost optimal transport problem.

        Parameters
        ----------
        X : torch.Tensor
            The input data for the X side.
        Y : torch.Tensor
            The input data for the Y side.
        S_X : torch.Tensor
            Sensitive attribute for the X data.
        S_Y : torch.Tensor
            Sensitive attribute for the Y data.
        a : torch.Tensor
            The weights for the X data, typically uniform weights.
        b : torch.Tensor
            The weights for the Y data, typically uniform weights.
        lr : float, optional
            Learning rate for the optimizer. Default is 0.001.
        pen : float, optional
            Penalty for the homogamy loss. Default is 0.1.
        n_iter : int, optional
            Number of iterations for the optimization. Default is 100.
        entropic_reg : float, optional
            Entropic regularization parameter for the optimal transport
            problem.
            Default is 1e-5.
        add_encoders : bool, optional
            If True, encoders for X and Y will be added to the model. Default
            is False.
        neural_encoder : bool, optional
            If True, uses a neural network as the cost model. Default is False.
        latent_dim : int, optional
            The dimensionality of the latent space. If None, it will be set to
            the dimensionality of the input data. Default is None.
        M_init : np.ndarray, optional
            Initial value for the linear transformation matrix M. If None, M
            will be initialized randomly. Default is None.
        X_encoder_init : np.ndarray, optional
            Initial value for the encoder for the X data. If None, the encoder
            will be initialized randomly. Default is None.
        Y_encoder_init : np.ndarray, optional
            Initial value for the encoder for the Y data. If None, the encoder
            will be initialized randomly. Default is None.
        log : bool, optional
            If True, logs the results of each iteration. Default is True.
        optimizer : str, optional
            The optimizer to use for the optimization. Can be either 'Adam' or
            'SGD'.
            Default is 'SGD'.

        Returns
        -------
        dict
            A dictionary containing the results of the optimization, including:
            - 'fair_ot_plan': The optimal transport plan that satisfies the
               fairness constraints.
            - 'M': The learned linear transformation matrix M.
            - 'encoder_X': The learned encoder for the X data (if add_encoders
              is True).
            - 'encoder_Y': The learned encoder for the Y data (if add_encoders
              is True).
            - 'fair_cost': The fair cost computed from the optimal transport
              plan.
            - 'log_run': A DataFrame containing the log of the optimization
              process.
        """

        if self.cost_model_name == "mahalanobis":
            cost_model = MahalanobisCostModel(d_in=X.shape[1])
        elif self.cost_model_name == "mlp":
            cost_model = MLPCostModel(
                d_in=X.shape[1],
                d_hidden=self.cost_model_kwargs["d_hidden"],
                d_out=self.cost_model_kwargs["d_out"],
                n_layers=self.cost_model_kwargs["n_layers"],
            )
        elif self.cost_model_name == "resnet":
            cost_model = ResNetCostModel(
                d_in=X.shape[1],
                d_hidden=self.cost_model_kwargs["d_hidden"],
                d_out=self.cost_model_kwargs["d_out"],
                n_layers=self.cost_model_kwargs["n_layers"],
            )
        elif self.cost_model_name == "weighted_distance":
            cost_model = WeightedDistanceFunction(**self.cost_model_kwargs)
        if pretrained_weights is not None:
            cost_model.load_state_dict(torch.load(pretrained_weights))

        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(cost_model.parameters(), lr=lr)
        elif optimizer == "SGD":
            optimizer = torch.optim.SGD(cost_model.parameters(), lr=lr)

        if log:
            log_run = {
                "iter": [],
                "loss": [],
                "fairness_loss_value": [],
                "inner_err": [],
                "n_iter_inner": [],
                "penalty_term": [],
            }
        else:
            log_run = None

        if true_cost_matrix is None:
            euclidean_cost = torch.sum(
                (X[:, None, :] - Y[None, :, :]) ** 2, dim=2
            )
        else:
            euclidean_cost = true_cost_matrix

        method = "sinkhorn_log"
        loss = 0
        if torch.cuda.is_available():
            cost_model = cost_model.cuda()
            if isinstance(X, torch.Tensor):
                X = X.cuda()
            if isinstance(Y, torch.Tensor):
                Y = Y.cuda()
            S_X = S_X.cuda()
            S_Y = S_Y.cuda()
            a = a.cuda()
            b = b.cuda()
            euclidean_cost = euclidean_cost.cuda()
        for iter in range(n_iter):
            optimizer.zero_grad()
            if id_X is None or id_Y is None:
                cost_matrix = cost_model(X, Y)
            else:
                cost_matrix = cost_model(X, Y, id_X, id_Y)

            if iter == 0:
                ot_plan, innerlog_ = ot.sinkhorn(
                    a,
                    b,
                    cost_matrix,
                    reg=entropic_reg,
                    warn=False,
                    verbose=False,
                    stopThr=1e-6,
                    numItermax=1000,
                    method=method,
                    log=True,
                )
            else:
                if method == "sinkhorn_log":
                    log_u = innerlog_["log_u"].detach()
                    log_v = innerlog_["log_v"].detach()
                    ot_plan, innerlog_ = ot.sinkhorn(
                        a,
                        b,
                        cost_matrix,
                        reg=entropic_reg,
                        warn=False,
                        verbose=False,
                        stopThr=1e-6,
                        numItermax=1000,
                        method=method,
                        log=True,
                        warmstart=(log_u, log_v),
                    )
                else:
                    u = innerlog_["u"].detach()
                    v = innerlog_["v"].detach()
                    ot_plan, innerlog_ = ot.sinkhorn(
                        a,
                        b,
                        cost_matrix,
                        reg=entropic_reg,
                        warn=False,
                        verbose=False,
                        stopThr=1e-6,
                        numItermax=1000,
                        method=method,
                        log=True,
                        warmstart=(u, v),
                    )
            penalty_term = torch.mean((cost_matrix - euclidean_cost) ** 2)
            _fairness_loss = self.fairness_loss(
                *self._check_args_fairness_loss(
                    ot_plan,
                    euclidean_cost,
                    S_X,
                    S_Y,
                    F,
                )
            )
            loss_old = loss
            loss = pen * _fairness_loss + penalty_term
            loss.backward()
            optimizer.step()

            # get gradient norm
            grad_norm = 0
            for param in cost_model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm**0.5
            print(f"Gradient norm: {grad_norm:.4f}")

            if hasattr(cost_model, "project"):
                cost_model.project()

            with torch.no_grad():

                if self.verbose:
                    print(
                        f"Iteration {iter+1}/{n_iter}, "
                        + f"Loss: {loss.item():.4f}, "
                        + f"Fairness Loss: {_fairness_loss.item():.4f}, "
                        + f"Penalty: {penalty_term.item():.4f}, "
                    )

                if torch.isnan(loss):
                    print("The algorithm produced NaNs")
                    break
                if log and iter % log_freq == 0:
                    iter_results = self._log_iteration(
                        iter, loss, _fairness_loss, innerlog_, penalty_term
                    )
                    for key, value in iter_results.items():
                        log_run[key].append(value)
                if iter >= 1:
                    if self.auto_stop and self._assess_stoping_criterion(
                        loss_old.item(), loss.item(), tol=self.tol
                    ):
                        print(
                            f"Stopping criterion met at iteration {iter+1}/"
                            + f"{n_iter}. Stopping optimization."
                        )
                        break

        else:
            if self.verbose:
                print(f"Reached maximum number of iterations ({n_iter}). ")

        log = {
            "fair_ot_plan": ot_plan.detach().cpu().numpy(),
            "cost_model_state": {
                key: value.detach().cpu().numpy()
                for key, value in cost_model.state_dict().items()
            },
            "fair_cost": torch.sum(ot_plan * cost_matrix)
            .detach()
            .cpu()
            .numpy(),
            # "fair_cost": torch.sum(ot_plan * cost_model(X, Y))
            # .detach()
            # .cpu()
            # .numpy(),
            "euclidean_cost": torch.sum(ot_plan * euclidean_cost)
            .detach()
            .cpu()
            .numpy(),
            "model": cost_model,
        }
        log.update(log_run)

        if not self.return_cost_model:
            cost_model = None
        return log, cost_model

    def _assess_stoping_criterion(self, a, b, tol=1e-8):
        """Assesses whether the stopping criterion is met based on the loss
        values.

        Parameters
        ----------
        loss_values : list of float
            The list of loss values from the optimization process.
        tol : float, optional
            The tolerance for the stopping criterion. Default is 1e-4.

        Returns
        -------
        bool
            True if the stopping criterion is met, False otherwise.
        """
        return np.abs(a - b) < tol

    def _log_iteration(
        self, iter, loss, fairness_loss, innerlog, penalty_term
    ):
        """Logs the results of a single iteration during the optimization
        process.

        Parameters
        ----------
        cost_model : CostModel
            The cost model used for the optimization.
        iter : int
            The current iteration number.
        loss : torch.Tensor
            The loss value computed for the current iteration.
        ot_plan : torch.Tensor
            The optimal transport plan computed for the current iteration.
        S_X : torch.Tensor
            Sensitive attribute for the X data.
        S_Y : torch.Tensor
            Sensitive attribute for the Y data.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the results of the current iteration,
            including:
            - 'iter': The current iteration number.
            - 'loss_value': The loss value for the current iteration.
            - 'fairness_loss_value': The fairness loss value for the current
               iteration.
            - 'fair_ot_plan': The optimal transport plan for the current
               iteration.
            - 'M': The learned linear transformation matrix M for the current
               iteration.
            - 'encoder_X': The learned encoder for the X data for the current
               iteration (if add_encoders is True).
            - 'encoder_Y': The learned encoder for the Y data for the current
               iteration (if add_encoders is True).
        """
        return {
            "iter": iter,
            "loss": loss.item(),
            "fairness_loss_value": fairness_loss.item(),
            "inner_err": innerlog["err"][-1].item(),
            "n_iter_inner": innerlog["niter"],
            "penalty_term": penalty_term.item(),
        }

    def solve(
        self,
        X: torch.Tensor = None,
        Y: torch.Tensor = None,
        S_X: torch.Tensor = None,
        S_Y: torch.Tensor = None,
        F: torch.Tensor = None,
        a: torch.Tensor = None,
        b: torch.Tensor = None,
        cost_matrix: torch.Tensor = None,
        n_iter=300,
        optimizer="SGD",
        auto_stop=True,
        pretrained_weights=None,
        tol=1e-8,
        n_jobs=1,
        use_cache=True,
        **kwargs_cost_model,
    ):
        """Solves the fair cost optimal transport problem for the given data
        and sensitive attributes.

        Parameters
        ----------
        X : torch.Tensor
            The input data for the X side.
        Y : torch.Tensor
            The input data for the Y side.
        S_X : torch.Tensor
            Sensitive attribute for the X data.
        S_Y : torch.Tensor
            Sensitive attribute for the Y data.
        a : torch.Tensor
            The weights for the X data, typically uniform weights.
        b : torch.Tensor
            The weights for the Y data, typically uniform weights.
        n_iter : int, optional
            Number of iterations for the optimization. Default is 100.
        optimizer : str, optional
            The optimizer to use for the optimization. Can be either 'Adam' or
            'SGD'.
            Default is 'SGD'.
        auto_stop : bool, optional
            If True, the optimization will stop automatically if the stopping
            criterion is met.
            Default is True.
        pretrained_weights : str, optional
            Path to the pretrained weights for the cost model. If None, the
            cost model will be trained from scratch. Default is None.
        tol : float, optional
            The tolerance for the stopping criterion. Default is 1e-4.
        n_jobs : int, optional
            The number of jobs to run in parallel. If -1, use all available
            cores. Default is 1.
        use_cache : bool, optional
            If True, uses caching to speed up repeated computations. Default is
            True.
        kwargs_cost_model : dict
            Additional keyword arguments to be passed to the cost model.

        Returns
        -------
        tuple
            A tuple containing:
            - results: A DataFrame containing the results of the optimization,
              including:
                - 'entropic_reg': The entropic regularization parameter used.
                - 'penalty': The penalty value used.
                - 'fair_cost': The fair cost computed from the optimal
                   transport plan.
                - 'true_cost': The true cost computed from the optimal
                   transport plan.
                - 'cost_diff': The absolute difference between the fair cost
                   and the true cost.
                - 'kl_div': The KL divergence between the fair and true
                   optimal transport plans.
                - 'fairness_loss_value': The fairness loss value computed fro
                   the optimal transport plan.
                - 'fair_ot_plan': The optimal transport plan that satisfies
                   the fairness constraints.
                - 'true_ot_plan': The true optimal transport plan computed
                   from the cost matrix.
            - log_dict: A dictionary containing logs of the optimization
              process, organized by entropic regularization parameter, penalty,
              and learning rate. Each entry contains the log of the run for
              that specific configuration.

        """
        results = pd.DataFrame(
            columns=[
                "entropic_reg",
                "penalty",
                "fair_cost",
                "true_cost",
                "euclidean_cost",
                "cost_diff",
                "kl_div",
                "fairness_loss_value",
                "fair_ot_plan",
                "true_ot_plan",
                "penalty_term",
                "model",
            ]
        )

        if a is None:
            a = torch.ones(X.shape[0]) / X.shape[0]
        if b is None:
            b = torch.ones(Y.shape[0]) / Y.shape[0]

        self._target_size_check(F, S_X, S_Y)

        self._check_cost_model_args(self.cost_model_kwargs)

        self.auto_stop = auto_stop
        self.tol = tol

        def run_one(eps, pen, lr):
            print(
                f"Solving with entropic_reg={eps}, "
                + f" and penalty={pen}, lr={lr}"
            )
            if cost_matrix is None:
                true_cost_matrix = (
                    ot.dist(X, Y, metric="sqeuclidean").detach().numpy()
                )
            else:
                true_cost_matrix = cost_matrix
                if isinstance(cost_matrix, torch.Tensor):
                    true_cost_matrix = true_cost_matrix.detach().numpy()
            true_ot_plan = ot.sinkhorn(
                a.detach().numpy(),
                b.detach().numpy(),
                true_cost_matrix,
                reg=eps,
                warn=False,
                stopThr=1e-6,
                numItermax=1000,
                method="sinkhorn_log" if eps < 1.0 else "sinkhorn",
            )
            true_cost = np.sum(true_ot_plan * true_cost_matrix)

            single_result, _ = self._single_solve(
                X=X,
                Y=Y,
                a=a,
                b=b,
                S_X=S_X,
                S_Y=S_Y,
                F=F,
                lr=lr,
                pen=pen,
                n_iter=n_iter,
                entropic_reg=eps,
                log=True,
                optimizer=optimizer,
                true_cost_matrix=torch.from_numpy(true_cost_matrix),
                pretrained_weights=pretrained_weights,
                **kwargs_cost_model,
            )

            fair_ot_plan = single_result["fair_ot_plan"]
            fair_cost = single_result["fair_cost"]
            fairness_loss_value = single_result["fairness_loss_value"]
            inner_err = single_result["inner_err"]
            n_iter_inner = single_result["n_iter_inner"]
            euclidean_cost = single_result["euclidean_cost"]

            cost_diff = np.abs(euclidean_cost - true_cost)
            kl_div = kl_divergence(fair_ot_plan, true_ot_plan)

            results = pd.DataFrame(
                {
                    "entropic_reg": [eps],
                    "penalty": [pen],
                    "lr": [lr],
                    "n_iter": [n_iter],
                    "fair_cost": [fair_cost],
                    "euclidean_cost": [euclidean_cost],
                    "true_cost": [true_cost],
                    "cost_diff": [cost_diff],
                    "kl_div": [kl_div],
                    "inner_err": [inner_err],
                    "n_iter_inner": [n_iter_inner],
                    "fair_ot_plan": [fair_ot_plan],
                    "true_ot_plan": [true_ot_plan],
                    "fairness_loss_value": [fairness_loss_value],
                    "loss": [single_result["loss"]],
                    "fairness_loss_name": [self.fairness_loss_name],
                    "cost_model_state": [single_result["cost_model_state"]],
                    "penalty_term": [single_result["penalty_term"]],
                    "model": single_result["model"],
                }
            )

            return results

        if use_cache:
            run_one = mem.cache(run_one)

        if n_jobs == 1:
            results = pd.concat(
                [
                    run_one(eps, reg, lr)
                    for eps in self.entropic_grid
                    for reg in self.penalty_grid
                    for lr in self.lr_grid
                ],
                ignore_index=True,
            )
        else:
            all_results = Parallel(n_jobs=n_jobs)(
                delayed(run_one)(eps, reg, lr)
                for eps in self.entropic_grid
                for reg in self.penalty_grid
                for lr in self.lr_grid
            )
            results = pd.concat(all_results, ignore_index=True)

        return self._numpyfy(results)

    def _numpyfy(self, df):
        """Convert all jax arrays in the dataframe to numpy arrays.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the results, which may include PyTorch
            tensors.

        Returns
        -------
        pd.DataFrame
            A DataFrame with all PyTorch tensors converted to NumPy arrays.
        """
        return df.map(
            lambda x: (
                x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            )
        )
