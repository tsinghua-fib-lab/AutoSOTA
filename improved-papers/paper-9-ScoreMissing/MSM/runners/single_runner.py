from abc import abstractmethod
import copy
from functools import partial
from typing import Literal, Union, Callable
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim as opt
from torch.optim.lr_scheduler import _LRScheduler
from typing import Union, TypeAlias  # noqa: F401, F811
from ..losses import score_matching as losses  # noqa: E402
from ..losses import marginal_score_matching as marg_losses  # noqa: E402
from ..models import density_models as models  # noqa: E402
from ..models import variational_models as var_models  # noqa: E402
from ..utils.regularisers import GenRegulariser  # noqa: E402
from .. import utils  # noqa: E402
RegType: TypeAlias = Union[GenRegulariser, list[GenRegulariser], None]


class GenericRunner():
    def __init__(self, q_theta: models.UDensity,
                 q_theta_opt: opt.Optimizer, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.q_theta = q_theta.to(device)
        self.q_theta_opt = q_theta_opt
        self.set_regularisers()
        self.set_grad_opts()
        self.set_schedulers()
        self.true_loss_diff = False

    def set_grad_opts(self, control_method=Literal["clip", "rerun"],
                      max_norm: float = 1., norm_type: Union[int, str] = 2):
        self.q_theta_grad_opts = {"control_method": control_method, "max_norm": max_norm, "norm_type": norm_type}

    def init_dataset(self, X, **kwargs):
        self.dataset = TensorDataset(X)
        self.dataloader = DataLoader(self.dataset, shuffle=True, **kwargs)

    def set_schedulers(self, theta_scheduler: _LRScheduler = None):
        self.q_theta_scheduler = theta_scheduler

    def set_regularisers(self, theta_regulariser: RegType = None):
        self.theta_regulariser = theta_regulariser

    @abstractmethod
    def loss(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass

    def true_loss(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.loss(X_obs, mask).mean()

    def _gradient_control(self, batch, grad_opts: dict, model: Literal["q", "p"] = "q"):

        model = self.q_theta
        optimizer = self.q_theta_opt
        loss_fn = self.loss
        regulariser = self.theta_regulariser

        grad_control_method = grad_opts.get("control_method", None)
        grad_max_norm = grad_opts.get("max_norm", None)
        grad_norm_type = grad_opts.get("norm_type", 2.0)
        if grad_control_method == "clip":
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm, grad_norm_type)
        # If large gradient, run again.
        if grad_control_method == "rerun":
            for iattempt in range(10):
                if utils.get_grad_norm(model, grad_norm_type) > grad_max_norm:
                    break
                elif iattempt == 9:
                    raise RuntimeError(
                        "Failed to get gradient norm below max_grad_norm"
                    )
                optimizer.zero_grad()
                loss = loss_fn(model(*batch))
                if regulariser is not None:
                    loss += regulariser()
                loss.backward()

    def loss_wreg(self, *args):
        self.q_theta_opt.zero_grad()
        outer_loss = self.loss(*args).mean()
        if self.theta_regulariser is not None:
            if type(self.theta_regulariser) is list:
                for reg in self.theta_regulariser:
                    outer_loss += reg(*args)
            else:
                outer_loss += self.theta_regulariser(*args)
        return outer_loss

    def step(self, *args):
        outer_loss = self.loss_wreg(*args)
        outer_loss.backward()
        self._gradient_control((*args, ), self.q_theta_grad_opts, model="q")
        if isinstance(self.q_theta_opt, opt.LBFGS):
            closure = partial(self.loss_wreg, *args)
            self.q_theta_opt.step(closure)
        else:
            self.q_theta_opt.step()
        return outer_loss

    def train(
        self, epochs=1000, niters=10000,
        snapshot_freq=100, loss_avg_length=100,
        min_loss_val=-torch.inf, max_loss_val=torch.inf,
        true_loss=False, verbose=False, **kwargs
    ):
        """Runs a training loop for the missing score matching model.

        Args:
            epochs (int, optional): Max number of runs through the data. Defaults to 1000.
            niters (int, optional): Max number of overall iterations. Defaults to 10000.
            snapshot_freq (int, optional): How often to store the values of the model. Defaults to 100.
            loss_avg_length (int, optional): Length of the moving average window for the loss. Defaults to 100.
            min_loss_val (float, optional): Minimum loss value to stop training. Defaults to -float("inf").
            verbose (bool, optional): Whether to print progress messages. Defaults to False.
            **score_args: Additional arguments to pass to the score model.

        Returns:
            list|none: If stored_vals is None, returns a list of the stored values. Otherwise, returns None
                and stores the results in stored_vals.
        """
        self.stored_vals = {"Losses": [],
                            "q_State_dicts": []}
        if true_loss:
            self.stored_vals["True_Losses"] = []
        counter = 0
        temp_loss_list = []
        # Set up data and batch size
        break_flag = False

        for epoch in range(epochs):
            if break_flag:
                break
            for batch in self.dataloader:
                # Send to device
                batch = [x.to(self.device) for x in batch]
                outer_loss = self.step(*batch)

                # Append loss
                temp_loss_list.append(outer_loss.item())

                # Check for snapshot
                if type(snapshot_freq) is int:
                    snapshot_bool = (counter % snapshot_freq == 0)
                elif type(snapshot_freq) is list:
                    snapshot_bool = (counter in snapshot_freq)
                else:
                    raise TypeError("snapshot_freq must be an int or a list of ints")
                if snapshot_bool:
                    if ((outer_loss <= min_loss_val) | (outer_loss >= max_loss_val)) | torch.isnan(outer_loss):
                        # Raise an error due to loss breaking down
                        raise utils.NaNModel("Model stopped training due to nan or inf loss")

                    if true_loss:
                        # If true loss is separate explicitly calculate it
                        if self.true_loss_diff:
                            true_loss = self.true_loss(*batch)
                        else:
                            true_loss = outer_loss

                        self.stored_vals["True_Losses"].append(true_loss.item())

                    # Get average internal loss within epoch
                    avg_loss = torch.mean(torch.tensor(temp_loss_list[-loss_avg_length:]))
                    if verbose:
                        print(f"Iteration {counter}: {avg_loss.item()}")
                    self.stored_vals["Losses"].append(avg_loss.item())

                    self.stored_vals["q_State_dicts"].append(
                        copy.deepcopy(self.q_theta.state_dict()))
                # Increment iteration counter
                counter += 1
                if counter >= niters:
                    break_flag = True
                    break
            if self.q_theta_scheduler is not None:
                self.q_theta_scheduler.step()


class ScoreMatchingRunner(GenericRunner):
    def loss(self, X, *args, **kwargs):
        return losses.score_matching(X, self.q_theta)


class MLRunner(GenericRunner):
    def loss(self, X, *args, **kwargs):
        return losses.ml_loss(X, self.q_theta)


class ImputingScoreMatchingRunner(GenericRunner):
    def __init__(self, q_theta: models.UDensity, p_phi: var_models.Imputer,
                 q_theta_opt: opt.Optimizer, device=None):
        super().__init__(q_theta, q_theta_opt, device)
        if isinstance(p_phi, nn.Module):
            self.p_phi = p_phi.to(device)
        else:
            self.p_phi = p_phi

    def init_dataset(self, X, mask, **kwargs):
        self.dataset = TensorDataset(X, mask)
        self.dataloader = DataLoader(self.dataset, shuffle=True, **kwargs)

    def loss(self, X_obs, mask, *args, **kwargs):
        return losses.imputed_score_matching(X_obs, mask, self.p_phi, self.q_theta, *args, **kwargs)


class ImputingZerodScoreMatchingRunner(ImputingScoreMatchingRunner):
    def loss(self, X_obs, mask, *args, **kwargs):
        return losses.imputed_zerod_score_matching(X_obs, mask, self.p_phi, self.q_theta, *args, **kwargs)


class IWScoreMatchingRunner(ImputingScoreMatchingRunner):
    def __init__(self, q_theta: models.UDensity, p_phi: var_models.VariationalDensity,
                 q_theta_opt: opt.Optimizer, ncopies=10, device=None):
        super().__init__(q_theta, p_phi, q_theta_opt, device)
        self.ncopies = ncopies

    def loss(self, X_obs, mask, *args, **kwargs):
        return marg_losses.scoreloss_marginal(
            X_obs, mask, self.p_phi, self.q_theta, do_iw=True,
            ncopies=self.ncopies, *args, **kwargs)


class EMScoreMatchingRunner(IWScoreMatchingRunner):
    def loss(self, X_obs, mask, *args, **kwargs):
        return marg_losses.scoreloss_em(
            X_obs, mask, self.p_phi, self.q_theta, do_iw=True, ncopies=self.ncopies,
            *args, **kwargs
        )


class TruncatedScoreMatchingRunner(ScoreMatchingRunner):
    def __init__(self, q_theta: models.UDensity,
                 q_theta_opt: opt.Optimizer, trunc_func: Callable[[torch.Tensor], torch.Tensor],
                 elementwise_trunc: bool = False, device=None):
        super().__init__(q_theta, q_theta_opt, device)
        self.trunc_func = trunc_func
        self.elementwise_trunc = elementwise_trunc

    def loss(self, X, *args, **kwargs):
        return losses.trunc_score_matching(
            X, self.q_theta, self.trunc_func, elementwise_trunc=self.elementwise_trunc,
            *args, **kwargs)


class TruncatedImputingScoreMatchingRunner(TruncatedScoreMatchingRunner):
    def __init__(self, q_theta: models.UDensity, p_phi: var_models.Imputer,
                 q_theta_opt: opt.Optimizer, trunc_func: Callable[[torch.Tensor], torch.Tensor],
                 elementwise_trunc: bool = False, device=None):
        super().__init__(q_theta, q_theta_opt, trunc_func, elementwise_trunc, device)
        if isinstance(p_phi, nn.Module):
            self.p_phi = p_phi.to(device)
        else:
            self.p_phi = p_phi

    init_dataset = ImputingScoreMatchingRunner.init_dataset

    def loss(self, X, mask, *args, **kwargs):
        return losses.trunc_imputed_score_matching(
            X, mask, self.p_phi, self.q_theta, self.trunc_func, elementwise_trunc=self.elementwise_trunc,
            *args, **kwargs)


class TruncatedZerodImputingScoreMatchingRunner(TruncatedImputingScoreMatchingRunner):
    def loss(self, X_obs, mask, *args, **kwargs):
        return losses.trunc_imputed_zerod_score_matching(
            X_obs, mask, self.p_phi, self.q_theta, self.trunc_func, elementwise_trunc=self.elementwise_trunc,
            *args, **kwargs)


class TruncatedIWScoreMatchingRunner(TruncatedImputingScoreMatchingRunner):
    def __init__(self, q_theta: models.UDensity, p_phi: var_models.Imputer,
                 q_theta_opt: opt.Optimizer, trunc_func: Callable[[torch.Tensor], torch.Tensor],
                 elementwise_trunc: bool = False, ncopies=10, device=None):
        super().__init__(q_theta, p_phi, q_theta_opt, trunc_func, elementwise_trunc, device)
        self.ncopies = ncopies

    def loss(self, X_obs, mask, *args, **kwargs):
        return marg_losses.scoreloss_marginal_trunc(
            X_obs, mask, self.p_phi, self.q_theta,
            trunc_func=self.trunc_func, elementwise_trunc=self.elementwise_trunc,
            do_iw=True, ncopies=self.ncopies, *args, **kwargs)


class TruncatedEMScoreMatchingRunner(TruncatedIWScoreMatchingRunner):
    def loss(self, X_obs, mask, *args, **kwargs):
        return marg_losses.scoreloss_em_trunc(
            X_obs, mask, self.p_phi, self.q_theta, trunc_func=self.trunc_func,
            elementwise_trunc=self.elementwise_trunc,
            do_iw=True, ncopies=self.ncopies, *args, **kwargs)
