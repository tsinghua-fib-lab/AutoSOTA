from abc import abstractmethod
import copy
from typing import Literal, Union, Callable
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim as opt
from torch.optim.lr_scheduler import _LRScheduler
from typing import Union, TypeAlias  # noqa: F401, F811
import higher
from ..losses import marginal_score_matching as losses  # noqa: E402
from ..models import density_models as models  # noqa: E402
from ..models import variational_models as var_models  # noqa: E402
from ..utils.regularisers import GenRegulariser  # noqa: E402
from .. import utils  # noqa: E402
RegType: TypeAlias = Union[GenRegulariser, list[GenRegulariser], None]


class BilevelGenericRunner():
    def __init__(self,
                 q_theta: models.UDensity, p_phi: var_models.VariationalDensity,
                 q_theta_opt: opt.Optimizer, p_phi_opt: opt.Optimizer,
                 n_phi_step: int = 10, n_theta_step: int = 1,
                 inner_loss: Literal["kl", "fisher"] = "fisher", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.q_theta = q_theta.to(device)
        self.p_phi = p_phi.to(device)
        self.q_theta_opt = q_theta_opt
        self.p_phi_opt = p_phi_opt
        self.n_phi_step = int(n_phi_step)
        self.n_theta_step = int(n_theta_step)
        self.set_regularisers()
        self.set_grad_opts()
        self.set_schedulers()
        self.outer_step_counter = 0
        self.true_loss_diff = False

        # Set up inner loss func
        self.inner_loss_str = inner_loss.lower()
        if self.inner_loss_str == "kl":
            self.inner_loss_func = losses.variational_kl
        elif self.inner_loss_str == "fisher":
            self.inner_loss_func = losses.variational_fisher
        else:
            raise ValueError("inner_loss must be either 'kl' or 'fisher'")

    def set_grad_opts(self, object: Literal["q", "p", "both"] = None, control_method=Literal["clip", "rerun"],
                      max_norm: float = 1., norm_type: Union[int, str] = 2):
        if object in ["q", "both"]:
            self.q_theta_grad_opts = {"control_method": control_method, "max_norm": max_norm, "norm_type": norm_type}
        else:
            self.q_theta_grad_opts = {}
        if object in ["p", "both"]:
            self.p_phi_grad_opts = {"control_method": control_method, "max_norm": max_norm, "norm_type": norm_type}
        else:
            self.p_phi_grad_opts = {}

    def init_dataset(self, X, mask, **kwargs):
        self.dataset = TensorDataset(X, mask)
        self.dataloader = DataLoader(self.dataset, shuffle=True, **kwargs)

    def set_schedulers(self, phi_scheduler: _LRScheduler = None,
                       theta_scheduler: _LRScheduler = None):
        self.p_phi_scheduler = phi_scheduler
        self.q_theta_scheduler = theta_scheduler

    def set_regularisers(self, phi_regulariser: RegType = None,
                         theta_regulariser: RegType = None):
        self.phi_regulariser = phi_regulariser
        self.theta_regulariser = theta_regulariser

    def inner_loss(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.inner_loss_func(X_obs, mask, self.p_phi, self.q_theta)

    @abstractmethod
    def loss(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass

    def true_loss(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.loss(X_obs, mask).mean()

    def _gradient_control(self, batch, grad_opts: dict, model: Literal["q", "p"] = "q"):
        if model == "q":
            model = self.q_theta
            optimizer = self.q_theta_opt
            loss_fn = self.loss
            regulariser = self.theta_regulariser

        elif model == "p":
            model = self.p_phi
            optimizer = self.p_phi_opt
            loss_fn = self.inner_loss
            regulariser = self.phi_regulariser
        else:
            raise ValueError("model must be either 'q' or 'p'")
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

    def inner_step(self, *args):
        self.p_phi_opt.zero_grad()
        inner_loss = self.inner_loss(*args).mean()
        # Regularise
        if self.phi_regulariser is not None:
            if type(self.phi_regulariser) is list:
                for reg in self.phi_regulariser:
                    inner_loss += reg(*args)
            else:
                inner_loss += self.phi_regulariser(*args)
        inner_loss.backward()
        self._gradient_control((*args,), self.p_phi_grad_opts, model="p")
        self.p_phi_opt.step()

    def outer_step(self, *args) -> torch.Tensor:
        self.q_theta_opt.zero_grad()
        outer_loss = self.loss(*args).mean()
        if self.theta_regulariser is not None:
            if type(self.theta_regulariser) is list:
                for reg in self.theta_regulariser:
                    outer_loss += reg(*args)
            else:
                outer_loss += self.theta_regulariser(*args)
        outer_loss.backward()
        self._gradient_control((*args, ), self.q_theta_grad_opts, model="q")
        self.q_theta_opt.step()
        return outer_loss

    def step(self, *args):
        # Update Phi parameter
        self.q_theta.requires_grad_(False)
        self.p_phi.requires_grad_(True)
        # Optimise variational model
        if self.outer_step_counter % self.n_theta_step == 0:
            for _ in range(self.n_phi_step):
                self.inner_step(*args)

        # Update final model
        self.q_theta.requires_grad_(True)
        self.p_phi.requires_grad_(False)
        outer_loss = self.outer_step(*args)

        self.outer_step_counter += 1
        return outer_loss

    def train(
        self, epochs=1000, niters=10000,
        snapshot_freq=100, loss_avg_length=100,
        burn_in=0, min_loss_val=-torch.inf,
        true_loss=False, verbose=False, **kwargs
    ):
        """Runs a training loop for the missing score matching model.

        Args:
            epochs (int, optional): Max number of runs through the data. Defaults to 1000.
            niters (int, optional): Max number of overall iterations. Defaults to 10000.
            snapshot_freq (int, optional): How often to store the values of the model. Defaults to 100.
            loss_avg_length (int, optional): Length of the moving average window for the loss. Defaults to 100.
            burn_in (int, optional): Number of inner iterations to run before starting outer iterations. Defaults to 0.
            min_loss_val (float, optional): Minimum loss value to stop training. Defaults to -float("inf").
            verbose (bool, optional): Whether to print progress messages. Defaults to False.
            **score_args: Additional arguments to pass to the score model.

        Returns:
            list|none: If stored_vals is None, returns a list of the stored values. Otherwise, returns None
                and stores the results in stored_vals.
        """
        self.stored_vals = {"Losses": [],
                            "q_State_dicts": [], "p_State_dicts": []}
        if true_loss:
            self.stored_vals["True_Losses"] = []
        counter = 0
        temp_loss_list = []
        # Set up data and batch size
        break_flag = False
        for i in range(burn_in):
            batch = next(iter(self.dataloader))
            batch = [x.to(self.device) for x in batch]
            self.inner_step(*batch)

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
                    if (outer_loss <= min_loss_val) | torch.isnan(outer_loss):
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
                    self.stored_vals["p_State_dicts"].append(
                        copy.deepcopy(self.p_phi.state_dict()))
                # Increment iteration counter
                counter += 1
                if counter >= niters:
                    break_flag = True
                    break
            # Update learning rate at the end of each epoch
            if self.p_phi_scheduler is not None:
                self.p_phi_scheduler.step()
            if self.q_theta_scheduler is not None:
                self.q_theta_scheduler.step()


class BilevelUnrolledGenericRunner(BilevelGenericRunner):
    def __init__(self,
                 q_theta: models.UDensity, p_phi: var_models.VariationalDensity,
                 q_theta_opt: opt.Optimizer, p_phi_opt: opt.Optimizer,
                 n_phi_step: int = 10, n_theta_step: int = 1,
                 n_unroll: int = 5, inner_loss: Literal["kl", "fisher"] = "fisher", device=None):
        super().__init__(q_theta, p_phi, q_theta_opt, p_phi_opt, n_phi_step, n_theta_step, inner_loss, device)
        self.n_unroll = n_unroll
        self.in_unroll = False
        # self.p_phi_unroll will be copy for use in inner loss in unrolling
        self.p_phi_unroll: var_models.VariationalDensity

    def inner_loss(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        phi_model = self.p_phi_unroll if self.in_unroll else self.p_phi
        return self.inner_loss_func(X_obs, mask, phi_model, self.q_theta)

    def outer_step(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Update Theta parameter (by first unrolling additional phi steps)
        self.p_phi.requires_grad_(True)
        # backup_p_phi_state_dict = self.p_phi.state_dict()
        # phi_temp = self.p_phi
        with higher.innerloop_ctx(self.p_phi, self.p_phi_opt) as (temp_p_phi, temp_p_phi_opt):
            self.in_unroll = True
            self.p_phi_unroll = temp_p_phi
            for _ in range(self.n_unroll):
                unroll_loss = self.inner_loss(X_obs, mask)
                temp_p_phi_opt.step(unroll_loss)
            self.q_theta_opt.zero_grad()
            outer_loss = self.loss(X_obs, mask).mean()
            outer_loss.backward()
            self._gradient_control((X_obs, mask), self.q_theta_grad_opts, model="q")
            if self.theta_regulariser is not None:
                outer_loss += self.theta_regulariser()
            self.q_theta_opt.step()
        # Update "constant" p_phi (not put into higher)
        self.in_unroll = False
        return outer_loss


class BilevelScoreUnrolledMarginal(BilevelUnrolledGenericRunner):
    def __init__(self,
                 q_theta: models.UDensity, p_phi: var_models.VariationalDensity,
                 q_theta_opt: opt.Optimizer, p_phi_opt: opt.Optimizer,
                 n_phi_step: int = 10, n_theta_step: int = 1,
                 n_unroll: int = 5, inner_loss: Literal["kl", "fisher"] = "fisher", device=None):

        super().__init__(q_theta, p_phi, q_theta_opt, p_phi_opt, n_phi_step, n_theta_step, n_unroll, inner_loss, device)
        self.inner_loss_str = inner_loss.lower()
        if self.inner_loss_str == "kl":
            self.inner_loss_func = losses.variational_kl
        elif self.inner_loss_str == "fisher":
            self.inner_loss_func = losses.variational_fisher
        else:
            raise ValueError("inner_loss must be either 'kl' or 'fisher'")

    def loss(self, X_obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return losses.bilevel_scoreloss(X_obs, mask, self.p_phi, self.q_theta)


class BiLevelMarginal(BilevelGenericRunner):
    def __init__(self, q_theta: models.UDensity, p_phi: models.UDensity,
                 q_theta_opt: opt.Optimizer, p_phi_opt: opt.Optimizer,
                 n_phi_step: int = 10, n_theta_step: int = 1,
                 do_iw=True, ncopies=10, inner_loss: Literal["kl", "fisher"] = "fisher", device=None):
        self.true_loss_diff = True

        super(BiLevelMarginal, self).__init__(
            q_theta, p_phi, q_theta_opt, p_phi_opt, n_phi_step, n_theta_step, inner_loss, device)
        self.do_iw = do_iw
        self.ncopies = ncopies
        self.inner_loss_str = inner_loss.lower()

    def loss(self, X_obs: torch.Tensor, mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return losses.scoreloss_marginal(
            X_obs, mask, self.p_phi, self.q_theta,
            self.do_iw, self.ncopies, *args, **kwargs)

    def true_loss(self, X_obs: torch.Tensor, mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return losses.true_scoreloss_marginal(
            X_obs, mask, self.p_phi, self.q_theta,
            self.do_iw, self.ncopies, *args, **kwargs)


class BiLevelMarginalTrunc(BiLevelMarginal):
    def __init__(self, q_theta: models.UDensity, p_phi: models.UDensity,
                 q_theta_opt: opt.Optimizer, p_phi_opt: opt.Optimizer,
                 trunc_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 elementwise_trunc: bool = False,
                 n_phi_step: int = 10, n_theta_step: int = 1,
                 do_iw=True, ncopies=10, inner_loss: Literal["kl", "fisher"] = "fisher", device=None):

        super().__init__(
            q_theta, p_phi, q_theta_opt, p_phi_opt, n_phi_step,
            n_theta_step, do_iw, ncopies, inner_loss, device)
        self.trunc_func = trunc_func
        self.elementwise_trunc = elementwise_trunc

    def loss(self, X_obs: torch.Tensor, mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return losses.scoreloss_marginal_trunc(
            X_obs, mask, self.p_phi, self.q_theta, self.trunc_func,
            self.elementwise_trunc, self.do_iw, self.ncopies, *args, **kwargs)

    def true_loss(self, X_obs: torch.Tensor, mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return losses.scoreloss_marginal_trunc(
            X_obs, mask, self.p_phi, self.q_theta, self.trunc_func,
            self.elementwise_trunc, self.do_iw, self.ncopies, *args, **kwargs)


class BiLevelEM(BilevelGenericRunner):
    def __init__(self, q_theta: models.UDensity, p_phi: models.UDensity,
                 q_theta_opt: opt.Optimizer, p_phi_opt: opt.Optimizer,
                 n_phi_step: int = 10, n_theta_step: int = 1,
                 do_iw=True, ncopies=10, inner_loss: Literal["kl", "fisher"] = "fisher", device=None):

        super(BiLevelMarginal, self).__init__(q_theta, p_phi, q_theta_opt, p_phi_opt, n_phi_step, n_theta_step,
                                              inner_loss, device)
        self.do_iw = do_iw
        self.ncopies = ncopies
        self.inner_loss_str = inner_loss.lower()

    def loss(self, X_obs: torch.Tensor, mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return losses.scoreloss_em(X_obs, mask, self.p_phi, self.q_theta,
                                   self.do_iw, self.ncopies, *args, **kwargs)


class BiLevelMSCMarginal(BiLevelMarginal):
    def __init__(self, q_theta: models.UDensity, p_phi: var_models.VariationalDensity,
                 q_theta_opt: opt.Optimizer, p_phi_opt: opt.Optimizer,
                 n_candidates: int = 10,
                 n_phi_step: int = 10, n_theta_step: int = 1,
                 do_iw=True, ncopies=10, inner_loss: Literal["kl", "fisher"] = "fisher", device=None):

        super().__init__(
            q_theta, p_phi, q_theta_opt, p_phi_opt, n_phi_step,
            n_theta_step, do_iw, ncopies, inner_loss, device)

        # Re set-up inner loss func
        # Set up inner loss func
        self.n_candidates = n_candidates
        self.inner_loss_str = inner_loss.lower()
        if self.inner_loss_str == "kl":
            self.inner_loss_func = losses.variational_forward_kl
        elif self.inner_loss_str == "fisher":
            self.inner_loss_func = losses.variational_forward_fisher
        else:
            raise ValueError("inner_loss must be either 'kl' or 'fisher'")

    def init_dataset(self, X, mask, **kwargs):
        indices = torch.arange(X.shape[0])
        self.markov_chain = self.p_phi.impute_sample(X, mask, ncopies=1).squeeze(0)
        self.dataset = TensorDataset(X, mask, indices)
        self.dataloader = DataLoader(self.dataset, shuffle=True, **kwargs)

    def inner_loss(self, X_obs: torch.Tensor, mask: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self.inner_loss_func(
            X_obs, mask, indices, self.p_phi, self.q_theta,
            markov_chain=self.markov_chain, n_candidates=self.n_candidates)

    def loss(self, X_obs: torch.Tensor, mask: torch.Tensor, batch: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return losses.scoreloss_marginal(
            X_obs, mask, self.p_phi, self.q_theta,
            self.do_iw, self.ncopies, *args, **kwargs)


class BiLevelMSCMarginalTrunc(BiLevelMSCMarginal):
    def __init__(self, q_theta: models.UDensity, p_phi: var_models.VariationalDensity,
                 q_theta_opt: opt.Optimizer, p_phi_opt: opt.Optimizer,
                 trunc_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 elementwise_trunc: bool = False,
                 n_candidates: int = 10, n_phi_step: int = 10, n_theta_step: int = 1,
                 do_iw=True, ncopies=10, inner_loss: Literal["kl", "fisher"] = "fisher", device=None):

        super().__init__(
            q_theta, p_phi, q_theta_opt, p_phi_opt, n_candidates, n_phi_step,
            n_theta_step, do_iw, ncopies, inner_loss, device)

        self.trunc_func = trunc_func
        self.elementwise_trunc = elementwise_trunc

    def loss(self, X_obs: torch.Tensor, mask: torch.Tensor, batch: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return losses.scoreloss_marginal_trunc(
            X_obs, mask, self.p_phi, self.q_theta, self.trunc_func,
            self.elementwise_trunc, self.do_iw, self.ncopies, *args, **kwargs)
