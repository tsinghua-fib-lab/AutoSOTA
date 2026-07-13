"""The VAE trainers module houses trainer classes used with VAE variants."""

import copy
import warnings
from typing import Literal

import numpy as np
import pyro
import pyro.optim as opt
import torch
import torch.utils.data as utils
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.poutine import uncondition
from torch import nn



class _BasePyroTrainerMixin:
    """Mixin for basic Pyro models. All losses should be available through a pass of model & guide. Do NOT instantiate.

    Parameters
    ----------
    model : :class:`~torch.nn.Module`
        The model to train.

    train_loader : :class:`~torch.utils.data.DataLoader`
        Data loader for the training set.

    test_loader : :class:`~torch.utils.data.DataLoader`
        Data loader for the test set.

    optim : :class:`~pyro.optim.PyroOptim`, default: None
        Optimizer to be used for training. Defaults to :class:`~pyro.optim.Adam` with learning rate 1e-3 and default parameters.

    verbose : `bool`, default: False
        Determine whether to print losses after every epoch

    Methods
    -------
    get_variables(which='train')
        Obtain encodings for the attached set specified.

    get_trace(which='train')
        Obtain model trace for the attached set specified.

    get_weights()
        Get interpretable coefficients relating input dimensions with latent variables.

    train_single_epoch()
        Trains the attached model for a single pass through the dataset.

    reset()
        Resets Pyro parameter storage for continued training.

    train()
        Trains the attached model until the designated stop condition is reached.

    is_stop_condition()
        Defines the stop condition for the model.

    save(pth)
        Saves best model to disk.

    load(pth)
        Loads model from disk.

    """

    @staticmethod
    def _send_args_to_device(args, device):
        return tuple(
            [arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args]
        )

    def __init__(
        self,
        model: nn.Module,
        train_loader: utils.DataLoader,
        test_loader: utils.DataLoader,
        optim=opt.AdamW({"lr": 1e-3}),
        verbose: bool = True,
    ):
        self.reset()
        self.train_losses, self.test_losses, self.epochs, self.predictive = (
            [],
            [],
            0,
            None,
        )
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader, self.test_loader = train_loader, test_loader
        self.model, self.elbo, self.optim = model.to(self.device), Trace_ELBO(), optim
        self.best_model = copy.deepcopy(self.model)
        self.svi = SVI(self.model.model, self.model.guide, self.optim, self.elbo)

        if hasattr(self.optim, 'pt_scheduler_constructor'):
            print("Registered optimizer with scheduler.")

    def get_variables(self, which: Literal["train", "test"] = "train"):
        """Obtain encodings for the attached set specified.

        Parameters
        ----------
        which : `Literal["train", "test"]`, default 'train'
            Specifies which of the attached sets to encode.


        """
        try:
            assert self.predictive is not None
        except AssertionError:
            warnings.warn(
                "Predictive missing, please run training before calling the trainer for predictions, returning None.",
                stacklevel=2,
            )
            return None

        match which:
            case "train":
                return self.predictive(
                    *self._send_args_to_device(
                        self.train_loader.dataset[:], self.device
                    )
                )

            case "test":
                return self.predictive(
                    *self._send_args_to_device(self.test_loader.dataset[:], self.device)
                )

            case _:
                warnings.warn(
                    'Invalid args for "which". Please enter either "train" or "test", returning None.',
                    stacklevel=2,
                )
                return None


    def get_trace(self, which: Literal["train", "test"] = "train"):
        """Obtain the full trace for specified inputs.

        Parameters
        ----------
        which : `Literal["train", "test"]`, default 'train'
            Specifies which of the attached sets to run through the model.


        """
        try:
            match which:
                case "train":
                    return self.best_model.get_model_trace(
                        *self._send_args_to_device(
                            self.train_loader.dataset[:], self.device
                        )
                    )
            
                case "test":
                    return self.best_model.get_model_trace(
                        *self._send_args_to_device(self.test_loader.dataset[:], self.device)
                    )
            
                case _:
                    warnings.warn(
                        'Invalid args for "which". Please enter either "train" or "test", returning None.',
                        stacklevel=2,
                    )
                    return None


        except AttributeError:
            print("You need to run the model before attempting to get model trace.")
            
            


        

    def _predictive_setup(self, s=1):
        try:
            self.best_model.eval()
            
        except AttributeError:
            print("Best model not tracked, defaulting to model from last epoch.")

            self.best_model = self.model
            self.best_model.eval()

        self.predictive = Predictive(
                uncondition(self.best_model.model),
                guide=self.best_model.guide,
                num_samples=s,
        )


    def train_single_epoch(self):
        """Trains the attached model for a single pass through the dataset."""
        self.model.train()

        train_losses, test_losses = [], []

        for args in self.train_loader:
            args = self._send_args_to_device(args, self.device)
            train_losses.append(self.svi.step(*args))

        self.model.eval()
        with torch.no_grad():
            for args in self.test_loader:
                args = self._send_args_to_device(args, self.device)
                test_losses.append(
                    self.elbo.loss(self.model.model, self.model.guide, *args)
                )

        if self.verbose:
            print(
                f"Epoch : {self.epochs + 1} || Train Loss: {np.mean(train_losses).round(3)} || Test Loss: {np.mean(test_losses).round(3)}"
            )

        
        if hasattr(self.optim, 'pt_scheduler_constructor'):
            self.optim.step(np.mean(test_losses)) # is actually a scheduler step so no problems!

        self.train_losses.append(np.mean(train_losses))
        self.test_losses.append(np.mean(test_losses))
        self.epochs += 1

    def reset(self):
        """Resets Pyro parameter storage for continued training."""
        pyro.clear_param_store()
        self.train_losses, self.test_losses, self.epochs = [], [], 0

    def train(self):
        """Trains the attached model until the designated stop condition is reached."""
        while not self.is_stop_condition():
            self.train_single_epoch()

        self._predictive_setup()

    def is_stop_condition(self):
        """Defines the stop condition for the model."""
        pass

    def save(self, pth: str):
        """Saves best model to disk.

        Parameters
        ----------
        pth : `str`
            Save path.
        """
        try:
            self.best_model.save(pth)

        except AttributeError:
            warnings.warn(
                "Best model not found! Please run training before saving the model through trainer interface.",
                stacklevel=2,
            )

    def load(self, pth: str):
        """Loads model from disk.

        Parameters
        ----------
        pth : `str`
            Load path.
        """
        self.model.load(pth)
        self.best_model = self.model
        self._predictive_setup()


class _AdversarialPyroTrainerMixin(_BasePyroTrainerMixin):
    """Mixin for Pyro models with adversarial loss (CSVAE). Do NOT instantiate.
    
    Parameters
    ----------
    m_step : `int`, default: 1
        Number of model iterations to train for. If m > 1, m-1 additional samples are drawn from the training set.
        
    d_step : `int`, default: 1
        Number of classifier iterations to train for. If d > 1, d-1 additional samples are drawn from the training set.
        
    model : :class:`~torch.nn.Module`
        The model to train.

    train_loader : :class:`~torch.utils.data.DataLoader`
        Data loader for the training set.

    test_loader : :class:`~torch.utils.data.DataLoader`
        Data loader for the test set.

    optim : :class:`~pyro.optim.PyroOptim`, default: None
        Optimizer to be used for training. Defaults to :class:`~pyro.optim.AdamW` with learning rate 1e-3 and default parameters.

    verbose : `bool`, default: False
        Determine whether to print losses after every epoch

    

    Methods
    -------
    get_variables(which='train')
        Obtain encodings for the attached set specified.

    get_trace(which='train')
        Obtain model trace for the attached set specified.

    get_weights()
        Get interpretable coefficients relating input dimensions with latent variables.

    train_single_epoch()
        Trains the attached model for a single pass through the dataset.

    reset()
        Resets Pyro parameter storage for continued training.

    train()
        Trains the attached model until the designated stop condition is reached.

    is_stop_condition()
        Defines the stop condition for the model.

    save(pth)
        Saves best model to disk.

    load(pth)
        Loads model from disk.
    
    """

    def __init__(
        self,
        m_step: int = 1,
        d_step: int = 1,
        *args,
        cycle_kl: bool = False,
        cycle_length: int = 10,
        z_kl_range: tuple = (0.5, 0.9),
        w_kl_range: tuple = (0.1, 0.2),
    ):
        _BasePyroTrainerMixin.__init__(self, *args)
        self.cycle_kl = cycle_kl
        self.cycle_length = cycle_length
        self.z_kl_min, self.z_kl_max = z_kl_range
        self.w_kl_min, self.w_kl_max = w_kl_range
        self._z_kl_base = getattr(self.model, 'z_kl_weight', 0.9)
        self._w_kl_base = getattr(self.model, 'w_kl_weight', 0.2)
        self.contrastive_weight = 0.0
        self.contrastive_tau = 0.1
        params = dict(self.model.named_parameters())

        classifier_params, model_params = [], []

        for k, v in params.items():
            if "classifiers" in k:
                classifier_params.append(v)

            else:
                model_params.append(v)

        self.model_opt = self.optim.pt_optim_constructor(
            model_params, **self.optim.pt_optim_args
        )
        self.classifier_opt = self.optim.pt_optim_constructor(
            classifier_params, **self.optim.pt_optim_args
        )

        if hasattr(self.optim, 'pt_scheduler_constructor'):

            self.model_scheduler = self.optim.pt_scheduler_constructor(
                self.model_opt, eps=1e-5
            )
    
            self.classifier_scheduler = self.optim.pt_scheduler_constructor(
                self.classifier_opt, eps=1e-5
            )

        
        self.svi = None
        self.m_step, self.d_step = m_step, d_step

    def _train_step_classifier(self, *args):
        self.classifier_opt.zero_grad()
        self.model.classification(*args).mean().backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.classifier_opt.step()

    def _train_step_model(self, *args):
        loss = self.elbo.differentiable_loss(self.model.model, self.model.guide, *args)
        if self.contrastive_weight > 0:
            x = args[0]
            y = args[1]
            z_loc, _ = self.model.encoder(x)
            contrast_loss = self._compute_infonce_loss(z_loc, y)
            loss = loss + self.contrastive_weight * contrast_loss
        self.model_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.model_opt.step()

        return loss.detach().cpu()

    def _compute_infonce_loss(self, z, y):
        """InfoNCE contrastive loss on z latent to encourage label-invariant representations."""
        import torch.nn.functional as F
        tau = self.contrastive_tau
        z_norm = F.normalize(z, dim=-1)
        sim = z_norm @ z_norm.T / tau
        y_flat = y.squeeze()
        pos_mask = (y_flat.unsqueeze(0) == y_flat.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)
        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        diag_exp = torch.exp(torch.diag(sim))
        all_sum = exp_sim.sum(dim=1) - diag_exp
        valid = pos_mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        loss = -torch.log(pos_sum[valid] / (all_sum[valid] + 1e-8)).mean()
        return loss

    def _update_kl_weights(self):
        """Update KL weights using cyclical schedule to prevent KL collapse."""
        if not self.cycle_kl:
            return
        import math
        progress = (self.epochs % self.cycle_length) / max(self.cycle_length, 1)
        z_w = self.z_kl_min + (self.z_kl_max - self.z_kl_min) * (1 - math.cos(2 * math.pi * progress)) / 2
        w_w = self.w_kl_min + (self.w_kl_max - self.w_kl_min) * (1 - math.cos(2 * math.pi * progress)) / 2
        self.model.z_kl_weight = z_w
        self.model.w_kl_weight = w_w

    def train_single_epoch(self):
        """Trains the attached model with adversarial loss for a single pass through the dataset."""
        self._update_kl_weights()
        self.model.train()

        train_losses, test_losses_classifier, test_losses_model = [], [], []

        for args in self.train_loader:
            args = self._send_args_to_device(args, self.device)

            self._train_step_classifier(*args) # Classifier
            
            for _ in range(self.d_step - 1): # Additional steps if hyperparameter declared
                sub_args = self._send_args_to_device(self.train_loader.dataset[np.random.choice(range(len(self.train_loader.dataset)), size=self.train_loader.batch_size)], self.device)
                self._train_step_classifier(*sub_args)
            
            
            
            train_losses.append(self._train_step_model(*args)) # Model

            for _ in range(self.m_step - 1): # Additional steps if hyperparameter declared
                sub_args = self._send_args_to_device(self.train_loader.dataset[np.random.choice(range(len(self.train_loader.dataset)), size=self.train_loader.batch_size)], self.device)
                self._train_step_model(*sub_args)

        

        self.model.eval()
        with torch.no_grad():
            for args in self.test_loader:
                args = self._send_args_to_device(args, self.device)
                test_losses_model.append(
                    (
                        self.elbo.differentiable_loss(
                            self.model.model, self.model.guide, *args
                        )
                    )
                    .detach()
                    .cpu()
                )

                test_losses_classifier.append(
                    (
                        self.model.classification(*args).mean()
                    )
                    .detach()
                    .cpu()
                )

        
        if hasattr(self.optim, 'pt_scheduler_constructor'):
            self.model_scheduler.step(np.mean(test_losses_model))
            self.classifier_scheduler.step(np.mean(test_losses_classifier))


        if self.verbose:
            print(
                "Epoch :",
                self.epochs + 1,
                "|| Train Loss:",
                np.mean(train_losses).round(3),
                "|| Test Loss:",
                (np.mean(test_losses_model)).round(3),
                "|| Classifier Loss:",
                (np.mean(test_losses_classifier)).round(3),
            )

        self.train_losses.append(np.mean(train_losses))
        self.test_losses.append(np.mean(test_losses_model))
        self.epochs += 1
            


class _EpochMixin:
    """Adds epoch based stopping capability."""

    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs

    def is_stop_condition(self):
        """Stop upon reaching the designated number of epochs."""
        try:

            if min(self.test_losses[:-1]) - self.test_losses[-1] > 0:
                del self.best_model
                self.best_model = copy.deepcopy(self.model)

        except ValueError:
            pass

        return self.epochs >= self.max_epochs


class _ThresholdMixin:
    """Adds threshold based stopping capability"""

    def __init__(self, convergence_threshold, patience):
        self.convergence_threshold = convergence_threshold
        self.patience, self.max_patience = 0, patience

    def is_stop_condition(self):
        """Stop when patience runs out without improvement."""
        if not self.patience < self.max_patience:
            return True

        try:

            if (
                min(self.test_losses[:-1]) - self.test_losses[-1]
                > self.convergence_threshold
            ):
                if self.verbose:
                    print("New minimum attained, resetting patience...")
                self.patience = 0
                del self.best_model
                self.best_model = copy.deepcopy(self.model)

            else:
                self.patience += 1

        except ValueError:
            self.patience += 1

        return False


class EpochPyroTrainer(_EpochMixin, _BasePyroTrainerMixin):
    """Trainer class that stops upon reaching the designated number of epochs.

    Parameters
    ----------
    max_epochs : int
        Number of epochs to run the model for.

    *args :
        All other arguments passed to :class:`BasePyroTrainerMixin`
    """

    def __init__(self, max_epochs: int, *args):
        _BasePyroTrainerMixin.__init__(self, *args)
        _EpochMixin.__init__(self, max_epochs)


class ThresholdPyroTrainer(_ThresholdMixin, _BasePyroTrainerMixin):
    """Trainer class that stops upon fulfilling convergence criteria.

    Parameters
    ----------
    convergence_threshold : float, default: 1e-3
        Minimum improvement required to keep the model running.

    patience : int, default: 15
        Number of epochs allowed for minimum improvement to be observed.

    *args :
        All other arguments passed to :class:`BasePyroTrainerMixin`
    """

    def __init__(self, convergence_threshold: float = 1e-3, patience: int = 15, *args):
        _BasePyroTrainerMixin.__init__(self, *args)
        _ThresholdMixin.__init__(self, convergence_threshold, patience)

    def reset(self):
        """Resets Pyro parameter storage for continued training."""
        _BasePyroTrainerMixin.reset(self)
        self.best_model = None
        self.patience = 0


class AdversarialEpochPyroTrainer(_EpochMixin, _AdversarialPyroTrainerMixin):
    """Adversarial trainer class that stops upon reaching the designated number of epochs.

    Parameters
    ----------
    max_epochs : int
        Number of epochs to run the model for.

    *args :
        All other arguments passed to :class:`BasePyroTrainerMixin`
    """

    def __init__(self, max_epochs: int, *args):
        _AdversarialPyroTrainerMixin.__init__(self, *args)
        _EpochMixin.__init__(self, max_epochs)


class AdversarialThresholdPyroTrainer(_ThresholdMixin, _AdversarialPyroTrainerMixin):
    """Adversarial trainer class that stops upon fulfilling convergence criteria.

    Parameters
    ----------
    convergence_threshold : float, default: 1e-3
        Minimum improvement required to keep the model running.

    patience : int, default: 15
        Number of epochs allowed for minimum improvement to be observed.

    *args :
        All other arguments passed to :class:`BasePyroTrainerMixin`
    """

    def __init__(self, convergence_threshold: float = 1e-3, patience: int = 15, *args, **kwargs):
        _AdversarialPyroTrainerMixin.__init__(self, *args, **kwargs)
        _ThresholdMixin.__init__(self, convergence_threshold, patience)

    def reset(self):
        """Resets Pyro parameter storage for continued training."""
        _AdversarialPyroTrainerMixin.reset(self)
        self.patience = 0
