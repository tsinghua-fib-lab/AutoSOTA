"""
Variational inference routine
"""

import torch
from torch import Tensor
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.distributions import Distribution, MultivariateNormal
from torch.distributions.utils import vec_to_tril_matrix
from torch.func import vmap, jacrev

from typing import Callable, Optional
from tqdm import tqdm

from distributions.distributions import ObservationModel


class VI(nn.Module):

    def __init__(self, state_size: int,
                 prior: Distribution,
                 likelihood: dict[str, ObservationModel],
                 inverse_transform: Optional[Callable[[Tensor], Tensor]] = None,
                 initial_loc_std: float = 0.01,
                 initial_scale_tril_std: float = 0.01,
                 lr: float = 0.1,
                 lr_decay: float = 0.999,
                 *args, **kwargs):
        """
        Variational inference routine, fitting a Gaussian to the posterior by maximising an unbiased
        estimator of the ELBO.

        Args:
            state_size: Size of sample space of posterior.
            prior: Prior distribution.
            likelihood: Dictionary of likelihood distributions / observation models.
            inverse_transform: Transform from sample space of GMM approximation to prior.
                Defaults to nn.Identity().
            initial_logits_std: Standard deviation of zero-mean normal logits initialisation.
                Defaults to 1.
            initial_loc_std: Standard deviation of zero-mean normal loc element initialisation.
                Defaults to 0.01.
            initial_scale_tril_std: Standard deviation of zero-mean normal scale_tril element initialisation.
                Defaults to 0.01.
            lr: Learning rate for optimizer.
                Defaults to 0.1.
            lr_decay: Decay constant for lr per iteration.
                Defaults to 0.999.

        """
        super().__init__()
        self.state_size = state_size
        self.inverse_transform = nn.Identity() if inverse_transform is None else inverse_transform
        self.prior = prior
        self.likelihood = likelihood

        # Distribution params
        self.loc = nn.Parameter(initial_loc_std * torch.randn(*prior.batch_shape, state_size))
        self.scale_flat = nn.Parameter(initial_scale_tril_std * torch.randn(*prior.batch_shape,
                                                                            state_size * (state_size + 1) // 2))

        self.lr = lr
        self.lr_decay = lr_decay

    def distribution(self) -> MultivariateNormal:
        """
        Get the current fitted distribution.

        Returns:
            Fitted Gaussian.

        """
        loc = self.loc
        diag = self.scale_flat[..., :self.state_size].exp()
        scale = vec_to_tril_matrix(self.scale_flat[..., self.state_size:], -1) + torch.diag_embed(diag)
        return MultivariateNormal(loc, scale_tril=scale)

    def prior_loss(self, n_samples: int = 1,
                   distribution: Optional[Distribution] = None) -> Tensor:
        """
        KL divergence between Gaussian approximation for prior and the prior itself.

        Args:
            n_samples: Number of samples with which to estimate ELBO.
            distribution: External distribution to compute ELBO for. Set to None to use internal distribution.
                Defaults to None.

        Returns:
            KL divergence loss.

        """
        x = self.distribution().sample((n_samples,)) if distribution is None else distribution.sample((n_samples,))
        kl = (self.distribution().log_prob(x)
              - self.prior.log_prob(self.inverse_transform(x).reshape(n_samples, *self.prior.batch_shape,
                                                                      *self.prior.event_shape)))
        kl -= torch.logdet(vmap(jacrev(self.inverse_transform))(x.reshape(-1, self.state_size)
                                                                ).reshape(n_samples, *self.prior.batch_shape,
                                                                          self.state_size, self.state_size))
        prob = self.distribution().log_prob(x)
        kl *= torch.exp(prob - prob.clone().detach())  # Likelihood ratio / log derivative trick
        return kl.mean(dim=0)

    def posterior_loss(self, z: dict[str, Tensor],
                       n_samples: int = 1,
                       distribution: Optional[Distribution] = None,
                       inverse_transform: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
        """
        Negative ELBO for p(x|z).

        Args:
            z: Dictionary of observation values.
            n_samples: Number of samples with which to estimate ELBO.
                Defaults to 1.
            distribution: External distribution to compute ELBO for. Set to None to use internal distribution.
                Defaults to None.
            inverse_transform: Inverse transform corresponding to provided external distribution.
                Defaults to None.

        Returns:
            Negative ELBO loss.

        """
        distribution = self.distribution() if distribution is None else distribution
        inverse_transform = self.inverse_transform if inverse_transform is None else inverse_transform
        x = distribution.sample((n_samples,))
        elbo = self.prior.log_prob(inverse_transform(x).reshape(n_samples, *self.prior.batch_shape,
                                                                *self.prior.event_shape))
        for key, likelihood in self.likelihood.items():
            likelihood.condition_(inverse_transform(x).reshape(n_samples, *self.prior.batch_shape,
                                                               *self.prior.event_shape))
            elbo += likelihood.log_prob(z[key]).reshape(elbo.shape)
        elbo -= distribution.log_prob(x)
        elbo += torch.logdet(vmap(jacrev(inverse_transform))(x.reshape(-1, self.state_size)
                                                             ).reshape(n_samples, *self.prior.batch_shape,
                                                                       self.state_size, self.state_size))
        if torch.is_grad_enabled():
            prob = distribution.log_prob(x)
            elbo *= torch.exp(prob - prob.clone().detach())  # Likelihood ratio / log derivative trick
        return -elbo.nanmean(dim=0)

    def fit(self, z: Optional[dict[str, Tensor]] = None,
            fit_prior: bool = False,
            n_iters: int = 10000,
            n_samples: int = 1,
            ewma_gamma: float = 0.01,
            progress_bar: bool = True,
            epoch: int = 1,
            num_epochs: int = 1,
            *args, **kwargs) -> dict[str, Tensor]:
        """
        Fit either prior GMM by minimising KL divergence or posterior GMM by maximising ELBO.

        Args:
            z: Dictionary of observation values. Does not need to be specified if fit_prior is True.
                Defaults to None.
            fit_prior: Whether to fit prior (True) or posterior (False).
                Defaults to posterior.
            n_iters: Number of optimization steps to carry out.
                Defaults to 1000.
            n_samples: Number of samples with which to estimate ELBO.
                Defaults to 1.
            ewma_gamma: Decay constant for EWMA of loss.
                Defaults to 0.1.
            progress_bar: Whether to include a progress bar.
                Defaults to true.
            epoch: Epoch number for progress tracking.
                Defaults to 1.
            num_epochs: Number of epochs for progress tracking.
                Defaults to 1.

        Returns:
            Dictionary of GMM parameters, using the scale_tril scale parametrisation.

        """
        if fit_prior:
            z = None
        assert (z is None) == fit_prior, "z must be specified if fitting posterior"

        average_loss = 0
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, self.lr_decay)
        tqdm_iter = tqdm(range(n_iters), desc=f'VI Epoch {epoch}/{num_epochs}') if progress_bar else None

        for i in range(n_iters):
            tqdm_iter.update() if tqdm_iter is not None else None
            optimizer.zero_grad()
            loss = self.posterior_loss(z, n_samples) if z is not None else self.prior_loss(n_samples)
            loss.nanmean().backward()
            optimizer.step()
            scheduler.step()

            if tqdm_iter:
                average_loss = ((1-ewma_gamma) * average_loss + ewma_gamma * loss)
                average_loss[torch.logical_or(torch.isinf(average_loss), torch.isnan(average_loss))] = 1e6
                if fit_prior:
                    tqdm_iter.set_postfix({"Mean KL Divergence": average_loss.mean().item(),
                                           "LR": scheduler.get_last_lr()[0]})
                else:
                    tqdm_iter.set_postfix({"Mean ELBO": -average_loss.mean().item(),
                                           "LR": scheduler.get_last_lr()[0]})

        self.lr *= self.lr_decay ** n_iters
        distribution = self.distribution()
        return {
            "loc": distribution.loc,
            "scale_tril": distribution.scale_tril
        }