from collections import namedtuple

import torch
from torch import nn
from torch import distributions as D
from torch.nn import functional as F

from models.uncertainty.components.eps_ctx_encode import FcModel

MNDCoef = namedtuple("MDNCoef", ['pi', 'mu', 'sigma'])


class MDN(nn.Module):
    """
    Models for up to k variables with 1D Gaussian Mixtures (coefficients are modeled independent for each variable)
    """

    def __init__(self, input_dim, n_targets, n_components, **kwargs) -> None:
        super().__init__()
        self._n_values = n_targets
        self._n_components = n_components
        self._fc_pi = FcModel(input_dim=input_dim, out_dim=n_components * n_targets, hidden=())
        self._fc2_mu = FcModel(input_dim=input_dim, out_dim=n_components * n_targets, hidden=())
        self._fc_sig = FcModel(input_dim=input_dim, out_dim=n_components * n_targets, hidden=())

    @property
    def n_components(self):
        return self._n_components

    def forward(self, x, draw_samples=1, **kwargs):
        """
        :param x:             [n_batches, latent_dim]
        :param draw_samples:  int
        :return:
         - sample [n_batches, n_targets]
         - coef   (each [n_batches, n_targets, n_components])
        """
        coef = self.get_mixture_coef(x)
        if draw_samples > 0:
            batch_size = x.shape[0]
            d = self.get_mixture_distribution_from_coef(*coef)
            sample = d.sample((draw_samples,)).reshape(batch_size, self._n_values)
        else:
            sample = None
        return sample, coef

    def get_mixture_coef(self, z):
        """
        :param z:   [n_batches, latent_dim]
        :return:
            pi:     [n_batches, n_targets, n_components]
            mu:     [n_batches, n_targets, n_components]
            sigma:  [n_batches, n_targets, n_components]
        """
        batches = z.shape[0]
        pi = F.softmax(self._fc_pi(z).view(batches * self._n_values, self._n_components), dim=1)\
            .view(batches, self._n_values, self._n_components)
        mu = self._fc2_mu(z).view(batches, self._n_values, self._n_components)
        sigma = torch.exp(self._fc_sig(z))\
            .view(batches, self._n_values, self._n_components)  # Maybe use ELU instead exp
        return MNDCoef(pi=pi, mu=mu, sigma=sigma)

    @staticmethod
    def get_mixture_distribution_from_coef(pi, mu, sigma) -> D.Distribution:
        """
        :param pi:      [n_batches, n_targets, n_components]
        :param mu:      [n_batches, n_targets, n_components]
        :param sigma:   [n_batches, n_targets, n_components]
        :return:        Distributions (n_batches * n_targets)
        """
        assert pi.shape == mu.shape and mu.shape == sigma.shape
        n_components = pi.shape[2]
        n_gm = pi.shape[0] * pi.shape[1]
        mix = D.Categorical(pi.view(n_gm, n_components))
        comp = D.Independent(D.Normal(mu.view(n_gm, n_components, 1), sigma.view(n_gm, n_components, 1)), 1)
        return D.MixtureSameFamily(mix, comp)


class MDNMutliVariate:
    """
    Models a kD Gaussian Mixture
    """

    def __init__(self, input_dim, n_targets, n_components, **kwargs) -> None:
        super().__init__()
        self._n_values = n_targets
        self._n_components = n_components
        self._fc_pi = FcModel(input_dim=input_dim, out_dim=n_components, hidden=())
        self._fc2_mu = FcModel(input_dim=input_dim, out_dim=n_components * n_targets, hidden=())
        self._fc_sig = FcModel(input_dim=input_dim, out_dim=n_components * n_targets * n_targets, hidden=())


    def get_mixture_coef(self, z):
        pass

    @staticmethod
    def get_mixture_distribution_from_coef(pi, mu, sigma) -> D.Distribution:
        """
        :param pi:      [n_batches, n_components]
        :param mu:      [n_batches, n_targets, n_components]
        :param sigma:   [n_batches, n_targets, n_targets, n_components]
        :return:        Distributions (n_batches * n_targets)
        """
        batch_size = pi.shape[0]
        n_components = pi.shape[1]
        n_vars = mu.shape[0]
        assert batch_size == mu.shape[0] and batch_size == sigma.shape[0]
        assert n_components == mu.shape[2] and n_components == sigma.shape[3]
        assert n_vars == sigma.shape[1] and n_vars == sigma.shape[2]
        mix = D.Categorical(pi.view(batch_size, n_components, ))
        comp = D.MultivariateNormal(mu, sigma)
        #comp = D.Independent(D.Normal(mu.view(n_gm, n_components, 1), sigma.view(n_gm, n_components, 1)), 1)
        return D.MixtureSameFamily(mix, comp)


class MDNLatent:
    """
    MDN where the output is a latent
    """
    def __init__(self, input_dim, out_dim, n_components, **kwargs) -> None:
        pass
