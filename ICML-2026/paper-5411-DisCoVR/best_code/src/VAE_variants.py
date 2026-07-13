"""The VAE variants module houses base model definitions for VAE style models."""

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from torch import nn

from MLP_variants import MLP, GaussianMLP
from VAE_mixins import (
    _AdversarialMixin,
    _AdversarialEntropyMixin,
    _CSVAENAMixin,
    _ClassificationMixin,
    _DIVAMixin,
    _DoubleLossMixin,
    _HCSVAENAMixin,
    _VAEMixin,
)

class _BaseReconstructionMixin:
    def _reconstruct(self, z, *args):
        x_rec = self.decoder(self._get_latent_args(z, *args))
        pyro.deterministic('rec', x_rec)

        with poutine.scale(None, self.recon_weight):
            pyro.factor(
                "reconstruction_loss",
                -1 * torch.nn.functional.mse_loss(x_rec, self._get_output_args(*args)).mean()
            )
    

class VAE(_BaseReconstructionMixin, _VAEMixin, nn.Module):
    """Base VAE class.

    Parameters
    ----------
    in_dim : `int`
        Size of the input / output space.

    hidden_dim : `int` or array_like, default: 128
        Size of the hidden layers.

    num_layers : `int` or array_like, default: 2
        Number of hidden layers.

    latent_dim : `int`, default: 10
        Size of the latent variable `z`.

    recon_weight : `float`, default: 1.0
        Weight of the reconstruction loss for the VAE.

    kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the VAE.


    Methods
    -------
    __init__(in_dim, hidden_dim=128, num_layers=2, latent_dim=10, recon_weight=20.0, kl_weight=1.0)
        Constructor for the base VAE.

    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
    ):
        nn.Module.__init__(self)
        self.latent_dim, self.in_dim, self.recon_weight, self.kl_weight = (
            latent_dim,
            in_dim,
            recon_weight,
            kl_weight,
        )

        self.encoder = GaussianMLP(self.in_dim, [hidden_dim] * num_layers, self.latent_dim)
        self.decoder = MLP(self.latent_dim, [hidden_dim] * num_layers, self.in_dim)

    def model(self, *args):
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)
        
        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _VAEMixin.model(self, *args)

    def guide(self, *args):
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)
        
        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _VAEMixin.guide(self, *args)



class CVAE(VAE, nn.Module):
    """Conditional VAE class.

    Parameters
    ----------
    in_dim : `int`
        Size of the input / output space.

    label_dim : `int`
        Size of the labels.

    hidden_dim : `int` or array_like, default: 128
        Size of the hidden layers.

    num_layers : `int` or array_like, default: 2
        Number of hidden layers.

    latent_dim : `int`, default: 10
        Size of the latent variable `z`.

    recon_weight : `float`, default: 1.0
        Weight of the reconstruction loss for the CVAE.

    kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the CVAE.


    Methods
    -------
    __init__(in_dim, labels_dim, hidden_dim=128, num_layers=2, latent_dim=10, recon_weight=20.0, kl_weight=1.0)
        Constructor for the CVAE.

    """

    def __init__(
        self,
        in_dim: int,
        label_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
    ):
        VAE.__init__(
            self, in_dim, hidden_dim, num_layers, latent_dim, recon_weight, kl_weight
        )

        self.encoder = GaussianMLP(
            self.in_dim + label_dim, [hidden_dim] * num_layers, self.latent_dim
        )
        self.decoder = MLP(
            self.latent_dim + label_dim, [hidden_dim] * num_layers, self.in_dim
        )

    @staticmethod
    def _get_input_args(*args):
        return torch.concatenate(args[:2], dim=-1)

    @staticmethod
    def _get_latent_args(z, *args):
        return torch.concatenate((z, args[1]), dim=-1)

    @staticmethod
    def _get_output_args(*args):
        return args[0]


class CSVAENA(_BaseReconstructionMixin, _CSVAENAMixin, nn.Module):
    """CSVAE without adversarial loss.

    Parameters
    ----------
    in_dim : `int`
        Size of the input / output space.

    label_dims : array-like of `int`
        List where each element is the number of subconditions for the given condition group.

    hidden_dim : `int` or array-like, default: 128
        Size of the hidden layers.

    num_layers : `int` or array-like, default: 2
        Number of hidden layers.

    latent_dim : `int`, default: 10
        Size of the latent variable `z`, assumed to be decorrelated from condition labels.

    w_dim : `int`, default: 2
        Size of the latent variable `w`, assumed to be correlated with condition labels

    w_locs : `list` of `float`, default: [0., 3.]
        Prior means for the corresponding label dimension being 0 or 1 respectively.

    w_scales : `list` of `float`, default: [0.1, 1.]
        Prior variances for the corresponding label dimension being 0 or 1 respectively.

    recon_weight : `float`, default: 1.0
        Weight of the reconstruction loss for the CSVAE.

    z_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the common latent variable.

    w_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the conditional latent variable.


    Methods
    -------
    __init__(in_dim, label_dims, hidden_dim=128, num_layers=2, latent_dim=10, w_dim=2, w_locs=None, w_scale=None, recon_weight=20.0, z_kl_weight=0.2, w_kl_weight=1.0)
        Constructor.

    """

    def __init__(
        self,
        in_dim: int,
        label_dims,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        w_dim: int = 2,
        w_locs: list = None,
        w_scales: list = None,
        recon_weight: float = 1.0,
        z_kl_weight: float = 1.0,
        w_kl_weight: float = 1.0,
    ):
        nn.Module.__init__(self)
        (
            self.latent_dim,
            self.in_dim,
            self.w_dim,
            self.label_dims,
            self.recon_weight,
            self.z_kl_weight,
            self.w_kl_weight,
        ) = (
            latent_dim,
            in_dim,
            w_dim,
            label_dims,
            recon_weight,
            z_kl_weight,
            w_kl_weight,
        )

        self.encoder = self.encoder_z = GaussianMLP(
            self.in_dim,
            [hidden_dim] * num_layers,
            self.latent_dim,
        )
        self.encoder_w = GaussianMLP(
            self.in_dim + sum(self.label_dims),
            [hidden_dim] * num_layers,
            sum(self.label_dims) * self.w_dim,
        )
        self.decoder = MLP(
            self.latent_dim + sum(self.label_dims) * self.w_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )

        if w_locs is None:
            w_locs = [0.0, 3.0]
        if w_scales is None:
            w_scales = [0.1, 1.0]
            
        self.w_locs, self.w_scales = w_locs, w_scales
        self.learnable_prior = False


    def model(self, *args):
        """Generative model.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _CSVAENAMixin.model(self, *args)

    def guide(self, *args):
        """Approximate variational posterior.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _CSVAENAMixin.guide(self, *args)
    

class CSVAE(CSVAENA, _AdversarialEntropyMixin, nn.Module):
    """Conditional Subspace VAE class.

    Parameters
    ----------
    in_dim : `int`
        Size of the input / output space.

    label_dims : array-like of `int`
        List where each element is the number of subconditions for the given condition group.

    hidden_dim : `int` or array-like, default: 128
        Size of the hidden layers.

    num_layers : `int` or array-like, default: 2
        Number of hidden layers.

    latent_dim : `int`, default: 10
        Size of the latent variable `z`, assumed to be decorrelated from condition labels.

    w_dim : `int`, default: 2
        Size of the latent variable `w`, assumed to be correlated with condition labels

    w_locs : `list` of `float`, default: [0., 3.0]
        Prior means for the corresponding label dimension being 0 or 1 respectively.

    w_scales : `list` of `float`, default: [0.1, 1.0]
        Prior variances for the corresponding label dimension being 0 or 1 respectively.

    recon_weight : `float`, default: 1.0
        Weight of the reconstruction loss for the CSVAE.

    z_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the common latent variable of the CSVAE.

    w_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the conditional latent variable of the CSVAE.

    adversarial_weight : `float`, default: 1.0
        Weight of the adversarial loss.


    Methods
    -------
    __init__(in_dim, label_dims, hidden_dim=128, num_layers=2, latent_dim=10,  w_dim=2, w_locs=None, w_scale=None, recon_weight=20.0, z_kl_weight=0.2, w_kl_weight=1.0, adversarial_weight=1.0)
        Constructor for the CSVAE.

    """

    def __init__(
        self,
        in_dim: int,
        label_dims,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        w_dim: int = 2,
        w_locs: list = None,
        w_scales: list = None,
        recon_weight: float = 1.0,
        z_kl_weight: float = 1.0,
        w_kl_weight: float = 1.0,
        adversarial_weight: float = 1.0,
    ):
        CSVAENA.__init__(
            self,
            in_dim,
            label_dims,
            hidden_dim,
            num_layers,
            latent_dim,
            w_dim,
            w_locs,
            w_scales,
            recon_weight,
            z_kl_weight,
            w_kl_weight,
        )

        self.adversarial_weight = adversarial_weight

        classifier_dims = [dim if dim != 1 else dim*2 for dim in self.label_dims]
        for i in range(len(classifier_dims)):
            setattr(
                self,
                f"classifiers_{i}",
                MLP(self.latent_dim, [hidden_dim] * num_layers, classifier_dims[i]),
            )

    def guide(self, *args):
        """Approximate variational posterior for the CSVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            z = _CSVAENAMixin.guide(self, *args)

            with poutine.scale(None, self.adversarial_weight):
                pyro.factor(
                    "adversarial_loss",
                    self._entropy_from_encodings(z),
                    has_rsample=True,
                )


class HCSVAENA(_BaseReconstructionMixin, _HCSVAENAMixin, nn.Module):
    """Hierarchical Structured Conditional VAE class.

    Parameters
    ----------
    in_dim : `int`
        Size of the input / output space.

    label_dims : array-like of `int`
        List where each element is the number of subconditions for the given condition group.

    hidden_dim : `int` or array-like, default: 128
        Size of the hidden layers.

    num_layers : `int` or array-like, default: 2
        Number of hidden layers.

    latent_dim : `int`, default: 10
        Size of the latent variable `z`, assumed to be decorrelated from condition labels.

    w_dim : `int`, default: 2
        Size of the latent variable `w`, assumed to be correlated with condition labels.

    w_locs : `list` of `float`, default: [0., 3.0]
        Prior means for the corresponding label dimension being 0 or 1 respectively.

    w_scales : `list` of `float`, default: [0.1, 1.0]
        Prior variances for the corresponding label dimension being 0 or 1 respectively.

    recon_weight : `float`, default: 1.0
        Weight of the reconstruction loss.

    z_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the common latent variable.

    w_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the conditional latent variable.


    Methods
    -------
    __init__(in_dim, label_dims, hidden_dim=128, num_layers=2, latent_dim=10,  w_dim=10, , w_locs=None, w_scale=None, recon_weight=20.0, z_kl_weight=0.2, w_kl_weight=1.0)
        Constructor.

    """

    def __init__(
        self,
        in_dim: int,
        label_dims,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        w_dim: int = 2,
        w_locs: list = None,
        w_scales: list = None,
        recon_weight: float = 1.0,
        z_kl_weight: float = 1.0,
        w_kl_weight: float = 1.0,
    ):
        nn.Module.__init__(self)
        (
            self.in_dim,
            self.latent_dim,
            self.w_dim,
            self.label_dims,
            self.recon_weight,
            self.z_kl_weight,
            self.w_kl_weight,
        ) = (
            in_dim,
            latent_dim,
            w_dim,
            label_dims,
            recon_weight,
            z_kl_weight,
            w_kl_weight,
        )

        self.rho_dim = self.latent_dim + sum(self.label_dims) * self.w_dim

        self.encoder_rho = GaussianMLP(
            self.in_dim,
            [hidden_dim] * num_layers,
            self.rho_dim,
        )
        self.encoder = self.encoder_z = GaussianMLP(
            self.rho_dim,
            [hidden_dim] * num_layers,
            self.latent_dim,
        )
        self.encoder_w = GaussianMLP(
            self.rho_dim + sum(self.label_dims),
            [hidden_dim] * num_layers,
            sum(self.label_dims) * self.w_dim,
        )
        self.decoder_rho = GaussianMLP(
            self.rho_dim,
            [hidden_dim] * num_layers,
            self.rho_dim,
        )
        self.decoder = MLP(
            self.rho_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )

        if w_locs is None:
            w_locs = [0.0, 3.0]
        if w_scales is None:
            w_scales = [0.1, 1.0]
        self.w_locs, self.w_scales = w_locs, w_scales
        self.learnable_prior = False

    def model(self, *args):
        """Generative model.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _HCSVAENAMixin.model(self, *args)

    def guide(self, *args):
        """Approximate variational posterior.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _HCSVAENAMixin.guide(self, *args)


class HCSVAE(HCSVAENA, _AdversarialEntropyMixin, nn.Module):
    """Hierarchical Conditional Subspace VAE class.

    Parameters
    ----------
    in_dim : `int`
        Size of the input / output space.

    label_dims : array-like of `int`
        List where each element is the number of subconditions for the given condition group.

    hidden_dim : `int` or array-like, default: 128
        Size of the hidden layers.

    num_layers : `int` or array-like, default: 2
        Number of hidden layers.

    latent_dim : `int`, default: 10
        Size of the latent variable `z`, assumed to be decorrelated from condition labels.

    w_dim : `int`, default: 2
        Size of the latent variable `w`, assumed to be correlated with condition labels.

    w_locs : `list` of `float`, default: [0., 3.0]
        Prior means for the corresponding label dimension being 0 or 1 respectively.

    w_scales : `list` of `float`, default: [0.1, 1.0]
        Prior variances for the corresponding label dimension being 0 or 1 respectively.

    recon_weight : `float`, default: 1.0
        Weight of the reconstruction loss.

    z_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the common latent variable of the HCSVAE.

    w_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the conditional latent variable of the HCSVAE.

    adversarial_weight : `float`, default: 1.0
        Weight of the adversarial loss term.


    Methods
    -------
    __init__(in_dim, label_dims, hidden_dim=128, num_layers=2, latent_dim=10, w_dim=10, recon_weight=20.0, z_kl_weight=0.2, w_kl_weight=1.0, adversarial_weight=1.0)
        Constructor for the HCSVAE.

    """

    def __init__(
        self,
        in_dim: int,
        label_dims,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        w_dim: int = 2,
        w_locs: list = None,
        w_scales: list = None,
        recon_weight: float = 1.0,
        z_kl_weight: float = 1.0,
        w_kl_weight: float = 1.0,
        adversarial_weight: float = 1.0,
    ):
        HCSVAENA.__init__(
            self,
            in_dim,
            label_dims,
            hidden_dim,
            num_layers,
            latent_dim,
            w_dim,
            w_locs,
            w_scales,
            recon_weight,
            z_kl_weight,
            w_kl_weight,
        )

        self.adversarial_weight = adversarial_weight

        classifier_dims = [dim if dim != 1 else dim*2 for dim in self.label_dims]
        for i in range(len(classifier_dims)):
            setattr(
                self,
                f"classifiers_{i}",
                MLP(self.latent_dim, [hidden_dim] * num_layers, classifier_dims[i]),
            )

    def guide(self, *args):
        """Approximate variational posterior for the HCSVAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            z = _HCSVAENAMixin.guide(self, *args)

            with poutine.scale(None, self.adversarial_weight):
                pyro.factor(
                    "adversarial_loss",
                    self._entropy_from_encodings(z),
                    has_rsample=True,
                )

    def classification(self, *args):
        """Calculates classifier / adversarial loss from inputs."""
        x = self._get_input_args(*args)

        rho_loc, rho_scale = self.encoder_rho(x)
        rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))

        z_loc, z_scale = self.encoder_z(rho)

        z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        return -1 * self._adversarial_from_encodings(z, self._get_label_args(*args))



class SDIVA(_BaseReconstructionMixin, _ClassificationMixin, _DIVAMixin, nn.Module):
    """Supervised DIVA implementation (https://proceedings.mlr.press/v121/ilse20a/ilse20a.pdf, https://arxiv.org/pdf/2006.10102).

    Parameters
    ----------
    in_dim : `int`
        Size of the input / output space.

    label_dims : array-like of `int`
        List where each element is the number of subconditions for the given condition group.

    hidden_dim : `int` or array-like, default: 128
        Size of the hidden layers.

    num_layers : `int` or array-like, default: 2
        Number of hidden layers.

    latent_dim : `int`, default: 10
        Size of the latent variable `z`, assumed to be decorrelated from condition labels.

    w_dim : `int`, default: 2
        Size of the latent variable `w`, assumed to be correlated with condition labels

    recon_weight : `float`, default: 1.0
        Weight of the reconstruction loss for the CSVAE.

    kl_weight : `float`, default: 1.0
        Weight of the KL divergence losses.

    classifier_weight : `float`, default: 1.0
        Weight of the classifier losses to encourage separation.


    Methods
    -------
    __init__()
        Constructor.

    """

    def __init__(
        self,
        in_dim: int,
        label_dims,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        w_dim: int = 2,
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
        classifier_weight: float = 1.0
    ):
        nn.Module.__init__(self)
        (
            self.latent_dim,
            self.in_dim,
            self.w_dim,
            self.label_dims,
            self.recon_weight,
            self.z_kl_weight,
            self.w_kl_weight,
            self.classifier_weight
        ) = (
            latent_dim,
            in_dim,
            w_dim,
            label_dims,
            recon_weight,
            kl_weight,
            kl_weight,
            classifier_weight,
        )

        self.encoder = self.encoder_z = GaussianMLP(
            self.in_dim,
            [hidden_dim] * num_layers,
            self.latent_dim,
        )
        self.encoder_w = GaussianMLP(
            self.in_dim,
            [hidden_dim] * num_layers,
            sum(self.label_dims) * self.w_dim,
        )
        self.decoder = MLP(
            self.latent_dim + sum(self.label_dims) * self.w_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )

        self.prior_w = GaussianMLP(
                sum(self.label_dims),
                [hidden_dim] * num_layers,
                sum(self.label_dims) * self.w_dim,
            )

        self.learnable_prior_z = False

        

        classifier_dims = [dim if dim != 1 else dim*2 for dim in self.label_dims]
        for i in range(len(classifier_dims)):
            setattr(
                self,
                f"classifiers_{i}",
                MLP(self.w_dim*self.label_dims[i], [hidden_dim] * num_layers, classifier_dims[i]),
            )

    def model(self, *args):
        """Generative model.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _DIVAMixin.model(self, *args)

    def guide(self, *args):
        """Approximate variational posterior.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _, w = _DIVAMixin.guide(self, *args)

            with poutine.scale(None, self.classifier_weight):
                pyro.factor(
                    "classifier_loss",
                    self._classification_from_encodings(w, self._get_label_args(*args)),
                    has_rsample=True,
                )


class CCVAE(SDIVA):
    """Supervised CCVAE implementation (https://arxiv.org/pdf/2006.10102). Since the correct DIVA 
     formulation should include the additional classifier loss, not much is different except the prior on z.

    Parameters
    ----------
    in_dim : `int`
        Size of the input / output space.

    label_dims : array-like of `int`
        List where each element is the number of subconditions for the given condition group.

    hidden_dim : `int` or array-like, default: 128
        Size of the hidden layers.

    num_layers : `int` or array-like, default: 2
        Number of hidden layers.

    latent_dim : `int`, default: 10
        Size of the latent variable `z`, assumed to be decorrelated from condition labels.

    w_dim : `int`, default: 2
        Size of the latent variable `w`, assumed to be correlated with condition labels

    recon_weight : `float`, default: 1.0
        Weight of the reconstruction loss for the CSVAE.

    kl_weight : `float`, default: 1.0
        Weight of the KL divergence losses.

    classifier_weight : `float`, default: 1.0
        Weight of the classifier losses to encourage separation.


    Methods
    -------
    __init__()
        Constructor.

    """

    def __init__(
        self,
        in_dim: int,
        label_dims,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        w_dim: int = 2,
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
        classifier_weight: float = 1.0
    ):

        SDIVA.__init__(self, in_dim, label_dims, hidden_dim, num_layers, latent_dim, w_dim, recon_weight , kl_weight , classifier_weight)
        self.learnable_prior_z = True
        self.prior_z = GaussianMLP(
            sum(self.label_dims),
            [hidden_dim] * num_layers,
            self.latent_dim,
        )
    


# TODO: Make this smaller by incorporating _BaseReconMixin
class DLVAE(_DoubleLossMixin, _AdversarialEntropyMixin, nn.Module):
    """Our model, based on CSVAE as the backbone (without strong prior).

    Parameters
    ----------
    in_dim : `int`
        Size of the input / output space.

    label_dims : array-like of `int`
        List where each element is the number of subconditions for the given condition group.

    hidden_dim : `int` or array-like, default: 128
        Size of the hidden layers.

    num_layers : `int` or array-like, default: 2
        Number of hidden layers.

    latent_dim : `int`, default: 10
        Size of the latent variable `z`, assumed to be decorrelated from condition labels.

    w_dim : `int`, default: 10
        Size of the latent variable `w`, assumed to be correlated with condition labels.

    recon_weight : `float`, default: 1.0
        Weight of the reconstruction loss.

    recon_weight_z : `float`, default: 1.0
        Weight of the secondary reconstruction loss.

    z_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the common latent variable.

    w_kl_weight : `float`, default: 1.0
        Weight of the KL divergence loss for the conditional latent variable.

    adversarial_weight : `float`, default: 1.0
        Weight of the adversarial loss term.

    classifier_layers : `int`, default: 1
        Number of hidden layers for the classifier.

    Methods
    -------
    __init__(in_dim, labels_dim, hidden_dim=128, num_layers=2, latent_dim=10,  w_dim=10, recon_weight=20.0, z_kl_weight=1.0, w_kl_weight=1.0, adversarial_weight=1.0)
        Constructor.

    """

    def __init__(
        self,
        in_dim: int,
        label_dims,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        w_dim: int = 10,
        recon_weight: float = 1.0,
        recon_weight_z: float = 1.0,
        z_kl_weight: float = 1.0,
        w_kl_weight: float = 1.0,
        adversarial_weight: float = 1.0,
        classifier_layers: int = 1,
        learnable_prior: bool = False,
    ):
        nn.Module.__init__(self)
        (
            self.in_dim,
            self.latent_dim,
            self.w_dim,
            self.label_dims,
            self.recon_weight,
            self.recon_weight_z,
            self.z_kl_weight,
            self.w_kl_weight,
            self.classifier_layers,
            self.learnable_prior
        ) = (
            in_dim,
            latent_dim,
            w_dim,
            label_dims,
            recon_weight,
            recon_weight_z,
            z_kl_weight,
            w_kl_weight,
            classifier_layers,
            learnable_prior
        )

        self.encoder = GaussianMLP(
            self.in_dim,
            [hidden_dim] * num_layers,
            self.latent_dim,
        )
        self.encoder_w = GaussianMLP(
            self.in_dim + sum(self.label_dims),
            [hidden_dim] * num_layers,
            self.w_dim,
        )
        self.decoder = MLP(
            self.w_dim + self.latent_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )
        self.decoder_z = MLP(
            self.latent_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )

        if self.learnable_prior:
            self.prior_w = GaussianMLP(
                self.latent_dim + sum(self.label_dims),
                [hidden_dim] * num_layers,
                self.w_dim,
            )


        self.adversarial_weight = adversarial_weight

        classifier_dims = [dim if dim != 1 else dim*2 for dim in self.label_dims]
        for i in range(len(classifier_dims)):
            setattr(
                self,
                f"classifiers_{i}",
                MLP(self.in_dim, [hidden_dim] * self.classifier_layers, classifier_dims[i]),
            )
            

    def model(self, *args):
        """Generative model.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            x_rec_z = _DoubleLossMixin.model(self, *args)

            with poutine.scale(None, self.adversarial_weight):
                pyro.factor(
                    "adversarial_loss",
                    -1*self._entropy_from_encodings(x_rec_z),
                )


    def guide(self, *args):
        """Approximate variational posterior.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)

        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            _DoubleLossMixin.guide(self, *args)

    
    def classification(self, *args):
        """Calculates classifier / adversarial loss from inputs."""
        x = self._get_input_args(*args)

        z_loc, z_scale = self.encoder(x)
        
        z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        x_rec_z = self._reconstruct_z(z, *args)

        return -1 * self._adversarial_from_encodings(x_rec_z, self._get_label_args(*args))
    

    def _reconstruct(self, w, *args):
        x_rec_w = self.decoder(w)

        pyro.deterministic('rec_w', x_rec_w)

        with poutine.scale(None, self.recon_weight):
            pyro.factor(
                "reconstruction_loss_w",
                -1 * torch.nn.functional.mse_loss(x_rec_w, self._get_output_args(*args)).mean()
            ) 

    def _reconstruct_z(self, z, *args):
        x_rec_z = self.decoder_z(z)

        pyro.deterministic('rec_z', x_rec_z)

        with poutine.scale(None, self.recon_weight_z):
            pyro.factor(
                "reconstruction_loss_z",
                -1 * torch.nn.functional.mse_loss(x_rec_z, self._get_output_args(*args)).mean()
            ) 
             
        return x_rec_z