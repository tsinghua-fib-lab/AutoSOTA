"""
The ZINBVAE variants module houses VAE variants extended with a generative model that can model gene expression.
Following the recent defaults of SCVI, all models treat size factors as observed and make direct use of inputs
instead of encoding them as additional parameters.  
"""

import pyro, copy
import pyro.distributions as dist
import torch
from pyro import poutine
from torch import nn
from torch.distributions import constraints
from functools import partial

from MLP_variants import MLP, GaussianMLP, ZINBMLP, NBMLP
from VAE_mixins import (
    _AdversarialMixin,
    _AdversarialEntropyMixin,
    _ClassificationMixin,
    _CovariateMixin,
    _CSVAENAMixin,
    _DoubleLossMixin,
    _DIVAMixin, 
    _HCSVAENAMixin,
    _VAEMixin,
)
from VAE_variants import (
    VAE, 
    CVAE, 
    CSVAENA, 
    CSVAE, 
    HCSVAENA, 
    HCSVAE,
    SDIVA,
    CCVAE,
    DLVAE,
)


class _ScBaseMixin:
    """Base class for shared capabilities across all variants."""

    def __init__(self):
        pass
        #    self.factor_estimator = MLP(1, [8]*2, 1) #NLL better w/ this but takes way longer to converge

    def get_size_factor(self, l):
        return l
    #   return torch.nn.functional.softplus(self.factor_estimator(l))
    
    def get_theta(self, x):
        theta = pyro.param(
            "inverse_dispersion",
            10.0 * x.new_ones(self.in_dim),
            constraint=constraints.positive,
        )

        return theta


    @staticmethod
    def _get_theta(*args):
        return args[-1]
        


class _NBReconMixin:
    """Appends log probability under NB for batch to loss."""
    
    def _reconstruct(self, z, *args):
        theta = self._get_theta(*args)
        mu = self.decoder(self._get_latent_args(z, *args))

        size_factors = self.get_size_factor(self._get_input_args(*args).sum(-1).reshape(-1,1))
        nb_logits = (size_factors*mu + 1e-6).log() - (theta + 1e-6).log()
        
        x_dist = dist.NegativeBinomial(
            total_count=theta,
            logits=nb_logits,
            validate_args=False,
        )

        with poutine.scale(None, self.recon_weight):
            pyro.sample("rec", x_dist.to_event(1), obs=self._get_output_args(*args))



class _ZINBReconMixin:
    """Appends log probability under ZINB for batch to loss."""
    
    def _reconstruct(self, z, *args):
        theta = self._get_theta(*args)
        gate_logits, mu = self.decoder(self._get_latent_args(z, *args))

        size_factors = self.get_size_factor(self._get_input_args(*args).sum(-1).reshape(-1,1))
        nb_logits = (size_factors*mu + 1e-6).log() - (theta + 1e-6).log()

        
        x_dist = dist.ZeroInflatedNegativeBinomial(
            gate_logits=gate_logits,
            total_count=theta,
            logits=nb_logits,
            validate_args=False,
        )

        with poutine.scale(None, self.recon_weight):
            pyro.sample("rec", x_dist.to_event(1), obs=self._get_output_args(*args))
        




    



class NBVAE(_NBReconMixin, _CovariateMixin, VAE, _ScBaseMixin):

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
        covariate_dim: int = 0,
        linear_decoder: bool = False,
    ):
        VAE.__init__(self, in_dim, hidden_dim, num_layers, latent_dim, recon_weight, kl_weight)
        _CovariateMixin.__init__(self, covariate_dim, covariate_arg_index=1)
        _ScBaseMixin.__init__(self)
        decoder_hidden_dims = [hidden_dim] * num_layers
        decoder_bias = True

        if linear_decoder:
            decoder_hidden_dims = []
            decoder_bias = False
        
        self.decoder = NBMLP(
            self.latent_dim + self.covariate_dim,
            decoder_hidden_dims,
            self.in_dim,
            bias=decoder_bias,
        )

    def model(self, *args):
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)
        
        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            theta = _ScBaseMixin.get_theta(self, x)
            _VAEMixin.model(self, *(args + tuple([theta])))


class ZINBVAE(_ZINBReconMixin, NBVAE):

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
        covariate_dim: int = 0,
        linear_decoder: bool = False,
    ):
        NBVAE.__init__(
            self,
            in_dim,
            hidden_dim,
            num_layers,
            latent_dim,
            recon_weight,
            kl_weight,
            covariate_dim,
            linear_decoder,
        )
        decoder_hidden_dims = [hidden_dim] * num_layers
        decoder_bias = True

        if linear_decoder:
            decoder_hidden_dims = []
            decoder_bias = False
        
        self.decoder = ZINBMLP(
            self.latent_dim + self.covariate_dim,
            decoder_hidden_dims,
            self.in_dim,
            bias=decoder_bias,
        )



class NBCVAE(NBVAE, _ScBaseMixin):

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
        NBVAE.__init__(self, in_dim, hidden_dim, num_layers, latent_dim, recon_weight, kl_weight)
        _ScBaseMixin.__init__(self)

        self.encoder = GaussianMLP(
            self.in_dim + label_dim, [hidden_dim] * num_layers, self.latent_dim
        )

        
        self.decoder = NBMLP(self.latent_dim + label_dim, [hidden_dim] * num_layers, self.in_dim)


    @staticmethod
    def _get_input_args(*args):
        return torch.concatenate(args[:2], dim=-1)

    @staticmethod
    def _get_latent_args(z, *args):
        return torch.concatenate((z, args[1]), dim=-1)

    @staticmethod
    def _get_output_args(*args):
        return args[0]



class ZINBCVAE(_ZINBReconMixin, NBCVAE, _ScBaseMixin):

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
        NBCVAE.__init__(self, in_dim, label_dim, hidden_dim, num_layers, latent_dim, recon_weight, kl_weight)
        
        self.decoder = ZINBMLP(self.latent_dim + label_dim, [hidden_dim] * num_layers, self.in_dim)



class NBCSVAENA(_NBReconMixin, CSVAENA, _ScBaseMixin):

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
        CSVAENA.__init__(self, in_dim, label_dims, hidden_dim, num_layers, latent_dim, w_dim, w_locs, w_scales, recon_weight, z_kl_weight, w_kl_weight)
        _ScBaseMixin.__init__(self)
     
        self.decoder = NBMLP(
            self.latent_dim + sum(self.label_dims) * self.w_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )

    
    def model(self, *args):
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)
        
        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            theta = _ScBaseMixin.get_theta(self, x)
            _CSVAENAMixin.model(self, *(args + tuple([theta])))



class ZINBCSVAENA(_ZINBReconMixin, NBCSVAENA, _ScBaseMixin):

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
        NBCSVAENA.__init__(self, in_dim, label_dims, hidden_dim, num_layers, latent_dim, w_dim, w_locs, w_scales, recon_weight, z_kl_weight, w_kl_weight)
     
        self.decoder = ZINBMLP(
            self.latent_dim + sum(self.label_dims) * self.w_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )

class NBCSVAE(NBCSVAENA, _ScBaseMixin, _AdversarialEntropyMixin):

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
        NBCSVAENA.__init__(
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
        
        _ScBaseMixin.__init__(self)
        
        self.adversarial_weight = adversarial_weight

        classifier_dims = [dim if dim != 1 else dim*2 for dim in self.label_dims]
        for i in range(len(classifier_dims)):
            setattr(
                self,
                f"classifiers_{i}",
                MLP(self.latent_dim, [hidden_dim] * num_layers, classifier_dims[i]),
            )


    
    def guide(self, *args):
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



class ZINBCSVAE(_ZINBReconMixin, NBCSVAE):
    
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
        NBCSVAE.__init__(
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
            adversarial_weight,
        )


        self.decoder = ZINBMLP(
            self.latent_dim + sum(self.label_dims) * self.w_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )


class NBHCSVAENA(_NBReconMixin, HCSVAENA, _ScBaseMixin):

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
        HCSVAENA.__init__(self, in_dim, label_dims, hidden_dim, num_layers, latent_dim, w_dim, w_locs, w_scales, recon_weight, z_kl_weight, w_kl_weight)

        _ScBaseMixin.__init__(self)
     
        self.decoder = NBMLP(
            self.latent_dim + sum(self.label_dims) * self.w_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )

    
    def model(self, *args):
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)
        
        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            theta = _ScBaseMixin.get_theta(self, x)
            _HCSVAENAMixin.model(self, *(args + tuple([theta])))



class ZINBHCSVAENA(_ZINBReconMixin, NBHCSVAENA):

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
        NBHCSVAENA.__init__(self, in_dim, label_dims, hidden_dim, num_layers, latent_dim, w_dim, w_locs, w_scales, recon_weight, z_kl_weight, w_kl_weight)

     
        self.decoder = ZINBMLP(
            self.latent_dim + sum(self.label_dims) * self.w_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )


class NBHCSVAE(NBHCSVAENA, _ScBaseMixin, _AdversarialEntropyMixin):

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
        NBHCSVAENA.__init__(
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

        _ScBaseMixin.__init__(self)

        self.adversarial_weight = adversarial_weight

        classifier_dims = [dim if dim != 1 else dim*2 for dim in self.label_dims]
        for i in range(len(classifier_dims)):
            setattr(
                self,
                f"classifiers_{i}",
                MLP(self.latent_dim, [hidden_dim] * num_layers, classifier_dims[i]),
            )


    
    def guide(self, *args):
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


class ZINBHCSVAE(_ZINBReconMixin, NBHCSVAE):

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
        NBHCSVAE.__init__(
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
            adversarial_weight,
        )

        self.decoder = ZINBMLP(
            self.latent_dim + sum(self.label_dims) * self.w_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )

class NBDIVA(_NBReconMixin, SDIVA, _ScBaseMixin):
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
        SDIVA.__init__(self, in_dim, label_dims, hidden_dim, num_layers, latent_dim, w_dim, recon_weight, kl_weight, classifier_weight)

        _ScBaseMixin.__init__(self)
     
        self.decoder = NBMLP(
            self.latent_dim + sum(self.label_dims) * self.w_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )

    
    def model(self, *args):
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)
        
        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            theta = _ScBaseMixin.get_theta(self, x)
            _DIVAMixin.model(self,  *(args + tuple([theta])))


class ZINBDIVA(_ZINBReconMixin, NBDIVA):
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
        NBDIVA.__init__(self, in_dim, label_dims, hidden_dim, num_layers, latent_dim, w_dim, recon_weight, kl_weight, classifier_weight)
     
        self.decoder = ZINBMLP(
            self.latent_dim + sum(self.label_dims) * self.w_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )


class NBCCVAE(NBDIVA):
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

        NBDIVA.__init__(self, in_dim, label_dims, hidden_dim, num_layers, latent_dim, w_dim, recon_weight , kl_weight , classifier_weight)
        self.learnable_prior_z = True
        self.prior_z = GaussianMLP(
            sum(self.label_dims),
            [hidden_dim] * num_layers,
            self.latent_dim,
        )


class ZINBCCVAE(_ZINBReconMixin, NBCCVAE):
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
        NBCCVAE.__init__(self, in_dim, label_dims, hidden_dim, num_layers, latent_dim, w_dim, recon_weight , kl_weight , classifier_weight)
        self.decoder = ZINBMLP(
            self.latent_dim + sum(self.label_dims) * self.w_dim,
            [hidden_dim] * num_layers,
            self.in_dim,
        )
    


# TODO: Make this and the ZINB version shorter somehow by integrating ReconMixins
class NBDLVAE(_CovariateMixin, DLVAE, _ScBaseMixin):

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
        covariate_dim: int = 0,
        linear_decoder: bool = False,
    ):
        DLVAE.__init__(self, in_dim, label_dims, hidden_dim, num_layers, latent_dim, w_dim, recon_weight, recon_weight_z, z_kl_weight, w_kl_weight, adversarial_weight, classifier_layers, learnable_prior)
        _CovariateMixin.__init__(self, covariate_dim, covariate_arg_index=2)

        _ScBaseMixin.__init__(self)
        decoder_hidden_dims = [hidden_dim] * num_layers
        decoder_bias = True

        if linear_decoder:
            decoder_hidden_dims = []
            decoder_bias = False
     
        self.decoder = NBMLP(
            self.w_dim + self.latent_dim + self.covariate_dim,
            decoder_hidden_dims,
            self.in_dim,
            bias=decoder_bias,
        )
        self.decoder_z = NBMLP(
            self.latent_dim + self.covariate_dim,
            decoder_hidden_dims,
            self.in_dim,
            bias=decoder_bias,
        )

    
    def model(self, *args):
        x = self._get_input_args(*args)

        pyro.module(self.__class__.__name__, self)
        
        with (
            pyro.plate("batch", x.shape[0]),
            poutine.scale(scale=1.0 / x.shape[0]),
        ):
            theta = _ScBaseMixin.get_theta(self, x)
            x_rec_z = _DoubleLossMixin.model(self, *(args + tuple([theta])))

            with poutine.scale(None, self.adversarial_weight):
                pyro.factor(
                    "adversarial_loss",
                    -1 * self._entropy_from_encodings(x_rec_z),
                )

    
    def classification(self, *args):
        """Calculates classifier / adversarial loss from inputs."""
        x = self._get_input_args(*args)

        z_loc, z_scale = self.encoder(x)
       
        z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        theta = _ScBaseMixin.get_theta(self, x)
        x_rec_z = self._reconstruct_z(z, *(args + tuple([theta])))

        return -1 * self._adversarial_from_encodings(x_rec_z, self._get_label_args(*args))
    

    def _reconstruct(self, w, *args):
        theta = self._get_theta(*args)
        mu = self.decoder(self._get_latent_args(w, *args))

        size_factors = self.get_size_factor(self._get_input_args(*args).sum(-1).reshape(-1,1))
        nb_logits = (size_factors*mu + 1e-6).log() - (theta + 1e-6).log()
        
        x_dist = dist.NegativeBinomial(
            total_count=theta,
            logits=nb_logits,
            validate_args=False,
        )

        with poutine.scale(None, self.recon_weight):
            pyro.sample("rec_w", x_dist.to_event(1), obs=self._get_output_args(*args))


    def _reconstruct_z(self, z, *args):
        theta = self._get_theta(*args)
        mu = self.decoder_z(self._get_latent_args(z, *args))

        size_factors = self.get_size_factor(self._get_input_args(*args).sum(-1).reshape(-1,1))
        nb_logits = (size_factors*mu + 1e-6).log() - (theta + 1e-6).log()
        
        x_dist = dist.NegativeBinomial(
            total_count=theta,
            logits=nb_logits,
            validate_args=False,
        )

        with poutine.scale(None, self.recon_weight_z):
            x_rec_z = pyro.sample("rec_z", x_dist.to_event(1), obs=self._get_output_args(*args))
             
        return x_rec_z



# TODO: Make this and the ZINB version shorter somehow by integrating ReconMixins
class ZINBDLVAE(NBDLVAE):

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
        covariate_dim: int = 0,
        linear_decoder: bool = False,
    ):
        NBDLVAE.__init__(
            self,
            in_dim,
            label_dims,
            hidden_dim,
            num_layers,
            latent_dim,
            w_dim,
            recon_weight,
            recon_weight_z,
            z_kl_weight,
            w_kl_weight,
            adversarial_weight,
            classifier_layers,
            learnable_prior,
            covariate_dim,
            linear_decoder,
        )
        decoder_hidden_dims = [hidden_dim] * num_layers
        decoder_bias = True

        if linear_decoder:
            decoder_hidden_dims = []
            decoder_bias = False

     
        self.decoder = ZINBMLP(
            self.w_dim + self.latent_dim + self.covariate_dim,
            decoder_hidden_dims,
            self.in_dim,
            bias=decoder_bias,
        )
        self.decoder_z = ZINBMLP(
            self.latent_dim + self.covariate_dim,
            decoder_hidden_dims,
            self.in_dim,
            bias=decoder_bias,
        )


    def _reconstruct(self, w, *args):        
        theta = self._get_theta(*args)
        gate_logits, mu = self.decoder(self._get_latent_args(w, *args))

        size_factors = self.get_size_factor(self._get_input_args(*args).sum(-1).reshape(-1,1))
        nb_logits = (size_factors*mu + 1e-6).log() - (theta + 1e-6).log()
        
        x_dist = dist.ZeroInflatedNegativeBinomial(
            gate_logits=gate_logits,
            total_count=theta,
            logits=nb_logits,
            validate_args=False,
        )

        with poutine.scale(None, self.recon_weight):
            pyro.sample("rec_w", x_dist.to_event(1), obs=self._get_output_args(*args))


    def _reconstruct_z(self, z, *args):
        theta = self._get_theta(*args)
        gate_logits, mu = self.decoder_z(self._get_latent_args(z, *args))
        
        size_factors = self.get_size_factor(self._get_input_args(*args).sum(-1).reshape(-1,1))
        nb_logits = (size_factors*mu + 1e-6).log() - (theta + 1e-6).log()
        
        x_dist = dist.ZeroInflatedNegativeBinomial(
            gate_logits=gate_logits,
            total_count=theta,
            logits=nb_logits,
            validate_args=False,
        )

         
        with poutine.scale(None, self.recon_weight_z):
           pyro.sample("rec_z", x_dist.to_event(1), obs=self._get_output_args(*args))
             
        return mu # Best estimate is normalized expression, as ZINB cannot be reparameterized
