"""The VAE variants module houses base model definitions for VAE style models."""

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from MLP_variants import GaussianMLP


class _ModelMixin:
    """Includes shared capabilities for models.

    Methods
    -------
    save(path)
        Saves model parameters to disk.

    load(path, map_location=None)
        Loads model parameters from disk.

    """

    def save(self, path):
        """Saves model parameters to disk.

        Parameters
        ----------
        path : `str`
            Path to save model parameters.
        """
        torch.save(self.state_dict(), path + "_torch.pth")
        pyro.get_param_store().save(path + "_pyro.pth")

    def load(self, path, map_location=None):
        """Loads model parameters from disk.

        Parameters
        ----------
        path : `str`
            Path to find model parameters. Should not include the extensions `_torch.pth` or `_pyro.pth`.

        map_location : `str`, default: None
            Specifies where the model should be loaded. See :class:`~torch.device` for details.
        """
        pyro.clear_param_store()

        if map_location is None:
            self.load_state_dict(torch.load(path + "_torch.pth"))
            pyro.get_param_store().load(path + "_pyro.pth")

        else:
            self.load_state_dict(
                torch.load(path + "_torch.pth", map_location=map_location)
            )
            pyro.get_param_store().load(path + "_pyro.pth", map_location=map_location)


class _CovariateMixin:
    """Batch / covariate handling for VAE-style models.

    Parameters
    ----------
    covariate_dim : `int`
        Dimension of the user-provided decoder covariates.
        For categorical batch correction, pass one-hot encoded batch indicators and set this to the number of batches. 
        Scalar batch codes are accepted when `covariate_dim=1`, but are treated as continuous / ordinal covariates.

    covariate_arg_index: `int`
        The index of the covariate argument s in the model input. For VAE, (x, s) index is 1. For DLVAE, (x, y, s) index is 2.
    """

    def __init__(
        self,
        covariate_dim: int = 0,
        covariate_arg_index: int | None = None,
    ):
        self.covariate_dim = covariate_dim
        self._covariate_arg_index = covariate_arg_index

    def _get_covariate_args(self, *args):
        if self.covariate_dim == 0:
            return None

        if self._covariate_arg_index is None or len(args) <= self._covariate_arg_index:
            raise ValueError(
                f"Expected covariates at positional argument {self._covariate_arg_index}; "
                f"received {len(args)} positional arguments."
            )

        covariates = args[self._covariate_arg_index]
        if covariates.shape[-1] != self.covariate_dim:
            raise ValueError(
                f"Expected covariates with last dimension {self.covariate_dim}, "
                f"received {covariates.shape[-1]}."
            )

        return covariates

    def _get_latent_args(self, z, *args):
        covariates = self._get_covariate_args(*args)
        if covariates is None:
            return z

        covariates = covariates.to(device=z.device, dtype=z.dtype)
        return torch.concatenate((z, covariates), dim=-1)


class _VAEMixin(_ModelMixin):
    """Includes shared capabilities for VAE based models.

    Methods
    -------
    model(*args)
        Generative model for the VAE.

    guide(*args)
        Approximate variational posterior for the VAE.

    """

    @staticmethod
    def _get_input_args(*args):
        return args[0]

    @staticmethod
    def _get_latent_args(z, *args):
        return z

    @staticmethod
    def _get_output_args(*args):
        return args[0]


    def model(self, *args):
        """Generative model for the base VAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)
        
        z_loc, z_scale = torch.zeros((x.shape[0], self.latent_dim)).to(
            x.device
        ), torch.ones((x.shape[0], self.latent_dim)).to(x.device)

        with poutine.scale(None, self.kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        self._reconstruct(z, *args)

    def guide(self, *args):
        """Approximate variational posterior for the base VAE.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        z_loc, z_scale = self.encoder(x)

        with poutine.scale(None, self.kl_weight):
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))


    def get_model_trace(self, *args):
        """Get the full model trace from pyro for downstream calculations."""
        guide_trace = poutine.trace(self.guide).get_trace(*args)
        model_trace = poutine.trace(
            poutine.replay(self.model, trace=guide_trace)
        ).get_trace(*args) 

        return model_trace



class _CSVAENAMixin(_VAEMixin):
    """Includes shared capabilities for CSVAE based models.

    Methods
    -------
    model(*args)
        Generative model for the CSVAENA.

    guide(*args)
        Approximate variational posterior for the CSVAENA.

    """

    @staticmethod
    def _get_input_args(*args):
        return args[0]

    @staticmethod
    def _get_output_args(*args):
        return args[0]

    @staticmethod
    def _get_label_args(*args):
        return args[1]

    @staticmethod
    def _concat_lat_dims(labels, ref_list, dim):
        idxs = labels.int()
        return (
            torch.tensor(
                np.array(
                    [
                        np.concatenate([[ref_list[num]] * dim for num in elem])
                        for elem in idxs
                    ]
                )
            )
            .type_as(labels)
            .to(labels.device)
        )

    def model(self, *args):
        """Generative model for the CSVAENA.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x, y = self._get_input_args(*args), self._get_label_args(*args)

        z_loc, z_scale = torch.zeros((x.shape[0], self.latent_dim)).to(
            x.device
        ), torch.ones((x.shape[0], self.latent_dim)).to(x.device)

    
        w_loc, w_scale = self._concat_lat_dims(
            y, self.w_locs, self.w_dim
        ), self._concat_lat_dims(y, self.w_scales, self.w_dim)

        with poutine.scale(None, self.z_kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        with poutine.scale(None, self.w_kl_weight):
            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        self._reconstruct(torch.concatenate((z, w), dim=-1), *args)
        
            
    def guide(self, *args):
        """Approximate variational posterior for the CSVAENA.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x, y = self._get_input_args(*args), self._get_label_args(*args)

        z_loc, z_scale = self.encoder_z(x)
        w_loc, w_scale = self.encoder_w(torch.concatenate((x, y), dim=-1))

        with poutine.scale(None, self.z_kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        with poutine.scale(None, self.w_kl_weight):
            pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        return z


class _HCSVAENAMixin(_CSVAENAMixin):
    """Includes shared capabilities for HCSVAENA based models.

    Methods
    -------
    model(*args)
        Generative model for the HCSVAENA.

    guide(*args)
        Approximate variational posterior for the HCSVAENA.

    """

    def model(self, *args):
        """Generative model for the HCSVAENA.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x, y = self._get_input_args(*args), self._get_label_args(*args)

        z_loc, z_scale = torch.zeros((x.shape[0], self.latent_dim)).to(
            x.device
        ), torch.ones((x.shape[0], self.latent_dim)).to(x.device)

        with poutine.scale(None, self.z_kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        w_loc, w_scale = self._concat_lat_dims(
            y, self.w_locs, self.w_dim
        ), self._concat_lat_dims(y, self.w_scales, self.w_dim)

        with poutine.scale(None, self.w_kl_weight):
            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        rho_loc, rho_scale = self.decoder_rho(torch.concatenate((z, w), dim=-1))
        rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))

        self._reconstruct(rho, *args)

        return z

    def guide(self, *args):
        """Approximate variational posterior for the HCSVAENA.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        rho_loc, rho_scale = self.encoder_rho(x)
        rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))

        z_loc, z_scale = self.encoder_z(rho)

        with poutine.scale(None, self.z_kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        w_loc, w_scale = self.encoder_w(
            torch.concatenate((rho, self._get_label_args(*args)), dim=-1)
        )

        with poutine.scale(None, self.w_kl_weight):
            pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        return z


class _DoubleLossMixin(_CSVAENAMixin):
    """Base with additional MLP parameterizing W and additional reconstruction loss.
    
    Methods
    -------
    model(*args)
        Generative model.

    guide(*args)
        Approximate variational posterior.
    """

    
    def _concat_lat_dims(self, labels, z):
        batch_means = {tuple(level.tolist()) : z[torch.all(torch.tensor(tuple(level.tolist())).to(labels.device) == labels, dim=-1)].mean(dim=0) for level in labels.int().unique(dim=0)}
        
        batched_priors = torch.vstack([batch_means[tuple(label.tolist())] for label in labels.int()])
        
        return batched_priors.type_as(labels).to(labels.device)

    def model(self, *args):
        """Generative model.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x, y = self._get_input_args(*args), self._get_label_args(*args)

        # Z
        z_loc, z_scale = torch.zeros((x.shape[0], self.latent_dim)).to(
            x.device
        ), torch.ones((x.shape[0], self.latent_dim)).to(x.device)

        with poutine.scale(None, self.z_kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
        
        # W
        if self.learnable_prior:
            w_loc, w_scale = self.prior_w(torch.concatenate((z, y), dim=-1))

        else:
            w_loc, w_scale = self._concat_lat_dims(y, z), torch.ones((x.shape[0], self.w_dim)).to(x.device)
        
        with poutine.scale(None, self.w_kl_weight):
            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        self._reconstruct(torch.concatenate((w, z), dim=-1), *args)
        x_rec_z = self._reconstruct_z(z, *args)
        
        return x_rec_z

    def guide(self, *args):
        """Approximate variational posterior.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        
        x, y = self._get_input_args(*args), self._get_label_args(*args)

        z_loc, z_scale = self.encoder(x)
        
        with poutine.scale(None, self.z_kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        w_loc, w_scale = self.encoder_w(
             torch.concatenate((x, y), dim=-1)
        ) 

        with poutine.scale(None, self.w_kl_weight):
            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))
        
        return z


class _DIVAMixin(_VAEMixin):
    """Includes shared capabilities for SDIVA based models.

    Methods
    -------
    model(*args)
        Generative model for the SDIVA.

    guide(*args)
        Approximate variational posterior for the SDIVA.

    """

    @staticmethod
    def _get_input_args(*args):
        return args[0]

    @staticmethod
    def _get_output_args(*args):
        return args[0]

    @staticmethod
    def _get_label_args(*args):
        return args[1]

    
    def model(self, *args):
        """Generative model for the SDIVA.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x, y = self._get_input_args(*args), self._get_label_args(*args)

        if self.learnable_prior_z:
            z_loc, z_scale = self.prior_z(y)

        else:
            z_loc, z_scale = torch.zeros((x.shape[0], self.latent_dim)).to(
                x.device
            ), torch.ones((x.shape[0], self.latent_dim)).to(x.device)
    
        w_loc, w_scale = self.prior_w(y)

        with poutine.scale(None, self.z_kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        with poutine.scale(None, self.w_kl_weight):
            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        self._reconstruct(torch.concatenate((z, w), dim=-1), *args)
        
            
    def guide(self, *args):
        """Approximate variational posterior for the CSVAENA.

        Parameters
        ----------
        *args :
            Static methods are used to pick the correct args from multiple args.
        """
        x = self._get_input_args(*args)

        z_loc, z_scale = self.encoder_z(x)
        w_loc, w_scale = self.encoder_w(x)

        with poutine.scale(None, self.z_kl_weight):
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        with poutine.scale(None, self.w_kl_weight):
            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        return z, w


class _ClassificationMixin:
    """Includes shared capabilities for models making use of a classifier to encourage separation.

    Methods
    -------
    adversarial(*args)
        Calculates classifier / adversarial loss from inputs.

    """
    def _classification_from_encodings(self, w, y):
        classification_loss, attr_track = 0, 0

        for i in range(len(self.label_dims)):
            label = y[..., attr_track : attr_track + self.label_dims[i]]
            
            if self.label_dims[i] == 1:
                label = torch.nn.functional.one_hot(label.type(torch.long), num_classes=2).squeeze()

            classification_loss += dist.OneHotCategorical(
                logits=getattr(self, f"classifiers_{i}")(w[..., attr_track*self.w_dim : (attr_track + self.label_dims[i])*self.w_dim])
            ).log_prob(label)

            attr_track = attr_track + self.label_dims[i]

        return -1 * classification_loss.mean()


    def classification(self, *args, reparam=False):
        """Calculates classifier loss from inputs. Uses `encoder_w` as function for inputs to pass through."""

        w_loc, w_scale = self.encoder_w(self._get_input_args(*args))

        if reparam:
            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        else: # CCVAE recommends not reparameterizing classifier inputs to avoid gradient variance.
            w = w_loc

        return self._classification_from_encodings(w, self._get_label_args(*args))
    


class _AdversarialMixin:
    """Includes shared capabilities for models making use of adversarial information loss to encourage mixing.

    Methods
    -------
    adversarial(*args)
        Calculates classifier / adversarial loss from inputs.

    """
    def _adversarial_from_encodings(self, z, y):
        classification_loss, attr_track = 0, 0

        for i in range(len(self.label_dims)):
            label = y[..., attr_track : attr_track + self.label_dims[i]]
            
            if self.label_dims[i] == 1:
                label = torch.nn.functional.one_hot(label.type(torch.long), num_classes=2).squeeze()

            classification_loss += dist.OneHotCategorical(
                logits=getattr(self, f"classifiers_{i}")(z)
            ).log_prob(label)

            attr_track = attr_track + self.label_dims[i]

        return classification_loss.mean()

    def classification(self, *args, reparam=False):
        """Calculates classifier / adversarial loss from inputs. Uses `encoder` as function for inputs to pass through. """
        x = self._get_input_args(*args)

        z_loc, z_scale = self.encoder(x)

        if reparam:
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        else: # CCVAE recommends not reparameterizing classifier inputs to avoid gradient variance.
            z = z_loc

        return -1 * self._adversarial_from_encodings(z, self._get_label_args(*args))


class _AdversarialEntropyMixin(_AdversarialMixin):
    """Includes shared capabilities for models making use of adversarial cross-entropy loss to encourage mixing.

    Methods
    -------
    adversarial(*args)
        Calculates classifier / adversarial loss from inputs.

    """

    def _entropy_from_encodings(self, z):
        entropy_loss_z = 0

        for i in range(len(self.label_dims)):
            p = torch.nn.functional.softmax(getattr(self, f"classifiers_{i}")(z), dim=-1)
            entropy_loss_z += (p * (p+1e-6).log()).sum(dim=-1).mean()

        return entropy_loss_z
