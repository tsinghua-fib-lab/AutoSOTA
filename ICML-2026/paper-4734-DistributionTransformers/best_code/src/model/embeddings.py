"""
Learnable embeddings from input space to model latent space. Must respect information geometry where applicable.
"""

import torch
from torch import nn, Tensor
from torch.nn import Identity
from torch.distributions.utils import vec_to_tril_matrix

from typing import Union, Sequence, Callable, Optional

from model.components import MLP


class Embedding(nn.Module):

    def __init__(self,
                 d_model: int,
                 transform: Optional[Callable[[Tensor], Tensor]] = None,
                 inverse_transform: Optional[Callable[[Tensor], Tensor]] = None,
                 embedding_model: nn.Module = Identity(),
                 de_embedding_model: Optional[nn.Module] = Identity()):
        """
        Base class for learnable embeddings to and from model latent space.

        Args:
            d_model: Dimensionality of model latent space.
            transform: Deterministic transform applied to input space.
                Defaults to None.
            inverse_transform: Inverse operation to transform.
                Defaults to None.
            embedding_model: Learnable model from transformed input space to latent space.
                Defaults to Identity().
            de_embedding_model: Learnable model from latent space to transformed input space, acting as inverse of
                embedding_model.
                Defaults to Identity().

        """
        super().__init__()
        self.d_model = d_model
        self.transform = transform
        self.inverse_transform = inverse_transform
        self.embedding_model = embedding_model
        self.de_embedding_model = de_embedding_model

    def forward(self, x: Tensor, reverse: bool = False) -> Tensor:
        """
        Forward call of module.

        Args:
            x: Input tensor, either in input space (forward) or latent space (reverse).
            reverse: Whether to run in forward (embedding) or reverse (de-embedding) mode.

        Returns:
            Embedded / de-embedded tensor.

        """
        if reverse:
            return self.de_embed(x)
        else:
            return self.embed(x)

    def embed(self, x: Tensor) -> Tensor:
        """
        Embed input tensor into model latent space.

        Args:
            x: Input tensor.

        Returns:
            Embedded tensor.

        """
        if self.transform is not None:
            x = self.transform(x)
        x = self.embedding_model(x)
        return x

    def de_embed(self, x: Tensor) -> Tensor:
        """
        De-embed input tensor from model latent space.

        Args:
            x: Input tensor.

        Returns:
            De-embedded tensor.

        """
        x = self.de_embedding_model(x)
        if self.inverse_transform is not None:
            x = self.inverse_transform(x)
        return x


class ComponentEmbedding(Embedding):

    def __init__(self,
                 state_size: int,
                 d_model: int,
                 hidden_layer_sizes: Optional[Sequence[int]] = None,
                 activation: Union[str, nn.Module] = nn.GELU(),
                 jitter: float = 1e-6,
                 eps: Optional[float] = None,
                 scale_parametrisation: str = None):
        """
        Learnable embedding from GMM component parameter space, ie the cartesian product of [0, 1] and SG, to model
        latent space.

        Args:
            state_size: Dimensionality of component variable.
            d_model: Dimensionality of model latent space.
            hidden_layer_sizes: Sequence of hidden layer sizes, if used.
                Defaults to None.
            activation: Activation function between hidden layers. "relu", "gelu" or a nn.Module subclass.
                Defaults to GELU.
            jitter: Small value to add to diagonal of scale matrix for conditioning.
                Defaults to 1e-6.
            eps: Weight clamp bound.
                Defaults to None.
            scale_parametrisation: Parametrisation of Gaussian scale parameter.
                Must be one of "covariance_matrix", "precision_matrix" of "scale_tril".
                Defaults to "covariance_matrix".

        """
        self.state_size = state_size

        assert scale_parametrisation in {"covariance_matrix", "precision_matrix", "scale_tril", None}, \
            'Scale parametrisation must be one of "covariance_matrix", "precision_matrix" of "scale_tril"'
        self.scale_parametrisation = "covariance_matrix" if scale_parametrisation is None else scale_parametrisation

        def transform(x: Tensor) -> Tensor:
            batch_shape = x.shape[:-1]
            w = x[..., 0:1]
            w = torch.logit(w, eps)
            loc = x[..., 1:1 + state_size]
            scale = x[..., 1 + state_size:].reshape(batch_shape + (state_size, state_size))
            if self.scale_parametrisation != "scale_tril":
                scale = torch.linalg.cholesky(scale)
            diag = torch.diagonal(scale, dim1=-2, dim2=-1).log()
            tril_idx = torch.tril_indices(state_size, state_size, -1)
            scale_flat = torch.cat([diag,
                                    scale[..., tril_idx[0], tril_idx[1]]],
                                   dim=-1)
            scale_flat = scale_flat.reshape(batch_shape + (-1,))
            return torch.cat([w, loc, scale_flat], dim=-1)

        def inverse_transform(x: Tensor) -> Tensor:
            batch_shape = x.shape[:-1]
            w = x[..., 0:1]
            w = torch.sigmoid(w)
            loc = x[..., 1:1 + state_size]
            scale_flat = x[..., 1 + state_size:]
            diag = scale_flat[..., :self.state_size].clamp(-11., 14.).exp() + jitter
            scale = vec_to_tril_matrix(scale_flat[..., state_size:], -1) + torch.diag_embed(diag)
            if self.scale_parametrisation != "scale_tril":
                scale = scale @ scale.mT
            scale = scale.reshape(batch_shape + (-1,))
            return torch.cat([w, loc, scale], dim=-1)

        n_in = 1 + state_size + state_size * (state_size + 1) // 2

        if not isinstance(activation, nn.Module):
            match activation:
                case "relu":
                    activation = nn.ReLU()
                case "gelu":
                    activation = nn.GELU()
                case _:
                    raise ValueError('activation must be one of "relu", "gelu" or a nn.Module subclass')

        if hidden_layer_sizes is None:
            embedding_model = nn.Linear(n_in, d_model)
            de_embedding_model = nn.Linear(d_model, n_in)
        else:
            layer_sizes = [n_in] + list(hidden_layer_sizes) + [d_model]
            embedding_model = MLP(layer_sizes, activation)
            de_embedding_model = MLP(layer_sizes[::-1], activation)

        super().__init__(d_model=d_model,
                         transform=transform,
                         inverse_transform=inverse_transform,
                         embedding_model=embedding_model,
                         de_embedding_model=de_embedding_model)


class DistributionEmbedding(Embedding):

    def __init__(self,
                 n_params: int,
                 n_components: int,
                 d_model: int,
                 transform: Optional[Callable[[Tensor], Tensor]] = None,
                 embedding_hidden_layer_sizes: Optional[Sequence[int]] = None,
                 embedding_activation: Union[str, nn.Module] = nn.GELU(),
                 conversion_hidden_layer_sizes: Optional[Sequence[int]] = None,
                 conversion_activation: Union[str, nn.Module] = nn.GELU()
                 ):
        """
        Learnable embedding from arbitrary parametric distribution to GMM representation in model latent space.
        Note that this embedding is not invertible.

        Args:
            n_params: Number of distribution parameters.
            n_components: Number of GMM components.
            d_model: Dimensionality of model latent space.
            transform: Deterministic transform applied to input space.
                Defaults to None.
            embedding_hidden_layer_sizes: Sequence of hidden layer sizes in MLP embedding from transformed parameter
                space to model latent space, if used.
                Defaults to None
            embedding_activation: Activation function between hidden layers in MLP embedding from transformed parameter
                space to model latent space. "relu", "gelu" or a nn.Module subclass.
                Defaults to GELU.
            conversion_hidden_layer_sizes: Sequence of hidden layer sizes in MLP conversion to latent GMM
                representation, if used.
                Defaults to None.
            conversion_activation: Activation function between hidden layers in MLP conversion to latent GMM
                representation. "relu", "gelu" or a nn.Module subclass.
                Defaults to GELU.

        """
        self.n_params = n_params
        self.n_components = n_components

        if embedding_hidden_layer_sizes is None:
            embedding_model = nn.Linear(n_params, d_model)
        else:
            layer_sizes = [n_params] + list(embedding_hidden_layer_sizes) + [d_model]
            embedding_model = MLP(layer_sizes, embedding_activation)

        super().__init__(d_model=d_model,
                         transform=transform,
                         inverse_transform=None,
                         embedding_model=embedding_model,
                         de_embedding_model=None)

        if conversion_hidden_layer_sizes is None:
            self.conversion_model = nn.ModuleList([nn.Linear(d_model, d_model)
                                                   for _ in range(n_components)])
        else:
            layer_sizes = [d_model] + list(conversion_hidden_layer_sizes) + [d_model]
            self.conversion_model = nn.ModuleList([MLP(layer_sizes, conversion_activation)
                                                   for _ in range(n_components)])

    def embed(self, x: Tensor) -> Tensor:
        """
        Embed input tensor into model latent space.

        Args:
            x: Input parameter tensor.

        Returns:
            Embedded GMM representation of input parameter tensor.

        """
        x = super().embed(x)
        x = torch.stack([conversion_model(x) for conversion_model in self.conversion_model], dim=-2)
        return x

    def de_embed(self, x: Tensor) -> Tensor:
        """
        De_embedding not supported by DistributionEmbedding.

        """
        raise NotImplementedError


class ObservationEmbedding(Embedding):

    def __init__(self,
                 observation_size: int,
                 d_model: int,
                 transform: Optional[Callable[[Tensor], Tensor]] = None,
                 hidden_layer_sizes: Optional[Sequence[int]] = None,
                 activation: Union[str, nn.Module] = nn.GELU(),
                 sequential: bool = False):
        """
        Learnable embedding from arbitrary parametric distribution to GMM representation in model latent space.
        Note that this embedding is not invertible.

        Args:
            observation_size: Dimensionality of observation.
            d_model: Dimensionality of model latent space.
            transform: Deterministic transform applied to input space.
                Defaults to None.
            hidden_layer_sizes: Sequence of hidden layer sizes in MLP embedding from transformed parameter
                space to model latent space, if used.
                Defaults to None
            activation: Activation function between hidden layers in MLP embedding from transformed parameter
                space to model latent space. "relu", "gelu" or a nn.Module subclass.
                Defaults to GELU.
            sequential: Whether to expect a sequence-ready observation input or not. If not, reshape input appropriately
                for subsequent attention calculations.
                Defaults to False.

        """
        self.observation_size = observation_size
        self.sequential = sequential

        if hidden_layer_sizes is None:
            embedding_model = nn.Linear(observation_size, d_model)
        else:
            layer_sizes = [observation_size] + list(hidden_layer_sizes) + [d_model]
            embedding_model = MLP(layer_sizes, activation)

        super().__init__(d_model=d_model,
                         transform=transform,
                         inverse_transform=None,
                         embedding_model=embedding_model,
                         de_embedding_model=None)

    def embed(self, x: Tensor) -> Tensor:
        """
        Embed input tensor into model latent space.

        Args:
            x: Input tensor.

        Returns:
            Embedded tensor.

        """
        if not self.sequential:
            x = x.unsqueeze(-2)
        return super().embed(x)

    def de_embed(self, x: Tensor) -> Tensor:
        """
        De_embedding not supported by ObservationEmbedding.

        """
        raise NotImplementedError


class GammaEmbedding(DistributionEmbedding):

    def __init__(self,
                 d_model: int,
                 n_components: int,
                 embedding_hidden_layer_sizes: Optional[Sequence[int]] = None,
                 embedding_activation: Union[str, nn.Module] = nn.GELU,
                 conversion_hidden_layer_sizes: Optional[Sequence[int]] = None,
                 conversion_activation: Union[str, nn.Module] = nn.GELU):
        """
        Learnable embedding from Gamma/Inverse Gamma distributions to GMM representation in model latent space.
        Note that this embedding is not invertible.

        Args:
            d_model: Dimensionality of model latent space.
            n_components: Number of GMM components.
            embedding_hidden_layer_sizes: Sequence of hidden layer sizes in MLP embedding from transformed parameter
                space to model latent space, if used.
                Defaults to None
            embedding_activation: Activation function between hidden layers in MLP embedding from transformed parameter
                space to model latent space. "relu", "gelu" or a nn.Module subclass.
                Defaults to GELU.
            conversion_hidden_layer_sizes: Sequence of hidden layer sizes in MLP conversion to latent GMM
                representation, if used.
                Defaults to None.
            conversion_activation: Activation function between hidden layers in MLP conversion to latent GMM
                representation. "relu", "gelu" or a nn.Module subclass.
                Defaults to GELU.

        """
        super().__init__(n_params=2,
                         n_components=n_components,
                         d_model=d_model,
                         transform=torch.log,
                         embedding_hidden_layer_sizes=embedding_hidden_layer_sizes,
                         embedding_activation=embedding_activation,
                         conversion_hidden_layer_sizes=conversion_hidden_layer_sizes,
                         conversion_activation=conversion_activation)
