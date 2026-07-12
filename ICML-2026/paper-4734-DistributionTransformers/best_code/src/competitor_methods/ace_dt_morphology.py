"""
Intermediate models morphing from DTs to ACE one step at a time
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions.utils import vec_to_tril_matrix

from typing import Optional, Callable, Union, Sequence, Literal
from model.components import MLP

from model.distribution_transformer import DistributionTransformer, ConditionalTransformer, TransformerKwargs
from model.embeddings import Embedding, DistributionEmbedding, ComponentEmbedding, ObservationEmbedding
from competitor_methods.ace import ACEBaseTransformer


class UnconditionalTransformer(nn.TransformerEncoder):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 num_layers: int = 6,
                 norm: Optional[nn.Module] = None,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5,
                 norm_first: bool = False,
                 bias: bool = True,
                 device=None,
                 dtype=None):
        """
        This standard decoder layer is based on the paper "Attention Is All You Need".
        Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
        Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
        Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
        in a different way during application.

        Args:
            d_model: The number of expected features in the input (required).
            n_head: The number of heads in the multiheadattention models (required).
            num_layers: The number of sub-decoder-layers in the decoder.
                Defaults to 6.
            norm: The layer normalization component.
                Defaults to None.
            dim_feedforward: The dimension of the feedforward network model.
                Defaults to 2048.
            dropout: The dropout value.
                Defaults to 0.1.
            activation: The activation function of the intermediate layer, can be a string
                ("relu" or "gelu") or a unary callable.
                Defaults to gelu.
            layer_norm_eps: The eps value in layer normalization components (default=1e-5).
            norm_first: If ``True``, layer norm is done prior to self attention, multihead
                attention and feedforward operations, respectively. Otherwise it's done after.
                Defaults to ``False`` (after).
            bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive bias.
                Defaults to ``True``.

        """
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation,
                                                   layer_norm_eps=layer_norm_eps,
                                                   batch_first=True,
                                                   norm_first=norm_first,
                                                   bias=bias,
                                                   device=device,
                                                   dtype=dtype)

        super().__init__(encoder_layer=encoder_layer,
                         num_layers=num_layers,
                         norm=norm)
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)

class DistributionTransformerWithEncoder(DistributionTransformer):
    """
    Distribution Transformer with a transformer encoder applied to the observation track
    """

    def __init__(self,
                 component_embedding: ComponentEmbedding,
                 transformer_decoder_kwargs: TransformerKwargs,
                 transformer_encoder_kwargs: TransformerKwargs,
                 n_components: Optional[int] = None,
                 prior_embedding: Optional[DistributionEmbedding] = None,
                 sample_space_transform: Optional[Callable[[Tensor], Tensor]] = None,
                 **observation_embeddings: ObservationEmbedding):
        """
        Conditional Transformer model for dealing with Gaussian mixture model priors and posteriors.
        Treats the Gaussian mixture model as a permutation-invariant sequence of (weight, Gaussian density) pairs,
        and conditions the output sequence on the encoded and self-attended sequence of observations using a transformer
        decoder. Uses prior component encodings that respect the information geometry of the component densities.
        Differs from regular Distribution Transformer by also processing observation embeddings with a transformer encoder

        Args:
            component_embedding: Embedding model from sequential GMM representation to model latent space.
            transformer_kwargs: Kwargs for conditional transformer.
            n_components: Number of GMM components. Does not need to be specified if prior_embedding is provided.
                Defaults to None.
            prior_embedding: Embedding model from prior parameter space to model latent space. If None, assumes
                sequential representation GMM prior.
                Defaults to None.
            sample_space_transform: Transform from sample space of prior to sample space of approximating GMM.
                Defaults to Identity().
            observation_embeddings: Dictionary of embedding models.
        """
        super().__init__(component_embedding, transformer_decoder_kwargs, n_components, prior_embedding, sample_space_transform, **observation_embeddings)
        self.unconditional_transformer = UnconditionalTransformer(**transformer_encoder_kwargs)

    def forward(self, phi: Tensor, **z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of conditional transformer model.

        Args:
            phi: Batched prior parameter tensor.
            z: Dictionary of batched observation tensors. Must have keys with corresponding observation embeddings.

        Returns:
            Sequential GMM representation of approximate prior parameter tensor.
            Sequential GMM representation of approximate posterior parameter tensor.

        """
        assert z.keys() <= dict(self.observation_embeddings).keys(), \
            "Observation keys must have matching keys in observation_embeddings"
        phi_in_embedded = self.prior_embedding.embed(phi)
        phi_in = self.component_embedding.de_embed(phi_in_embedded)
        phi_in[..., 0] /= phi_in[..., 0].sum(dim=-1, keepdim=True)  # Normalise weights
        z_embedded = torch.cat([observation_embedding(z[key])
                                for key, observation_embedding in self.observation_embeddings.items()], dim=-2)
        z_encoded = self.unconditional_transformer(z_embedded, is_causal=False)
        phi_out_embedded = self.conditional_transformer(phi_in_embedded, z_encoded, tgt_is_causal=False)
        phi_out = self.component_embedding.de_embed(phi_out_embedded)
        phi_out[..., 0] /= phi_out[..., 0].sum(dim=-1, keepdim=True)  # Normalise weights
        return phi_in, phi_out


class GMMEmbedding(ComponentEmbedding):

    def __init__(self,
                 state_size: int,
                 n_components: int,
                 d_model: int,
                 hidden_layer_sizes: Optional[Sequence[int]] = None,
                 activation: Union[str, nn.Module] = nn.GELU(),
                 jitter: float = 1e-6,
                 eps: Optional[float] = None,
                 scale_parametrisation: str = None):
        """
        Learnable embedding from total GMM parameter space, ie the cartesian product of the simplex and SG^n, to a
        single vector in model latent space.

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
        self.n_components = n_components

        assert scale_parametrisation in {"covariance_matrix", "precision_matrix", "scale_tril", None}, \
            'Scale parametrisation must be one of "covariance_matrix", "precision_matrix" of "scale_tril"'
        self.scale_parametrisation = "covariance_matrix" if scale_parametrisation is None else scale_parametrisation

        def transform(x: Tensor) -> Tensor:
            batch_shape = x.shape[:-2]
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
            components = torch.cat([w, loc, scale_flat], dim=-1)
            return components.flatten(start_dim=-2)

        def inverse_transform(x: Tensor) -> Tensor:
            batch_shape = x.shape[:-2]  # Second to last dimension is tokens, always 1 if using this class
            x = x.reshape(batch_shape + (self.n_components, -1))
            w = x[..., 0:1]
            w = torch.sigmoid(w)
            loc = x[..., 1:1 + state_size]
            scale_flat = x[..., 1 + state_size:]
            diag = scale_flat[..., :self.state_size].clamp(-11., 14.).exp() + jitter
            scale = vec_to_tril_matrix(scale_flat[..., state_size:], -1) + torch.diag_embed(diag)
            if self.scale_parametrisation != "scale_tril":
                scale = scale @ scale.mT
            scale = scale.reshape(batch_shape + (self.n_components, -1))
            return torch.cat([w, loc, scale], dim=-1)

        n_in = n_components * (1 + state_size + state_size * (state_size + 1) // 2)

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

        Embedding.__init__(self,
                           d_model=d_model,
                           transform=transform,
                           inverse_transform=inverse_transform,
                           embedding_model=embedding_model,
                           de_embedding_model=de_embedding_model)

class SingleChannelDistributionTransformer(DistributionTransformerWithEncoder):
    """
    Distribution Transformer with a transformer encoder applied to the observation track
    """

    def __init__(self,
                 component_embedding: GMMEmbedding,
                 transformer_decoder_kwargs: TransformerKwargs,
                 transformer_encoder_kwargs: TransformerKwargs,
                 prior_embedding: Optional[DistributionEmbedding] = None,
                 sample_space_transform: Optional[Callable[[Tensor], Tensor]] = None,
                 **observation_embeddings: ObservationEmbedding):
        """
        Conditional Transformer model for dealing with Gaussian mixture model priors and posteriors.
        Treats the Gaussian mixture model as a permutation-invariant sequence of (weight, Gaussian density) pairs,
        and conditions the output sequence on the encoded and self-attended sequence of observations using a transformer
        decoder. Uses prior component encodings that respect the information geometry of the component densities.
        Differs from regular Distribution Transformer by also processing observation embeddings with a transformer
        encoder, and also by using a single channel in latent space to represent the prior/posterior

        Args:
            component_embedding: Embedding model from sequential GMM representation to model latent space.
            transformer_kwargs: Kwargs for conditional transformer.
            prior_embedding: Embedding model from prior parameter space to model latent space. If None, assumes
                sequential representation GMM prior, and uses single-latent-token embedding.
                Defaults to None.
            sample_space_transform: Transform from sample space of prior to sample space of approximating GMM.
                Defaults to Identity().
            observation_embeddings: Dictionary of embedding models.
        """
        super().__init__(component_embedding, transformer_decoder_kwargs, transformer_encoder_kwargs, 1, prior_embedding,
                         sample_space_transform, **observation_embeddings)


class LatentDecodedDistributionTransformer(SingleChannelDistributionTransformer):

    def __init__(self,
                 component_embedding: GMMEmbedding,
                 transformer_decoder_kwargs: TransformerKwargs,
                 transformer_encoder_kwargs: TransformerKwargs,
                 prior_embedding: Optional[DistributionEmbedding] = None,
                 sample_space_transform: Optional[Callable[[Tensor], Tensor]] = None,
                 **observation_embeddings: ObservationEmbedding):
        """
        Conditional Transformer model for dealing with Gaussian mixture model priors and posteriors.
        Treats the Gaussian mixture model as a permutation-invariant sequence of (weight, Gaussian density) pairs,
        and conditions the output sequence on the encoded and self-attended sequence of observations using a transformer
        decoder. Uses prior component encodings that respect the information geometry of the component densities.
        Differs from regular Distribution Transformer by also processing observation embeddings with a transformer
        encoder, and also by using a single channel in latent space to represent the prior/posterior, and treating the
        prior as an observation and seeding the single readout channel with a fixed, learnable vector.

        Args:
            component_embedding: Embedding model from sequential GMM representation to model latent space.
            transformer_kwargs: Kwargs for conditional transformer.
            prior_embedding: Embedding model from prior parameter space to model latent space. If None, assumes
                sequential representation GMM prior, and uses single-latent-token embedding.
                Defaults to None.
            sample_space_transform: Transform from sample space of prior to sample space of approximating GMM.
                Defaults to Identity().
            observation_embeddings: Dictionary of embedding models.
        """
        super().__init__(component_embedding, transformer_decoder_kwargs, transformer_encoder_kwargs, prior_embedding,
                         sample_space_transform, **observation_embeddings)
        self.latent_readout_token = torch.nn.Parameter(torch.randn(1, component_embedding.d_model))

    def forward(self, phi: Tensor, **z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of conditional transformer model.

        Args:
            phi: Batched prior parameter tensor.
            z: Dictionary of batched observation tensors. Must have keys with corresponding observation embeddings.

        Returns:
            Sequential GMM representation of approximate prior parameter tensor.
            Sequential GMM representation of approximate posterior parameter tensor.

        """
        assert z.keys() <= dict(self.observation_embeddings).keys(), \
            "Observation keys must have matching keys in observation_embeddings"
        phi_in_embedded = self.prior_embedding.embed(phi)
        phi_in = self.component_embedding.de_embed(phi_in_embedded)
        phi_in[..., 0] /= phi_in[..., 0].sum(dim=-1, keepdim=True)  # Normalise weights
        z_and_phi_embedded = torch.cat([observation_embedding(z[key])
                                        for key, observation_embedding in self.observation_embeddings.items()]
                                        + [phi_in_embedded], dim=-2)
        z_encoded = self.unconditional_transformer(z_and_phi_embedded, is_causal=False)
        phi_out_embedded = self.conditional_transformer(self.latent_readout_token.broadcast_to(phi_in_embedded.shape), z_encoded, tgt_is_causal=False)
        phi_out = self.component_embedding.de_embed(phi_out_embedded)
        phi_out[..., 0] /= phi_out[..., 0].sum(dim=-1, keepdim=True)  # Normalise weights
        return phi_in, phi_out


class SoftplusLatentDecodedDistributionTransformer(LatentDecodedDistributionTransformer):

    def __init__(self,
                 component_embedding: GMMEmbedding,
                 transformer_decoder_kwargs: TransformerKwargs,
                 transformer_encoder_kwargs: TransformerKwargs,
                 prior_embedding: Optional[DistributionEmbedding] = None,
                 sample_space_transform: Optional[Callable[[Tensor], Tensor]] = None,
                 **observation_embeddings: ObservationEmbedding):
        """
        Conditional Transformer model for dealing with Gaussian mixture model priors and posteriors.
        Treats the Gaussian mixture model as a permutation-invariant sequence of (weight, Gaussian density) pairs,
        and conditions the output sequence on the encoded and self-attended sequence of observations using a transformer
        decoder. Uses prior component encodings that respect the information geometry of the component densities.
        Differs from regular Distribution Transformer by also processing observation embeddings with a transformer
        encoder, and also by using a single channel in latent space to represent the prior/posterior, and treating the
        prior as an observation and seeding the single readout channel with a fixed, learnable vector. Uses softplus
        instead of exp to parametrise variances.

        Args:
            component_embedding: Embedding model from sequential GMM representation to model latent space.
            transformer_kwargs: Kwargs for conditional transformer.
            prior_embedding: Embedding model from prior parameter space to model latent space. If None, assumes
                sequential representation GMM prior, and uses single-latent-token embedding.
                Defaults to None.
            sample_space_transform: Transform from sample space of prior to sample space of approximating GMM.
                Defaults to Identity().
            observation_embeddings: Dictionary of embedding models.
        """
        super().__init__(component_embedding, transformer_decoder_kwargs, transformer_encoder_kwargs, prior_embedding,
                         sample_space_transform, **observation_embeddings)


    def forward(self, phi: Tensor, **z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of conditional transformer model.

        Args:
            phi: Batched prior parameter tensor.
            z: Dictionary of batched observation tensors. Must have keys with corresponding observation embeddings.

        Returns:
            Sequential GMM representation of approximate prior parameter tensor.
            Sequential GMM representation of approximate posterior parameter tensor.

        """
        phi_in_provisional, phi_out_provisional = super().forward(phi, **z)

        phi_in = torch.copy(phi_in_provisional)
        phi_out = torch.copy(phi_out_provisional)

        phi_in[..., 1 + self.state_size:] = F.softplus(torch.log(phi_in_provisional[..., 1 + self.state_size:])) + 1e-3
        phi_out[..., 1 + self.state_size:] = F.softplus(torch.log(phi_out_provisional[..., 1 + self.state_size:])) + 1e-3

        return phi_in, phi_out


def distribution_transformer_factory(
    kind: Literal["DistributionTransformer", "DistributionTransformerWithEncoder",
                  "SingleChannelDistributionTransformer", "LatentDecodedDistributionTransformer"],
    n_components: int,
    state_size: int,
    component_embedding_kwargs: dict,
    observation_embedding_kwargs: dict[str, dict],
    transformer_decoder_kwargs: TransformerKwargs,
    transformer_encoder_kwargs: Optional[TransformerKwargs] = None,
    prior_embedding: Optional[DistributionEmbedding] = None,
    sample_space_transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> DistributionTransformer:
    """

    Args:
        kind: Kind of distribution transformer to construct.
        state_size: Dimensionality of distribution state.
        component_embedding_kwargs: Kwargs for component embedding.
        observation_embedding_kwargs: Kwargs for observation embeddings.
        transformer_decoder_kwargs: Kwargs for conditional transformer.
        transformer_encoder_kwargs: Kwargs for unconditional transformer.
        prior_embedding: Embedding model from prior parameter space to model latent space.
        sample_space_transform: Transform from sample space of approximating GMM.

    Returns:
        distribution transformer of corresponding kind.

    """
    d_model = prior_embedding.d_model

    observation_embedding = {key: ObservationEmbedding(d_model=d_model, observation_size=1, **kwargs)
                             for key, kwargs in observation_embedding_kwargs.items()}

    match kind:
        case "DistributionTransformer":
            component_embedding = ComponentEmbedding(state_size=state_size, d_model=d_model, **component_embedding_kwargs)
            model = DistributionTransformer(
                component_embedding=component_embedding,
                transformer_kwargs=transformer_decoder_kwargs,
                n_components=n_components,
                prior_embedding=prior_embedding,
                sample_space_transform=sample_space_transform,
                **observation_embedding
            )

        case "DistributionTransformerWithEncoder":
            component_embedding = ComponentEmbedding(state_size=state_size, d_model=d_model, **component_embedding_kwargs)
            model = DistributionTransformerWithEncoder(
                component_embedding=component_embedding,
                transformer_decoder_kwargs=transformer_decoder_kwargs,
                transformer_encoder_kwargs=transformer_encoder_kwargs if transformer_encoder_kwargs else transformer_decoder_kwargs,
                n_components=n_components,
                prior_embedding=prior_embedding,
                sample_space_transform=sample_space_transform,
                **observation_embedding
            )

        case "SingleChannelDistributionTransformer":
            component_embedding = GMMEmbedding(state_size=state_size, n_components=n_components, d_model=d_model,
                                               **component_embedding_kwargs)
            model = SingleChannelDistributionTransformer(
                component_embedding=component_embedding,
                transformer_decoder_kwargs=transformer_decoder_kwargs,
                transformer_encoder_kwargs=transformer_encoder_kwargs if transformer_encoder_kwargs else transformer_decoder_kwargs,
                prior_embedding=prior_embedding,
                sample_space_transform=sample_space_transform,
                **observation_embedding
            )


        case "LatentDecodedDistributionTransformer":
            component_embedding = GMMEmbedding(state_size=state_size, n_components=n_components, d_model=d_model,
                                               **component_embedding_kwargs)
            model = LatentDecodedDistributionTransformer(
                component_embedding=component_embedding,
                transformer_decoder_kwargs=transformer_decoder_kwargs,
                transformer_encoder_kwargs=transformer_encoder_kwargs if transformer_encoder_kwargs else transformer_decoder_kwargs,
                prior_embedding=prior_embedding,
                sample_space_transform=sample_space_transform,
                **observation_embedding
            )

        case "SoftplusLatentDecodedDistributionTransformer":
            component_embedding = GMMEmbedding(state_size=state_size, n_components=n_components, d_model=d_model,
                                               **component_embedding_kwargs)
            model = SoftplusLatentDecodedDistributionTransformer(
                component_embedding=component_embedding,
                transformer_decoder_kwargs=transformer_decoder_kwargs,
                transformer_encoder_kwargs=transformer_encoder_kwargs if transformer_encoder_kwargs else transformer_decoder_kwargs,
                prior_embedding=prior_embedding,
                sample_space_transform=sample_space_transform,
                **observation_embedding
            )

        case _:
            raise NotImplementedError("Unknown kind of distribution transformer")

    return model