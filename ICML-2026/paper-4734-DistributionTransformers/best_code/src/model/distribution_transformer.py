"""
Transformer model architectures for each prior/posterior distribution parametrisation
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Union, Callable, Optional, TypedDict
from typing_extensions import NotRequired

from model.embeddings import DistributionEmbedding, ComponentEmbedding, ObservationEmbedding


class TransformerKwargs(TypedDict):
    d_model: int
    n_head: int
    num_layers: NotRequired[int]
    dim_feedforward: NotRequired[int]
    norm: NotRequired[nn.Module]
    dropout: NotRequired[float]
    activation: NotRequired[Union[str, Callable[[Tensor], Tensor]]]
    layer_norm_eps: NotRequired[float]
    norm_first: NotRequired[bool]
    bias: NotRequired[bool]


class ConditionalTransformer(nn.TransformerDecoder):
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
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
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

        super().__init__(decoder_layer=decoder_layer,
                         num_layers=num_layers,
                         norm=norm)
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)


class DistributionTransformer(nn.Module):
    def __init__(self,
                 component_embedding: ComponentEmbedding,
                 transformer_kwargs: TransformerKwargs,
                 n_components: Optional[int] = None,
                 prior_embedding: Optional[DistributionEmbedding] = None,
                 sample_space_transform: Optional[Callable[[Tensor], Tensor]] = None,
                 **observation_embeddings: ObservationEmbedding):
        """
        Conditional Transformer model for dealing with Gaussian mixture model priors and posteriors.
        Treats the Gaussian mixture model as a permutation-invariant sequence of (weight, Gaussian density) pairs,
        and conditions the output sequence on the encoded and self-attended sequence of observations using a transformer
        decoder. Uses prior component encodings that respect the information geometry of the component densities.

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
        super().__init__()
        self.component_embedding = component_embedding
        self.prior_embedding = prior_embedding if prior_embedding is not None else component_embedding
        self.sample_space_transform = nn.Identity() if sample_space_transform is None else sample_space_transform
        self.observation_embeddings = nn.ModuleDict(observation_embeddings)
        self.conditional_transformer = ConditionalTransformer(**transformer_kwargs)

        assert prior_embedding is not None or n_components is not None, \
            "n_components must be specified if prior_embedding is not given"
        if prior_embedding is not None and n_components is not None:
            assert prior_embedding.n_components == n_components, \
                "if specified, n_components must agree with prior_embedding.n_components"
        self.n_components = n_components if prior_embedding is None else prior_embedding.n_components
        self.state_size = component_embedding.state_size

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
        phi_out_embedded = self.conditional_transformer(phi_in_embedded, z_embedded, tgt_is_causal=False)
        phi_out = self.component_embedding.de_embed(phi_out_embedded)
        phi_out[..., 0] /= phi_out[..., 0].sum(dim=-1, keepdim=True)  # Normalise weights
        return phi_in, phi_out
