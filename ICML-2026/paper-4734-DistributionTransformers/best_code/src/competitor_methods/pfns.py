"""
Prior-fitted networks, as per paper "Transformers can do Bayesian Inference - Muller et al 2021"
"""

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, Normal, Uniform, constraints
from torch.distributions.utils import lazy_property
from torch.func import vmap
from torch.nn.modules.module import T

from math import log, pi
from typing import Optional, Union, Callable, Sequence

from torch.types import _size

from model.distribution_transformer import TransformerKwargs
from model.embeddings import ObservationEmbedding
from model.components import MLP


class RiemannDistribution(Distribution):
    arg_constraints = {
        "borders": constraints.real_vector,
        "weights": constraints.simplex
    }

    def __init__(self, probs: Tensor,
                 borders: Tensor,
                 infinite_support: Union[bool, tuple[bool, bool]] = True):
        """
        Riemann Distribution as described in "Transformers can do Bayesian Inference - Muller et al 2021"

        Args:
            probs: Bucket probabilities.
            borders: Bucket borders. Must correspond to length of probs accounting for infinite support.
            infinite_support: Whether the distribution has finite (False), left or right half-infinite ((True, False)
                etc) or infinite (True) support.
                Defaults to True (infinite support).

        """
        super().__init__(event_shape=torch.Size(), batch_shape=probs.shape[:-1])

        assert borders.shape[-1] == probs.shape[-1] + 1, "number of borders must equal to number of probs + 1"
        self.probs = probs / probs.sum(dim=-1, keepdim=True)

        if isinstance(infinite_support, bool):
            self.left_infinite_support = self.right_infinite_support = infinite_support
        else:
            self.left_infinite_support = infinite_support[0]
            self.right_infinite_support = infinite_support[1]

        borders = borders.broadcast_to(*self.probs.shape[:-1], borders.shape[-1])
        self.left_variance = 2.2 * borders.diff()[..., 0] ** 2 \
            if self.left_infinite_support else torch.ones(self.batch_shape, device=borders.device)
        self.right_variance = 2.2 * borders.diff()[..., -1] ** 2 \
            if self.right_infinite_support else torch.ones(self.batch_shape, device=borders.device)
        self.complete_borders = borders
        self.borders = borders[..., self.left_infinite_support:-1 if self.right_infinite_support else None]

        self.bucket_log_probs = self.probs[..., self.left_infinite_support:-1
                                           if self.right_infinite_support else None].log() - self.borders.diff().log()

    def log_prob(self, value: Tensor) -> Tensor:
        device = value.device
        value_shape = value.shape
        value = value.broadcast_to(1, *self.batch_shape) if len(value.shape) <= len(self.batch_shape) \
            else value.flatten(end_dim=-len(self.batch_shape)-1)
        idx = torch.searchsorted(self.borders, value.swapdims(0, -1))
        max_idx = self.borders.shape[-1]

        bucket_log_probs = torch.hstack([torch.zeros(*self.batch_shape, 1, device=device),
                                         self.bucket_log_probs,
                                         torch.zeros(*self.batch_shape, 1, device=device)])

        ret = ((((0.5 * (log(2) - log(pi) - self.left_variance.log()
                         - (value - self.borders[..., 0]) ** 2 / self.left_variance) + self.probs[..., 0].log()
                if self.left_infinite_support else -torch.inf)
               * (idx.swapdims(0, -1) == 0))).nan_to_num()
               + ((0.5 * (log(2) - log(pi) - self.right_variance.log()
                          - (value - self.borders[..., -1]) ** 2 / self.right_variance) + self.probs[..., -1].log()
                  if self.right_infinite_support else -torch.inf)
               * (idx.swapdims(0, -1) == max_idx)).nan_to_num()
               + bucket_log_probs.gather(-1, idx).swapdims(0, -1))
        return ret.reshape(torch.broadcast_shapes(value_shape, self.batch_shape))

    def conf(self, percentile: float) -> Tensor:
        lower = torch.zeros(*self.batch_shape, self.probs.shape[-1]) + (1 - percentile) / 2
        upper = torch.ones(*self.batch_shape, self.probs.shape[-1]) - (1 - percentile) / 2
        cdf = self.probs.cumsum(dim=-1)
        lower_idx = torch.argmin((cdf - lower) + ((cdf - lower) < 1e-6), dim=-1)
        upper_idx = torch.argmin((cdf - upper) + ((cdf - upper) < -1e-6), dim=-1)
        lower_lower_boundary = torch.gather(self.complete_borders, -1, lower_idx.unsqueeze(-1))
        lower_upper_boundary = torch.gather(self.complete_borders, -1, lower_idx.unsqueeze(-1) + 1)
        lower_cdf = torch.hstack([torch.zeros(*cdf.shape[:-1], 1), cdf])
        lower_lower_boundary_cdf = torch.gather(lower_cdf, -1, lower_idx.unsqueeze(-1))
        lower_upper_boundary_cdf = torch.gather(cdf, -1, lower_idx.unsqueeze(-1))
        upper_lower_boundary = torch.gather(self.complete_borders, -1, upper_idx.unsqueeze(-1))
        upper_upper_boundary = torch.gather(self.complete_borders, -1, upper_idx.unsqueeze(-1) + 1)
        upper_lower_boundary_cdf = torch.gather(cdf, -1, upper_idx.unsqueeze(-1) - 1)
        upper_upper_boundary_cdf = torch.gather(cdf, -1, upper_idx.unsqueeze(-1))
        lower_confidence = (lower_upper_boundary - (lower_upper_boundary - lower_lower_boundary) *
                            (lower_upper_boundary_cdf - lower[..., 0].reshape(lower_lower_boundary.shape)) /
                            (lower_upper_boundary_cdf - lower_lower_boundary_cdf)
                            * (1 + (lower_idx.unsqueeze(-1) == 0) * self.left_infinite_support))
        upper_confidence = (upper_lower_boundary + (upper_upper_boundary - upper_lower_boundary) *
                            (upper[..., 0].reshape(upper_lower_boundary.shape) - upper_lower_boundary_cdf) /
                            (upper_upper_boundary_cdf - upper_lower_boundary_cdf)
                            * (1 + (upper_idx.unsqueeze(-1) == cdf.shape[-1] - 1) * self.right_infinite_support))
        return torch.stack([lower_confidence.squeeze(-1), upper_confidence.squeeze(-1)], dim=-1)

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        device = self.borders.device
        bucket_dist = Categorical(probs=self.probs)
        max_bucket = self.probs.shape[-1] - 1
        buckets = bucket_dist.sample(sample_shape).to(device).unsqueeze(-1)
        borders = torch.hstack([(self.borders[..., 0:1] - 1e-6) * (1 - self.borders[..., 0:1].sign() * 1e-3)]
                               * self.left_infinite_support +
                               [self.borders] +
                               [(self.borders[..., -1:] + 1e-6) * (1 + self.borders[..., -1:].sign() * 1e-4)]
                               * self.right_infinite_support)
        uniform_sample = Uniform(borders.gather(-1, buckets.swapdims(0, -1)).swapdims(0, -1),
                                 borders.gather(-1, buckets.swapdims(0, -1) + 1).swapdims(0, -1)).sample().squeeze(-1)

        left_normal_sample = -Normal(torch.zeros(self.batch_shape, device=self.left_variance.device),
                                     self.left_variance).sample(sample_shape).to(device).abs() \
            if self.left_infinite_support else torch.zeros(sample_shape + self.batch_shape, device=device)
        right_normal_sample = Normal(torch.zeros(self.batch_shape, device=self.right_variance.device),
                                     self.right_variance).sample(sample_shape).to(device).abs() \
            if self.right_infinite_support else torch.zeros(sample_shape + self.batch_shape, device=device)

        return (uniform_sample
                + left_normal_sample * ((buckets.squeeze(-1) == 0) * self.left_infinite_support)
                + right_normal_sample * (buckets.squeeze(-1) == max_bucket) * self.right_infinite_support)

    @property
    def mean(self) -> Tensor:
        mean = torch.sum((self.borders[..., :-1] + self.borders.diff(dim=-1) / 2)
                         * self.probs[..., self.left_infinite_support:-1 if self.right_infinite_support else None],
                         dim=-1)
        if self.left_infinite_support:
            mean += self.probs[..., 0] * (self.borders[..., 0] - 0.8 * self.left_variance.sqrt())
        if self.right_infinite_support:
            mean += self.probs[..., -1] * (self.borders[..., -1] + 0.8 * self.right_variance.sqrt())
        return mean

    @lazy_property
    def weights(self):
        return self.probs

    @lazy_property
    def borders(self):
        return self.borders


def get_borders_from_prior(prior: Distribution,
                           n_buckets: int,
                           infinite_support: Union[bool, tuple[bool, bool]],
                           leftmost_border: Optional[float] = None,
                           rightmost_border: Optional[float] = None,
                           n_samples: int = 10000
                           ) -> Tensor:
    """
    Get initial borders for Riemann Distribution by matching prior quantiles.

    Args:
        prior: Prior distribution to match.
        n_buckets: Number of buckets.
        infinite_support: Whether the distribution has finite (False), left or right half-infinite ((True, False)
                etc) or infinite (True) support.
                Defaults to True (infinite support).
        leftmost_border: Hard leftmost border to assign if specified.
            Defaults to None.
        rightmost_border: Hard rightmost border to assign if specified.
            Defaults to None.
        n_samples: Number of samples with which to approximate quantiles.
            Defaults to 10000.

    Returns:
        Tensor of borders.

    """
    assert torch.prod(torch.tensor(prior.event_shape)) == 1, "only defined for univariate distributions"

    samples = prior.sample((n_samples,)).reshape(*prior.batch_shape, n_samples)

    if isinstance(infinite_support, bool):
        left_infinite_support = right_infinite_support = infinite_support
    else:
        left_infinite_support = infinite_support[0]
        right_infinite_support = infinite_support[1]

    quantiles = torch.linspace(0., 1., n_buckets+1)
    # Account for 50% probability mass of infinite tails
    if left_infinite_support:
        quantiles[..., 0] = 0.5 * (quantiles[..., 0] + quantiles[..., 1])
    if right_infinite_support:
        quantiles[..., -1] = 0.5 * (quantiles[..., -1] + quantiles[..., -2])

    quantile_func = torch.quantile
    for i in range(len(prior.batch_shape)):
        quantile_func = vmap(quantile_func)
    borders = quantile_func(samples, quantiles.broadcast_to(*prior.batch_shape, n_buckets+1), dim=-1)

    if leftmost_border is not None:
        borders[..., 0] = leftmost_border
    if rightmost_border is not None:
        borders[..., -1] = rightmost_border

    return borders


class Transformer(nn.TransformerEncoder):
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
        This standard encoder layer is based on the paper "Attention Is All You Need".
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


class PFN(nn.Module):

    def __init__(self,
                 transformer_kwargs: TransformerKwargs,
                 n_buckets: int = 10,
                 infinite_support: Union[bool, tuple[bool, bool]] = True,
                 leftmost_border: Optional[float] = None,
                 rightmost_border: Optional[float] = None,
                 hidden_layer_sizes: Optional[Sequence[int]] = None,
                 activation: Union[str, nn.Module] = "gelu",
                 **observation_embeddings: ObservationEmbedding
                 ):
        """
        Prior Fitted Network (PFN) as described in "Transformers can do Bayesian Inference - Muller et al 2021"

        Args:
            transformer_kwargs: Kwargs for transformer.
            n_buckets: Number of buckets for output Riemann distribution.
                Defaults to 10.
            infinite_support: Whether the distribution has finite (False), left or right half-infinite ((True, False)
                etc) or infinite (True) support.
                Defaults to True (infinite support).
            leftmost_border: Hard leftmost border to assign if specified.
                Defaults to None.
            rightmost_border: Hard rightmost border to assign if specified.
                Defaults to None.
            observation_embeddings: Dictionary of observation embeddings.
        """
        super().__init__()
        self.observation_embeddings = nn.ModuleDict(observation_embeddings)
        self.transformer = Transformer(**transformer_kwargs)
        self.n_buckets = n_buckets
        self.infinite_support = infinite_support
        self.leftmost_border = leftmost_border
        self.rightmost_border = rightmost_border

        self.borders: Optional[Tensor] = None

        if not isinstance(activation, nn.Module):
            match activation:
                case "relu":
                    activation = nn.ReLU()
                case "gelu":
                    activation = nn.GELU()
                case _:
                    raise ValueError('activation must be one of "relu", "gelu" or a nn.Module subclass')

        d_model = transformer_kwargs["d_model"]

        if hidden_layer_sizes is None:
            self.de_embedding_model = nn.Linear(d_model, n_buckets)
        else:
            layer_sizes = [d_model] + list(hidden_layer_sizes) + [n_buckets]
            self.de_embedding_model = MLP(layer_sizes, activation)

    def forward(self, **z: Tensor) -> Tensor:
        assert z.keys() <= dict(self.observation_embeddings).keys(), \
            "Observation keys must have matching keys in observation_embeddings"

        z_embedded = torch.cat([observation_embedding(z[key])
                                for key, observation_embedding in self.observation_embeddings.items()], dim=-2)
        phi_out_embedded = self.transformer(z_embedded, is_causal=False)
        logits = self.de_embedding_model(phi_out_embedded[..., -1, :])
        return logits.softmax(dim=-1)

    def to(self: T, *args, **kwargs) -> T:
        super().to(*args, **kwargs)
        if self.borders is not None:
            self.borders = self.borders.to(*args, **kwargs)
        return self

    def cpu(self: T) -> T:
        super().cpu()
        if self.borders is not None:
            self.borders = self.borders.cpu()
        return self
