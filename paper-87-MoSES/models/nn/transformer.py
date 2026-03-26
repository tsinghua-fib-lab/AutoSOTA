import sys
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#from rl4co.models.nn.attention import MultiHeadAttention
#from rl4co.models.nn.moe import MoE
from rl4co.models.nn.mlp import MLP
from rl4co.utils.pylogger import get_pylogger
from torch import Tensor

from models.nn.attention import MultiHeadAttention

log = get_pylogger(__name__)


class RMSNorm(nn.Module):
    """From https://github.com/meta-llama/llama-models"""

    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()
        if normalization != "layer":
            normalizer_class = {
                "batch": nn.BatchNorm1d,
                "instance": nn.InstanceNorm1d,
                "rms": RMSNorm,
            }.get(normalization, None)
            self.normalizer = (
                normalizer_class(embed_dim, affine=True)
                if normalizer_class is not None
                else None
            )
        else:
            self.normalizer = "layer"
        if self.normalizer is None:
            log.error(
                "Normalization type {} not found. Skipping normalization.".format(
                    normalization
                )
            )
        self.normalization = normalization

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d) or self.normalization == 'batch':
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d) or self.normalization == 'instance':
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.normalizer == "layer":
            return (x - x.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(
                x.var((1, 2)).view(-1, 1, 1) + 1e-05
            )
        elif isinstance(self.normalizer, RMSNorm) or self.normalization == 'rms':
            return self.normalizer(x)
        else:
            assert self.normalizer is None, "Unknown normalizer type {}".format(
                self.normalizer
            )
            return x



class ParallelGatedMLP(nn.Module):
    """From https://github.com/togethercomputer/stripedhyena"""

    def __init__(
        self,
        hidden_size: int = 128,
        inner_size_multiple_of: int = 256,
        mlp_activation: str = "silu",
        model_parallel_size: int = 1,
    ):
        super().__init__()

        multiple_of = inner_size_multiple_of
        self.act_type = mlp_activation
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError

        self.multiple_of = multiple_of * model_parallel_size

        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        )

        self.l1 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        return self.l3(self.act(z1) * z2)




class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
        normalization: Optional[str] = "instance",
        use_prenorm: bool = False,
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
        parallel_gated_kwargs: Optional[dict] = None,
        attn_sparse_ratio: float = 1.0,
        sparse_applied_to_score: bool = None,
    ):
        super(TransformerBlock, self).__init__()
        feedforward_hidden = (
            4 * embed_dim if feedforward_hidden is None else feedforward_hidden
        )
        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if moe_kwargs is not None:
            raise NotImplementedError
            #ffn = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        elif parallel_gated_kwargs is not None:
            ffn = ParallelGatedMLP(embed_dim, **parallel_gated_kwargs)
        else:
            raise NotImplementedError
        #    ffn = MLP(
        #        input_dim=embed_dim,
        #        output_dim=embed_dim,
        #        num_neurons=num_neurons,
        #        hidden_act="ReLU",
        #    )

        self.norm_attn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.attention = MultiHeadAttention(
            embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn,
            attn_sparse_ratio=attn_sparse_ratio, sparse_applied_to_score=sparse_applied_to_score
        )
        self.norm_ffn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.ffn = ffn
        self.use_prenorm = use_prenorm

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feedforward_hidden = feedforward_hidden
        self.normalization = normalization
        self.bias = bias
        self.sdpa_fn = sdpa_fn
        self.moe_kwargs = moe_kwargs
        self.parallel_gated_kwargs = parallel_gated_kwargs
        self.attn_sparse_ratio = attn_sparse_ratio
        self.sparse_applied_to_score = sparse_applied_to_score


    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.use_prenorm:
            # more modern transformer structure
            # https://arxiv.org/abs/2002.04745
            h = x + self.attention(self.norm_attn(x), mask)
            h = h + self.ffn(self.norm_ffn(h))
        else:
            # from Kool et al. (2019)
            # i.e. from Attention is All You Need
            h = self.norm_attn(x + self.attention(x, mask))
            h = self.norm_ffn(h + self.ffn(h))
        return h

