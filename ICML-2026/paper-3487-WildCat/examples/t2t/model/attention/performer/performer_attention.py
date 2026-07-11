# Adapted from https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
import math
import torch
from torch import nn
from einops import rearrange

from functools import partial

# NOTE: Original imports
# from src.models.attention.projection_utils import gaussian_orthogonal_random_matrix
# from src.models.attention.performer_utils import (
#     softmax_kernel, generalized_kernel,
#     causal_linear_attention, causal_linear_attention_noncuda, linear_attention
# )
# NOTE: The original imports have been modified to import correct dependencies
from ..projection_utils import gaussian_orthogonal_random_matrix
from .performer_utils import (
    softmax_kernel, generalized_kernel,
    causal_linear_attention, causal_linear_attention_noncuda, linear_attention
)

# helpers

def default(val, d):
    return val if val is not None else d


class PerformerAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
    """
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, generalized_attention=False,
                 kernel_fn=nn.ReLU(), no_projection=False, softmax_temp=None, softmax_eps=1e-4,
                 normalization_eps=1e-6, device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix,
                                         nrows=self.nb_features, ncols=dim_heads,
                                         scaling=ortho_scaling, **factory_kwargs)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.softmax_temp = softmax_temp
        self.softmax_eps = softmax_eps  # Stabilizer for softmax kernel
        self.normalization_eps = normalization_eps  # Stabilizer for normalization step

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            query: (B, T, H, E) The tensor containing the query
            key: (B, S, H, E) The tensor containing the key
            value: (B, S, H, D) The tensor containing the value
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            key_padding_mask: An implementation of BaseMask that encodes how
                         many query each sequence in the batch consists of
        """
        query = rearrange(query, 'b t h e -> b h t e')
        key = rearrange(key, 'b s h e -> b h s e')
        value = rearrange(value, 'b s h d -> b h s d')
        if self.no_projection:
            query = query.softmax(dim=-1)
            key = torch.exp(key) if self.causal else key.softmax(dim=-2)
        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn,
                                    projection_matrix=self.projection_matrix)
            query, key = map(create_kernel, (query, key))
        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix,
                                    softmax_temp=self.softmax_temp, eps=self.softmax_eps)
            query = create_kernel(query, is_query=True)
            key = create_kernel(key, is_query=False)

        if key_padding_mask is not None and not key_padding_mask.all_ones:
            # performer-pytorch chooses to zero out the value instead of the key
            # https://github.com/lucidrains/performer-pytorch/blob/457dade217c900b6c972c77731c7bbbf55cf5b8a/performer_pytorch/performer_pytorch.py#L393
            value = value.masked_fill(rearrange(~key_padding_mask.bool_matrix, 'b s -> b 1 s 1'),
                                      0.0)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones or is causal
        causal = attn_mask is not None and attn_mask.lower_triangular
        if not (attn_mask is None or attn_mask.all_ones or causal):
            raise RuntimeError(("PerformerAttention does not support arbitrary attention masks"))
        if causal:
            assert query.shape[1] == key.shape[1], 'query and key must have the same sequence length'
            try:
                import fast_transformers.causal_product.causal_product_cuda
                attn_fn = causal_linear_attention
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                attn_fn = causal_linear_attention_noncuda
        else:
            attn_fn = linear_attention
        out, attn = attn_fn(query, key, value, eps=self.normalization_eps,
                            need_weights=need_weights)
        out = rearrange(out, 'b h s d -> b s h d')
        return out, attn
