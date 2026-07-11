import math
import torch
import torch.nn as nn

from einops import rearrange

# from fast_transformers.local_product import local_dot_product, local_weighted_average
# from src.models.modules.masking import FullMask, LengthMask

# from src.models.attention.feature_maps_sb import SBPerformerFeatures
# from src.models.attention.scatterbrain_utils import linear_attention_normalization
# from src.models.attention.scatterbrain_utils import causal_linear_attention, linear_attention

# Albert: fix dependencies
from fast_transformers.local_product import local_dot_product, local_weighted_average
from .masking import FullMask, LengthMask

from .feature_maps_sb import SBPerformerFeatures
from .scatterbrain_utils import linear_attention_normalization
from .scatterbrain_utils import causal_linear_attention, linear_attention


class SBLocalAttention(nn.Module):
    """Implement fast local attention where a query can only attend to
    neighboring keys.
    In this attention module the query Q_i can only attend to a key K_j if
    |i-j| < local_context/2.
    Arguments
    ---------
        local_context: The neighborhood to consider for local attention.
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, local_context, dim_heads, nb_features=None, ortho_scaling=0,
                 causal=False, softmax_temp=None, attention_dropout=0.0, softmax_eps=0.0,
                 device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.feature_map = SBPerformerFeatures(dim_heads, nb_features, ortho_scaling=ortho_scaling,
                                               softmax_temp=softmax_temp, eps=softmax_eps,
                                               **factory_kwargs)
        self.local_context = local_context
        self.causal = causal
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax_eps = softmax_eps

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False,
                return_attn_unnormalized=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            query: (B, T, H, E) The tensor containing the query
            key: (B, S, H, E) The tensor containing the key
            value: (B, S, H, D) The tensor containing the value
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            key_padding_mask: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # Extract some shapes and compute the temperature
        B, T, H, E = query.shape
        _, S, _, D = value.shape
        softmax_temp = self.softmax_temp or 1 / math.sqrt(E)

        # TODO: check causal
        if attn_mask is None:
            attn_mask_additive_matrix = torch.zeros(T, S, device=query.device)
        else:
            attn_mask_additive_matrix = attn_mask.additive_matrix_finite
        if key_padding_mask is None:
            key_padding_mask_lengths = torch.full(size=(B,), fill_value=S, dtype=torch.long,
                                                  device=key.device)
        else:
            key_padding_mask_lengths = key_padding_mask.lengths

        # Permute the dimensions to BHTE instead of BTHE
        query = rearrange(query, 'b t h e -> b h t e').contiguous()
        key = rearrange(key, 'b s h e -> b h s e').contiguous()
        value = rearrange(value, 'b s h d -> b h s d').contiguous()

        self.feature_map.new_feature_map(query.device)
        q_prime, q_prime_log_scale = self.feature_map.forward_queries(query)
        k_prime, k_prime_log_scale = self.feature_map.forward_keys(key)

        prime_log_scale = q_prime_log_scale + k_prime_log_scale
        m = q_prime.shape[-1]
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            k_prime = k_prime.masked_fill(~rearrange(key_padding_mask.bool_matrix,
                                                     'b s -> b 1 s 1'), 0.0)
        attn_fn = linear_attention if not self.causal else causal_linear_attention
        q_prime_k_prime_1 = linear_attention_normalization(q_prime, k_prime, causal=self.causal)
        q_prime_k_prime_v, attn_prime = attn_fn(q_prime, k_prime, value, need_weights=need_weights)

        QK = softmax_temp * local_dot_product(
            query, key, attn_mask_additive_matrix, key_padding_mask_lengths,
            self.local_context
        )
        dots_prime = local_dot_product(
            q_prime, k_prime, attn_mask_additive_matrix, key_padding_mask_lengths,
            self.local_context
        )
        # local_dot_product fills in -1e24 for invalid locations. We want to set them to zero.
        # dots_prime[dots_prime <= -1e24] = 0.0
        i = rearrange(torch.arange(T, device=query.device), 't -> 1 1 t 1')
        j = torch.arange(self.local_context, device=query.device)
        local_idx = i - self.local_context // 2 + j
        valid_idx_mask = ((local_idx >= 0)
                          & (local_idx < rearrange(key_padding_mask_lengths, 'b -> b 1 1 1')))
        dots_prime.masked_fill_(~valid_idx_mask, 0.0)
        assert torch.all(dots_prime >= 0)

        # Compute the normalization first
        QK_lse = torch.logsumexp(QK, dim=-1, keepdim=True)
        dots_prime_sum = dots_prime.sum(dim=-1, keepdim=True)
        lr_log_normalization = torch.log((rearrange(q_prime_k_prime_1, 'b h s -> b h s 1')
                                          - dots_prime_sum).clamp_min_(1e-24)) + prime_log_scale
        log_normalization = torch.logaddexp(QK_lse, lr_log_normalization)

        prime_scale = torch.exp(prime_log_scale - log_normalization)
        # When we drop out, we want that location in the attn matrix to be zero.
        # So here we dropout just torch.exp(QK) and leave -dots_prime, so that when we add it back
        # to attn_prime it's equivalent to setting that location to zero.
        dots = self.dropout(torch.exp(QK - log_normalization)) - dots_prime * prime_scale

        out_local = local_weighted_average(dots, value)
        out = out_local + q_prime_k_prime_v * prime_scale

        attn = None
        if need_weights:
            attn_local = torch.zeros(B, H, T, S, device=query.device)
            k = torch.arange(S, device=key.device)
            idx = k - i
            local_mask = ((idx >= -(self.local_context // 2))
                          & (idx < (self.local_context + 1) // 2)
                          & (k < rearrange(key_padding_mask_lengths, 'b -> b 1 1 1')))
            attn_local.masked_scatter_(local_mask, dots.masked_select(valid_idx_mask))
            attn = attn_local + attn_prime * prime_scale
            if return_attn_unnormalized:  # For testing purpose
                attn = (attn, attn * torch.exp(log_normalization),
                        attn_prime * torch.exp(prime_log_scale))
        return rearrange(out, 'b h t d -> b t h d'), attn
