import math
import torch
import torch.nn as nn

from einops import rearrange


# Adapted from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/full_attention.py
class FullAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, softmax_temp=None, attention_dropout=0.0, **kwargs):
        super().__init__()
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=True):
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
        # Extract some shapes and compute the temperature
        B, T, H, E = query.shape
        _, S, _, D = value.shape
        softmax_temp = self.softmax_temp or 1 / math.sqrt(E)

        # Scale the query instead of applying the softmax temperature to the
        # dot products
        query = query * softmax_temp

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("bthe,bshe->bhts", query, key)
        if attn_mask is not None and not attn_mask.all_ones:
            QK.masked_fill_(~attn_mask.bool_matrix, float('-inf'))
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            QK.masked_fill_(rearrange(~key_padding_mask.bool_matrix, 'b s -> b 1 1 s'),
                            float('-inf'))

        # Compute the attention and the weighted average
        # if T > 3000:
        #     import os
        #     # if os.path.exists("./t2t_imagenet_condition.pth"):
        #     #     D = QK.exp().sum(dim=-1).detach().cpu()
        #     #     kap = D.max(dim=-1).values / D.min(dim=-1).values
        #     #     kap_all = torch.load("./t2t_imagenet_condition.pth")
        #     #     kap_all = torch.cat((kap_all, kap))
        #     #     torch.save(kap_all, "./t2t_imagenet_condition.pth")
        #     #     # import pdb; pdb.set_trace()
        #     # else:
        #     #     D = QK.exp().sum(dim=-1).detach().cpu()
        #     #     kap = D.max(dim=-1).values / D.min(dim=-1).values
        #     #     torch.save(kap, "./t2t_imagenet_condition.pth")
        # import pdb; pdb.set_trace()
        attn = torch.softmax(QK, dim=-1)
        A = self.dropout(attn)
        output = torch.einsum("bhts,bshd->bthd", A, value)
        return output, attn if need_weights else None
