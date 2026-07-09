# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.
#
# Modifications copyright (c) 2025 Huawei Noah's Ark Lab.
# This file contains code adapted from the Toto project and modified for use in Mantis.

import torch

from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange


class TimeWiseMultiheadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, rotary_emb, dim_head: int):
        """
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        rotary_emb: Optional[TimeAwareRotaryEmbedding],
        dim_head: int,
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        if dim_head is None:
            self.head_dim = embed_dim // num_heads
        else:
            self.head_dim = dim_head
        self.rotary_emb = rotary_emb
        self.wQKV = torch.nn.Linear(embed_dim, self.head_dim * num_heads * 3)
        self.dropout = dropout
        self.wO = torch.nn.Linear(self.head_dim * num_heads, embed_dim)

    def rearrange_inputs(self, inputs):
        """
        inputs: Float[torch.Tensor, "batch variate seq_len embed_dim"]
        -> Float[torch.Tensor, "... embed_dim"]
        """
        return rearrange(inputs, "batch variate seq_len embed_dim -> (batch variate) seq_len embed_dim")

    def get_qkv(self, inputs):
        """
        inputs: torch.Tensor
        returns tuple[torch.Tensor, ...]
        """
        qkv = self.wQKV(inputs.contiguous())
        pattern = "batch_X_variate seq_len (qkv head_dim n_heads) -> qkv batch_X_variate n_heads seq_len head_dim"
        return rearrange(qkv, pattern, qkv=3, head_dim=self.head_dim, n_heads=self.num_heads).unbind(dim=0)

    def positional_embedding(self, q, k, v):
        seq_pos_offset = 0
        if self.rotary_emb is not None:
            q, k = self.rotary_emb.rotate_queries_and_keys(q, k, seq_pos_offset=seq_pos_offset)

        q = q.contiguous()
        k = k.contiguous().to(q.dtype)
        v = v.contiguous().to(q.dtype)
        return q, k, v, seq_pos_offset

    def rearrange_output(self, output, batch, variate, seq_len):
        """
        output: torch.Tensor, batch: int, variate: int, seq_len: int
        returns Float[torch.Tensor, "batch variate seq_len embed_dim"]
        """
        pattern = "(batch variate) n_heads seq_len head_dim -> batch variate seq_len (n_heads head_dim)"
        return rearrange(output, pattern, batch=batch, variate=variate, seq_len=seq_len)

    def run_attention(self, q, k, v, seq_pos_offset, dropout, seq_len):
        q_dim_start, q_dim_end = seq_pos_offset, seq_pos_offset + seq_len
        kv_dim_end = v.shape[2]
        return scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=False)

    def forward(self, inputs):
        batch_size, variate, seq_len, _ = inputs.shape
        dropout = self.dropout if self.training else 0.0
        rearranged_inputs = self.rearrange_inputs(inputs)
        q, k, v = self.get_qkv(rearranged_inputs)
        q, k, v, seq_pos_offset = self.positional_embedding(q, k, v)
        output = self.run_attention(q, k, v, seq_pos_offset, dropout, seq_len)
        output = self.rearrange_output(output, batch_size, variate, seq_len)
        return self.wO(output)
