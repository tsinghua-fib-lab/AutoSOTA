"""
Implementation of scaled dot-product attention and multi-head attention as described in 'Attention Is All You Need' (Vaswani et al., 2017) based on https://www.tensorflow.org/tutorials/text/transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import rotate_half, LlamaRotaryEmbedding


def apply_rotary_pos_emb(q, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query/key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    return (q * cos.unsqueeze(0).unsqueeze(0)) + (rotate_half(q) * sin.unsqueeze(0).unsqueeze(0))


def scaled_dot_product_attention(queries, keys, values, mask=None):
    """
    Args:
        queries: (..., num_queries, d_queries)
        keys: (..., num_keys, d_queries)
        values: (..., num_keys, d_values)
        mask: (..., num_queries, num_keys)
    Returns:
        attention: (..., num_queries, d_values)
        attention_weights: (..., num_queries, num_keys)
    """
    attention_logits = torch.matmul(queries, keys.transpose(-2, -1))  # (..., num_queries, num_keys)

    # scale by square root of d_queries
    d_queries = queries.size(-1)
    scaled_attention_logits = attention_logits / torch.sqrt(torch.tensor(d_queries, dtype=queries.dtype))

    # mask scaled values
    if mask is not None:
        scaled_attention_logits += (mask * torch.finfo(queries.dtype).min)

    # perform softmax over key axis and multiply resulting attention weights with values
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (..., num_queries, num_keys)
    attention = torch.matmul(attention_weights, values)  # (..., num_queries, d_values)
    return attention, attention_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, d_embedding, num_heads, pe_type: str):
        if d_embedding % num_heads != 0:
            raise ValueError(f"Embedding dimension {d_embedding} must be divisible by number of heads {num_heads}.")

        super().__init__()

        self.d_embedding = d_embedding
        self.num_heads = num_heads
        self.d_heads = d_embedding // num_heads

        self.Q = nn.Linear(d_embedding, d_embedding)
        self.K = nn.Linear(d_embedding, d_embedding)
        self.V = nn.Linear(d_embedding, d_embedding)

        self.final_projection = nn.Linear(d_embedding, d_embedding)

        if pe_type == "rope":
            self.rope = LlamaRotaryEmbedding(self.d_heads)
        else:
            self.rope = None

    def _split_heads(self, input, batch_size, ap_count):
        """
        Splits last dimension d_embedding into (num_heads, d_heads) and transposes result
        Args:
            input: (batch_size, ap_count, num_inputs, d_embedding)
        Returns:
            (batch_size, ap_count, num_heads, num_inputs, d_heads)
        """
        input = input.view(batch_size, ap_count, -1, self.num_heads, self.d_heads)
        return input.transpose(2, 3)

    def forward(self, queries, keys, values, mask=None, cache=None, past_queries=0, no_pe_keys: bool = False):
        """
        Args:
            queries: (batch_size, ap_count, num_queries, d_embedding)
            keys: (batch_size, ap_count, num_keys, d_embedding)
            values: (batch_size, ap_count, num_keys, d_embedding)
            mask: (batch_size, ap_count, num_queries, num_keys)
            cache: a dictionary with attention from previous decoding steps that is used for fast decoding and has the following form:
                {'keys': [batch_size, i, num_heads, d_heads]
                 'values': [batch_size, i, num_heads, d_heads]}
                where i is the number of previous decoding steps
            past_queries: the number of previous decoding steps
            no_pe_keys: if True, no rotary positional encoding is applied to the keys
        Returns:
            attention: (batch_size, ap_count, num_queries, d_embedding)
            attention_weights: (batch_size, ap_count, num_queries, num_keys)
        """
        batch_size = queries.size(0)
        query_ap_count = queries.size(1)  # number of atomic propositions
        key_ap_count = keys.size(1)

        queries = self.Q(queries)
        keys = self.K(keys)
        values = self.V(values)

        queries = self._split_heads(queries, batch_size, query_ap_count)  # (batch_size, ap_count, num_heads, num_queries, d_heads)
        keys = self._split_heads(keys, batch_size, key_ap_count)  # (batch_size, ap_count, num_heads, num_keys, d_heads)
        values = self._split_heads(values, batch_size, key_ap_count)  # (batch_size, ap_count, num_heads, num_keys, d_heads)

        if self.rope is not None:
            if not no_pe_keys:
                # Apply rotary positional encoding to keys
                # Note that cache is transposed compared to attention: (batch_size, seq_len, num_heads, d_heads)
                prev_tokens = cache['keys'].size(3) if cache is not None else 0
                sequence_length = keys.size(3)
                positional_ids = torch.arange(prev_tokens, prev_tokens + sequence_length, dtype=torch.long, device=queries.device)
                positional_ids = positional_ids.unsqueeze(0)
                cos, sin = self.rope(keys, positional_ids)
                keys = apply_rotary_pos_emb(keys, cos, sin)

            # Do the same for queries
            sequence_length = queries.size(3)
            prev_tokens = past_queries
            positional_ids = torch.arange(prev_tokens, prev_tokens + sequence_length, dtype=torch.long, device=queries.device)
            positional_ids = positional_ids.unsqueeze(0)
            cos, sin = self.rope(queries, positional_ids)
            queries = apply_rotary_pos_emb(queries, cos, sin)

        if cache is not None:
            # concatenate cached keys and values
            keys = torch.cat([cache['keys'], keys], dim=-2)
            values = torch.cat([cache['values'], values], dim=-2)
            # update cache
            cache['keys'] = keys
            cache['values'] = values

        scaled_attention, attention_weights = scaled_dot_product_attention(queries, keys, values, mask)  # (batch_size, ap_count, num_heads, num_queries, d_heads), (batch_size, ap_count, num_heads, num_queries, num_keys)
        scaled_attention = scaled_attention.transpose(2, 3)  # (batch_size, ap_count, num_queries, num_heads, d_heads)
        concat_attention = scaled_attention.contiguous().view(batch_size, query_ap_count, -1, self.d_embedding)  # (batch_size, ap_count, num_queries, d_embedding)
        attention = self.final_projection(concat_attention)  # (batch_size, ap_count, num_queries, d_embedding)
        return attention, attention_weights