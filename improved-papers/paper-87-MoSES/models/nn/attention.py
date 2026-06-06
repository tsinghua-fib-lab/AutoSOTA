import torch
import torch.nn as nn
from typing import Optional, Callable
from einops import rearrange
from torch.nn import functional as F

try:
    from torch.nn.functional import scaled_dot_product_attention
except ImportError:
    log.warning(
        "torch.nn.functional.scaled_dot_product_attention not found. Make sure you are using PyTorch >= 2.0.0."
        "Alternatively, install Flash Attention https://github.com/HazyResearch/flash-attention ."
        "Using custom implementation of scaled_dot_product_attention without Flash Attention. "
    )
    from rl4co.models.nn.attention import scaled_dot_product_attention_simple as scaled_dot_product_attention



def sparsify_tensors(scores, fill_value, top_k):
    top_k_scores, top_k_indices = scores.topk(top_k, dim=-1)
    sparse_scores = torch.full_like(scores, fill_value=fill_value, requires_grad=True)
    sparse_scores = sparse_scores.scatter(dim=-1, index=top_k_indices, src=top_k_scores)
    return sparse_scores


def sparse_scaled_dot_product_attention(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, attn_sparse_ratio=None, sparse_applied_to_score=False
):
    """Simple (exact) Scaled Dot-Product Attention in RL4CO without customized kernels (i.e. no Flash Attention)."""

    # Check for causal and attn_mask conflict
    if is_causal and attn_mask is not None:
        raise ValueError("Cannot set both is_causal and attn_mask")

    # Calculate scaled dot product
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)

    # Apply the provided attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, float("-inf"))
        else:
            scores += attn_mask

    # Apply causal mask
    if is_causal:
        s, l_ = scores.size(-2), scores.size(-1)
        mask = torch.triu(torch.ones((s, l_), device=scores.device), diagonal=1)
        scores.masked_fill_(mask.bool(), float("-inf"))

    attn_top_k = int(attn_sparse_ratio * scores.size(-1))
    assert isinstance(sparse_applied_to_score, bool)
    if sparse_applied_to_score:
        scores = sparsify_tensors(scores, fill_value=float('-inf'), top_k=attn_top_k)

    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    if not sparse_applied_to_score:
        attn_weights = sparsify_tensors(attn_weights, fill_value=0, top_k=attn_top_k)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Compute the weighted sum of values
    return torch.matmul(attn_weights, v)



class MultiHeadAttention(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        causal: whether to apply causal mask to attention scores
        device: torch device
        dtype: torch dtype
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attention_dropout: float = 0.0,
        causal: bool = False,
        device: str = None,
        dtype: torch.dtype = None,
        sdpa_fn: Optional[Callable] = None,
        attn_sparse_ratio: float = 1.0,
        sparse_applied_to_score: bool = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.attention_dropout = attention_dropout

        self.attn_sparse_ratio = attn_sparse_ratio
        self.sparse_applied_to_score = sparse_applied_to_score
        if sdpa_fn is not None:
            self.sdpa_fn = sdpa_fn
        else:
            if attn_sparse_ratio < 1.0:
                self.sdpa_fn = sparse_scaled_dot_product_attention
            else:
                self.sdpa_fn = scaled_dot_product_attention
        #self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        #self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        #q, k, v = rearrange(
        #    self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
        #).unbind(dim=0)
        q = rearrange(self.Wq(x), "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(self.Wk(x), "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(self.Wv(x), "b s (h d) -> b h s d", h=self.num_heads)

        # Scaled dot product attention
        if self.attn_sparse_ratio < 1.0:
            out = self.sdpa_fn(
                q,
                k,
                v,
                attn_mask=key_padding_mask,
                dropout_p=self.attention_dropout,
                attn_sparse_ratio=self.attn_sparse_ratio,
                sparse_applied_to_score=self.sparse_applied_to_score
            )
        else:
            out = self.sdpa_fn(
                q,
                k,
                v,
                attn_mask=key_padding_mask,
                dropout_p=self.attention_dropout,
            )

        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))


