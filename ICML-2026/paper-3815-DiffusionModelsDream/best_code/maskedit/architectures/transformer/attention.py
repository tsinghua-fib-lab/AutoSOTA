#from __future__ import annotations
from .utils import ensure_mask_shape
from typing import Optional

import jax
import jax.numpy as jnp

from flax import nnx

Array = jax.Array

def dense_dot_product_attention(
    q: Array,  # [B, Tq, H, K]
    k: Array,  # [B, Tk, H, K]
    v: Array,  # [B, Tk, H, V]
    key_size: int,
    mask: Array | None,  # [B, H, Tq, Tk] True=keep
) -> tuple[Array, Array | None]:
    B, Tq, H, K = q.shape
    Tk = k.shape[1]
    # [B, H, Tq, Tk]
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(jnp.asarray(key_size, k.dtype))
    if mask is not None:
        logits = jnp.where(mask, logits, jnp.asarray(-1e30, logits.dtype))
    attn_weights = jax.nn.softmax(logits, axis=-1)  # [B, H, Tq, Tk]
    # [B, Tq, H, V]
    attn = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
    # Merge heads: [B, Tq, H*V]
    attn = attn.reshape(B, Tq, -1)
    return attn, attn_weights


def _query_chunk_attention(
    query: Array,  # [B, Qc, H, K]
    key: Array,    # [B, Tk, H, K]
    value: Array,  # [B, Tk, H, V]
    mask: Array | None,  # [B, H, Qc, Tk] or [B, 1, Qc, 1]
    precision= jax.lax.Precision.HIGHEST,
) -> Array:  # [B, Qc, H, V]
    # Stable exp-sum across key chunks
    logits = jnp.einsum("bqhd,bkhd->bhqk", query, key, precision=precision)
    max_score = jnp.max(logits, axis=-1, keepdims=True)
    max_score = jax.lax.stop_gradient(max_score)
    if mask is not None:
        logits = jnp.where(mask, logits, jnp.asarray(-1e30, logits.dtype))
    exp_w = jnp.exp(logits - max_score)
    # [B, H, Qc, V]
    exp_v = jnp.einsum("bhqk,bkhd->bhqd", exp_w, value, precision=precision)
    exp_w_sum = jnp.sum(exp_w, axis=-1, keepdims=False)  # [B, H, Qc]
    # Normalize
    out = exp_v / jnp.clip(exp_w_sum[..., None], a_min=1e-9)
    # Return [B, Qc, H, V]
    return jnp.swapaxes(out, 1, 2)


def memory_efficient_dot_product_attention(
    q: Array,  # [B, Tq, H, K]
    k: Array,  # [B, Tk, H, K]
    v: Array,  # [B, Tk, H, V]
    mask: Array | None,  # [B, H, Tq, Tk]
    precision=jax.lax.Precision.HIGHEST,
    query_chunk_size: int = 512,
    key_chunk_size: int = 2048,
) -> Array:  # [B, Tq, H*V]
    B, Tq, H, K = q.shape
    Tk = k.shape[1]

    # Chunk along queries
    qcs = max(1, query_chunk_size)
    num_chunks = (Tq + qcs - 1) // qcs

    def do_chunk(i):
        start = i * qcs
        end = jnp.minimum(start + qcs, Tq)
        q_chunk = jax.lax.dynamic_slice(q, (0, start, 0, 0), (B, end - start, H, K))
        if mask is None:
            m_chunk = None
        else:
            m_chunk = jax.lax.dynamic_slice(mask, (0, 0, start, 0), (B, H, end - start, Tk))
        out = _query_chunk_attention(q_chunk, k, v, m_chunk, precision=precision)
        return out  # [B, Qc, H, V]

    outs = jax.vmap(do_chunk)(jnp.arange(num_chunks))  # [C, B, Qc, H, V]
    # Stitch back along Tq
    outs = jnp.concatenate([outs[i] for i in range(num_chunks)], axis=1)  # [B, Tq, H, V]
    return outs.reshape(B, Tq, -1)  # [B, Tq, H*V]


def sparse_dot_product_attention(
    q: Array,  # [B, Tq, H, K]
    k: Array,  # [B, Tk, H, K]
    v: Array,  # [B, Tk, H, V]
    mask: Array,  # [B, H, Tq, Tk] (boolean)
) -> Array:
    # Convert boolean mask into edge indices and aggregate.
    # This is a simple (dense->sparse) fallback; not optimized.
    B, Tq, H, K = q.shape
    Tk = k.shape[1]
    outputs = []
    for b in range(B):
        # [H, Tq, Tk]
        mb = mask[b]
        qb = q[b]  # [Tq, H, K]
        kb = k[b]
        vb = v[b]
        # logits
        logits = jnp.einsum("qhd,khd->hqk", qb, kb) / jnp.sqrt(jnp.asarray(K, qb.dtype))  # [H,Tq,Tk]
        logits = jnp.where(mb, logits, jnp.asarray(-1e30, logits.dtype))
        w = jax.nn.softmax(logits, axis=-1)  # [H,Tq,Tk]
        attn = jnp.einsum("hqk,khd->qhd", w, vb)  # [Tq,H,V]
        outputs.append(attn)
    out = jnp.stack(outputs, axis=0)  # [B,Tq,H,V]
    return out.reshape(B, Tq, -1)


# ------------------------------------------------------------
# Custom Multi-Head Attention (NNX)
# ------------------------------------------------------------

class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        model_size: int,
        num_heads: int,
        key_size: int,
        value_size: Optional[int] = None,
        *,
        w_init: nnx.Initializer | None = None,
        b_init: nnx.Initializer | None = None,
        with_bias: bool = True,
        attention_method: str = "dense",  # "dense" | "mem_eff" | "sparse"
        save_attention_weights: bool = False,
        rngs: nnx.Rngs,
    ):
        self.model_size = model_size
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = key_size if value_size is None else value_size
        self.attention_method = attention_method
        self.save_attention_weights = save_attention_weights
        self.with_bias = with_bias

        if w_init is None:
            w_init = nnx.initializers.xavier_uniform()
        if b_init is None:
            b_init = nnx.initializers.zeros_init()

        # Projections
        self.q_proj = nnx.Linear(model_size, num_heads * self.key_size, kernel_init=w_init, bias_init=b_init, use_bias=with_bias, rngs=rngs)
        self.k_proj = nnx.Linear(model_size, num_heads * self.key_size, kernel_init=w_init, bias_init=b_init, use_bias=with_bias, rngs=rngs)
        self.v_proj = nnx.Linear(model_size, num_heads * self.value_size, kernel_init=w_init, bias_init=b_init, use_bias=with_bias, rngs=rngs)
        self.o_proj = nnx.Linear(num_heads * self.value_size, model_size, kernel_init=w_init, bias_init=b_init, use_bias=with_bias, rngs=rngs)

        # Optional state to store attention weights
        if self.save_attention_weights:
            self.attn_weights = nnx.State(jnp.zeros((1, 1, 1, 1), dtype=jnp.float32))

    def __call__(self, x_q: Array, x_k: Array | None = None, x_v: Array | None = None, mask: Array | None = None) -> Array:
        # x_*: [B, T, D]
        if x_k is None:
            x_k = x_q
        if x_v is None:
            x_v = x_k
        B, Tq, D = x_q.shape
        Tk = x_k.shape[1]

        q = self.q_proj(x_q).reshape(B, Tq, self.num_heads, self.key_size)
        k = self.k_proj(x_k).reshape(B, Tk, self.num_heads, self.key_size)
        v = self.v_proj(x_v).reshape(B, Tk, self.num_heads, self.value_size)

        mask4 = ensure_mask_shape(mask, B, self.num_heads, Tq, Tk)

        if self.attention_method == "dense":
            attn, attn_w = dense_dot_product_attention(q, k, v, self.key_size, mask4)
        elif self.attention_method == "mem_eff":
            attn = memory_efficient_dot_product_attention(q, k, v, mask4)
            attn_w = None
        elif self.attention_method == "sparse":
            if mask4 is None:
                raise ValueError("Sparse attention requires a boolean mask.")
            attn = sparse_dot_product_attention(q, k, v, mask4)
            attn_w = None
        else:
            raise NotImplementedError(f"Unknown attention_method: {self.attention_method}")

        if self.save_attention_weights and (attn_w is not None):
            self.attn_weights.value = attn_w  # [B, H, Tq, Tk]

        out = self.o_proj(attn)  # [B, Tq, D]
        return out