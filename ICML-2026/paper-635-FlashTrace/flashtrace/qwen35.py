"""Qwen3.5 hybrid-architecture support for FlashTrace.

Qwen3.5 interleaves Gated-DeltaNet linear-attention layers with full softmax
attention layers. FlashTrace's IFR attribution is built on per-layer token
mixing matrices; linear-attention layers expose no softmax attention matrix.

This module derives, for a Gated-DeltaNet layer, the *effective* lower-triangular
token mixing matrix ``A`` such that ``core_out[i] == sum_{j<=i} A[i, j] value[j]``.
Because the gated delta rule is linear in the value sequence, ``A`` is recovered
exactly by running the layer's own delta-rule kernel once on an identity-valued
value sequence (one probe column per source position).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .core import LayerAttnInput, ModelMetadata, linearize_norm


class _CapturingDeltaRule:
    """Wraps a delta-rule kernel to record its inputs and output for one call."""

    def __init__(self, fn):
        self._fn = fn
        self.query: torch.Tensor | None = None
        self.key: torch.Tensor | None = None
        self.value: torch.Tensor | None = None
        self.kwargs: dict | None = None
        self.core_out: torch.Tensor | None = None

    def __call__(self, query, key, value, **kwargs):
        self.query = query
        self.key = key
        self.value = value
        self.kwargs = kwargs
        result = self._fn(query, key, value, **kwargs)
        # The kernel returns ``(core_attn_out, last_recurrent_state)``.
        self.core_out = result[0] if isinstance(result, tuple) else result
        return result


@torch.no_grad()
def gated_deltanet_effective_attention(
    linear_attn: nn.Module,
    hidden_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive the effective token-mixing matrix for one Gated-DeltaNet layer.

    Args:
        linear_attn: a ``Qwen3_5GatedDeltaNet`` module.
        hidden_states: input of shape ``[1, S, hidden_size]``.

    Returns:
        ``(attn, value, core_ref)`` where

        - ``attn`` is ``[n_v_heads, S, S]``, lower triangular, the effective
          per-head token mixing matrix.
        - ``value`` is ``[S, n_v_heads, head_v_dim]``, the post-conv value
          sequence fed to the delta-rule kernel.
        - ``core_ref`` is ``[S, n_v_heads, head_v_dim]``, the layer's true
          pre-output-norm core attention output, satisfying
          ``core_ref[i] == sum_j attn[:, i, j] * value[j]``.
    """

    if hidden_states.dim() != 3 or hidden_states.shape[0] != 1:
        raise ValueError("gated_deltanet_effective_attention expects a [1, S, H] input.")

    original = linear_attn.chunk_gated_delta_rule
    capture = _CapturingDeltaRule(original)
    linear_attn.chunk_gated_delta_rule = capture
    try:
        linear_attn(hidden_states)
    finally:
        linear_attn.chunk_gated_delta_rule = original

    if capture.value is None or capture.core_out is None:
        raise RuntimeError("Gated-DeltaNet layer did not invoke its chunked kernel.")

    query = capture.query
    key = capture.key
    value = capture.value  # [1, S, n_v_heads, head_v_dim]
    kwargs = dict(capture.kwargs or {})
    core_ref = capture.core_out  # [1, S, n_v_heads, head_v_dim]

    seq_len = value.shape[1]
    n_v_heads = value.shape[2]

    # Probe the (value-linear) kernel with an identity value sequence: column j
    # carries a unit impulse at source position j, so the output reads off A.
    identity = torch.eye(seq_len, dtype=value.dtype, device=value.device)
    value_probe = identity[None, :, None, :].expand(1, seq_len, n_v_heads, seq_len).contiguous()

    probe_out = original(query, key, value_probe, **kwargs)
    probe_core = probe_out[0] if isinstance(probe_out, tuple) else probe_out

    # probe_core[0, i, h, j] == A[h, i, j]
    attn = probe_core[0].permute(1, 0, 2).contiguous()
    # Numerical guard: enforce strict causality (no mass on future tokens).
    attn = torch.tril(attn)

    return attn, value[0].contiguous(), core_ref[0].contiguous()


def is_hybrid_stack(metadata: ModelMetadata) -> bool:
    """True if the decoder mixes linear-attention layers (needs the hybrid path)."""

    return any(spec.kind == "linear" for spec in metadata.layer_specs)


def _full_layer_input(
    layer: nn.Module,
    spec,
    x_prev: torch.Tensor,
    attn_li: torch.Tensor,
    d_model: int,
    model_dtype: torch.dtype,
) -> LayerAttnInput:
    """Build the IFR input triple for a softmax full-attention layer.

    The value path mirrors the legacy IFR computation: a linearized input norm
    followed by the value projection, then GQA expansion to query heads.
    """

    v_w = layer.self_attn.v_proj.weight.detach().to(model_dtype)
    o_w = layer.self_attn.o_proj.weight.detach().to(model_dtype)

    s_prev = linearize_norm(layer.input_layernorm, x_prev.unsqueeze(0)).squeeze(0)
    x_prev_lin = x_prev.float() * s_prev
    V_all = torch.matmul(x_prev_lin.to(model_dtype), v_w.T)
    V_kv = V_all.view(x_prev.shape[0], spec.n_kv_heads, spec.head_dim).contiguous()
    V_q = V_kv.repeat_interleave(spec.group_size, dim=1)
    O_blocks = o_w.view(d_model, spec.n_heads_q, spec.head_dim).permute(1, 2, 0).contiguous()
    return LayerAttnInput(attn=attn_li.to(model_dtype), V_q=V_q, O_blocks=O_blocks)


def _linear_layer_input(
    layer: nn.Module,
    x_prev: torch.Tensor,
    d_model: int,
    model_dtype: torch.dtype,
) -> LayerAttnInput:
    """Build the IFR input triple for a Gated-DeltaNet linear-attention layer."""

    hidden = layer.input_layernorm(x_prev.unsqueeze(0))
    attn, value, _core_ref = gated_deltanet_effective_attention(layer.linear_attn, hidden)
    n_v_heads, head_v_dim = value.shape[1], value.shape[2]

    o_w = layer.linear_attn.out_proj.weight.detach().to(model_dtype)
    O_blocks = o_w.view(d_model, n_v_heads, head_v_dim).permute(1, 2, 0).contiguous()
    return LayerAttnInput(
        attn=attn.to(model_dtype),
        V_q=value.to(model_dtype),
        O_blocks=O_blocks,
    )


@torch.no_grad()
def build_layer_inputs(
    metadata: ModelMetadata,
    cache: Dict[str, List[Optional[torch.Tensor]]],
    attentions: Optional[Sequence[torch.Tensor]],
    model_dtype: torch.dtype,
) -> List[LayerAttnInput]:
    """Build precomputed per-layer IFR inputs for a hybrid Qwen3.5 decoder.

    Full-attention layers reuse the softmax attention captured via
    ``output_attentions``; linear-attention layers derive an effective
    token-mixing matrix from the Gated-DeltaNet recurrence.

    Note: ``attentions`` holds one entry per *full-attention* layer (linear
    layers contribute nothing), so it is indexed by full-attention order, not
    by global layer index.
    """

    d_model = metadata.d_model
    layer_inputs: List[LayerAttnInput] = []
    full_idx = 0
    for li, (layer, spec) in enumerate(zip(metadata.layers, metadata.layer_specs)):
        x_prev_full = cache["pre_attn_resid"][li]
        if x_prev_full is None:
            raise RuntimeError(f"Missing captured residual for layer {li}.")
        x_prev = x_prev_full[0]

        if spec.kind == "linear":
            layer_inputs.append(_linear_layer_input(layer, x_prev, d_model, model_dtype))
        else:
            if attentions is None or full_idx >= len(attentions) or attentions[full_idx] is None:
                raise RuntimeError(
                    f"Full-attention layer {li} has no captured attention weights; "
                    "load the model with attn_implementation='eager'."
                )
            attn_li = attentions[full_idx][0]
            layer_inputs.append(
                _full_layer_input(layer, spec, x_prev, attn_li, d_model, model_dtype)
            )
            full_idx += 1
    return layer_inputs
