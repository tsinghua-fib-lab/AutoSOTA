"""Information Flow Routes (IFR) utilities integrated for CAGE.

This module is adapted from the original agenttrace implementation and provides the
core tensor utilities required to compute IFR token attributions.  It exposes
model-agnostic helpers that assume a Llama/Qwen style stack with the attributes
used below.  The code is intentionally self-contained so it can be imported
directly by the attribution pipeline without depending on the agenttrace repo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm


@dataclass
class LayerSpec:
    """Per-layer geometry for a (possibly hybrid) decoder stack.

    ``kind`` is ``"full"`` for softmax self-attention layers and ``"linear"``
    for Gated-DeltaNet style linear-attention layers (e.g. Qwen3.5). The head
    counts describe the value/output geometry IFR attributes through: for
    linear layers this is the Gated-DeltaNet value head layout.
    """

    kind: str
    n_heads_q: int
    n_kv_heads: int
    head_dim: int
    group_size: int


def _detect_layer_kind(layer: nn.Module, config: Any, idx: int) -> str:
    """Classify a decoder layer as ``"full"`` or ``"linear"`` attention."""

    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and idx < len(layer_types):
        return "linear" if "linear" in str(layer_types[idx]) else "full"
    if hasattr(layer, "linear_attn") and not hasattr(layer, "self_attn"):
        return "linear"
    return "full"


def _build_layer_spec(
    layer: nn.Module,
    config: Any,
    idx: int,
    *,
    full_n_heads_q: int,
    full_n_kv_heads: int,
    full_head_dim: int,
) -> "LayerSpec":
    """Build the per-layer geometry descriptor for one decoder layer."""

    kind = _detect_layer_kind(layer, config, idx)
    if kind == "linear":
        n_v_heads = int(getattr(config, "linear_num_value_heads"))
        head_dim = int(getattr(config, "linear_value_head_dim"))
        # Gated-DeltaNet mixes value heads independently: no GQA grouping at
        # the value/output stage IFR attributes through.
        return LayerSpec(
            kind="linear",
            n_heads_q=n_v_heads,
            n_kv_heads=n_v_heads,
            head_dim=head_dim,
            group_size=1,
        )
    return LayerSpec(
        kind="full",
        n_heads_q=full_n_heads_q,
        n_kv_heads=full_n_kv_heads,
        head_dim=full_head_dim,
        group_size=full_n_heads_q // full_n_kv_heads,
    )


@dataclass
class ModelMetadata:
    """Structural details extracted from the transformer decoder stack."""

    decoder: nn.Module
    layers: Sequence[nn.Module]
    n_layers: int
    d_model: int
    n_heads_q: int
    n_kv_heads: int
    head_dim: int
    group_size: int
    rotary_emb: Optional[nn.Module] = None
    layer_specs: Sequence[LayerSpec] = field(default_factory=tuple)


def extract_model_metadata(model: nn.Module) -> ModelMetadata:
    """Derive metadata for models with Llama/Qwen style decoder blocks."""

    if not hasattr(model, "model"):
        raise AttributeError(
            "Expected a causal LM with `model` attribute exposing the decoder stack."
        )

    decoder = model.model
    if not hasattr(decoder, "layers"):
        raise AttributeError("Decoder does not expose `layers`; IFR assumes a layer list.")

    layers: Sequence[nn.Module] = decoder.layers
    n_layers = len(layers)
    if n_layers == 0:
        raise ValueError("Decoder contains no layers; cannot run IFR.")

    d_model = getattr(model.config, "hidden_size", None)
    if d_model is None:
        raise AttributeError("Model config is missing `hidden_size`, required for IFR.")

    try:
        n_heads_q = model.config.num_attention_heads
        n_kv_heads = model.config.num_key_value_heads
    except AttributeError:
        first_attn = layers[0].self_attn
        n_heads_q = getattr(first_attn, "num_heads")
        n_kv_heads = getattr(first_attn, "num_key_value_groups", n_heads_q)

    group_size = n_heads_q // n_kv_heads
    if n_heads_q % n_kv_heads != 0:
        raise ValueError("IFR assumes grouped-query attention with integer group size.")

    head_dim = getattr(model.config, "head_dim", None)
    if head_dim is None:
        first_attn = layers[0].self_attn
        head_dim = getattr(first_attn, "head_dim", None)
    if head_dim is None:
        # Fallback: infer from V projection rows.
        v_rows = layers[0].self_attn.v_proj.weight.shape[0]
        head_dim = v_rows // n_kv_heads

    rotary_emb = getattr(decoder, "rotary_emb", None)
    if rotary_emb is None:
        first_attn = getattr(layers[0], "self_attn", None)
        rotary_emb = getattr(first_attn, "rotary_emb", None) if first_attn is not None else None

    layer_specs = tuple(
        _build_layer_spec(
            layer,
            model.config,
            idx,
            full_n_heads_q=n_heads_q,
            full_n_kv_heads=n_kv_heads,
            full_head_dim=head_dim,
        )
        for idx, layer in enumerate(layers)
    )

    return ModelMetadata(
        decoder=decoder,
        layers=layers,
        n_layers=n_layers,
        d_model=d_model,
        n_heads_q=n_heads_q,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        group_size=group_size,
        rotary_emb=rotary_emb,
        layer_specs=layer_specs,
    )


def build_weight_pack(metadata: ModelMetadata, model_dtype: torch.dtype) -> List[Dict[str, torch.Tensor | nn.Module]]:
    """Collect per-layer tensors/modules required for IFR."""

    specs = metadata.layer_specs or [LayerSpec("full", 0, 0, 0, 1)] * len(metadata.layers)
    weight_pack: List[Dict[str, torch.Tensor | nn.Module]] = []
    for layer, spec in zip(metadata.layers, specs):
        if spec.kind == "linear":
            # Linear-attention (Gated-DeltaNet) layers have no q/k/v/o_proj;
            # their IFR inputs are supplied separately via layer_inputs.
            weight_pack.append(
                {
                    "kind": "linear",
                    "in_ln": layer.input_layernorm,
                    "post_attn_ln": layer.post_attention_layernorm,
                    "mlp": layer.mlp,
                }
            )
            continue
        attn = layer.self_attn
        pack: Dict[str, torch.Tensor | nn.Module] = {
            "v_w": attn.v_proj.weight.detach().to(dtype=model_dtype),
            "o_w": attn.o_proj.weight.detach().to(dtype=model_dtype),
            "q_w": attn.q_proj.weight.detach().to(dtype=model_dtype),
            "k_w": attn.k_proj.weight.detach().to(dtype=model_dtype),
            "in_ln": layer.input_layernorm,
            "post_attn_ln": layer.post_attention_layernorm,
            "mlp": layer.mlp,
        }
        q_bias = getattr(attn.q_proj, "bias", None)
        k_bias = getattr(attn.k_proj, "bias", None)
        if q_bias is not None:
            pack["q_bias"] = q_bias.detach().to(dtype=model_dtype)
        if k_bias is not None:
            pack["k_bias"] = k_bias.detach().to(dtype=model_dtype)
        weight_pack.append(pack)
    return weight_pack


# ---------------------------------------------------------------------------
# Attention recomputation utilities
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half — standard RoPE helper."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K tensors."""
    # cos/sin shape from HF: [1, S, head_dim] or [S, head_dim] — broadcast to [1, 1, S, head_dim]
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif sin.dim() == 3:
        sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


@torch.no_grad()
def recompute_layer_attention(
    x_prev: torch.Tensor,
    layer_weights: Dict[str, torch.Tensor | nn.Module],
    rotary_emb: nn.Module,
    params: IFRParameters,
) -> torch.Tensor:
    """Recompute attention weights for a single layer from cached activations.

    Returns attention weights of shape ``[n_heads_q, S, S]`` (post-softmax, causal masked).
    This avoids the need to store all layers' attention maps simultaneously.
    """
    device = x_prev.device
    model_dtype = params.model_dtype
    S = x_prev.shape[0]
    n_heads_q = params.n_heads_q
    n_kv_heads = params.n_kv_heads
    head_dim = params.head_dim
    group_size = params.group_size

    in_ln_mod = layer_weights["in_ln"]
    q_w = layer_weights["q_w"].to(device, non_blocking=True)
    k_w = layer_weights["k_w"].to(device, non_blocking=True)

    # Apply layernorm (actual, not linearized) to get the true normed input
    x_normed = in_ln_mod(x_prev.unsqueeze(0)).squeeze(0).to(model_dtype)

    # Project Q and K
    Q = torch.matmul(x_normed, q_w.T)  # [S, n_heads_q * head_dim]
    K = torch.matmul(x_normed, k_w.T)  # [S, n_kv_heads * head_dim]

    q_bias = layer_weights.get("q_bias")
    k_bias = layer_weights.get("k_bias")
    if q_bias is not None:
        Q = Q + q_bias.to(device, non_blocking=True)
    if k_bias is not None:
        K = K + k_bias.to(device, non_blocking=True)

    # Reshape to [1, n_heads, S, head_dim]
    Q = Q.view(S, n_heads_q, head_dim).transpose(0, 1).unsqueeze(0)
    K = K.view(S, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)

    # Apply rotary position embeddings
    position_ids = torch.arange(S, device=device).unsqueeze(0)
    cos, sin = rotary_emb(K, position_ids)
    Q, K = _apply_rotary_pos_emb(Q, K, cos, sin)

    # GQA: repeat K for grouped-query attention
    K = K.repeat_interleave(group_size, dim=1)  # [1, n_heads_q, S, head_dim]

    # Compute attention scores
    attn_weights = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(head_dim)

    # Apply causal mask
    causal_mask = torch.triu(
        torch.full((S, S), float("-inf"), device=device, dtype=attn_weights.dtype),
        diagonal=1,
    )
    attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

    # Softmax
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_weights = attn_weights.to(model_dtype)

    return attn_weights[0]  # [n_heads_q, S, S]


@dataclass
class IFRParameters:
    """Static configuration describing model geometry and chunk sizes."""

    n_layers: int
    n_heads_q: int
    n_kv_heads: int
    head_dim: int
    group_size: int
    d_model: int
    sequence_length: int
    model_dtype: torch.dtype
    chunk_tokens: int
    sink_chunk_tokens: int
    # Precomputed per-layer IFR inputs for hybrid stacks (None = legacy path).
    layer_inputs: Optional[Sequence["LayerAttnInput"]] = None


@dataclass
class IFRLayerResult:
    """Layer-level contributions for a single sink position."""

    e_attn_tokens: torch.Tensor
    e_resid_attn: float
    head_importance: torch.Tensor
    e_ffn: float
    e_resid_ffn: float


@dataclass
class IFRAggregate:
    """Aggregate IFR statistics for one or more sink positions."""

    per_layer: List[IFRLayerResult]
    token_importance_total: torch.Tensor
    head_importance_total: torch.Tensor
    ffn_importance_per_layer: torch.Tensor
    resid_ffn_importance_per_layer: torch.Tensor


@dataclass
class IFRAllPositions:
    """Batch of IFR outputs across a contiguous range of sink positions."""

    token_importance_matrix: torch.Tensor
    head_importance_matrix: torch.Tensor
    resid_attn_fraction_total: torch.Tensor
    sink_indices: List[int]
    per_layer_results: Optional[List[List[IFRLayerResult]]]
    note: str = ""


class MultiHopIFRResult(NamedTuple):
    """Container returned by ``compute_multi_hop_ifr``."""

    raw_attributions: List[IFRAggregate]
    thinking_ratios: List[float]
    observation: Dict[str, torch.Tensor | List[torch.Tensor]]


@torch.no_grad()
def attach_hooks(
    layers: Sequence[nn.Module],
    model_dtype: torch.dtype,
) -> Tuple[Dict[str, List[Optional[torch.Tensor]]], List[RemovableHandle]]:
    """Attach forward hooks to capture residual streams and MLP activations."""

    cache: Dict[str, List[Optional[torch.Tensor]]] = {
        "pre_attn_resid": [None for _ in range(len(layers))],
        "mid_resid": [None for _ in range(len(layers))],
        "post_resid": [None for _ in range(len(layers))],
        "mlp_out": [None for _ in range(len(layers))],
    }
    hooks: List[RemovableHandle] = []

    def make_pre_ln_hook(li: int):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            x_in = inputs[0]
            cache["pre_attn_resid"][li] = x_in.detach().to(model_dtype)

        return hook

    def make_post_attn_ln_pre_hook(li: int):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            x_mid = inputs[0]
            cache["mid_resid"][li] = x_mid.detach().to(model_dtype)

        return hook

    def make_mlp_hook(li: int):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            cache["mlp_out"][li] = output.detach().to(model_dtype)

        return hook

    def make_block_output_hook(li: int):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output) -> None:
            x_out = output[0] if isinstance(output, (tuple, list)) else output
            cache["post_resid"][li] = x_out.detach().to(model_dtype)

        return hook

    for li, layer in enumerate(layers):
        hooks.append(layer.input_layernorm.register_forward_hook(make_pre_ln_hook(li)))
        hooks.append(
            layer.post_attention_layernorm.register_forward_pre_hook(make_post_attn_ln_pre_hook(li))
        )
        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(li)))
        hooks.append(layer.register_forward_hook(make_block_output_hook(li)))

    return cache, hooks


def linearize_norm(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Linearize LayerNorm/RMSNorm to obtain per-token scaling vectors."""

    if x.dtype != torch.float32:
        x = x.float()

    if hasattr(module, "weight") and module.weight is not None:
        w = module.weight.detach().to(device=x.device, dtype=torch.float32).view(1, 1, -1)
    else:
        w = torch.ones(1, 1, x.shape[-1], dtype=torch.float32, device=x.device)

    name = module.__class__.__name__.lower()
    if name.endswith("rmsnorm"):
        eps = getattr(module, "eps", 1e-6)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
        return w / rms

    eps = getattr(module, "eps", 1e-5)
    mu = x.mean(dim=-1, keepdim=True)
    sigma = ((x - mu).pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    return w / sigma


def l1_norm(x: torch.Tensor) -> torch.Tensor:
    """Return the L1 norm reduced over the last dimension."""

    return x.abs().sum(dim=-1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """Return the L2 norm reduced over the last dimension."""

    return x.pow(2).sum(dim=-1).sqrt()


def proximity(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute proximity contributions used in IFR attribution."""

    return torch.clamp(-l2_norm(a - x) + l2_norm(x), min=0.0)


@dataclass
class LayerAttnInput:
    """Precomputed per-layer IFR inputs for hybrid (or any) decoder stacks.

    Supplying these lets the IFR core treat every layer uniformly regardless of
    its attention mechanism: ``attn`` is the token-mixing matrix, ``V_q`` the
    per-token per-head value vectors, ``O_blocks`` the output projection.
    """

    attn: torch.Tensor  # [n_heads, S, S]
    V_q: torch.Tensor  # [T, n_heads, head_dim]
    O_blocks: torch.Tensor  # [n_heads, head_dim, d_model]


def _resolve_layer_attn_vo(
    li: int,
    *,
    cache: Dict[str, List[Optional[torch.Tensor]]],
    attentions: Optional[Sequence[torch.Tensor]],
    weight_pack: Sequence[Dict[str, torch.Tensor | nn.Module]],
    params: "IFRParameters",
    rotary_emb: Optional[nn.Module],
    layer_inputs: Optional[Sequence["LayerAttnInput"]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(attn, V_q, O_blocks)`` for layer ``li`` on the layer's device.

    With ``layer_inputs`` supplied (hybrid path) the precomputed tensors are
    used directly; otherwise the legacy weight-based computation runs, which is
    numerically identical to the pre-refactor inline code.
    """

    x_prev = cache["pre_attn_resid"][li][0]
    device = x_prev.device
    model_dtype = params.model_dtype

    if layer_inputs is not None:
        inp = layer_inputs[li]
        attn_li = inp.attn.to(device=device, dtype=model_dtype, non_blocking=True)
        V_q = inp.V_q.to(device=device, dtype=model_dtype, non_blocking=True)
        O_blocks = inp.O_blocks.to(device=device, dtype=model_dtype, non_blocking=True)
        return attn_li, V_q, O_blocks

    if attentions is not None:
        attn_li = attentions[li][0]
        if attn_li.device != device or attn_li.dtype != model_dtype:
            attn_li = attn_li.to(device=device, dtype=model_dtype, non_blocking=True)
    else:
        assert rotary_emb is not None, "rotary_emb is required when attentions is None"
        attn_li = recompute_layer_attention(x_prev, weight_pack[li], rotary_emb, params)

    v_w = weight_pack[li]["v_w"].to(device, non_blocking=True)
    o_w = weight_pack[li]["o_w"].to(device, non_blocking=True)
    in_ln_mod = weight_pack[li]["in_ln"]

    s_prev = linearize_norm(in_ln_mod, x_prev.unsqueeze(0)).squeeze(0)
    x_prev_lin = x_prev.float() * s_prev
    V_all = torch.matmul(x_prev_lin.to(model_dtype), v_w.T)
    V_kv = V_all.view(params.sequence_length, params.n_kv_heads, params.head_dim).contiguous()
    V_q = V_kv.repeat_interleave(params.group_size, dim=1)
    O_blocks = (
        o_w.view(params.d_model, params.n_heads_q, params.head_dim).permute(1, 2, 0).contiguous()
    )
    return attn_li, V_q, O_blocks


def _max_layer_heads(
    params: "IFRParameters", layer_inputs: Optional[Sequence["LayerAttnInput"]]
) -> int:
    """Width of the per-head accumulators (max head count across layers)."""

    if layer_inputs is not None:
        return max(int(inp.V_q.shape[1]) for inp in layer_inputs)
    return params.n_heads_q


@torch.no_grad()
def compute_ifr_for_position(
    focus_idx: int,
    cache: Dict[str, List[Optional[torch.Tensor]]],
    attentions: Optional[Sequence[torch.Tensor]],
    weight_pack: Sequence[Dict[str, torch.Tensor | nn.Module]],
    params: IFRParameters,
    renorm_threshold: float = 0.0,
    rotary_emb: Optional[nn.Module] = None,
) -> IFRAggregate:
    """Convenience wrapper computing IFR for a single sink position."""

    all_ifr = compute_ifr_for_all_positions(
        cache=cache,
        attentions=attentions,
        weight_pack=weight_pack,
        params=params,
        renorm_threshold=renorm_threshold,
        sink_range=(focus_idx, focus_idx),
        return_layerwise=True,
        rotary_emb=rotary_emb,
    )

    token_total_cpu = all_ifr.token_importance_matrix[0]
    head_total_cpu = all_ifr.head_importance_matrix[0]
    per_layer = all_ifr.per_layer_results[0] if all_ifr.per_layer_results is not None else []
    ffn_per_layer = torch.tensor([layer.e_ffn for layer in per_layer], dtype=torch.float32)
    resid_ffn_per_layer = torch.tensor(
        [layer.e_resid_ffn for layer in per_layer], dtype=torch.float32
    )

    return IFRAggregate(
        per_layer=per_layer,
        token_importance_total=token_total_cpu,
        head_importance_total=head_total_cpu,
        ffn_importance_per_layer=ffn_per_layer,
        resid_ffn_importance_per_layer=resid_ffn_per_layer,
    )


@torch.no_grad()
def compute_ifr_sentence_aggregate(
    sink_start: int,
    sink_end: int,
    cache: Dict[str, List[Optional[torch.Tensor]]],
    attentions: Optional[Sequence[torch.Tensor]],
    weight_pack: Sequence[Dict[str, torch.Tensor | nn.Module]],
    params: IFRParameters,
    renorm_threshold: float = 0.0,
    sink_weights: Optional[torch.Tensor] = None,
    rotary_emb: Optional[nn.Module] = None,
) -> IFRAggregate:
    """Aggregate IFR contributions over an inclusive sink span [sink_start, sink_end]."""

    assert 0 <= sink_start <= sink_end < params.sequence_length, "Invalid sink span."
    sink_end_exclusive = sink_end + 1

    layer_inputs = params.layer_inputs
    n_layers = params.n_layers
    T = params.sequence_length
    model_dtype = params.model_dtype

    per_layer: List[IFRLayerResult] = []
    head_total_cpu = torch.zeros(_max_layer_heads(params, layer_inputs), dtype=torch.float32)
    token_total_cpu = torch.zeros(T, dtype=torch.float32)
    ffn_per_layer = torch.zeros(n_layers, dtype=torch.float32)
    resid_ffn_per_layer = torch.zeros(n_layers, dtype=torch.float32)

    J_max = sink_end_exclusive

    for li in range(n_layers):
        x_prev_full = cache["pre_attn_resid"][li]
        x_mid_full = cache["mid_resid"][li]
        x_out_full = cache["post_resid"][li]
        mlp_out_full = cache["mlp_out"][li]

        assert x_prev_full is not None
        assert x_mid_full is not None
        assert x_out_full is not None
        assert mlp_out_full is not None

        x_prev = x_prev_full[0]
        x_mid = x_mid_full[0]
        x_out = x_out_full[0]
        mlp_out = mlp_out_full[0]
        layer_device = x_prev.device

        if x_mid.device != layer_device:
            x_mid = x_mid.to(layer_device, non_blocking=True)
        if x_out.device != layer_device:
            x_out = x_out.to(layer_device, non_blocking=True)
        if mlp_out.device != layer_device:
            mlp_out = mlp_out.to(layer_device, non_blocking=True)

        attn_li, V_q, O_blocks = _resolve_layer_attn_vo(
            li,
            cache=cache,
            attentions=attentions,
            weight_pack=weight_pack,
            params=params,
            rotary_emb=rotary_emb,
            layer_inputs=layer_inputs,
        )
        n_heads_q = int(V_q.shape[1])

        if sink_weights is not None:
            w = sink_weights.to(layer_device).to(model_dtype)
            if w.numel() != (sink_end_exclusive - sink_start):
                raise ValueError("sink_weights length must equal number of sink positions.")
            w = w / (w.sum() + 1e-12)
            w_f32 = w.to(torch.float32)
            xS = (
                x_mid[sink_start:sink_end_exclusive]
                .to(torch.float32)
                .mul(w_f32.view(-1, 1))
                .sum(dim=0)
            )
            y_resid_S = (
                x_prev[sink_start:sink_end_exclusive]
                .to(torch.float32)
                .mul(w_f32.view(-1, 1))
                .sum(dim=0)
            )
        else:
            xS = x_mid[sink_start:sink_end_exclusive].to(torch.float32).sum(dim=0)
            y_resid_S = x_prev[sink_start:sink_end_exclusive].to(torch.float32).sum(dim=0)
        xS_l1 = xS.pow(2).sum().sqrt()
        resid_attn_prox_S = torch.clamp(xS_l1 - (y_resid_S - xS).pow(2).sum().sqrt(), min=0.0)

        P = sink_end_exclusive - sink_start
        alpha_slice = attn_li[:, sink_start:sink_end_exclusive, :J_max]
        i_abs = torch.arange(sink_start, sink_end_exclusive, device=layer_device).view(P, 1)
        j_abs = torch.arange(0, J_max, device=layer_device).view(1, J_max)
        mask = (j_abs <= i_abs).to(alpha_slice.dtype)
        if sink_weights is not None:
            w = sink_weights.to(layer_device).to(alpha_slice.dtype)
            w = w / (w.sum() + 1e-12)
            alpha_weight = alpha_slice * w.view(1, -1, 1)
            alpha_sum = (alpha_weight * mask.unsqueeze(0)).sum(dim=1).contiguous()
        else:
            alpha_sum = (alpha_slice * mask.unsqueeze(0)).sum(dim=1).contiguous()

        numer_tok_sum = torch.zeros((J_max,), device=layer_device, dtype=model_dtype)
        numer_head_sum = torch.zeros((n_heads_q,), device=layer_device, dtype=model_dtype)

        for j0 in range(0, J_max, params.chunk_tokens):
            j1 = min(J_max, j0 + params.chunk_tokens)
            V_chunk = V_q[j0:j1]
            F_chunk = torch.einsum("jhd,hdk->jhk", V_chunk, O_blocks)
            A_chunk = alpha_sum[:, j0:j1].permute(1, 0).unsqueeze(-1)
            W_chunk = F_chunk * A_chunk
            dist = (W_chunk.float() - xS).pow(2).sum(dim=-1).sqrt()
            prox = torch.clamp(xS_l1 - dist, min=0.0)
            if renorm_threshold > 0.0:
                prox = prox * (prox >= renorm_threshold)
            numer_tok_sum[j0:j1] += prox.sum(dim=1).to(model_dtype)
            numer_head_sum += prox.sum(dim=0).to(model_dtype)

        denom_S = numer_tok_sum.float().sum() + resid_attn_prox_S + 1e-12
        e_attn_tokens_full = torch.zeros((T,), dtype=torch.float32)
        e_attn_tokens_full[:J_max] = (numer_tok_sum.float() / denom_S).to(torch.float32).cpu()
        e_resid_attn_S = float((resid_attn_prox_S / denom_S).item())
        head_importance_S = (numer_head_sum.float() / denom_S).to(torch.float32).cpu()

        x_out_sum = x_out[sink_start:sink_end_exclusive].to(torch.float32).sum(dim=0)
        y_ffn_sum = mlp_out[sink_start:sink_end_exclusive].to(torch.float32).sum(dim=0)
        x_mid_sum = x_mid[sink_start:sink_end_exclusive].to(torch.float32).sum(dim=0)
        prox_ffn_S = proximity(y_ffn_sum, x_out_sum)
        prox_resid_ffn_S = proximity(x_mid_sum, x_out_sum)
        if renorm_threshold > 0.0:
            if prox_ffn_S < renorm_threshold:
                prox_ffn_S = torch.zeros((), dtype=torch.float32, device=layer_device)
            if prox_resid_ffn_S < renorm_threshold:
                prox_resid_ffn_S = torch.zeros((), dtype=torch.float32, device=layer_device)
        denom_ffn_S = prox_ffn_S + prox_resid_ffn_S + 1e-12
        e_ffn_S = float((prox_ffn_S / denom_ffn_S).item())
        e_resid_ffn_S = float((prox_resid_ffn_S / denom_ffn_S).item())

        per_layer.append(
            IFRLayerResult(
                e_attn_tokens=e_attn_tokens_full,
                e_resid_attn=e_resid_attn_S,
                head_importance=head_importance_S,
                e_ffn=e_ffn_S,
                e_resid_ffn=e_resid_ffn_S,
            )
        )
        token_total_cpu += e_attn_tokens_full
        head_total_cpu[:n_heads_q] += head_importance_S
        ffn_per_layer[li] = e_ffn_S
        resid_ffn_per_layer[li] = e_resid_ffn_S

    return IFRAggregate(
        per_layer=per_layer,
        token_importance_total=token_total_cpu,
        head_importance_total=head_total_cpu,
        ffn_importance_per_layer=ffn_per_layer,
        resid_ffn_importance_per_layer=resid_ffn_per_layer,
    )


@torch.no_grad()
def compute_multi_hop_ifr(
    sink_start: int,
    sink_end: int,
    thinking_span: Tuple[int, int],
    n_hops: int,
    cache: Dict[str, List[Optional[torch.Tensor]]],
    attentions: Optional[Sequence[torch.Tensor]],
    weight_pack: Sequence[Dict[str, torch.Tensor | nn.Module]],
    params: IFRParameters,
    renorm_threshold: float = 0.0,
    observation_mask: Optional[torch.Tensor] = None,
    rotary_emb: Optional[nn.Module] = None,
) -> MultiHopIFRResult:
    """Compute the base and multi-hop IFR distribution for a sink span."""

    hop_count = max(0, int(n_hops))
    sink_start = int(sink_start)
    sink_end = int(sink_end)
    think_start = int(thinking_span[0])
    think_end = int(thinking_span[1])

    if think_start > think_end:
        raise ValueError("thinking_span start must be <= end")

    base_ifr = compute_ifr_sentence_aggregate(
        sink_start=sink_start,
        sink_end=sink_end,
        cache=cache,
        attentions=attentions,
        weight_pack=weight_pack,
        params=params,
        renorm_threshold=renorm_threshold,
        rotary_emb=rotary_emb,
    )

    raw_attributions: List[IFRAggregate] = [base_ifr]

    token_total = base_ifr.token_importance_total
    T = token_total.shape[0]
    if observation_mask is None:
        obs_mask = torch.ones((T,), dtype=torch.float32)
        obs_mask[think_start : min(think_end + 1, T)] = 0.0
        obs_mask[sink_start : min(sink_end + 1, T)] = 0.0
        if think_end + 1 < T:
            obs_mask[think_end + 1 :] = 0.0
    else:
        obs_mask = observation_mask.clone().to(dtype=torch.float32)
        if obs_mask.shape[0] != T:
            raise ValueError("observation_mask must match sequence length.")

    base_obs = token_total.clone().to(torch.float32) * obs_mask
    obs_accum = base_obs.clone()
    per_hop_obs: List[torch.Tensor] = []

    thinking_slice = token_total[think_start : think_end + 1]
    w_thinking = thinking_slice.detach().clone().to(params.model_dtype)
    denom_base = float(token_total.sum().item())
    current_ratio = float(w_thinking.sum().item()) / (denom_base + 1e-12) if denom_base > 0 else 0.0
    ratios: List[float] = [current_ratio]

    for hop in range(1, hop_count + 1):
        hop_ifr = compute_ifr_sentence_aggregate(
            sink_start=think_start,
            sink_end=think_end,
            cache=cache,
            attentions=attentions,
            weight_pack=weight_pack,
            params=params,
            renorm_threshold=renorm_threshold,
            sink_weights=w_thinking,
            rotary_emb=rotary_emb,
        )

        raw_attributions.append(hop_ifr)
        hop_total = hop_ifr.token_importance_total.clone().to(torch.float32)
        obs_only = hop_total * obs_mask * current_ratio
        obs_accum += obs_only
        per_hop_obs.append(obs_only)

        thinking_slice = hop_total[think_start : think_end + 1]
        w_thinking = thinking_slice.detach().clone().to(params.model_dtype)
        hop_denom = float(hop_total.sum().item())
        if hop_denom <= 0.0:
            current_ratio = 0.0
        else:
            current_ratio *= float(w_thinking.sum().item()) / (hop_denom + 1e-12)
        ratios.append(current_ratio)

    obs_avg = obs_accum / float(max(1, hop_count))
    observation = {
        "mask": obs_mask,
        "base": base_obs,
        "per_hop": per_hop_obs,
        "sum": obs_accum,
        "avg": obs_avg,
    }

    return MultiHopIFRResult(raw_attributions=raw_attributions, thinking_ratios=ratios, observation=observation)


@torch.no_grad()
def compute_ifr_for_all_positions(
    cache: Dict[str, List[Optional[torch.Tensor]]],
    attentions: Optional[Sequence[torch.Tensor]],
    weight_pack: Sequence[Dict[str, torch.Tensor | nn.Module]],
    params: IFRParameters,
    renorm_threshold: float = 0.0,
    sink_range: Optional[Tuple[int, int]] = None,
    return_layerwise: bool = False,
    rotary_emb: Optional[nn.Module] = None,
) -> IFRAllPositions:
    """Compute IFR importances for every sink position in the selected range."""

    layer_inputs = params.layer_inputs
    n_layers = params.n_layers
    T = params.sequence_length
    model_dtype = params.model_dtype
    chunk_tokens = params.chunk_tokens
    sink_chunk_tokens = params.sink_chunk_tokens

    attn_start = 0 if sink_range is None else sink_range[0]
    attn_end = (T - 1) if sink_range is None else sink_range[1]
    assert 0 <= attn_start <= attn_end < T, "Invalid sink_range."
    S = attn_end - attn_start + 1

    max_heads = _max_layer_heads(params, layer_inputs)
    token_total = torch.zeros((S, T), dtype=torch.float32)
    head_total = torch.zeros((S, max_heads), dtype=torch.float32)
    resid_attn_total = torch.zeros((S,), dtype=torch.float32)
    per_layer_results: Optional[List[List[IFRLayerResult]]] = [list() for _ in range(S)] if return_layerwise else None

    for li in tqdm(range(n_layers), desc="IFR-all"):
        x_prev_full = cache["pre_attn_resid"][li]
        x_mid_full = cache["mid_resid"][li]
        x_out_full = cache["post_resid"][li]
        mlp_out_full = cache["mlp_out"][li]

        assert x_prev_full is not None
        assert x_mid_full is not None
        assert x_out_full is not None
        assert mlp_out_full is not None

        x_prev = x_prev_full[0]
        x_mid = x_mid_full[0]
        x_out = x_out_full[0]
        mlp_out = mlp_out_full[0]
        layer_device = x_prev.device

        if x_mid.device != layer_device:
            x_mid = x_mid.to(layer_device, non_blocking=True)
        if x_out.device != layer_device:
            x_out = x_out.to(layer_device, non_blocking=True)
        if mlp_out.device != layer_device:
            mlp_out = mlp_out.to(layer_device, non_blocking=True)

        attn_li, V_q, O_blocks = _resolve_layer_attn_vo(
            li,
            cache=cache,
            attentions=attentions,
            weight_pack=weight_pack,
            params=params,
            rotary_emb=rotary_emb,
            layer_inputs=layer_inputs,
        )
        n_heads_q = int(V_q.shape[1])

        xA_l1_vec = x_mid.float().pow(2).sum(dim=-1).sqrt()
        resid_diff_l1_vec = (x_prev.float() - x_mid.float()).pow(2).sum(dim=-1).sqrt()
        resid_prox_vec = torch.clamp(xA_l1_vec - resid_diff_l1_vec, min=0.0)

        for i0 in range(attn_start, attn_end + 1, sink_chunk_tokens):
            i1 = min(attn_end + 1, i0 + sink_chunk_tokens)
            P = i1 - i0

            numer_tok_sum = torch.zeros((P, T), device=layer_device, dtype=model_dtype)
            numer_head_sum = torch.zeros((P, n_heads_q), device=layer_device, dtype=model_dtype)

            for j0 in range(0, i1, chunk_tokens):
                j1 = min(i1, j0 + chunk_tokens)
                V_chunk = V_q[j0:j1]
                alpha_block = attn_li[:, i0:i1, j0:j1].permute(1, 2, 0).contiguous()

                i_abs = torch.arange(i0, i1, device=layer_device).view(P, 1, 1)
                j_abs = torch.arange(j0, j1, device=layer_device).view(1, j1 - j0, 1)
                mask = j_abs <= i_abs

                F_block = torch.einsum("jhd,hdk->jhk", V_chunk, O_blocks)
                diff = alpha_block.unsqueeze(-1).float() * F_block.unsqueeze(0).float()
                diff -= x_mid[i0:i1].float().unsqueeze(1).unsqueeze(2)
                dist_accum = diff.pow(2).sum(dim=-1).sqrt()

                xA_l1_chunk = xA_l1_vec[i0:i1].view(P, 1, 1)
                prox = torch.clamp(xA_l1_chunk - dist_accum, min=0.0)
                prox = prox * mask
                if renorm_threshold > 0.0:
                    prox = prox * (prox >= renorm_threshold)

                numer_tok_sum[:, j0:j1] += prox.sum(dim=2).to(model_dtype)
                numer_head_sum += prox.sum(dim=1).to(model_dtype)

            numer_total_i = numer_tok_sum.float().sum(dim=1)
            denom = resid_prox_vec[i0:i1] + numer_total_i + 1e-12

            e_tokens_chunk = (numer_tok_sum.float() / denom[:, None]).to(torch.float32).cpu()
            e_heads_chunk = (numer_head_sum.float() / denom[:, None]).to(torch.float32).cpu()

            s0 = i0 - attn_start
            s1 = i1 - attn_start
            token_total[s0:s1, :] += e_tokens_chunk
            head_total[s0:s1, :n_heads_q] += e_heads_chunk
            resid_attn_total[s0:s1] += (resid_prox_vec[i0:i1] / denom).to(torch.float32).cpu()

            if return_layerwise and per_layer_results is not None:
                for p in range(P):
                    pos_abs = i0 + p
                    x_out_i = x_out[pos_abs].to(torch.float32)
                    y_ffn_i = mlp_out[pos_abs].to(torch.float32)
                    x_mid_i = x_mid[pos_abs].to(torch.float32)
                    prox_ffn_t = proximity(y_ffn_i, x_out_i)
                    prox_resid_ffn_t = proximity(x_mid_i, x_out_i)
                    if renorm_threshold > 0.0:
                        if prox_ffn_t < renorm_threshold:
                            prox_ffn_t = torch.zeros((), dtype=torch.float32, device=layer_device)
                        if prox_resid_ffn_t < renorm_threshold:
                            prox_resid_ffn_t = torch.zeros((), dtype=torch.float32, device=layer_device)
                    denom_ffn_t = prox_ffn_t + prox_resid_ffn_t + 1e-12
                    e_ffn = float((prox_ffn_t / denom_ffn_t).item())
                    e_resid_ffn = float((prox_resid_ffn_t / denom_ffn_t).item())
                    e_resid_attn = float((resid_prox_vec[pos_abs] / denom[p]).item())
                    layer_result = IFRLayerResult(
                        e_attn_tokens=e_tokens_chunk[p],
                        e_resid_attn=e_resid_attn,
                        head_importance=e_heads_chunk[p],
                        e_ffn=e_ffn,
                        e_resid_ffn=e_resid_ffn,
                    )
                    per_layer_results[s0 + p].append(layer_result)

    return IFRAllPositions(
        token_importance_matrix=token_total,
        head_importance_matrix=head_total,
        resid_attn_fraction_total=resid_attn_total,
        sink_indices=list(range(attn_start, attn_end + 1)),
        per_layer_results=per_layer_results,
        note="Sum over layers of layer-normalized importances per sink.",
    )


__all__ = [
    "ModelMetadata",
    "extract_model_metadata",
    "build_weight_pack",
    "recompute_layer_attention",
    "IFRParameters",
    "IFRLayerResult",
    "IFRAggregate",
    "IFRAllPositions",
    "MultiHopIFRResult",
    "attach_hooks",
    "compute_ifr_for_position",
    "compute_ifr_sentence_aggregate",
    "compute_multi_hop_ifr",
    "compute_ifr_for_all_positions",
]
