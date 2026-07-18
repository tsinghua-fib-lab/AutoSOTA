# modeling_gpt.py
#
# circuit-oriented modeling of gpt-2 small.

from __future__ import annotations

from ..utils import DEVICE

import copy
import math
import warnings
from dataclasses import dataclass
from typing import Callable, cast

import einops
import torch
import torch.nn as nn

from transformer_lens import HookedTransformer
import transformer_lens.loading_from_pretrained as loading

from transformer_lens.components import (
    Embed,
    PosEmbed,
    LayerNormPre,
    Unembed,
    MLP,
)
from transformer_lens.utilities.addmm import batch_addmm

from ..circuit import (
    Circuit,
    Edge,
    Node,
    edge_key,
    node_key,
    create_circuit_from_nodes_and_edges,
)

warnings.filterwarnings(
    "ignore",
    message=r".*torch_dtype.*",
    category=FutureWarning,
)

# --------------------------------------------------------------------------------------
# masks and edge intervention
# --------------------------------------------------------------------------------------

EdgeIntervention = Callable[
    [Edge, torch.Tensor, node_key, node_key],
    torch.Tensor | None,
]


@dataclass(frozen=True)
class EdgeLogitGroupSpec:
    keys: tuple[edge_key, ...]
    shape: tuple[int, ...]
    name: object | None = None


@dataclass(frozen=True)
class WeightLogitGroupSpec:
    items: tuple[tuple[node_key, str], ...]
    shape: tuple[int, ...]
    name: object | None = None


@dataclass
class GPTEdgeMasks:
    """
    Dense edge masks for the GPT execution layout.

    Circuit remains the public structural representation. During DiscoGP
    training, this pack avoids rebuilding a soft Circuit and then reconstructing
    these same dense masks inside every forward pass.
    """

    attention_qkv: tuple[torch.Tensor, ...]
    mlp: tuple[torch.Tensor, ...]
    output: torch.Tensor


@dataclass
class GPTAttentionWeightMasks:
    W_Q: torch.Tensor | None = None
    b_Q: torch.Tensor | None = None
    W_K: torch.Tensor | None = None
    b_K: torch.Tensor | None = None
    W_V: torch.Tensor | None = None
    b_V: torch.Tensor | None = None
    W_O: torch.Tensor | None = None
    b_O: torch.Tensor | None = None


@dataclass
class GPTMLPWeightMasks:
    W_in: torch.Tensor | None = None
    b_in: torch.Tensor | None = None
    W_out: torch.Tensor | None = None
    b_out: torch.Tensor | None = None


@dataclass
class GPTWeightMasks:
    attention: tuple[GPTAttentionWeightMasks, ...]
    mlp: tuple[GPTMLPWeightMasks, ...]


@dataclass
class GPTRuntimeMasks:
    edge_masks: GPTEdgeMasks | None = None
    weight_masks: GPTWeightMasks | None = None


@dataclass
class PerLayerCache:
    layer_inputs: tuple[tuple[torch.Tensor, tuple[node_key, ...]], ...]
    output_input: tuple[torch.Tensor, tuple[node_key, ...]]


ATTENTION_NODE_WEIGHT_KEYS: dict[str, tuple[str, ...]] = {
    "attn_q": ("W_Q", "b_Q"),
    "attn_k": ("W_K", "b_K"),
    "attn_v": ("W_V", "b_V"),
    "attn_o": ("W_O", "b_O"),
}
MLP_WEIGHT_KEYS = ("W_in", "b_in", "W_out", "b_out")


def apply_gate(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x

    gate = mask.to(device=x.device, dtype=x.dtype)
    return x * gate


def apply_weight_mask(
    weight: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return weight

    gate = mask.to(device=weight.device, dtype=weight.dtype)
    return weight * gate


def source_axis(x: torch.Tensor) -> torch.Tensor:
    return einops.rearrange(
        x,
        "batch pos d_model -> batch pos 1 d_model",
    )


def expand_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    return x.unsqueeze(2).expand(-1, -1, n_heads, -1)


def attention_keys(layer_id: int, n_heads: int, kind: str) -> list[node_key]:
    return [
        (layer_id, head, kind)
        for head in range(n_heads)
    ]


def apply_edge_intervention(
    *,
    edge: Edge,
    current_src: torch.Tensor,
    dst: node_key,
    src: node_key,
    edge_intervention: EdgeIntervention | None,
) -> torch.Tensor:
    """
    edge contribution rule.

    edge_mask is None:
        implicit on, use current source activation.

    edge_mask is explicit:
        gate * current_src + (1 - gate) * replacement_src.

    by default, replacement_src is zero, so off edges are zero-ablated.
    """
    if edge.edge_mask is None:
        return current_src

    gate = edge.edge_mask.to(device=current_src.device, dtype=current_src.dtype)

    replacement = (
        None
        if edge_intervention is None
        else edge_intervention(edge, current_src, dst, src)
    )
    if replacement is None:
        replacement = torch.zeros_like(current_src)

    replacement = replacement.to(device=current_src.device, dtype=current_src.dtype)
    return gate * current_src + (1.0 - gate) * replacement


def edge_gate(
    edge: Edge,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if edge.edge_mask is None:
        return torch.ones((), device=device, dtype=dtype)

    return edge.edge_mask.to(device=device, dtype=dtype)


def gather_for_dst(
    *,
    circuit: Circuit,
    residual: torch.Tensor,
    src_keys: list[node_key],
    dst: node_key,
    edge_intervention: EdgeIntervention | None = None,
) -> torch.Tensor:
    """
    gather residual contributors for a destination node.

    residual:
        [batch, pos, src_node_id, d_model]
    """
    src_index = {src: i for i, src in enumerate(src_keys)}
    out = torch.zeros_like(residual[:, :, 0, :])

    for edge in circuit.incoming_edges_of(dst):
        if edge.src not in src_index:
            continue

        current_src = residual[:, :, src_index[edge.src], :]

        contrib = apply_edge_intervention(
            edge=edge,
            current_src=current_src,
            dst=dst,
            src=edge.src,
            edge_intervention=edge_intervention,
        )
        out = out + contrib

    return out


def dense_edge_mask_for_dsts(
    *,
    circuit: Circuit,
    src_keys: list[node_key],
    dst_keys: list[node_key],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build a dense [src, dst] edge-mask matrix from the modular Circuit.

    The Circuit remains the public representation. This helper is only an
    execution cache for vectorized forwards, so the autograd graph has one
    grouped masking op instead of one activation-sized op per edge.
    """
    zero = torch.zeros((), device=device, dtype=dtype)
    columns: list[torch.Tensor] = []

    for dst in dst_keys:
        edges_by_src = {
            edge.src: edge
            for edge in circuit.incoming_edges_of(dst)
        }
        columns.append(
            torch.stack(
                [
                    edge_gate(edges_by_src[src], device=device, dtype=dtype)
                    if src in edges_by_src
                    else zero
                    for src in src_keys
                ]
            )
        )

    return torch.stack(columns, dim=1)


def gather_for_dsts_dense(
    *,
    circuit: Circuit,
    residual: torch.Tensor,
    src_keys: list[node_key],
    dst_keys: list[node_key],
) -> torch.Tensor:
    mask = dense_edge_mask_for_dsts(
        circuit=circuit,
        src_keys=src_keys,
        dst_keys=dst_keys,
        device=residual.device,
        dtype=residual.dtype,
    )

    return torch.einsum("bpsd,st->bptd", residual, mask)


def gather_for_dst_dense(
    *,
    circuit: Circuit,
    residual: torch.Tensor,
    src_keys: list[node_key],
    dst: node_key,
) -> torch.Tensor:
    mask = dense_edge_mask_for_dsts(
        circuit=circuit,
        src_keys=src_keys,
        dst_keys=[dst],
        device=residual.device,
        dtype=residual.dtype,
    )[:, 0]

    return torch.einsum("bpsd,s->bpd", residual, mask)


def node_gate_vector(
    *,
    circuit: Circuit,
    keys: list[node_key],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.ones((), device=device, dtype=dtype)
            if (node_mask := circuit.nodes[key].node_mask) is None
            else node_mask.to(device=device, dtype=dtype)
            for key in keys
        ]
    )


def apply_node_gates_many(
    x: torch.Tensor,
    *,
    circuit: Circuit,
    keys: list[node_key],
) -> torch.Tensor:
    gates = node_gate_vector(
        circuit=circuit,
        keys=keys,
        device=x.device,
        dtype=x.dtype,
    )
    return x * gates.view(1, 1, -1, 1)

# --------------------------------------------------------------------------------------
# circuit construction
# --------------------------------------------------------------------------------------

def gpt_src_keys_before_layer(cfg, layer_id: int) -> list[node_key]:
    keys: list[node_key] = [(-1, 0, "emb")]

    for layer in range(layer_id):
        for head in range(cfg.n_heads):
            keys.append((layer, head, "attn_o"))
        keys.append((layer, 0, "mlp"))

    return keys


def build_full_gpt_circuit(cfg) -> Circuit:
    nodes: list[Node] = []
    edges: list[Edge] = []

    emb = Node(layer=-1, index=0, kind="emb")
    nodes.append(emb)

    for layer in range(cfg.n_layers):
        pre_srcs = gpt_src_keys_before_layer(cfg, layer)

        for head in range(cfg.n_heads):
            for kind, weight_keys in ATTENTION_NODE_WEIGHT_KEYS.items():
                node = Node(
                    layer=layer,
                    index=head,
                    kind=kind,
                    weight_masks={key: None for key in weight_keys},
                )
                nodes.append(node)

                if kind != "attn_o":
                    for src in pre_srcs:
                        edges.append(Edge(dst=node.key, src=src))

        mlp = Node(
            layer=layer,
            index=0,
            kind="mlp",
            weight_masks={
                key: None
                for key in MLP_WEIGHT_KEYS
            },
        )
        nodes.append(mlp)

        mlp_srcs = pre_srcs + [
            (layer, head, "attn_o")
            for head in range(cfg.n_heads)
        ]

        for src in mlp_srcs:
            edges.append(Edge(dst=mlp.key, src=src))

    output = Node(layer=cfg.n_layers, index=0, kind="output")
    nodes.append(output)

    for src in gpt_src_keys_before_layer(cfg, cfg.n_layers):
        edges.append(Edge(dst=output.key, src=src))

    return create_circuit_from_nodes_and_edges(nodes, edges)


def finalize_gpt_circuit(circuit: Circuit) -> Circuit:
    if len(circuit.nodes) == 0:
        return circuit.clone()

    output_nodes = [key for key in circuit.nodes if key[2] == "output"]
    if len(output_nodes) == 0:
        return circuit.clone()

    output_key = max(output_nodes, key=lambda x: x[0])
    n_transformer_layers = output_key[0]
    n_heads = (
        max((key[1] for key in circuit.nodes if key[2] == "attn_o"), default=-1)
        + 1
    )

    source_dependencies: dict[node_key, tuple[node_key, ...]] = {}
    for layer in range(n_transformer_layers):
        for head in range(n_heads):
            o = (layer, head, "attn_o")
            if o in circuit.nodes and circuit.nodes[o].is_kept():
                source_dependencies[o] = tuple(
                    key
                    for key in (
                        (layer, head, "attn_q"),
                        (layer, head, "attn_k"),
                        (layer, head, "attn_v"),
                    )
                    if key in circuit.nodes and circuit.nodes[key].is_kept()
                )

        mlp = (layer, 0, "mlp")
        if mlp in circuit.nodes and circuit.nodes[mlp].is_kept():
            source_dependencies[mlp] = (mlp,)

    needed_dsts: set[node_key] = {output_key}
    needed_sources: set[node_key] = set()
    dst_worklist: list[node_key] = [output_key]
    src_worklist: list[node_key] = []
    expanded_dsts: set[node_key] = set()
    expanded_sources: set[node_key] = set()

    while dst_worklist or src_worklist:
        while dst_worklist:
            dst = dst_worklist.pop()
            if dst in expanded_dsts:
                continue
            expanded_dsts.add(dst)

            for edge in circuit.incoming_edges_of(dst):
                if edge.is_kept() and edge.src not in needed_sources:
                    needed_sources.add(edge.src)
                    src_worklist.append(edge.src)

        while src_worklist:
            src = src_worklist.pop()
            if src in expanded_sources:
                continue
            expanded_sources.add(src)

            for dst in source_dependencies.get(src, ()):
                if dst not in needed_dsts:
                    needed_dsts.add(dst)
                    dst_worklist.append(dst)

    false_mask = torch.tensor(False, dtype=torch.bool, device=DEVICE)
    out = circuit.clone()

    for edge in out.all_edges():
        keep = (
            edge.is_kept()
            and edge.dst in needed_dsts
            and edge.src in needed_sources
        )
        if not keep:
            edge.edge_mask = false_mask.clone()

    for key, node in out.nodes.items():
        live = key in needed_sources if node.is_src() else key in needed_dsts
        if not live:
            node.node_mask = false_mask.clone()

    return out

# --------------------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    causal_mask: torch.Tensor
    ignore: float

    def __init__(self, cfg, device: str):
        super().__init__()

        self.cfg = cfg

        self.W_Q = nn.Parameter(
            torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head), device=device)
        )
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head), device=device))

        self.W_K = nn.Parameter(
            torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head), device=device)
        )
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head), device=device))

        self.W_V = nn.Parameter(
            torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head), device=device)
        )
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head), device=device))

        self.W_O = nn.Parameter(
            torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model), device=device)
        )
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model), device=device))

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool, device=device),
                1,
            ),
            persistent=False,
        )
        self.ignore = -1e8

    def _masked_projection_weights(
        self,
        circuit: Circuit | None,
        layer_id: int,
        weight_masks: GPTAttentionWeightMasks | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if weight_masks is not None:
            def apply_runtime_mask(
                weight: torch.Tensor,
                mask: torch.Tensor | None,
            ) -> torch.Tensor:
                if mask is None:
                    return weight
                return weight * mask.to(device=weight.device, dtype=weight.dtype)

            b_O = (
                self.b_O
                if weight_masks.b_O is None
                else self.b_O.unsqueeze(0)
                * weight_masks.b_O.to(device=self.b_O.device, dtype=self.b_O.dtype)
            )

            return (
                apply_runtime_mask(self.W_Q, weight_masks.W_Q),
                apply_runtime_mask(self.b_Q, weight_masks.b_Q),
                apply_runtime_mask(self.W_K, weight_masks.W_K),
                apply_runtime_mask(self.b_K, weight_masks.b_K),
                apply_runtime_mask(self.W_V, weight_masks.W_V),
                apply_runtime_mask(self.b_V, weight_masks.b_V),
                apply_runtime_mask(self.W_O, weight_masks.W_O),
                b_O,
            )

        if circuit is None:
            return (
                self.W_Q,
                self.b_Q,
                self.W_K,
                self.b_K,
                self.W_V,
                self.b_V,
                self.W_O,
                self.b_O,
            )

        has_explicit_masks = False
        for head in range(self.cfg.n_heads):
            for kind in ("attn_q", "attn_k", "attn_v", "attn_o"):
                masks = circuit.nodes[(layer_id, head, kind)].weight_masks
                if any(mask is not None for mask in masks.values()):
                    has_explicit_masks = True
                    break
            if has_explicit_masks:
                break

        if not has_explicit_masks:
            return (
                self.W_Q,
                self.b_Q,
                self.W_K,
                self.b_K,
                self.W_V,
                self.b_V,
                self.W_O,
                self.b_O,
            )

        def mask_heads(
            weight: torch.Tensor,
            *,
            kind: str,
            w_key: str,
        ) -> torch.Tensor:
            masks = [
                circuit.nodes[(layer_id, head, kind)].weight_masks.get(w_key)
                for head in range(self.cfg.n_heads)
            ]
            if all(mask is None for mask in masks):
                return weight

            gates = torch.stack(
                [
                    torch.ones_like(weight[head])
                    if mask is None
                    else mask.to(device=weight.device, dtype=weight.dtype)
                    for head, mask in enumerate(masks)
                ],
                dim=0,
            )
            return weight * gates

        def mask_shared_per_head_bias(
            bias: torch.Tensor,
            *,
            kind: str,
            w_key: str,
        ) -> torch.Tensor:
            masks = [
                circuit.nodes[(layer_id, head, kind)].weight_masks.get(w_key)
                for head in range(self.cfg.n_heads)
            ]
            if all(mask is None for mask in masks):
                return bias

            gates = torch.stack(
                [
                    torch.ones_like(bias)
                    if mask is None
                    else mask.to(device=bias.device, dtype=bias.dtype)
                    for mask in masks
                ],
                dim=0,
            )
            return bias.unsqueeze(0) * gates

        return (
            mask_heads(self.W_Q, kind="attn_q", w_key="W_Q"),
            mask_heads(self.b_Q, kind="attn_q", w_key="b_Q"),
            mask_heads(self.W_K, kind="attn_k", w_key="W_K"),
            mask_heads(self.b_K, kind="attn_k", w_key="b_K"),
            mask_heads(self.W_V, kind="attn_v", w_key="W_V"),
            mask_heads(self.b_V, kind="attn_v", w_key="b_V"),
            mask_heads(self.W_O, kind="attn_o", w_key="W_O"),
            mask_shared_per_head_bias(self.b_O, kind="attn_o", w_key="b_O"),
        )

    def forward(
        self,
        normalized_q_input: torch.Tensor,
        normalized_k_input: torch.Tensor,
        normalized_v_input: torch.Tensor,
        *,
        circuit: Circuit | None,
        layer_id: int,
        weight_masks: GPTAttentionWeightMasks | None = None,
    ) -> torch.Tensor:
        (
            W_Q,
            b_Q,
            W_K,
            b_K,
            W_V,
            b_V,
            W_O,
            b_O,
        ) = self._masked_projection_weights(circuit, layer_id, weight_masks)

        q = torch.einsum("bphd,hde->bphe", normalized_q_input, W_Q) + b_Q
        k = torch.einsum("bphd,hde->bphe", normalized_k_input, W_K) + b_K

        attn_scores = torch.einsum("bqhe,bkhe->bhqk", q, k) / math.sqrt(
            self.cfg.d_head
        )

        seq_len = attn_scores.size(-1)
        attn_scores.masked_fill_(self.causal_mask[:seq_len, :seq_len], self.ignore)

        pattern = attn_scores.softmax(dim=-1)

        v = torch.einsum("bphd,hde->bphe", normalized_v_input, W_V) + b_V
        z = torch.einsum("bhqk,bkhe->bqhe", pattern, v)
        attn_out = torch.einsum("bqhe,hed->bqhd", z, W_O) + (
            b_O / self.cfg.n_heads
        )

        if circuit is not None:
            for head in range(self.cfg.n_heads):
                attn_o_key = (layer_id, head, "attn_o")
                attn_out[:, :, head, :] = apply_gate(
                    attn_out[:, :, head, :],
                    circuit.nodes[attn_o_key].node_mask,
                )

        return attn_out

# --------------------------------------------------------------------------------------
# block
# --------------------------------------------------------------------------------------

class CircuitGPTBlock(nn.Module):
    def __init__(self, cfg, layer_id: int, device: str):
        super().__init__()

        self.cfg = cfg
        self.layer_id = layer_id

        self.ln1: LayerNormPre = LayerNormPre(cfg)
        self.ln2: LayerNormPre = LayerNormPre(cfg)
        self.attn: MultiHeadAttention = MultiHeadAttention(cfg, device=device)
        self.mlp: MLP = MLP(cfg)

        for parameter in self.parameters():
            parameter.requires_grad = False

    def _masked_mlp(
        self,
        x: torch.Tensor,
        weight_masks: dict[str, torch.Tensor | None] | GPTMLPWeightMasks | None,
    ) -> torch.Tensor:
        if weight_masks is None:
            return self.mlp(x)

        if isinstance(weight_masks, GPTMLPWeightMasks):
            masks = {
                "W_in": weight_masks.W_in,
                "b_in": weight_masks.b_in,
                "W_out": weight_masks.W_out,
                "b_out": weight_masks.b_out,
            }
        else:
            masks = weight_masks

        if all(mask is None for mask in masks.values()):
            return self.mlp(x)

        W_in = apply_weight_mask(self.mlp.W_in, masks.get("W_in"))
        b_in = apply_weight_mask(self.mlp.b_in, masks.get("b_in"))
        W_out = apply_weight_mask(self.mlp.W_out, masks.get("W_out"))
        b_out = apply_weight_mask(self.mlp.b_out, masks.get("b_out"))

        pre_act = self.mlp.hook_pre(batch_addmm(b_in, W_in, x))

        hook_mid = getattr(self.mlp, "hook_mid", None)
        ln = getattr(self.mlp, "ln", None)
        if (
            self.mlp.cfg.is_layer_norm_activation()
            and hook_mid is not None 
            and ln is not None
        ):
            mid_act = hook_mid(self.mlp.act_fn(pre_act))
            post_act = self.mlp.hook_post(ln(mid_act))
        else:
            post_act = self.mlp.hook_post(self.mlp.act_fn(pre_act))

        return batch_addmm(b_out, W_out, post_act)

    def forward(
        self,
        residual: torch.Tensor,
        src_keys: list[node_key],
        circuit: Circuit | None,
        *,
        runtime_masks: GPTRuntimeMasks | None = None,
        edge_intervention: EdgeIntervention | None = None,
    ) -> tuple[torch.Tensor, list[node_key]]:
        edge_masks = runtime_masks.edge_masks if runtime_masks is not None else None
        weight_masks = runtime_masks.weight_masks if runtime_masks is not None else None

        q_keys = attention_keys(self.layer_id, self.cfg.n_heads, "attn_q")
        k_keys = attention_keys(self.layer_id, self.cfg.n_heads, "attn_k")
        v_keys = attention_keys(self.layer_id, self.cfg.n_heads, "attn_v")
        qkv_key_groups = (q_keys, k_keys, v_keys)

        if edge_masks is not None:
            qkv_input = torch.einsum(
                "bpsd,qsh->bqphd",
                residual,
                edge_masks.attention_qkv[self.layer_id],
            )
            q_input = qkv_input[:, 0]
            k_input = qkv_input[:, 1]
            v_input = qkv_input[:, 2]
        elif edge_intervention is None:
            if circuit is None:
                full_input = residual.sum(dim=2)
                q_input = k_input = v_input = expand_heads(
                    full_input,
                    self.cfg.n_heads,
                )
            else:
                q_input, k_input, v_input = [
                    apply_node_gates_many(
                        gather_for_dsts_dense(
                            circuit=circuit,
                            residual=residual,
                            src_keys=src_keys,
                            dst_keys=keys,
                        ),
                        circuit=circuit,
                        keys=keys,
                    )
                    for keys in qkv_key_groups
                ]
        else:
            if circuit is None:
                raise ValueError("circuit is required for edge interventions.")

            q_input, k_input, v_input = [
                torch.stack(
                    [
                        apply_gate(
                            gather_for_dst(
                                circuit=circuit,
                                residual=residual,
                                src_keys=src_keys,
                                dst=key,
                                edge_intervention=edge_intervention,
                            ),
                            circuit.nodes[key].node_mask,
                        )
                        for key in keys
                    ],
                    dim=2,
                )
                for keys in qkv_key_groups
            ]

        q_input = self.ln1.forward(q_input)
        k_input = self.ln1.forward(k_input)
        v_input = self.ln1.forward(v_input)

        attn_out = self.attn.forward(
            q_input,
            k_input,
            v_input,
            circuit=circuit,
            layer_id=self.layer_id,
            weight_masks=(
                None
                if weight_masks is None
                else weight_masks.attention[self.layer_id]
            ),
        )

        residual = torch.cat([residual, attn_out], dim=2)

        src_keys = src_keys + [
            (self.layer_id, head, "attn_o")
            for head in range(self.cfg.n_heads)
        ]

        mlp_key = (self.layer_id, 0, "mlp")

        if edge_masks is not None:
            mlp_input = torch.einsum(
                "bpsd,s->bpd",
                residual,
                edge_masks.mlp[self.layer_id],
            )
        elif edge_intervention is None:
            if circuit is None:
                mlp_input = residual.sum(dim=2)
            else:
                mlp_input = gather_for_dst_dense(
                    circuit=circuit,
                    residual=residual,
                    src_keys=src_keys,
                    dst=mlp_key,
                )
        else:
            if circuit is None:
                raise ValueError("circuit is required for edge interventions.")

            mlp_input = gather_for_dst(
                circuit=circuit,
                residual=residual,
                src_keys=src_keys,
                dst=mlp_key,
                edge_intervention=edge_intervention,
            )

        mlp_input = self.ln2.forward(mlp_input)

        mlp_out = self._masked_mlp(
            mlp_input,
            (
                weight_masks.mlp[self.layer_id]
                if weight_masks is not None
                else circuit.nodes[mlp_key].weight_masks
                if circuit is not None
                else None
            ),
        )

        if circuit is not None:
            mlp_out = apply_gate(mlp_out, circuit.nodes[mlp_key].node_mask)

        mlp_out = source_axis(mlp_out)

        residual = torch.cat([residual, mlp_out], dim=2)
        src_keys = src_keys + [mlp_key]

        return residual, src_keys


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------

class CircuitGPT(nn.Module):
    def __init__(self, cfg, device: str | None = None):
        super().__init__()

        self.cfg = copy.deepcopy(cfg)
        self.device_name = device if device is not None else DEVICE

        self.embed = Embed(self.cfg)
        self.pos_embed = PosEmbed(self.cfg)

        self.ln_final = LayerNormPre(self.cfg)
        self.unembed = Unembed(self.cfg)

        self.blocks = nn.ModuleList(
            [
                CircuitGPTBlock(self.cfg, layer_id, device=self.device_name)
                for layer_id in range(self.cfg.n_layers)
            ]
        )

        self.full_circuit = build_full_gpt_circuit(self.cfg)
        self._validated_circuit_ids: set[int] = {id(self.full_circuit)}

        for parameter in self.parameters():
            parameter.requires_grad = False

        self.to(device=self.device_name)

    def _assert_compatible_circuit(self, circuit: Circuit) -> None:
        circuit_id = id(circuit)
        if circuit_id in self._validated_circuit_ids:
            return

        self.full_circuit.assert_same_structure(circuit)
        self._validated_circuit_ids.add(circuit_id)

    def finalize_circuit(self, circuit: Circuit) -> Circuit:
        self._assert_compatible_circuit(circuit)
        return finalize_gpt_circuit(circuit)

    def lookup_weight(self, n_key: node_key, w_key: str) -> torch.Tensor:
        """
        map a circuit-local weight key to the actual model parameter tensor.

        this is the architecture-specific bridge used by architecture-agnostic
        algorithms such as DiscoGP.
        """
        layer, head, kind = n_key

        if (
            kind in ATTENTION_NODE_WEIGHT_KEYS
            and w_key in ATTENTION_NODE_WEIGHT_KEYS[kind]
        ):
            block = cast(CircuitGPTBlock, self.blocks[layer])
            parameter = getattr(block.attn, w_key)
            return parameter if w_key == "b_O" else parameter[head]

        if kind == "mlp":
            block = cast(CircuitGPTBlock, self.blocks[layer])
            param = getattr(block.mlp, w_key)
            if isinstance(param, torch.Tensor):
                return param

        raise KeyError(f"cannot map weight key {(n_key, w_key)} to model parameter.")

    def edge_logit_group_specs(self, circuit: Circuit) -> list[EdgeLogitGroupSpec]:
        self._assert_compatible_circuit(circuit)

        specs: list[EdgeLogitGroupSpec] = []
        seen: set[edge_key] = set()

        for layer in range(self.cfg.n_layers):
            heads = list(range(self.cfg.n_heads))
            qkv_dsts = {
                kind: [(layer, head, kind) for head in heads]
                for kind in ("attn_q", "attn_k", "attn_v")
            }
            srcs = circuit.incoming_srcs(qkv_dsts["attn_q"][0])

            for kind, dsts in qkv_dsts.items():
                for dst in dsts:
                    if circuit.incoming_srcs(dst) != srcs:
                        raise ValueError(
                            f"cannot pack GPT QKV edges for layer {layer}: "
                            f"incoming sources differ for {dst}."
                        )

            qkv_keys = tuple(
                (dst, src)
                for kind in ("attn_q", "attn_k", "attn_v")
                for src in srcs
                for dst in qkv_dsts[kind]
            )
            specs.append(
                EdgeLogitGroupSpec(
                    keys=qkv_keys,
                    shape=(3, len(srcs), self.cfg.n_heads),
                    name=(layer, "attention_qkv"),
                )
            )
            seen.update(qkv_keys)

            mlp_key = (layer, 0, "mlp")
            mlp_srcs = circuit.incoming_srcs(mlp_key)
            mlp_keys = tuple((mlp_key, src) for src in mlp_srcs)
            specs.append(
                EdgeLogitGroupSpec(
                    keys=mlp_keys,
                    shape=(len(mlp_srcs),),
                    name=(layer, "mlp"),
                )
            )
            seen.update(mlp_keys)

        output_key = (self.cfg.n_layers, 0, "output")
        output_srcs = circuit.incoming_srcs(output_key)
        output_keys = tuple((output_key, src) for src in output_srcs)
        specs.append(
            EdgeLogitGroupSpec(
                keys=output_keys,
                shape=(len(output_srcs),),
                name=("output",),
            )
        )
        seen.update(output_keys)

        if seen != circuit.all_edge_keys():
            missing = circuit.all_edge_keys() - seen
            extra = seen - circuit.all_edge_keys()
            raise ValueError(
                "GPT edge packing did not cover circuit exactly. "
                f"missing={len(missing)}, extra={len(extra)}."
            )

        return specs

    def weight_logit_group_specs(self, circuit: Circuit) -> list[WeightLogitGroupSpec]:
        self._assert_compatible_circuit(circuit)

        specs: list[WeightLogitGroupSpec] = []
        seen: set[tuple[node_key, str]] = set()

        for layer in range(self.cfg.n_layers):
            for kind, weight_keys in ATTENTION_NODE_WEIGHT_KEYS.items():
                for w_key in weight_keys:
                    items = tuple(
                        ((layer, head, kind), w_key)
                        for head in range(self.cfg.n_heads)
                    )
                    first_weight = self.lookup_weight(*items[0])
                    specs.append(
                        WeightLogitGroupSpec(
                            items=items,
                            shape=(self.cfg.n_heads, *tuple(first_weight.shape)),
                            name=(layer, "attention", w_key),
                        )
                    )
                    seen.update(items)

            mlp_key = (layer, 0, "mlp")
            for w_key in MLP_WEIGHT_KEYS:
                item = (mlp_key, w_key)
                specs.append(
                    WeightLogitGroupSpec(
                        items=(item,),
                        shape=tuple(self.lookup_weight(*item).shape),
                        name=(layer, "mlp", w_key),
                    )
                )
                seen.add(item)

        all_items = {
            (n_key, w_key)
            for n_key, node in circuit.nodes.items()
            for w_key in node.weight_masks.keys()
        }
        if seen != all_items:
            missing = all_items - seen
            extra = seen - all_items
            raise ValueError(
                "GPT weight packing did not cover circuit exactly. "
                f"missing={len(missing)}, extra={len(extra)}."
            )

        return specs

    def _runtime_edge_masks_from_logits(
        self,
        *,
        edge_logits,
        edge_group_specs,
        sample_mask_fn: Callable[..., torch.Tensor],
        reverse_edges: bool,
        random_mode,
        gs_temp_edge: float,
    ) -> GPTEdgeMasks:
        by_name = {
            spec.name: sample_mask_fn(
                logits,
                random_mode=random_mode,
                reverse=reverse_edges,
                gs_temp=gs_temp_edge,
            )
            for spec, logits in zip(edge_group_specs, edge_logits)
        }

        return GPTEdgeMasks(
            attention_qkv=tuple(
                by_name[(layer, "attention_qkv")]
                for layer in range(self.cfg.n_layers)
            ),
            mlp=tuple(
                by_name[(layer, "mlp")]
                for layer in range(self.cfg.n_layers)
            ),
            output=by_name[("output",)],
        )

    def _runtime_weight_masks_from_logits(
        self,
        *,
        weight_logits,
        weight_group_specs,
        sample_mask_fn: Callable[..., torch.Tensor],
        reverse_weights: bool,
        random_mode,
        gs_temp_weight: float,
    ) -> GPTWeightMasks:
        by_name = {
            spec.name: sample_mask_fn(
                logits,
                random_mode=random_mode,
                reverse=reverse_weights,
                gs_temp=gs_temp_weight,
            )
            for spec, logits in zip(weight_group_specs, weight_logits)
        }

        attention = []
        mlp = []

        for layer in range(self.cfg.n_layers):
            attention.append(
                GPTAttentionWeightMasks(
                    W_Q=by_name[(layer, "attention", "W_Q")],
                    b_Q=by_name[(layer, "attention", "b_Q")],
                    W_K=by_name[(layer, "attention", "W_K")],
                    b_K=by_name[(layer, "attention", "b_K")],
                    W_V=by_name[(layer, "attention", "W_V")],
                    b_V=by_name[(layer, "attention", "b_V")],
                    W_O=by_name[(layer, "attention", "W_O")],
                    b_O=by_name[(layer, "attention", "b_O")],
                )
            )
            mlp.append(
                GPTMLPWeightMasks(
                    W_in=by_name[(layer, "mlp", "W_in")],
                    b_in=by_name[(layer, "mlp", "b_in")],
                    W_out=by_name[(layer, "mlp", "W_out")],
                    b_out=by_name[(layer, "mlp", "b_out")],
                )
            )

        return GPTWeightMasks(attention=tuple(attention), mlp=tuple(mlp))

    @torch.no_grad()
    def boolean_runtime_weight_masks(
        self,
        *,
        weight_logits,
        weight_group_specs,
        boolean_mask_fn: Callable[..., torch.Tensor],
    ) -> GPTWeightMasks:
        def detached_boolean_mask(
            logits: torch.Tensor,
            **_: object,
        ) -> torch.Tensor:
            return boolean_mask_fn(logits).to(dtype=logits.dtype).detach()

        return self._runtime_weight_masks_from_logits(
            weight_logits=weight_logits,
            weight_group_specs=weight_group_specs,
            sample_mask_fn=detached_boolean_mask,
            reverse_weights=False,
            random_mode=None,
            gs_temp_weight=1.0,
        )

    def sample_runtime_masks(
        self,
        *,
        edge_logits=None,
        edge_group_specs=None,
        weight_logits=None,
        weight_group_specs=None,
        frozen_weight_runtime=None,
        sample_mask_fn: Callable[..., torch.Tensor],
        boolean_mask_fn: Callable[..., torch.Tensor] | None = None,
        mode: str,
        reverse_edges: bool = False,
        reverse_weights: bool = False,
        gs_temp_edge: float = 1.0,
        gs_temp_weight: float = 1.0,
        random_mode=None,
    ) -> GPTRuntimeMasks:
        edge_masks = None
        weight_masks = frozen_weight_runtime

        if mode == "edge":
            edge_masks = self._runtime_edge_masks_from_logits(
                edge_logits=edge_logits,
                edge_group_specs=edge_group_specs,
                sample_mask_fn=sample_mask_fn,
                reverse_edges=reverse_edges,
                random_mode=random_mode,
                gs_temp_edge=gs_temp_edge,
            )
        elif mode == "weight":
            weight_masks = self._runtime_weight_masks_from_logits(
                weight_logits=weight_logits,
                weight_group_specs=weight_group_specs,
                sample_mask_fn=sample_mask_fn,
                reverse_weights=reverse_weights,
                random_mode=random_mode,
                gs_temp_weight=gs_temp_weight,
            )
        else:
            raise ValueError(f"unknown runtime mask mode: {mode!r}.")

        return GPTRuntimeMasks(edge_masks=edge_masks, weight_masks=weight_masks)

    def embed_as_residual(
        self,
        tokens: torch.Tensor,
        circuit: Circuit | None,
    ) -> tuple[torch.Tensor, list[node_key]]:
        residual = self.embed(tokens) + self.pos_embed(tokens)
        emb_key = (-1, 0, "emb")

        if circuit is not None:
            residual = apply_gate(residual, circuit.nodes[emb_key].node_mask)

        return source_axis(residual), [emb_key]

    def forward_runtime(
        self,
        tokens: torch.Tensor,
        *,
        runtime_masks: GPTRuntimeMasks,
        return_residual: bool = False,
    ) -> torch.Tensor:
        embed = self.embed(tokens) + self.pos_embed(tokens)
        sources = [source_axis(embed)]
        edge_masks = runtime_masks.edge_masks
        weight_masks = runtime_masks.weight_masks

        for layer_id, block_module in enumerate(self.blocks):
            block = cast(CircuitGPTBlock, block_module)
            residual = torch.cat(sources, dim=2)

            if edge_masks is not None:
                qkv_input = torch.einsum(
                    "bpsd,qsh->bqphd",
                    residual,
                    edge_masks.attention_qkv[layer_id],
                )
                q_input = qkv_input[:, 0]
                k_input = qkv_input[:, 1]
                v_input = qkv_input[:, 2]
            else:
                full_input = residual.sum(dim=2)
                q_input = k_input = v_input = expand_heads(
                    full_input,
                    self.cfg.n_heads,
                )

            q_input = block.ln1.forward(q_input)
            k_input = block.ln1.forward(k_input)
            v_input = block.ln1.forward(v_input)

            attn_out = block.attn.forward(
                q_input,
                k_input,
                v_input,
                circuit=None,
                layer_id=layer_id,
                weight_masks=(
                    None
                    if weight_masks is None
                    else weight_masks.attention[layer_id]
                ),
            )

            sources.append(attn_out)
            residual = torch.cat(sources, dim=2)

            if edge_masks is not None:
                mlp_input = torch.einsum(
                    "bpsd,s->bpd",
                    residual,
                    edge_masks.mlp[layer_id],
                )
            else:
                mlp_input = residual.sum(dim=2)

            mlp_input = block.ln2.forward(mlp_input)
            mlp_out = block._masked_mlp(
                mlp_input,
                (
                    None
                    if weight_masks is None
                    else weight_masks.mlp[layer_id]
                ),
            )

            sources.append(source_axis(mlp_out))

        residual = torch.cat(sources, dim=2)
        if edge_masks is not None:
            final_residual = torch.einsum(
                "bpsd,s->bpd",
                residual,
                edge_masks.output,
            )
        else:
            final_residual = residual.sum(dim=2)

        if return_residual:
            return final_residual

        normalized = self.ln_final(final_residual)
        return self.unembed(normalized)

    @torch.inference_mode()
    def per_layer_cache(
        self,
        tokens: torch.Tensor,
        circuit: Circuit | None = None,
    ) -> PerLayerCache:
        """
        Cache residual-source states before each layer for algorithms that
        process edges from later layers to earlier layers.
        """
        if circuit is not None:
            self._assert_compatible_circuit(circuit)

        residual, src_keys = self.embed_as_residual(tokens, circuit)
        layer_inputs: list[tuple[torch.Tensor, tuple[node_key, ...]]] = []

        for block in self.blocks:
            layer_inputs.append((residual, tuple(src_keys)))
            residual, src_keys = block(
                residual,
                src_keys,
                circuit,
                runtime_masks=None,
                edge_intervention=None,
            )

        return PerLayerCache(
            layer_inputs=tuple(layer_inputs),
            output_input=(residual, tuple(src_keys)),
        )

    def _output_from_residual_sources(
        self,
        *,
        residual: torch.Tensor,
        src_keys: list[node_key],
        circuit: Circuit,
        return_residual: bool = False,
    ) -> torch.Tensor:
        output_key = (self.cfg.n_layers, 0, "output")
        final_residual = gather_for_dst_dense(
            circuit=circuit,
            residual=residual,
            src_keys=src_keys,
            dst=output_key,
        )
        final_residual = apply_gate(
            final_residual,
            circuit.nodes[output_key].node_mask,
        )

        if return_residual:
            return final_residual

        normalized = self.ln_final(final_residual)
        return self.unembed(normalized)

    @torch.inference_mode()
    def forward_from_per_layer_cache(
        self,
        cache: PerLayerCache,
        *,
        circuit: Circuit,
        edge_key: edge_key,
        return_residual: bool = False,
    ) -> torch.Tensor:
        """
        Recompute only the suffix affected by an edge candidate.

        This is optional algorithm support: generic algorithms can probe for
        this method when the caller enables model-specific ACDC acceleration.
        """
        self._assert_compatible_circuit(circuit)
        dst, _ = edge_key
        dst_layer, _, dst_kind = dst

        if dst_kind == "output":
            residual, src_keys_tuple = cache.output_input
            return self._output_from_residual_sources(
                residual=residual,
                src_keys=list(src_keys_tuple),
                circuit=circuit,
                return_residual=return_residual,
            )

        if dst_layer < 0 or dst_layer >= self.cfg.n_layers:
            raise ValueError(
                f"cannot use per-layer cache for edge destination {dst!r}."
            )

        residual, src_keys_tuple = cache.layer_inputs[dst_layer]
        src_keys = list(src_keys_tuple)

        for layer_id in range(dst_layer, self.cfg.n_layers):
            block = cast(CircuitGPTBlock, self.blocks[layer_id])
            residual, src_keys = block(
                residual,
                src_keys,
                circuit,
                runtime_masks=None,
                edge_intervention=None,
            )

        return self._output_from_residual_sources(
            residual=residual,
            src_keys=src_keys,
            circuit=circuit,
            return_residual=return_residual,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        circuit: Circuit | None = None,
        *,
        runtime_masks: GPTRuntimeMasks | None = None,
        edge_intervention: EdgeIntervention | None = None,
        return_residual: bool = False,
    ) -> torch.Tensor:
        if runtime_masks is not None and edge_intervention is not None:
            raise ValueError("runtime_masks cannot be combined with edge_intervention.")

        if circuit is not None and runtime_masks is not None:
            raise ValueError("runtime_masks and circuit are mutually exclusive.")

        if runtime_masks is not None:
            return self.forward_runtime(
                tokens,
                runtime_masks=runtime_masks,
                return_residual=return_residual,
            )

        if circuit is not None:
            self._assert_compatible_circuit(circuit)

        residual, src_keys = self.embed_as_residual(tokens, circuit)

        for block in self.blocks:
            residual, src_keys = block(
                residual,
                src_keys,
                circuit,
                runtime_masks=runtime_masks,
                edge_intervention=edge_intervention,
            )

        output_key = (self.cfg.n_layers, 0, "output")
        edge_masks = runtime_masks.edge_masks if runtime_masks is not None else None

        if edge_masks is not None:
            final_residual = torch.einsum(
                "bpsd,s->bpd",
                residual,
                edge_masks.output,
            )
        elif edge_intervention is None:
            if circuit is None:
                final_residual = residual.sum(dim=2)
            else:
                final_residual = gather_for_dst_dense(
                    circuit=circuit,
                    residual=residual,
                    src_keys=src_keys,
                    dst=output_key,
                )
        else:
            if circuit is None:
                raise ValueError("circuit is required for edge interventions.")

            final_residual = gather_for_dst(
                circuit=circuit,
                residual=residual,
                src_keys=src_keys,
                dst=output_key,
                edge_intervention=edge_intervention,
            )

        if circuit is not None:
            final_residual = apply_gate(
                final_residual,
                circuit.nodes[output_key].node_mask,
            )

        if return_residual:
            return final_residual

        normalized = self.ln_final(final_residual)
        logits = self.unembed(normalized)
        return logits

    @classmethod
    def load_model(
        cls,
        model_name: str = "gpt2-small",
        *,
        device: str | None = None,
    ) -> CircuitGPT:
        supported_models = {"gpt2-small", "gpt2-medium"}
        if model_name not in supported_models:
            raise ValueError(
                "this modeling_gpt.py currently supports "
                f"{sorted(supported_models)}; "
                f"got model_name={model_name!r}."
            )

        device = device if device is not None else DEVICE

        cfg = loading.get_pretrained_model_config(
            model_name=loading.get_official_model_name(model_name),
            hf_cfg=None,
            checkpoint_index=None,
            checkpoint_value=None,
            fold_ln=True,
            default_prepend_bos=True,
        )

        model = cls(cfg, device=device)

        source_model = HookedTransformer.from_pretrained(
            model_name,
            device="cpu",
        )
        state_dict = source_model.state_dict()

        model.load_state_dict(state_dict, strict=False)
        del state_dict, source_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif (
            hasattr(torch, "mps")
            and torch.backends.mps.is_available()
            and hasattr(torch.mps, "empty_cache")
        ):
            torch.mps.empty_cache()

        model.to(device=device)
        model.eval()

        return model
