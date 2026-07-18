# circuit.py
#
# core circuit representation, circuit algebra, and circuit tools.
# emb and pos_emb are designed to be represented as one inseparable emb node,
# if additive positional embedding is used in modeling.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, TypeAlias

import torch

# --------------------------------------------------------------------------------------
# kind sets
# --------------------------------------------------------------------------------------

src_kinds: set[str] = {"emb", "attn_o", "mlp"}
dst_kinds: set[str] = {"attn_q", "attn_k", "attn_v", "mlp", "output"}
node_kinds: set[str] = src_kinds | dst_kinds

# --------------------------------------------------------------------------------------
# type aliases
# --------------------------------------------------------------------------------------

node_key: TypeAlias = tuple[int, int, str]          # layer_idx, head_idx, kind
edge_key: TypeAlias = tuple[node_key, node_key]     # dst_key, src_key
weight_key: TypeAlias = str

# --------------------------------------------------------------------------------------
# mask helpers
# --------------------------------------------------------------------------------------

def scalar_true() -> torch.Tensor:
    return torch.tensor(True, dtype=torch.bool)


def scalar_false() -> torch.Tensor:
    return torch.tensor(False, dtype=torch.bool)


def as_bool_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    return mask.bool()


def clone_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    return mask.detach().clone()


def as_bool_weight_masks(
    masks: dict[weight_key, torch.Tensor | None],
) -> dict[weight_key, torch.Tensor | None]:
    return {key: as_bool_mask(mask) for key, mask in masks.items()}


def clone_weight_masks(
    masks: dict[weight_key, torch.Tensor | None],
) -> dict[weight_key, torch.Tensor | None]:
    return {key: clone_mask(mask) for key, mask in masks.items()}


def scalar_mask_is_on(mask: torch.Tensor | None) -> bool:
    """
    None means implicit on.
    """
    if mask is None:
        return True

    if mask.numel() != 1:
        raise ValueError(
            f"expected scalar mask with one element, got shape {tuple(mask.shape)}."
        )

    return bool(mask.bool().item())


def scalar_mask_not(mask: torch.Tensor | None) -> torch.Tensor:
    """
    complement for scalar masks.
    """
    if mask is None:
        return scalar_false()

    return torch.logical_not(mask.bool())


def scalar_mask_or(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
) -> torch.Tensor | None:
    """
    union for scalar masks.
    """
    if a is None or b is None:
        return None

    return torch.logical_or(a.bool(), b.bool())


def scalar_mask_and(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
) -> torch.Tensor | None:
    """
    intersection for scalar masks.
    """
    if a is None:
        return clone_mask(b)
    if b is None:
        return clone_mask(a)

    return torch.logical_and(a.bool(), b.bool())


def weight_mask_not(mask: torch.Tensor | None) -> torch.Tensor:
    """
    complement for one weight mask.
    """
    if mask is None:
        return scalar_false()

    return torch.logical_not(mask.bool())


def weight_mask_or(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
) -> torch.Tensor | None:
    """
    union for one weight mask.
    """
    if a is None or b is None:
        return None

    if a.shape != b.shape:
        raise ValueError(
            f"weight mask shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}."
        )

    return torch.logical_or(a.bool(), b.bool())


def weight_mask_and(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
) -> torch.Tensor | None:
    """
    intersection for one weight mask.
    """
    if a is None:
        return clone_mask(b)
    if b is None:
        return clone_mask(a)

    if a.shape != b.shape:
        raise ValueError(
            f"weight mask shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}."
        )

    return torch.logical_and(a.bool(), b.bool())


def weight_masks_not(
    masks: dict[weight_key, torch.Tensor | None],
) -> dict[weight_key, torch.Tensor | None]:
    return {
        key: weight_mask_not(mask)
        for key, mask in masks.items()
    }


def weight_masks_or(
    a: dict[weight_key, torch.Tensor | None],
    b: dict[weight_key, torch.Tensor | None],
) -> dict[weight_key, torch.Tensor | None]:
    if set(a.keys()) != set(b.keys()):
        raise ValueError(
            f"weight mask keys differ: {set(a.keys())} vs {set(b.keys())}."
        )

    return {
        key: weight_mask_or(a[key], b[key])
        for key in a.keys()
    }


def weight_masks_and(
    a: dict[weight_key, torch.Tensor | None],
    b: dict[weight_key, torch.Tensor | None],
) -> dict[weight_key, torch.Tensor | None]:
    if set(a.keys()) != set(b.keys()):
        raise ValueError(
            f"weight mask keys differ: {set(a.keys())} vs {set(b.keys())}."
        )

    return {
        key: weight_mask_and(a[key], b[key])
        for key in a.keys()
    }

# --------------------------------------------------------------------------------------
# core data structures
# --------------------------------------------------------------------------------------

@dataclass
class Node:
    """
    atomic circuit node.

    weight_masks maps local parameter names to masks, e.g.
    
        emb node:    {}
        attn_q node: {"W_Q": ..., "b_Q": ...}
        attn_k node: {"W_K": ..., "b_K": ...}
        attn_v node: {"W_V": ..., "b_V": ...}
        attn_o node: {"W_O": ..., "b_O": ...}
        mlp node:    {"W_in": ..., "b_in": ..., "W_out": ..., "b_out": ...}
        output node: {}
    """

    layer: int
    index: int
    kind: str

    node_mask: torch.Tensor | None = None
    weight_masks: dict[weight_key, torch.Tensor | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.layer, int):
            raise TypeError(f"layer must be int, got {type(self.layer)}.")

        if not isinstance(self.index, int):
            raise TypeError(f"index must be int, got {type(self.index)}.")

        if not isinstance(self.kind, str) or len(self.kind) == 0:
            raise ValueError("kind must be a non-empty string.")

        if self.kind not in node_kinds:
            raise ValueError(
                f"unknown node kind {self.kind}. "
                f"expected one of {sorted(node_kinds)}."
            )

        if self.node_mask is not None and self.node_mask.numel() != 1:
            raise ValueError(
                f"node_mask must have exactly one element, "
                f"got shape {tuple(self.node_mask.shape)}."
            )

        self.node_mask = as_bool_mask(self.node_mask)
        self.weight_masks = as_bool_weight_masks(self.weight_masks)

    @property
    def key(self) -> node_key:
        return (self.layer, self.index, self.kind)

    def is_src(self) -> bool:
        return self.kind in src_kinds

    def is_dst(self) -> bool:
        return self.kind in dst_kinds

    def is_kept(self) -> bool:
        return scalar_mask_is_on(self.node_mask)

    def has_weight_masks(self) -> bool:
        return len(self.weight_masks) > 0

    def has_explicit_weight_masks(self) -> bool:
        return any(mask is not None for mask in self.weight_masks.values())

    def clone(self) -> Node:
        return Node(
            layer=self.layer,
            index=self.index,
            kind=self.kind,
            node_mask=clone_mask(self.node_mask),
            weight_masks=clone_weight_masks(self.weight_masks),
        )


@dataclass
class Edge:
    """
    directed edge from src to dst, stored destination-first.

    key:
        (dst, src)
    """

    dst: node_key
    src: node_key

    edge_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.edge_mask is not None and self.edge_mask.numel() != 1:
            raise ValueError(
                f"edge_mask must have exactly one element, "
                f"got shape {tuple(self.edge_mask.shape)}."
            )

        self.edge_mask = as_bool_mask(self.edge_mask)

    @property
    def key(self) -> edge_key:
        return (self.dst, self.src)

    def has_valid_node_kinds(self) -> bool:
        return (
            len(self.dst) == 3
            and self.dst[2] in dst_kinds
            and len(self.src) == 3
            and self.src[2] in src_kinds
        )

    def is_kept(self) -> bool:
        return scalar_mask_is_on(self.edge_mask)

    def clone(self) -> Edge:
        return Edge(
            dst=self.dst,
            src=self.src,
            edge_mask=clone_mask(self.edge_mask),
        )


@dataclass
class Circuit:
    """
    circuit consists of nodes and edges.

    incoming_edges maps each destination node to its incoming edges:
        incoming_edges[dst] = [Edge(dst=dst, src=...), ...]
    """

    nodes: dict[node_key, Node]
    incoming_edges: dict[node_key, list[Edge]]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for key, node in self.nodes.items():
            if key != node.key:
                raise ValueError(
                    f"node dict key {key} does not match node.key {node.key}."
                )

            if node.node_mask is not None and node.node_mask.numel() != 1:
                raise ValueError(
                    f"node_mask for {node.key} must have exactly one element, "
                    f"got shape {tuple(node.node_mask.shape)}."
                )

        seen_edges: set[edge_key] = set()

        for dst, edges in self.incoming_edges.items():
            if dst not in self.nodes:
                raise ValueError(f"incoming edge bucket {dst} not found in nodes.")

            dst_node = self.nodes[dst]
            if not dst_node.is_dst():
                raise ValueError(
                    f"incoming edge bucket {dst} is not a destination node: "
                    f"kind={dst_node.kind}. "
                    f"destination kind must be one of {sorted(dst_kinds)}."
                )

            for edge in edges:
                if edge.dst != dst:
                    raise ValueError(
                        f"edge {edge.key} is stored under wrong destination {dst}."
                    )

                if edge.key in seen_edges:
                    raise ValueError(f"duplicate edge found: {edge.key}.")
                seen_edges.add(edge.key)

                if edge.src not in self.nodes:
                    raise ValueError(f"edge source {edge.src} not found in nodes.")

                if edge.dst not in self.nodes:
                    raise ValueError(f"edge destination {edge.dst} not found in nodes.")

                if edge.edge_mask is not None and edge.edge_mask.numel() != 1:
                    raise ValueError(
                        f"edge_mask for {edge.key} must have exactly one element, "
                        f"got shape {tuple(edge.edge_mask.shape)}."
                    )

                src_node = self.nodes[edge.src]
                dst_node = self.nodes[edge.dst]

                if not src_node.is_src():
                    raise ValueError(
                        f"invalid edge source {edge.src}: kind={src_node.kind}. "
                        f"source kind must be one of {sorted(src_kinds)}."
                    )

                if not dst_node.is_dst():
                    raise ValueError(
                        f"invalid edge destination {edge.dst}: kind={dst_node.kind}. "
                        f"destination kind must be one of {sorted(dst_kinds)}."
                    )

    def all_edges(self) -> list[Edge]:
        return [edge for edges in self.incoming_edges.values() for edge in edges]

    def all_edge_keys(self) -> set[edge_key]:
        return {edge.key for edge in self.all_edges()}

    def edge_dict(self) -> dict[edge_key, Edge]:
        return {edge.key: edge for edge in self.all_edges()}

    def assert_same_structure(self, other: Circuit) -> None:
        self_node_keys = set(self.nodes.keys())
        other_node_keys = set(other.nodes.keys())

        if self_node_keys != other_node_keys:
            raise ValueError(
                "node sets differ.\n"
                f"missing from other: {self_node_keys - other_node_keys}\n"
                f"missing from self: {other_node_keys - self_node_keys}"
            )

        self_edge_keys = self.all_edge_keys()
        other_edge_keys = other.all_edge_keys()

        if self_edge_keys != other_edge_keys:
            raise ValueError(
                "edge sets differ.\n"
                f"missing from other: {self_edge_keys - other_edge_keys}\n"
                f"missing from self: {other_edge_keys - self_edge_keys}"
            )

        for key, node in self.nodes.items():
            other_node = other.nodes[key]
            if set(node.weight_masks.keys()) != set(other_node.weight_masks.keys()):
                raise ValueError(
                    f"weight mask keys differ for node {key}: "
                    f"{set(node.weight_masks.keys())} vs "
                    f"{set(other_node.weight_masks.keys())}"
                )

    def clone(self) -> Circuit:
        return Circuit(
            nodes={key: node.clone() for key, node in self.nodes.items()},
            incoming_edges={
                dst: [edge.clone() for edge in edges]
                for dst, edges in self.incoming_edges.items()
            },
        )

    def node_keys(self) -> list[node_key]:
        return list(self.nodes.keys())

    def edge_keys(self) -> list[edge_key]:
        return [edge.key for edge in self.all_edges()]

    def get_node(self, key: node_key) -> Node:
        return self.nodes[key]

    def get_edge(self, key: edge_key) -> Edge:
        dst, src = key

        for edge in self.incoming_edges.get(dst, []):
            if edge.src == src:
                return edge

        raise KeyError(key)

    def src_nodes(self) -> list[Node]:
        return [node for node in self.nodes.values() if node.is_src()]

    def dst_nodes(self) -> list[Node]:
        return [node for node in self.nodes.values() if node.is_dst()]

    def kept_nodes(self) -> list[Node]:
        return [node for node in self.nodes.values() if node.is_kept()]

    def kept_edges(self) -> list[Edge]:
        return [edge for edge in self.all_edges() if edge.is_kept()]

    def pruned_nodes(self) -> list[Node]:
        return [node for node in self.nodes.values() if not node.is_kept()]

    def pruned_edges(self) -> list[Edge]:
        return [edge for edge in self.all_edges() if not edge.is_kept()]

    def incoming_edges_of(
        self,
        dst: node_key,
        kept_only: bool = False,
    ) -> list[Edge]:
        edges = list(self.incoming_edges.get(dst, []))

        if kept_only:
            edges = [edge for edge in edges if edge.is_kept()]

        return edges

    def outgoing_edges_of(
        self,
        src: node_key,
        kept_only: bool = False,
    ) -> list[Edge]:
        edges = [edge for edge in self.all_edges() if edge.src == src]

        if kept_only:
            edges = [edge for edge in edges if edge.is_kept()]

        return edges

    def incoming_srcs(
        self,
        dst: node_key,
        kept_only: bool = False,
    ) -> list[node_key]:
        return [
            edge.src
            for edge in self.incoming_edges_of(dst, kept_only=kept_only)
        ]

    def outgoing_dsts(
        self,
        src: node_key,
        kept_only: bool = False,
    ) -> list[node_key]:
        return [
            edge.dst
            for edge in self.outgoing_edges_of(src, kept_only=kept_only)
        ]

    def has_edge(self, dst: node_key, src: node_key) -> bool:
        return any(edge.src == src for edge in self.incoming_edges.get(dst, []))

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return sum(len(edges) for edges in self.incoming_edges.values())

    def num_kept_nodes(self) -> int:
        return sum(node.is_kept() for node in self.nodes.values())

    def num_kept_edges(self) -> int:
        return sum(edge.is_kept() for edge in self.all_edges())

    def node_density(self) -> float:
        if self.num_nodes() == 0:
            return 0.0
        return self.num_kept_nodes() / self.num_nodes()

    def edge_density(self) -> float:
        if self.num_edges() == 0:
            return 0.0
        return self.num_kept_edges() / self.num_edges()

    def has_weight_masks(self) -> bool:
        return any(node.has_weight_masks() for node in self.nodes.values())

    def has_explicit_weight_masks(self) -> bool:
        return any(node.has_explicit_weight_masks() for node in self.nodes.values())

    def num_weight_params(self) -> int:
        if not self.has_explicit_weight_masks():
            raise ValueError("no explicit weight masks found.")

        return sum(
            mask.numel()
            for node in self.nodes.values()
            for mask in node.weight_masks.values()
            if mask is not None
        )

    def num_kept_weight_params(self) -> int:
        if not self.has_explicit_weight_masks():
            raise ValueError("no explicit weight masks found.")

        return sum(
            int(mask.bool().sum().item())
            for node in self.nodes.values()
            for mask in node.weight_masks.values()
            if mask is not None
        )

    def weight_density(self) -> float:
        total = self.num_weight_params()

        if total == 0:
            raise ValueError("no weight parameters found.")

        return self.num_kept_weight_params() / total

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "num_nodes": self.num_nodes(),
            "num_edges": self.num_edges(),
            "num_kept_nodes": self.num_kept_nodes(),
            "num_kept_edges": self.num_kept_edges(),
            "node_density": self.node_density(),
            "edge_density": self.edge_density(),
        }

        if self.has_explicit_weight_masks():
            out.update(
                {
                    "num_weight_params": self.num_weight_params(),
                    "num_kept_weight_params": self.num_kept_weight_params(),
                    "weight_density": self.weight_density(),
                }
            )

        return out

    def full_like(self) -> Circuit:
        """
        same graph, all masks implicit on.
        """
        return Circuit(
            nodes={
                key: Node(
                    layer=node.layer,
                    index=node.index,
                    kind=node.kind,
                    node_mask=None,
                    weight_masks={w_key: None for w_key in node.weight_masks.keys()},
                )
                for key, node in self.nodes.items()
            },
            incoming_edges={
                dst: [
                    Edge(dst=edge.dst, src=edge.src, edge_mask=None)
                    for edge in edges
                ]
                for dst, edges in self.incoming_edges.items()
            },
        )

    def empty_like(self) -> Circuit:
        """
        same graph, node, edge, and known weight masks explicitly off.
        """
        return Circuit(
            nodes={
                key: Node(
                    layer=node.layer,
                    index=node.index,
                    kind=node.kind,
                    node_mask=scalar_false(),
                    weight_masks={
                        w_key: scalar_false()
                        for w_key in node.weight_masks.keys()
                    },
                )
                for key, node in self.nodes.items()
            },
            incoming_edges={
                dst: [
                    Edge(dst=edge.dst, src=edge.src, edge_mask=scalar_false())
                    for edge in edges
                ]
                for dst, edges in self.incoming_edges.items()
            },
        )
    

def create_circuit_from_nodes_and_edges(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
) -> Circuit:
    node_dict = {node.key: node for node in nodes}
    incoming_edges: dict[node_key, list[Edge]] = {}

    for edge in edges:
        if edge.dst not in incoming_edges:
            incoming_edges[edge.dst] = []
        incoming_edges[edge.dst].append(edge)

    return Circuit(nodes=node_dict, incoming_edges=incoming_edges)

# --------------------------------------------------------------------------------------
# circuit algebra
# --------------------------------------------------------------------------------------

def complement(circuit: Circuit) -> Circuit:
    """
    circuit complement.
    """
    return Circuit(
        nodes={
            key: Node(
                layer=node.layer,
                index=node.index,
                kind=node.kind,
                node_mask=scalar_mask_not(node.node_mask),
                weight_masks=weight_masks_not(node.weight_masks),
            )
            for key, node in circuit.nodes.items()
        },
        incoming_edges={
            dst: [
                Edge(
                    dst=edge.dst,
                    src=edge.src,
                    edge_mask=scalar_mask_not(edge.edge_mask),
                )
                for edge in edges
            ]
            for dst, edges in circuit.incoming_edges.items()
        },
    )


def union(a: Circuit, b: Circuit) -> Circuit:
    """
    circuit union.
    """
    a.assert_same_structure(b)
    b_edges = b.edge_dict()

    return Circuit(
        nodes={
            key: Node(
                layer=node_a.layer,
                index=node_a.index,
                kind=node_a.kind,
                node_mask=scalar_mask_or(
                    node_a.node_mask,
                    b.nodes[key].node_mask,
                ),
                weight_masks=weight_masks_or(
                    node_a.weight_masks,
                    b.nodes[key].weight_masks,
                ),
            )
            for key, node_a in a.nodes.items()
        },
        incoming_edges={
            dst: [
                Edge(
                    dst=edge_a.dst,
                    src=edge_a.src,
                    edge_mask=scalar_mask_or(
                        edge_a.edge_mask,
                        b_edges[edge_a.key].edge_mask,
                    ),
                )
                for edge_a in edges
            ]
            for dst, edges in a.incoming_edges.items()
        },
    )


def intersection(a: Circuit, b: Circuit) -> Circuit:
    """
    circuit intersection.
    """
    a.assert_same_structure(b)
    b_edges = b.edge_dict()

    return Circuit(
        nodes={
            key: Node(
                layer=node_a.layer,
                index=node_a.index,
                kind=node_a.kind,
                node_mask=scalar_mask_and(
                    node_a.node_mask,
                    b.nodes[key].node_mask,
                ),
                weight_masks=weight_masks_and(
                    node_a.weight_masks,
                    b.nodes[key].weight_masks,
                ),
            )
            for key, node_a in a.nodes.items()
        },
        incoming_edges={
            dst: [
                Edge(
                    dst=edge_a.dst,
                    src=edge_a.src,
                    edge_mask=scalar_mask_and(
                        edge_a.edge_mask,
                        b_edges[edge_a.key].edge_mask,
                    ),
                )
                for edge_a in edges
            ]
            for dst, edges in a.incoming_edges.items()
        },
    )


def difference(a: Circuit, b: Circuit) -> Circuit:
    """
    circuit difference: a minus b.
    """
    a.assert_same_structure(b)
    return intersection(a, complement(b))


def symmetric_difference(a: Circuit, b: Circuit) -> Circuit:
    """
    circuit symmetric difference.
    """
    return union(difference(a, b), difference(b, a))


def overlap_stats(a: Circuit, b: Circuit) -> dict[str, Any]:
    """
    overlap statistics for node, edge, and explicit weight masks.
    """
    inter = intersection(a, b)

    a_nodes = a.num_kept_nodes()
    b_nodes = b.num_kept_nodes()
    i_nodes = inter.num_kept_nodes()

    a_edges = a.num_kept_edges()
    b_edges = b.num_kept_edges()
    i_edges = inter.num_kept_edges()

    node_union = a_nodes + b_nodes - i_nodes
    edge_union = a_edges + b_edges - i_edges

    out: dict[str, Any] = {
        "node_intersection": i_nodes,
        "edge_intersection": i_edges,
        "node_overlap_over_a": i_nodes / a_nodes if a_nodes > 0 else None,
        "node_overlap_over_b": i_nodes / b_nodes if b_nodes > 0 else None,
        "edge_overlap_over_a": i_edges / a_edges if a_edges > 0 else None,
        "edge_overlap_over_b": i_edges / b_edges if b_edges > 0 else None,
        "node_jaccard": i_nodes / node_union if node_union > 0 else None,
        "edge_jaccard": i_edges / edge_union if edge_union > 0 else None,
    }

    if a.has_explicit_weight_masks() and b.has_explicit_weight_masks():
        a_weights = a.num_kept_weight_params()
        b_weights = b.num_kept_weight_params()
        i_weights = inter.num_kept_weight_params()

        weight_union = a_weights + b_weights - i_weights

        out.update(
            {
                "weight_intersection": i_weights,
                "weight_overlap_over_a": (
                    i_weights / a_weights if a_weights > 0 else None
                ),
                "weight_overlap_over_b": (
                    i_weights / b_weights if b_weights > 0 else None
                ),
                "weight_jaccard": (
                    i_weights / weight_union if weight_union > 0 else None
                ),
            }
        )

    return out

# --------------------------------------------------------------------------------------
# key formatting and parsing
# --------------------------------------------------------------------------------------

def node_key_to_str(key: node_key) -> str:
    layer, index, kind = key
    return f"l{layer}.i{index}.{kind}"


def edge_key_to_str(key: edge_key) -> str:
    dst, src = key
    return f"{node_key_to_str(dst)}<-{node_key_to_str(src)}"


def parse_node_key(s: str) -> node_key:
    try:
        layer_part, index_part, kind = s.split(".", maxsplit=2)
        layer = int(layer_part.removeprefix("l"))
        index = int(index_part.removeprefix("i"))
    except Exception as exc:
        raise ValueError(f"invalid node key string: {s}") from exc

    return (layer, index, kind)


def parse_edge_key(s: str) -> edge_key:
    try:
        dst_s, src_s = s.split("<-", maxsplit=1)
    except Exception as exc:
        raise ValueError(f"invalid edge key string: {s}") from exc

    return (parse_node_key(dst_s), parse_node_key(src_s))