"""Plotly visualization helpers for circuit node and edge masks.

The functions in this module work directly with the repo's ``Circuit`` object.
They intentionally ignore weight masks and algorithm-specific state: the visual
surface is just the node mask and edge mask structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .circuit import Circuit, edge_key, node_key

try:  # Plotly is optional for non-notebook use of the package.
    import plotly.graph_objects as go
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by users without plotly.
    go = None
    _PLOTLY_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _PLOTLY_IMPORT_ERROR = None


# Public defaults. Callers can override these module variables globally.
DEFAULT_WIDTH = 1500
DEFAULT_HEIGHT = 780


# Palette follows the OASR visualization style: blue attention heads, orange
# MLPs, green embedding, and red output.
_C_EMB = "#68b86b"
_C_OUTPUT = "#e56b6f"
_C_MLP = "#f39a5b"
_C_Q = "#5c8fbd"
_C_K = "#5c8fbd"
_C_V = "#5c8fbd"
_C_O = "#5c8fbd"
_C_ATTN = "#5c8fbd"
_C_INACTIVE = "rgba(185,185,185,0.30)"

_T_FROM_EMB = "rgba(104,184,107,{a})"
_T_FROM_ATTN = "rgba(92,143,189,{a})"
_T_FROM_MLP = "rgba(243,154,91,{a})"
_T_INACTIVE_EDGE = "rgba(170,170,170,{a})"
_T_OVERLAP = "rgba(0,168,107,{a})"
_T_ONLY_A = "rgba(92,143,189,{a})"
_T_ONLY_B = "rgba(229,107,111,{a})"

_HEAD_STEP = 1.12
_MLP_GAP = 1.25
_CURVE_PTS = 12
_NODE_BASE = 13.0
_NODE_SCALE = 4.0
_COLUMN_ALPHA = 0.045

_KIND_X_OFFSET = {
    "attn_q": -0.20,
    "attn_k": -0.08,
    "attn_v": 0.04,
    "attn_o": 0.22,
    "mlp": 0.0,
}
_KIND_Y_OFFSET = {
    "attn_q": -0.22,
    "attn_k": 0.00,
    "attn_v": 0.22,
    "attn_o": 0.44,
}
_KIND_LABEL = {
    "emb": "Emb",
    "attn": "H",
    "attn_q": "Q",
    "attn_k": "K",
    "attn_v": "V",
    "attn_o": "O",
    "mlp": "MLP",
    "output": "Out",
}
_KIND_COLOR = {
    "emb": _C_EMB,
    "attn": _C_ATTN,
    "attn_q": _C_Q,
    "attn_k": _C_K,
    "attn_v": _C_V,
    "attn_o": _C_O,
    "mlp": _C_MLP,
    "output": _C_OUTPUT,
}

_ATTN_COMPONENT_KINDS = {"attn_q", "attn_k", "attn_v", "attn_o"}


@dataclass(frozen=True)
class _NodeItem:
    key: node_key
    score: float
    active: bool
    components: tuple[tuple[node_key, float, bool], ...] = ()


@dataclass(frozen=True)
class _EdgeItem:
    key: edge_key
    score: float
    active: bool
    components: tuple[tuple[edge_key, float, bool], ...] = ()

    @property
    def dst(self) -> node_key:
        return self.key[0]

    @property
    def src(self) -> node_key:
        return self.key[1]


def _require_plotly() -> None:
    if _PLOTLY_IMPORT_ERROR is not None:
        raise ImportError(
            "visualization.py requires plotly. Install the notebook extras with "
            "`pip install -e .[notebook]` or install `plotly` directly."
        ) from _PLOTLY_IMPORT_ERROR


def _torch_load(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_circuit(
    payload: Circuit | Mapping[str, Any],
    *,
    circuit_key: str | None = None,
) -> Circuit:
    """Extract a ``Circuit`` from a saved experiment payload or return it directly."""
    if isinstance(payload, Circuit):
        return payload

    if circuit_key is not None:
        value = payload[circuit_key]
        if not isinstance(value, Circuit):
            raise TypeError(f"payload[{circuit_key!r}] is not a Circuit.")
        return value

    for key in ("circuit", "finalized_circuit", "raw_circuit"):
        value = payload.get(key)
        if isinstance(value, Circuit):
            return value

    raise TypeError(
        "expected a Circuit or a mapping containing one under 'circuit', "
        "'finalized_circuit', or 'raw_circuit'."
    )


def load_circuit(
    path: str | Path,
    *,
    circuit_key: str | None = None,
    map_location: str | torch.device = "cpu",
) -> Circuit:
    """Load a saved ``Circuit`` or experiment payload from a ``.pt`` file."""
    payload = _torch_load(path, map_location=map_location)
    return extract_circuit(payload, circuit_key=circuit_key)


def _mask_score(mask: torch.Tensor | None) -> float:
    if mask is None:
        return 1.0

    value = mask.detach()
    if value.numel() != 1:
        value = value.float().mean()
    else:
        value = value.reshape(())

    if value.dtype == torch.bool:
        return 1.0 if bool(value.item()) else 0.0

    return float(value.float().cpu().item())


def _mask_active(mask: torch.Tensor | None, threshold: float) -> bool:
    if mask is None:
        return True
    return abs(_mask_score(mask)) > threshold


def _node_items(circuit: Circuit, threshold: float) -> dict[node_key, _NodeItem]:
    return {
        key: _NodeItem(
            key=key,
            score=_mask_score(node.node_mask),
            active=_mask_active(node.node_mask, threshold),
        )
        for key, node in circuit.nodes.items()
    }


def _edge_items(
    circuit: Circuit,
    threshold: float,
    *,
    top_k_edges: int | None,
) -> dict[edge_key, _EdgeItem]:
    items = {
        edge.key: _EdgeItem(
            key=edge.key,
            score=_mask_score(edge.edge_mask),
            active=_mask_active(edge.edge_mask, threshold),
        )
        for edge in circuit.all_edges()
    }

    if top_k_edges is None:
        return items

    if top_k_edges < 0:
        raise ValueError("top_k_edges must be non-negative.")

    candidates = [
        item
        for item in items.values()
        if item.active
    ]
    candidates.sort(
        key=lambda item: (-abs(item.score), _format_edge_key(item.key)),
    )
    kept = {item.key for item in candidates[:top_k_edges]}
    return {
        key: _EdgeItem(key=item.key, score=item.score, active=key in kept)
        for key, item in items.items()
    }


def _visual_node_key(
    key: node_key,
    *,
    bind_attention_nodes: bool,
) -> node_key:
    layer, index, kind = key
    if bind_attention_nodes and kind in _ATTN_COMPONENT_KINDS:
        return (layer, index, "attn")
    return key


def _visual_edge_key(
    key: edge_key,
    *,
    bind_attention_nodes: bool,
) -> edge_key:
    dst, src = key
    return (
        _visual_node_key(dst, bind_attention_nodes=bind_attention_nodes),
        _visual_node_key(src, bind_attention_nodes=bind_attention_nodes),
    )


def _score_with_largest_abs(items: Sequence[_NodeItem | _EdgeItem]) -> float:
    if not items:
        return 0.0
    return max(items, key=lambda item: abs(item.score)).score


def _visual_node_items(
    items: Mapping[node_key, _NodeItem],
    *,
    bind_attention_nodes: bool,
) -> dict[node_key, _NodeItem]:
    if not bind_attention_nodes:
        return dict(items)

    grouped: dict[node_key, list[_NodeItem]] = {}
    for item in items.values():
        key = _visual_node_key(item.key, bind_attention_nodes=True)
        grouped.setdefault(key, []).append(item)

    visual_items: dict[node_key, _NodeItem] = {}
    for key, group in grouped.items():
        group = sorted(group, key=lambda item: _node_sort_key(item.key))
        active_group = [item for item in group if item.active]
        components = tuple((item.key, item.score, item.active) for item in group)
        if len(components) == 1 and components[0][0] == key:
            components = ()
        visual_items[key] = _NodeItem(
            key=key,
            score=_score_with_largest_abs(active_group or group),
            active=bool(active_group),
            components=components,
        )
    return visual_items


def _visual_edge_items(
    items: Mapping[edge_key, _EdgeItem],
    *,
    bind_attention_nodes: bool,
) -> dict[edge_key, _EdgeItem]:
    if not bind_attention_nodes:
        return dict(items)

    grouped: dict[edge_key, list[_EdgeItem]] = {}
    for item in items.values():
        key = _visual_edge_key(item.key, bind_attention_nodes=True)
        grouped.setdefault(key, []).append(item)

    visual_items: dict[edge_key, _EdgeItem] = {}
    for key, group in grouped.items():
        group = sorted(group, key=lambda item: _format_edge_key(item.key))
        active_group = [item for item in group if item.active]
        components = tuple((item.key, item.score, item.active) for item in group)
        if len(components) == 1 and components[0][0] == key:
            components = ()
        visual_items[key] = _EdgeItem(
            key=key,
            score=_score_with_largest_abs(active_group or group),
            active=bool(active_group),
            components=components,
        )
    return visual_items


def _active_node_keys(items: Mapping[node_key, _NodeItem]) -> set[node_key]:
    return {key for key, item in items.items() if item.active}


def _active_edge_keys(items: Mapping[edge_key, _EdgeItem]) -> set[edge_key]:
    return {key for key, item in items.items() if item.active}


def _active_counts(
    edge_items: Mapping[edge_key, _EdgeItem],
) -> tuple[dict[node_key, int], dict[node_key, int]]:
    in_deg: dict[node_key, int] = {}
    out_deg: dict[node_key, int] = {}
    for item in edge_items.values():
        if not item.active:
            continue
        out_deg[item.src] = out_deg.get(item.src, 0) + 1
        in_deg[item.dst] = in_deg.get(item.dst, 0) + 1
    return in_deg, out_deg


def _infer_dims(node_keys: Sequence[node_key]) -> tuple[int, int]:
    output_layers = [layer for layer, _, kind in node_keys if kind == "output"]
    if output_layers:
        n_layers = max(output_layers)
    else:
        transformer_layers = [layer for layer, _, kind in node_keys if layer >= 0 and kind != "output"]
        n_layers = max(transformer_layers, default=-1) + 1

    head_indices = [
        index
        for _, index, kind in node_keys
        if kind == "attn" or kind.startswith("attn_")
    ]
    n_heads = max(head_indices, default=0) + 1
    return max(n_layers, 0), max(n_heads, 1)


def _head_y(head: int, n_heads: int) -> float:
    return (n_heads - 1 - head) * _HEAD_STEP


def _center_y(n_heads: int) -> float:
    return (_head_y(n_heads - 1, n_heads) + _mlp_y(n_heads)) / 2.0


def _mlp_y(n_heads: int) -> float:
    return n_heads * _HEAD_STEP + _MLP_GAP


def _node_positions(node_keys: Sequence[node_key]) -> dict[node_key, tuple[float, float]]:
    n_layers, n_heads = _infer_dims(node_keys)
    output_x = n_layers + 1.58
    center_y = _center_y(n_heads)

    positions: dict[node_key, tuple[float, float]] = {}
    for key in sorted(node_keys, key=_node_sort_key):
        layer, index, kind = key
        if kind == "emb":
            positions[key] = (0.0, center_y)
        elif kind == "output":
            positions[key] = (output_x, center_y)
        elif kind == "mlp":
            positions[key] = (layer + 1.0 + _KIND_X_OFFSET["mlp"], _mlp_y(n_heads))
        elif kind == "attn":
            positions[key] = (layer + 1.0, _head_y(index, n_heads))
        elif kind.startswith("attn_"):
            y = _head_y(index, n_heads) + _KIND_Y_OFFSET.get(kind, 0.0)
            x = layer + 1.0 + _KIND_X_OFFSET.get(kind, 0.0)
            positions[key] = (x, y)
        else:
            positions[key] = (layer + 1.0, center_y)

    return positions


def _node_sort_key(key: node_key) -> tuple[int, int, int, str]:
    layer, index, kind = key
    kind_rank = {
        "emb": -1,
        "output": 10,
        "mlp": 8,
        "attn": 1,
        "attn_q": 1,
        "attn_k": 2,
        "attn_v": 3,
        "attn_o": 4,
    }.get(kind, 5)
    return (layer, kind_rank, index, kind)


def _format_node_key(key: node_key) -> str:
    layer, index, kind = key
    if kind == "emb":
        return "Embedding"
    if kind == "output":
        return "Output"
    if kind == "mlp":
        return f"L{layer} MLP"
    if kind == "attn":
        return f"L{layer} H{index}"
    if kind.startswith("attn_"):
        return f"L{layer} H{index} {_KIND_LABEL.get(kind, kind)}"
    return f"L{layer} {kind}[{index}]"


def _short_node_label(key: node_key) -> str:
    _, index, kind = key
    if kind == "emb":
        return "E"
    if kind == "output":
        return "O"
    if kind == "mlp":
        return "M"
    if kind == "attn":
        return f"H{index}"
    if kind.startswith("attn_"):
        return f"{_KIND_LABEL.get(kind, kind)}{index}"
    return kind[:2]


def _format_edge_key(key: edge_key) -> str:
    dst, src = key
    return f"{_format_node_key(src)} -> {_format_node_key(dst)}"


def _node_component_hover(item: _NodeItem) -> str:
    if not item.components:
        return ""

    lines = ["components:"]
    for key, score, active in item.components:
        label = _KIND_LABEL.get(key[2], key[2])
        state = "on" if active else "off"
        lines.append(f"{label}: {state} ({score:.6g})")
    return "<br>" + "<br>".join(lines)


def _edge_component_hover(item: _EdgeItem) -> str:
    if not item.components:
        return ""

    active = sum(component_active for _, _, component_active in item.components)
    total = len(item.components)
    lines = [f"raw component edges: {active}/{total} kept"]
    for key, score, component_active in item.components[:12]:
        state = "on" if component_active else "off"
        lines.append(f"{_format_edge_key(key)}: {state} ({score:.6g})")
    if total > 12:
        lines.append(f"... {total - 12} more")
    return "<br>" + "<br>".join(lines)


def _source_template(edge: _EdgeItem) -> tuple[str, str]:
    src_kind = edge.src[2]
    if src_kind == "emb":
        return _T_FROM_EMB, "From Emb"
    if src_kind == "mlp":
        return _T_FROM_MLP, "From MLP"
    if src_kind == "attn" or src_kind == "attn_o":
        return _T_FROM_ATTN, "From Attn"
    return _T_INACTIVE_EDGE, "From Other"


def _edge_opacity(count: int) -> float:
    return float(np.clip(80.0 / max(count, 1), 0.18, 0.9))


def _bezier_segment(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[list[float | None], list[float | None]]:
    dx = abs(x1 - x0) * 0.38
    ts = np.linspace(0.0, 1.0, _CURVE_PTS)
    cx0, cy0, cx1, cy1 = x0 + dx, y0, x1 - dx, y1
    xs = (
        (1 - ts) ** 3 * x0
        + 3 * (1 - ts) ** 2 * ts * cx0
        + 3 * (1 - ts) * ts**2 * cx1
        + ts**3 * x1
    )
    ys = (
        (1 - ts) ** 3 * y0
        + 3 * (1 - ts) ** 2 * ts * cy0
        + 3 * (1 - ts) * ts**2 * cy1
        + ts**3 * y1
    )
    return list(xs) + [None], list(ys) + [None]


def _edge_trace(
    edges: Sequence[_EdgeItem],
    positions: Mapping[node_key, tuple[float, float]],
    *,
    color_template: str,
    opacity: float,
    name: str,
    line_width: float = 1.35,
    showlegend: bool = True,
) -> Any:
    xs: list[float | None] = []
    ys: list[float | None] = []
    for edge in sorted(edges, key=lambda item: _format_edge_key(item.key)):
        if edge.src not in positions or edge.dst not in positions:
            continue
        seg_x, seg_y = _bezier_segment(*positions[edge.src], *positions[edge.dst])
        xs.extend(seg_x)
        ys.extend(seg_y)

    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color=color_template.format(a=opacity), width=line_width),
        name=name,
        showlegend=showlegend,
        hoverinfo="skip",
    )


def _edge_hover_trace(
    edges: Sequence[_EdgeItem],
    positions: Mapping[node_key, tuple[float, float]],
    *,
    name: str = "Edges",
) -> Any:
    xs: list[float] = []
    ys: list[float] = []
    hover: list[str] = []
    for edge in sorted(edges, key=lambda item: _format_edge_key(item.key)):
        if edge.src not in positions or edge.dst not in positions:
            continue
        x0, y0 = positions[edge.src]
        x1, y1 = positions[edge.dst]
        xs.append((x0 + x1) / 2.0)
        ys.append((y0 + y1) / 2.0)
        hover.append(
            f"{_format_edge_key(edge.key)}<br>"
            f"mask={edge.score:.6g}"
            f"{_edge_component_hover(edge)}"
        )
    return go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(size=5, color="rgba(0,0,0,0)"),
        hovertext=hover,
        hoverinfo="text",
        name=name,
        showlegend=False,
    )


def _node_traces(
    node_items: Mapping[node_key, _NodeItem],
    positions: Mapping[node_key, tuple[float, float]],
    in_deg: Mapping[node_key, int],
    out_deg: Mapping[node_key, int],
    *,
    show_inactive_nodes: bool,
) -> list[Any]:
    active_x: list[float] = []
    active_y: list[float] = []
    active_text: list[str] = []
    active_color: list[str] = []
    active_size: list[float] = []
    active_hover: list[str] = []
    inactive_x: list[float] = []
    inactive_y: list[float] = []
    inactive_hover: list[str] = []

    for key in sorted(node_items.keys(), key=_node_sort_key):
        item = node_items[key]
        if key not in positions:
            continue
        x, y = positions[key]
        if item.active:
            degree = in_deg.get(key, 0) + out_deg.get(key, 0)
            size = _NODE_BASE + _NODE_SCALE * min(float(np.sqrt(max(degree, 0))), 5.0)
            active_x.append(x)
            active_y.append(y)
            active_text.append(_short_node_label(key))
            active_color.append(_KIND_COLOR.get(key[2], "#666666"))
            active_size.append(size)
            active_hover.append(
                f"{_format_node_key(key)}<br>"
                f"kind={key[2]}<br>"
                f"mask={item.score:.6g}<br>"
                f"in={in_deg.get(key, 0)} out={out_deg.get(key, 0)}"
                f"{_node_component_hover(item)}"
            )
        elif show_inactive_nodes:
            inactive_x.append(x)
            inactive_y.append(y)
            inactive_hover.append(
                f"{_format_node_key(key)}<br>"
                f"mask={item.score:.6g}"
                f"{_node_component_hover(item)}"
            )

    traces: list[Any] = []
    if show_inactive_nodes:
        traces.append(
            go.Scatter(
                x=inactive_x,
                y=inactive_y,
                mode="markers",
                marker=dict(size=5, color=_C_INACTIVE, line_width=0),
                hovertext=inactive_hover,
                hoverinfo="text",
                name="Pruned nodes",
                showlegend=False,
            )
        )

    traces.append(
        go.Scatter(
            x=active_x,
            y=active_y,
            mode="markers+text",
            text=active_text,
            textposition="middle center",
            marker=dict(
                size=active_size,
                color=active_color,
                line=dict(width=1.3, color="white"),
            ),
            textfont=dict(size=8, color="white"),
            hovertext=active_hover,
            hoverinfo="text",
            name="Kept nodes",
            showlegend=False,
        )
    )
    return traces


def _compare_node_traces(
    all_node_keys: set[node_key],
    active_a: set[node_key],
    active_b: set[node_key],
    positions: Mapping[node_key, tuple[float, float]],
    in_deg: Mapping[node_key, int],
    out_deg: Mapping[node_key, int],
    *,
    label_a: str,
    label_b: str,
    show_inactive_nodes: bool,
) -> list[Any]:
    traces: list[Any] = []
    categories = {
        "overlap": active_a & active_b,
        "only_a": active_a - active_b,
        "only_b": active_b - active_a,
    }
    names = {
        "overlap": "shared node",
        "only_a": f"{label_a} only",
        "only_b": f"{label_b} only",
    }

    for category, keys in categories.items():
        xs: list[float] = []
        ys: list[float] = []
        labels: list[str] = []
        colors: list[str] = []
        sizes: list[float] = []
        hover: list[str] = []
        for key in sorted(keys, key=_node_sort_key):
            if key not in positions:
                continue
            x, y = positions[key]
            degree = in_deg.get(key, 0) + out_deg.get(key, 0)
            xs.append(x)
            ys.append(y)
            labels.append(_short_node_label(key))
            colors.append(_KIND_COLOR.get(key[2], "#666666"))
            sizes.append(_NODE_BASE + _NODE_SCALE * min(float(np.sqrt(max(degree, 0))), 5.0))
            hover.append(
                f"{_format_node_key(key)}<br>"
                f"{names[category]}<br>"
                f"in={in_deg.get(key, 0)} out={out_deg.get(key, 0)}"
            )

        if xs:
            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    text=labels,
                    textposition="middle center",
                    marker=dict(
                        size=sizes,
                        color=colors,
                        line=dict(width=1.3, color="white"),
                    ),
                    textfont=dict(size=8, color="white"),
                    hovertext=hover,
                    hoverinfo="text",
                    name=names[category],
                    showlegend=False,
                )
            )

    if show_inactive_nodes:
        inactive = all_node_keys - active_a - active_b
        xs = []
        ys = []
        hover = []
        for key in sorted(inactive, key=_node_sort_key):
            if key not in positions:
                continue
            x, y = positions[key]
            xs.append(x)
            ys.append(y)
            hover.append(_format_node_key(key))
        traces.insert(
            0,
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=5, color=_C_INACTIVE, line_width=0),
                hovertext=hover,
                hoverinfo="text",
                name="Inactive nodes",
                showlegend=False,
            ),
        )

    return traces


def _column_shapes(n_layers: int, n_heads: int) -> list[dict[str, Any]]:
    y0 = _head_y(n_heads - 1, n_heads) - 0.65
    y1 = _mlp_y(n_heads) + 0.65
    shapes: list[dict[str, Any]] = []
    for layer in range(n_layers):
        if layer % 2 != 0:
            continue
        x = layer + 1.0
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                layer="below",
                x0=x - 0.52,
                x1=x + 0.60,
                y0=y0,
                y1=y1,
                fillcolor=f"rgba(180,180,180,{_COLUMN_ALPHA})",
                line_width=0,
            )
        )
    return shapes


def _base_layout(
    node_keys: Sequence[node_key],
    *,
    title: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    n_layers, n_heads = _infer_dims(node_keys)
    output_x = n_layers + 1.58
    tickvals = [0.0] + [layer + 1.0 for layer in range(n_layers)] + [output_x]
    ticktext = ["Emb"] + [f"L{layer}" for layer in range(n_layers)] + ["Out"]

    return dict(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=dict(
            tickvals=tickvals,
            ticktext=ticktext,
            tickfont=dict(size=10),
            showgrid=False,
            zeroline=False,
            range=[-0.65, output_x + 0.65],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[_head_y(n_heads - 1, n_heads) - 0.85, _mlp_y(n_heads) + 0.85],
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        width=width,
        margin=dict(l=30, r=220, t=80, b=30),
        legend=dict(
            x=1.01,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
        ),
        shapes=_column_shapes(n_layers, n_heads),
    )


def _default_title(circuit: Circuit, *, prefix: str = "Circuit") -> str:
    stats = circuit.stats()
    return (
        f"{prefix}: {stats['num_kept_edges']}/{stats['num_edges']} edges "
        f"({stats['edge_density']:.2%}), "
        f"{stats['num_kept_nodes']}/{stats['num_nodes']} nodes "
        f"({stats['node_density']:.2%})"
    )


def _default_visual_title(
    circuit: Circuit,
    node_items: Mapping[node_key, _NodeItem],
    edge_items: Mapping[edge_key, _EdgeItem],
    *,
    prefix: str = "Circuit",
) -> str:
    stats = circuit.stats()
    n_nodes = len(node_items)
    n_edges = len(edge_items)
    kept_nodes = sum(item.active for item in node_items.values())
    kept_edges = sum(item.active for item in edge_items.values())
    if n_nodes == stats["num_nodes"] and n_edges == stats["num_edges"]:
        return _default_title(circuit, prefix=prefix)
    return (
        f"{prefix}: {kept_edges}/{n_edges} edges "
        f"({kept_edges / n_edges if n_edges else 0.0:.2%}), "
        f"{kept_nodes}/{n_nodes} nodes "
        f"({kept_nodes / n_nodes if n_nodes else 0.0:.2%})"
    )


def _selection_summary(
    node_items: Mapping[node_key, _NodeItem],
    edge_items: Mapping[edge_key, _EdgeItem],
) -> str:
    n_nodes = len(node_items)
    n_edges = len(edge_items)
    kept_nodes = sum(item.active for item in node_items.values())
    kept_edges = sum(item.active for item in edge_items.values())
    node_density = kept_nodes / n_nodes if n_nodes else 0.0
    edge_density = kept_edges / n_edges if n_edges else 0.0
    return (
        "<b>Mask summary</b><br>"
        f"Kept edges: {kept_edges}/{n_edges} ({edge_density:.2%})<br>"
        f"Kept nodes: {kept_nodes}/{n_nodes} ({node_density:.2%})"
    )


def _add_summary_annotation(fig: Any, text: str, *, y: float = 0.52) -> None:
    fig.update_layout(
        annotations=list(fig.layout.annotations or []) + [
            dict(
                text=text,
                xref="paper",
                yref="paper",
                x=1.01,
                y=y,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                align="left",
                font=dict(size=13, color="black"),
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="rgba(0,0,0,0.08)",
                borderwidth=1,
                borderpad=8,
            )
        ]
    )


def _prepare_circuit(
    circuit_or_payload: Circuit | Mapping[str, Any],
    *,
    circuit_key: str | None,
    mask_threshold: float,
    top_k_edges: int | None,
    bind_attention_nodes: bool,
) -> tuple[Circuit, dict[node_key, _NodeItem], dict[edge_key, _EdgeItem]]:
    circuit = extract_circuit(circuit_or_payload, circuit_key=circuit_key)
    raw_node_items = _node_items(circuit, mask_threshold)
    raw_edge_items = _edge_items(
        circuit,
        mask_threshold,
        top_k_edges=top_k_edges,
    )
    node_items = _visual_node_items(
        raw_node_items,
        bind_attention_nodes=bind_attention_nodes,
    )
    edge_items = _visual_edge_items(
        raw_edge_items,
        bind_attention_nodes=bind_attention_nodes,
    )
    return circuit, node_items, edge_items


def visualize_circuit(
    circuit_or_payload: Circuit | Mapping[str, Any],
    *,
    circuit_key: str | None = None,
    title: str | None = None,
    mask_threshold: float = 0.0,
    top_k_edges: int | None = None,
    show_inactive_nodes: bool = True,
    show_inactive_edges: bool = False,
    edge_hover: bool = False,
    bind_attention_nodes: bool = True,
    width: int | None = None,
    height: int | None = None,
) -> Any:
    """Visualize one circuit's node and edge masks.

    ``mask_threshold`` is applied to absolute mask values. ``top_k_edges`` can
    be used for float-valued intermediate masks to display only the largest
    active edge scores. By default Q/K/V/O components for each attention head
    are bound into one visual node; pass ``bind_attention_nodes=False`` to see
    the raw component-level graph.
    """
    _require_plotly()
    circuit, node_items, edge_items = _prepare_circuit(
        circuit_or_payload,
        circuit_key=circuit_key,
        mask_threshold=mask_threshold,
        top_k_edges=top_k_edges,
        bind_attention_nodes=bind_attention_nodes,
    )

    visual_node_keys = list(node_items.keys())
    positions = _node_positions(visual_node_keys)
    in_deg, out_deg = _active_counts(edge_items)
    active_edges = [item for item in edge_items.values() if item.active]
    inactive_edges = [item for item in edge_items.values() if not item.active]
    active_count = len(active_edges)
    alpha = _edge_opacity(active_count)

    fig = go.Figure()
    if show_inactive_edges and inactive_edges:
        fig.add_trace(
            _edge_trace(
                inactive_edges,
                positions,
                color_template=_T_INACTIVE_EDGE,
                opacity=0.08,
                name="Pruned edges",
                line_width=0.75,
                showlegend=True,
            )
        )

    grouped: dict[tuple[str, str], list[_EdgeItem]] = {}
    for edge in active_edges:
        grouped.setdefault(_source_template(edge), []).append(edge)

    for (template, name), edges in sorted(grouped.items(), key=lambda kv: kv[0][1]):
        fig.add_trace(
            _edge_trace(
                edges,
                positions,
                color_template=template,
                opacity=alpha,
                name=f"{name} ({len(edges)})",
            )
        )

    if edge_hover and active_edges:
        fig.add_trace(_edge_hover_trace(active_edges, positions))

    for trace in _node_traces(
        node_items,
        positions,
        in_deg,
        out_deg,
        show_inactive_nodes=show_inactive_nodes,
    ):
        fig.add_trace(trace)

    fig.update_layout(
        **_base_layout(
            visual_node_keys,
            title=title or _default_visual_title(circuit, node_items, edge_items),
            width=width or DEFAULT_WIDTH,
            height=height or DEFAULT_HEIGHT,
        )
    )
    _add_summary_annotation(fig, _selection_summary(node_items, edge_items))
    return fig


def circuit_mask_overlap(
    circuit_a: Circuit | Mapping[str, Any],
    circuit_b: Circuit | Mapping[str, Any],
    *,
    circuit_key_a: str | None = None,
    circuit_key_b: str | None = None,
    mask_threshold: float = 0.0,
    top_k_edges_a: int | None = None,
    top_k_edges_b: int | None = None,
    bind_attention_nodes: bool = True,
) -> dict[str, float | int]:
    """Return node/edge intersection-over-union stats for two circuits.

    By default this uses the same visual binding as the plot functions, where
    Q/K/V/O components for one attention head are treated as one node.
    """
    _, nodes_a, edges_a = _prepare_circuit(
        circuit_a,
        circuit_key=circuit_key_a,
        mask_threshold=mask_threshold,
        top_k_edges=top_k_edges_a,
        bind_attention_nodes=bind_attention_nodes,
    )
    _, nodes_b, edges_b = _prepare_circuit(
        circuit_b,
        circuit_key=circuit_key_b,
        mask_threshold=mask_threshold,
        top_k_edges=top_k_edges_b,
        bind_attention_nodes=bind_attention_nodes,
    )

    active_nodes_a = _active_node_keys(nodes_a)
    active_nodes_b = _active_node_keys(nodes_b)
    active_edges_a = _active_edge_keys(edges_a)
    active_edges_b = _active_edge_keys(edges_b)

    node_intersection = active_nodes_a & active_nodes_b
    node_union = active_nodes_a | active_nodes_b
    edge_intersection = active_edges_a & active_edges_b
    edge_union = active_edges_a | active_edges_b

    return {
        "num_nodes_a": len(active_nodes_a),
        "num_nodes_b": len(active_nodes_b),
        "num_node_intersection": len(node_intersection),
        "num_node_union": len(node_union),
        "node_iou": len(node_intersection) / len(node_union) if node_union else 1.0,
        "num_edges_a": len(active_edges_a),
        "num_edges_b": len(active_edges_b),
        "num_edge_intersection": len(edge_intersection),
        "num_edge_union": len(edge_union),
        "edge_iou": len(edge_intersection) / len(edge_union) if edge_union else 1.0,
    }


def visualize_circuit_comparison(
    circuit_a: Circuit | Mapping[str, Any],
    circuit_b: Circuit | Mapping[str, Any],
    *,
    circuit_key_a: str | None = None,
    circuit_key_b: str | None = None,
    label_a: str = "A",
    label_b: str = "B",
    title: str | None = None,
    mask_threshold: float = 0.0,
    top_k_edges_a: int | None = None,
    top_k_edges_b: int | None = None,
    show_inactive_nodes: bool = True,
    edge_hover: bool = False,
    bind_attention_nodes: bool = True,
    width: int | None = None,
    height: int | None = None,
) -> Any:
    """Compare two circuits using node and edge mask overlap.

    By default Q/K/V/O components for each attention head are bound into one
    visual node before plotting and IoU computation.
    """
    _require_plotly()
    circ_a, nodes_a, edges_a = _prepare_circuit(
        circuit_a,
        circuit_key=circuit_key_a,
        mask_threshold=mask_threshold,
        top_k_edges=top_k_edges_a,
        bind_attention_nodes=bind_attention_nodes,
    )
    circ_b, nodes_b, edges_b = _prepare_circuit(
        circuit_b,
        circuit_key=circuit_key_b,
        mask_threshold=mask_threshold,
        top_k_edges=top_k_edges_b,
        bind_attention_nodes=bind_attention_nodes,
    )

    all_node_keys = set(nodes_a) | set(nodes_b)
    positions = _node_positions(list(all_node_keys))

    active_nodes_a = _active_node_keys(nodes_a)
    active_nodes_b = _active_node_keys(nodes_b)
    active_edges_a = _active_edge_keys(edges_a)
    active_edges_b = _active_edge_keys(edges_b)

    overlap_edges = active_edges_a & active_edges_b
    only_a_edges = active_edges_a - active_edges_b
    only_b_edges = active_edges_b - active_edges_a
    all_active_edges = overlap_edges | only_a_edges | only_b_edges
    all_edge_items = {
        **{key: edges_a[key] for key in only_a_edges | overlap_edges if key in edges_a},
        **{key: edges_b[key] for key in only_b_edges if key in edges_b},
    }
    in_deg, out_deg = _active_counts(all_edge_items)
    alpha = _edge_opacity(len(all_active_edges))

    fig = go.Figure()
    if overlap_edges:
        fig.add_trace(
            _edge_trace(
                [edges_a[key] for key in overlap_edges if key in edges_a],
                positions,
                color_template=_T_OVERLAP,
                opacity=min(alpha * 1.45, 0.9),
                name="shared edges",
                line_width=2.3,
            )
        )
    if only_a_edges:
        fig.add_trace(
            _edge_trace(
                [edges_a[key] for key in only_a_edges if key in edges_a],
                positions,
                color_template=_T_ONLY_A,
                opacity=alpha,
                name=f"{label_a} only",
            )
        )
    if only_b_edges:
        fig.add_trace(
            _edge_trace(
                [edges_b[key] for key in only_b_edges if key in edges_b],
                positions,
                color_template=_T_ONLY_B,
                opacity=alpha,
                name=f"{label_b} only",
            )
        )

    if edge_hover:
        hover_items = [all_edge_items[key] for key in all_active_edges if key in all_edge_items]
        if hover_items:
            fig.add_trace(_edge_hover_trace(hover_items, positions))

    for trace in _compare_node_traces(
        all_node_keys,
        active_nodes_a,
        active_nodes_b,
        positions,
        in_deg,
        out_deg,
        label_a=label_a,
        label_b=label_b,
        show_inactive_nodes=show_inactive_nodes,
    ):
        fig.add_trace(trace)

    stats = circuit_mask_overlap(
        circ_a,
        circ_b,
        mask_threshold=mask_threshold,
        top_k_edges_a=top_k_edges_a,
        top_k_edges_b=top_k_edges_b,
        bind_attention_nodes=bind_attention_nodes,
    )
    summary = (
        "<b>Overlap</b><br>"
        f"Edge IoU: {stats['edge_iou']:.2%}<br>"
        f"Node IoU: {stats['node_iou']:.2%}<br>"
        f"{label_a} edges: {stats['num_edges_a']}<br>"
        f"{label_b} edges: {stats['num_edges_b']}<br>"
        f"shared edges: {stats['num_edge_intersection']}"
    )

    fig.update_layout(
        **_base_layout(
            list(all_node_keys),
            title=title or f"Circuit comparison: {label_a} vs {label_b}",
            width=width or DEFAULT_WIDTH,
            height=height or DEFAULT_HEIGHT,
        )
    )
    _add_summary_annotation(fig, summary)
    return fig


def write_circuit_html(
    circuit_or_payload: Circuit | Mapping[str, Any],
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Write ``visualize_circuit(...)`` to an interactive HTML file."""
    fig = visualize_circuit(circuit_or_payload, **kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    return out


__all__ = [
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "circuit_mask_overlap",
    "extract_circuit",
    "load_circuit",
    "visualize_circuit",
    "visualize_circuit_comparison",
    "write_circuit_html",
]
