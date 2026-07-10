"""Edge statistics for progressive graph structure adjustment."""
from __future__ import annotations

import numpy as np


def average_rw_stats(rows):
    """Average edge-stat dictionaries collected across reweighting steps."""
    if not rows:
        return None
    keys = [
        "original_edge_count",
        "new_edge_count",
        "num_added_edges",
        "num_removed_edges",
        "net_edge_delta",
        "edge_growth_rate_pct",
        "original_degree_mean",
        "new_degree_mean",
        "num_modified_nodes",
        "num_modified_edges",
        "avg_degree_modified",
    ]
    out = {k: float(np.mean([float(r.get(k, 0)) for r in rows])) for k in keys}
    om, nm = out["original_degree_mean"], out["new_degree_mean"]
    out["degree_growth_pct"] = (nm - om) / om * 100.0 if om > 0 else None
    out["n_rw_steps_averaged"] = int(len(rows))
    return out


def compute_detailed_edge_statistics(graph, original_adj, new_adj):
    """Compare original and adjusted adjacency; return edge sets and summary stats."""
    del graph  # kept for API compatibility

    original_rows, original_cols = original_adj.nonzero()
    new_rows, new_cols = new_adj.nonzero()
    original_edges = set(zip(original_rows.tolist(), original_cols.tolist()))
    new_edges = set(zip(new_rows.tolist(), new_cols.tolist()))

    added_edges = new_edges - original_edges
    removed_edges = original_edges - new_edges
    original_edge_count = len(original_edges)
    new_edge_count = len(new_edges)
    edge_growth_rate = (
        (new_edge_count - original_edge_count) / original_edge_count * 100
        if original_edge_count > 0
        else 0.0
    )

    original_degrees = np.array(original_adj.sum(axis=1)).flatten()
    new_degrees = np.array(new_adj.sum(axis=1)).flatten()
    orig_deg_mean = float(original_degrees.mean())
    new_deg_mean = float(new_degrees.mean())
    degree_growth_pct = (
        (new_deg_mean - orig_deg_mean) / orig_deg_mean * 100 if orig_deg_mean > 0 else None
    )

    nodes_touched = set()
    for u, v in added_edges:
        nodes_touched.add(int(u))
        nodes_touched.add(int(v))
    for u, v in removed_edges:
        nodes_touched.add(int(u))
        nodes_touched.add(int(v))

    stats = {
        "original_edge_count": int(original_edge_count),
        "new_edge_count": int(new_edge_count),
        "num_added_edges": int(len(added_edges)),
        "num_removed_edges": int(len(removed_edges)),
        "net_edge_delta": int(new_edge_count - original_edge_count),
        "edge_growth_rate_pct": float(edge_growth_rate),
        "original_degree_mean": orig_deg_mean,
        "new_degree_mean": new_deg_mean,
        "degree_growth_pct": float(degree_growth_pct) if degree_growth_pct is not None else None,
        "num_modified_nodes": int(len(nodes_touched)),
        "num_modified_edges": int(len(added_edges) + len(removed_edges)),
    }
    return added_edges, removed_edges, edge_growth_rate, stats


def format_edge_statistics(stats, *, averaged_over_steps=None):
    """Human-readable summary for logs."""
    title = "Edge statistics"
    if averaged_over_steps:
        title += f" (averaged over {averaged_over_steps} reweighting steps)"
    lines = [
        title,
        f"  original edges: {stats['original_edge_count']}",
        f"  new edges: {stats['new_edge_count']}",
        f"  added / removed: {stats['num_added_edges']} / {stats['num_removed_edges']}",
        f"  edge growth (%): {stats['edge_growth_rate_pct']:.2f}",
        f"  mean degree (orig / new): {stats['original_degree_mean']:.2f} / {stats['new_degree_mean']:.2f}",
    ]
    if stats.get("degree_growth_pct") is not None:
        lines.append(f"  degree growth (%): {stats['degree_growth_pct']:.2f}")
    lines.append(f"  modified nodes: {stats.get('num_modified_nodes', 'n/a')}")
    return "\n".join(lines)
