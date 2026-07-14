from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx
import random

from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    preprocess,
    select_useful_cols,
)


def df_to_prefix_graph(df, G):
    """
    Create directed graph H from df columns, matching G's prefix edges.
    
    Args:
    - df: DataFrame with columns like 'frontend_cpu_1', 'catalogue_mem_abc'
    - G: nx.DiGraph with prefix nodes (e.g. 'front-end' -> 'catalogue')
    
    Returns: nx.DiGraph H with exact df column names as nodes.
    """
    cols = df.columns.tolist()
    
    # Map G prefix -> matching df columns (exact prefix match)
    prefix_to_nodes = {}
    for prefix in G.nodes:
        prefix_to_nodes[prefix] = [col for col in cols if col.startswith(prefix)]
    
    H = nx.DiGraph()
    
    # Add all df columns as nodes
    H.add_nodes_from(cols)
    
    # Add edges: for every (u_prefix, v_prefix) in G, connect *all* u_nodes -> *all* v_nodes
    for u_prefix, v_prefix in G.edges:
        if u_prefix in prefix_to_nodes and v_prefix in prefix_to_nodes:
            u_nodes = prefix_to_nodes[u_prefix]
            v_nodes = prefix_to_nodes[v_prefix]
            for u in u_nodes:
                for v in v_nodes:
                    H.add_edge(u, v)
    
    return H



@dataclass
class MicroDigServiceResult:
    ranks: List[str]
    service_scores: Dict[str, float]
    anomaly_scores: Dict[str, float]


def _validate_inputs(
    normal_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    G: nx.DiGraph,
    issue_service: str,
) -> List[str]:
    if not isinstance(G, nx.DiGraph):
        raise TypeError("G must be a networkx.DiGraph")

    if issue_service not in G.nodes:
        raise ValueError(f"issue_service={issue_service!r} is not a node in G")

    common_cols = [c for c in normal_df.columns if c in abnormal_df.columns and c in G.nodes]
    if not common_cols:
        raise ValueError("No overlapping service columns across normal_df, abnormal_df, and G nodes")

    return common_cols


def _zshift_anomaly_scores(
    normal_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    cols: List[str],
    eps: float = 1e-8,
) -> Dict[str, float]:
    scores = {}
    for c in cols:
        n = pd.to_numeric(normal_df[c], errors="coerce").dropna()
        a = pd.to_numeric(abnormal_df[c], errors="coerce").dropna()
        if len(n) == 0 or len(a) == 0:
            scores[c] = 0.0
            continue
        mu_n = float(n.mean())
        mu_a = float(a.mean())
        sd_n = float(n.std(ddof=1)) if len(n) > 1 else 0.0
        scores[c] = abs(mu_a - mu_n) / (sd_n + eps)
    return scores


def _prepare_combined_timeseries(
    normal_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    cols: List[str],
) -> pd.DataFrame:
    x_n = normal_df[cols].apply(pd.to_numeric, errors="coerce")
    x_a = abnormal_df[cols].apply(pd.to_numeric, errors="coerce")
    x = pd.concat([x_n, x_a], axis=0, ignore_index=True)
    x = x.fillna(x.median(numeric_only=True))
    return x


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    if x.nunique(dropna=True) <= 1 or y.nunique(dropna=True) <= 1:
        return 0.0
    val = x.corr(y)
    if pd.isna(val):
        return 0.0
    return float(abs(val))


def _build_microdig_graph(
    G: nx.DiGraph,
    cols: List[str],
    issue_service: str,
    anomaly_scores: Dict[str, float],
    ts_df: pd.DataFrame,
    reverse_edge_weight: float = 0.2,
    service_self_loop_weight: float = 1.0,
    service_call_mix: float = 1.0,
) -> Tuple[nx.DiGraph, List[str], List[str]]:
    """
    Build a MicroDig-style heterogeneous graph.

    Nodes:
      - service nodes: ('svc', service_name)
      - call nodes: ('call', u, v) for each edge u -> v in G restricted to cols

    Construction:
      1) service -> call for caller and callee
      2) downstream_call -> upstream_call for adjacent calls u->v and v->w
      3) flip all edges for localization (effect -> cause)
      4) add weak reverse edges and service self-loops
    """
    services = [s for s in G.nodes if s in cols]
    restricted_edges = [(u, v) for u, v in G.edges if u in cols and v in cols]

    H = nx.DiGraph()

    service_nodes = [("svc", s) for s in services]
    call_nodes = [("call", u, v) for u, v in restricted_edges]

    H.add_nodes_from(service_nodes)
    H.add_nodes_from(call_nodes)

    # Service -> call edges, weighted by service anomaly
    for u, v in restricted_edges:
        cnode = ("call", u, v)
        au = anomaly_scores.get(u, 0.0)
        av = anomaly_scores.get(v, 0.0)

        # caller/callee as immediate "causes" of the call node
        H.add_edge(("svc", u), cnode, weight=max(au, 1e-8) * service_call_mix)
        H.add_edge(("svc", v), cnode, weight=max(av, 1e-8) * service_call_mix)

    # downstream_call -> upstream_call edges
    # if u->v and v->w then call(v,w) causes call(u,v)
    for u, v in restricted_edges:
        upstream_call = ("call", u, v)
        for w in G.successors(v):
            if w not in cols:
                continue
            if (v, w) not in restricted_edges:
                continue
            downstream_call = ("call", v, w)

            # service-level proxy for call correlation
            wgt = _safe_corr(ts_df[v], ts_df[w])
            H.add_edge(downstream_call, upstream_call, weight=max(wgt, 1e-8))

    # Flip graph for localization: effects -> causes
    R = nx.DiGraph()
    R.add_nodes_from(H.nodes)
    for a, b, data in H.edges(data=True):
        w = float(data.get("weight", 1.0))
        R.add_edge(b, a, weight=w)

    # Add weak reverse edges for robustness
    edges_now = list(R.edges(data=True))
    for a, b, data in edges_now:
        w = float(data.get("weight", 1.0))
        if not R.has_edge(b, a):
            R.add_edge(b, a, weight=reverse_edge_weight * w)

    # Add self-loops on service nodes
    for s in services:
        R.add_edge(("svc", s), ("svc", s), weight=service_self_loop_weight)

    return R, service_nodes, call_nodes


def _seed_distribution(
    R: nx.DiGraph,
    G: nx.DiGraph,
    issue_service: str,
    call_seed_weight: float = 1.0,
    self_seed_weight: float = 1.0,
) -> Dict[tuple, float]:
    """
    Seed on call nodes incident to issue_service, plus the issue service itself.
    """
    seed = {n: 0.0 for n in R.nodes}

    issue_node = ("svc", issue_service)
    if issue_node in seed:
        seed[issue_node] += self_seed_weight

    incident_calls = []
    for v in G.successors(issue_service):
        c = ("call", issue_service, v)
        if c in seed:
            incident_calls.append(c)
    for u in G.predecessors(issue_service):
        c = ("call", u, issue_service)
        if c in seed:
            incident_calls.append(c)

    if incident_calls:
        per = call_seed_weight / len(incident_calls)
        for c in incident_calls:
            seed[c] += per

    total = sum(seed.values())
    if total <= 0:
        seed[issue_node] = 1.0
        total = 1.0

    return {k: v / total for k, v in seed.items()}


def _transition_matrix(R: nx.DiGraph, nodes: List[tuple]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(nodes)}
    P = np.zeros((len(nodes), len(nodes)), dtype=float)

    for i, u in enumerate(nodes):
        out_edges = list(R.out_edges(u, data=True))
        if not out_edges:
            P[i, i] = 1.0
            continue
        weights = np.array([max(float(d.get("weight", 1.0)), 0.0) for _, _, d in out_edges], dtype=float)
        s = weights.sum()
        if s <= 0:
            P[i, i] = 1.0
            continue
        weights = weights / s
        for wgt, (_, v, _) in zip(weights, out_edges):
            j = idx[v]
            P[i, j] += wgt

    return P


def _personalized_random_walk(
    R: nx.DiGraph,
    seed_dist: Dict[tuple, float],
    alpha: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Dict[tuple, float]:
    nodes = list(R.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    P = _transition_matrix(R, nodes)

    s = np.zeros(len(nodes), dtype=float)
    for n, val in seed_dist.items():
        s[idx[n]] = float(val)

    x = s.copy()
    for _ in range(max_iter):
        x_new = alpha * (x @ P) + (1.0 - alpha) * s
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new

    return {n: float(x[idx[n]]) for n in nodes}


def microdig_service(
    normal_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    G: nx.DiGraph,
    issue_service: str,
    *,
    alpha: float = 0.85,
    reverse_edge_weight: float = 0.2,
    service_self_loop_weight: float = 1.0,
    call_seed_weight: float = 1.0,
    self_seed_weight: float = 1.0,
) -> Dict[str, List[str] | Dict[str, float]]:
    """
    MicroDig-style service-level RCA adaptation.

    Parameters
    ----------
    normal_df, abnormal_df:
        Rows should be time points, columns should be service names.
    G:
        networkx.DiGraph of service calls: u -> v means service u calls service v.
    issue_service:
        Alarmed / issue service used to seed the random walk.

    Returns
    -------
    {
      "ranks": [...],
      "service_scores": {...},
      "anomaly_scores": {...}
    }
    """
    cols = _validate_inputs(normal_df, abnormal_df, G, issue_service)
    anomaly_scores = _zshift_anomaly_scores(normal_df, abnormal_df, cols)
    ts_df = _prepare_combined_timeseries(normal_df, abnormal_df, cols)

    R, service_nodes, _ = _build_microdig_graph(
        G=G,
        cols=cols,
        issue_service=issue_service,
        anomaly_scores=anomaly_scores,
        ts_df=ts_df,
        reverse_edge_weight=reverse_edge_weight,
        service_self_loop_weight=service_self_loop_weight,
    )

    seed = _seed_distribution(
        R=R,
        G=G,
        issue_service=issue_service,
        call_seed_weight=call_seed_weight,
        self_seed_weight=self_seed_weight,
    )

    scores_all = _personalized_random_walk(R, seed_dist=seed, alpha=alpha)

    service_scores = {}
    for node in service_nodes:
        svc = node[1]
        service_scores[svc] = scores_all.get(node, 0.0)

    ranks = sorted(service_scores, key=lambda s: service_scores[s], reverse=True)

    return {
        "ranks": ranks,
        "service_scores": service_scores,
        "anomaly_scores": anomaly_scores,
    }


def microdig(
    data, inject_time=None, dataset=None, graph=None, target_node=None, num_loop=None, sli=None, anomalies=None, **kwargs
):
    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        anomal_df = data.tail(len(data) - anomalies[0])


    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    if target_node == "front-end_container_cpu" or  target_node == "front-end_container-memory":
        filtered = [s for s in intersects if s.startswith(target_node)]
        if filtered:                      # avoid error if none match
            target_node = random.choice(filtered)
        else:
            target_node = random.choice(intersects)                 

    if dataset == "re1-ss" or dataset == 'sock-shop-2' or dataset == "re1-ob" or dataset == 'online-boutique':
        granular_graph = df_to_prefix_graph(normal_df, graph)
    else:
        granular_graph = graph
        
    return microdig_service(normal_df, anomal_df, granular_graph, target_node)