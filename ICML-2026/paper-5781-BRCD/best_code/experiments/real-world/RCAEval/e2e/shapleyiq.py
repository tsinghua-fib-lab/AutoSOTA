from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import random
from sklearn.linear_model import Ridge

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

    prefix_to_nodes = {}
    for prefix in G.nodes:
        prefix_to_nodes[prefix] = [col for col in cols if col.startswith(prefix)]

    H = nx.DiGraph()
    H.add_nodes_from(cols)

    for u_prefix, v_prefix in G.edges:
        if u_prefix in prefix_to_nodes and v_prefix in prefix_to_nodes:
            u_nodes = prefix_to_nodes[u_prefix]
            v_nodes = prefix_to_nodes[v_prefix]
            for u in u_nodes:
                for v in v_nodes:
                    H.add_edge(u, v)

    return H


@dataclass
class LocalMechanism:
    parents: List[str]
    model: Optional[Ridge]
    parent_means: Optional[np.ndarray]
    parent_stds: Optional[np.ndarray]
    y_mean: float
    y_std: float
    intercept: float
    coefs_std: Optional[np.ndarray]


def _ensure_dag(G: nx.DiGraph) -> None:
    if not isinstance(G, nx.DiGraph):
        raise TypeError("G must be a networkx.DiGraph")
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("G must be a DAG")


def _shared_numeric_columns(
    normal_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    G: nx.DiGraph,
) -> List[str]:
    cols = []
    for c in normal_df.columns:
        if c in abnormal_df.columns and c in G.nodes:
            if pd.api.types.is_numeric_dtype(normal_df[c]) and pd.api.types.is_numeric_dtype(abnormal_df[c]):
                cols.append(c)
    return cols


def _median_impute_like_train(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train = train_df.loc[:, cols].copy().apply(pd.to_numeric, errors="coerce")
    other = other_df.loc[:, cols].copy().apply(pd.to_numeric, errors="coerce")
    med = train.median(numeric_only=True)
    train = train.fillna(med)
    other = other.fillna(med)
    return train, other, med


def _reference_states(
    normal_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    cols: Sequence[str],
) -> Tuple[pd.Series, pd.Series]:
    xN = normal_df.loc[:, cols].median(axis=0)
    xA = abnormal_df.loc[:, cols].median(axis=0)
    return xN, xA


def _normal_scale_stats(
    normal_df: pd.DataFrame,
    cols: Sequence[str],
) -> Tuple[pd.Series, pd.Series]:
    med = normal_df.loc[:, cols].median(axis=0)
    iqr = normal_df.loc[:, cols].quantile(0.75) - normal_df.loc[:, cols].quantile(0.25)
    iqr = iqr.replace(0, np.nan).fillna(1.0)
    return med, iqr


def _fit_local_mechanisms_linear(
    normal_df: pd.DataFrame,
    cols: Sequence[str],
    G: nx.DiGraph,
    alpha: float = 1.0,
) -> Dict[str, LocalMechanism]:
    """
    Fast local linear mechanisms fit only on normal data.

    Standardized-parent coefficients are stored for propagation:
        X_std -> y_std
    """
    mechs: Dict[str, LocalMechanism] = {}

    for node in nx.topological_sort(G):
        if node not in cols:
            continue

        parents = [p for p in G.predecessors(node) if p in cols]
        y = normal_df[node].to_numpy(dtype=float)
        y_mean = float(np.mean(y))
        y_std = float(np.std(y))
        if y_std <= 1e-12:
            y_std = 1.0

        if len(parents) == 0:
            mechs[node] = LocalMechanism(
                parents=[],
                model=None,
                parent_means=None,
                parent_stds=None,
                y_mean=y_mean,
                y_std=y_std,
                intercept=y_mean,
                coefs_std=None,
            )
            continue

        X = normal_df[parents].to_numpy(dtype=float)
        parent_means = np.mean(X, axis=0)
        parent_stds = np.std(X, axis=0)
        parent_stds[parent_stds <= 1e-12] = 1.0

        Xs = (X - parent_means) / parent_stds
        ys = (y - y_mean) / y_std

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(Xs, ys)

        mechs[node] = LocalMechanism(
            parents=parents,
            model=model,
            parent_means=parent_means,
            parent_stds=parent_stds,
            y_mean=y_mean,
            y_std=y_std,
            intercept=float(model.intercept_),
            coefs_std=np.asarray(model.coef_, dtype=float),
        )

    return mechs


def _predict_from_mechanism(
    node: str,
    parent_values: Dict[str, float],
    mechs: Dict[str, LocalMechanism],
) -> float:
    mech = mechs[node]
    if len(mech.parents) == 0 or mech.model is None:
        return mech.y_mean

    x = np.array([parent_values[p] for p in mech.parents], dtype=float)
    xs = (x - mech.parent_means) / mech.parent_stds
    ys = mech.model.predict(xs.reshape(1, -1))[0]
    return float(mech.y_mean + mech.y_std * ys)


def _compute_residual_shifts(
    cols: Sequence[str],
    G: nx.DiGraph,
    mechs: Dict[str, LocalMechanism],
    xN: pd.Series,
    xA: pd.Series,
) -> Dict[str, float]:
    """
    residual_shift[node] =
        total abnormal shift - inherited shift from parents
    """
    residual_shifts: Dict[str, float] = {}

    for node in nx.topological_sort(G):
        if node not in cols:
            continue

        mech = mechs[node]
        total_shift = float(xA[node] - xN[node])

        if len(mech.parents) == 0 or mech.model is None:
            residual_shifts[node] = total_shift
            continue

        pred_N = _predict_from_mechanism(
            node=node,
            parent_values={p: float(xN[p]) for p in mech.parents},
            mechs=mechs,
        )
        pred_A_from_Aparents = _predict_from_mechanism(
            node=node,
            parent_values={p: float(xA[p]) for p in mech.parents},
            mechs=mechs,
        )

        inherited_shift = pred_A_from_Aparents - pred_N
        residual_shifts[node] = total_shift - inherited_shift

    return residual_shifts


def _ancestor_candidates(
    cols: Sequence[str],
    G: nx.DiGraph,
    target: str,
) -> List[str]:
    anc = set(nx.ancestors(G, target))
    return [n for n in nx.topological_sort(G) if n in anc and n in cols and n != target]


def _anomaly_scores(
    xN: pd.Series,
    xA: pd.Series,
    iqrN: pd.Series,
    cols: Sequence[str],
) -> Dict[str, float]:
    scores = {}
    for c in cols:
        scores[c] = float(abs(xA[c] - xN[c]) / (iqrN[c] + 1e-8))
    return scores


def _build_edge_weight_map(
    cols: Sequence[str],
    G: nx.DiGraph,
    mechs: Dict[str, LocalMechanism],
    max_abs_weight: float = 0.95,
) -> Dict[Tuple[str, str], float]:
    """
    Edge weights are standardized local sensitivities.
    Stored as w[(parent, child)].
    """
    w: Dict[Tuple[str, str], float] = {}

    for child in nx.topological_sort(G):
        if child not in cols:
            continue

        mech = mechs[child]
        if len(mech.parents) == 0 or mech.coefs_std is None:
            continue

        for p, coef in zip(mech.parents, mech.coefs_std):
            val = float(np.clip(coef, -max_abs_weight, max_abs_weight))
            w[(p, child)] = val

    return w


def _propagation_strength_to_target(
    cols: Sequence[str],
    G: nx.DiGraph,
    target: str,
    edge_w: Dict[Tuple[str, str], float],
    mode: str = "sum",
) -> Dict[str, float]:
    """
    Dynamic programming over DAG.

    strength[target] = 1
    strength[node] = sum or max over children of
                     |w(node, child)| * strength[child]
    """
    strength = {node: 0.0 for node in cols}
    strength[target] = 1.0

    topo = [n for n in nx.topological_sort(G) if n in cols]

    for node in reversed(topo):
        if node == target:
            continue
        child_vals = []
        for child in G.successors(node):
            if child not in cols:
                continue
            wij = abs(edge_w.get((node, child), 0.0))
            if wij <= 0:
                continue
            child_vals.append(wij * strength[child])

        if not child_vals:
            strength[node] = 0.0
        elif mode == "max":
            strength[node] = float(max(child_vals))
        else:
            strength[node] = float(sum(child_vals))

    return strength


def _screen_candidates(
    candidates: Sequence[str],
    anomaly_scores: Dict[str, float],
    propagation_strength: Dict[str, float],
    residual_shifts: Dict[str, float],
    top_k: int = 12,
    min_anom: float = 0.25,
) -> List[str]:
    scored = []
    for node in candidates:
        if propagation_strength.get(node, 0.0) <= 0:
            continue
        if anomaly_scores.get(node, 0.0) < min_anom and abs(residual_shifts.get(node, 0.0)) < 1e-12:
            continue

        pre_score = (
            abs(residual_shifts.get(node, 0.0))
            * (1.0 + anomaly_scores.get(node, 0.0))
            * propagation_strength.get(node, 0.0)
        )
        scored.append((node, pre_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored[:top_k]]


def causal_shapleyiq_scaled(
    normal_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    G: nx.DiGraph,
    target: str,
    *,
    candidate_nodes: Optional[Sequence[str]] = None,
    top_k_candidates: int = 12,
    ridge_alpha: float = 1.0,
    propagation_mode: str = "sum",
    min_anom_score: float = 0.25,
) -> Dict[str, object]:
    """
    Scalable ShapleyIQ-style RCA for tabular metrics.

    Main idea:
      1) fit local mechanisms on normal data
      2) compute local residual abnormal shifts
      3) propagate influence to target through DAG using local sensitivities
      4) rank by propagated contribution

    Returns
    -------
    dict with keys:
      - "ranks"
      - "scores"
      - "candidate_nodes"
      - "screened_candidates"
      - "normal_reference"
      - "abnormal_reference"
      - "residual_shifts"
      - "anomaly_scores"
      - "propagation_strength"
      - "edge_weights"
    """
    _ensure_dag(G)

    if target not in G.nodes:
        raise ValueError(f"target={target!r} is not in G")

    cols = _shared_numeric_columns(normal_df, abnormal_df, G)
    if target not in cols:
        raise ValueError(
            f"target={target!r} must be a numeric column shared by normal_df, abnormal_df, and G"
        )

    normal_clean, abnormal_clean, _ = _median_impute_like_train(normal_df, abnormal_df, cols)
    xN, xA = _reference_states(normal_clean, abnormal_clean, cols)
    _, iqrN = _normal_scale_stats(normal_clean, cols)

    mechs = _fit_local_mechanisms_linear(
        normal_df=normal_clean,
        cols=cols,
        G=G,
        alpha=ridge_alpha,
    )

    residual_shifts = _compute_residual_shifts(
        cols=cols,
        G=G,
        mechs=mechs,
        xN=xN,
        xA=xA,
    )

    anomaly_scores = _anomaly_scores(
        xN=xN,
        xA=xA,
        iqrN=iqrN,
        cols=cols,
    )

    edge_w = _build_edge_weight_map(
        cols=cols,
        G=G,
        mechs=mechs,
    )

    propagation_strength = _propagation_strength_to_target(
        cols=cols,
        G=G,
        target=target,
        edge_w=edge_w,
        mode=propagation_mode,
    )

    if candidate_nodes is None:
        candidates = _ancestor_candidates(cols=cols, G=G, target=target)
    else:
        cand_set = set(candidate_nodes)
        candidates = [
            n for n in nx.topological_sort(G)
            if n in cand_set and n in cols and n != target
        ]

    screened = _screen_candidates(
        candidates=candidates,
        anomaly_scores=anomaly_scores,
        propagation_strength=propagation_strength,
        residual_shifts=residual_shifts,
        top_k=top_k_candidates,
        min_anom=min_anom_score,
    )

    scores: Dict[str, float] = {}
    for node in screened:
        # signed score keeps direction; ranking uses absolute magnitude
        scores[node] = float(
            residual_shifts[node] * propagation_strength[node]
        )

    ranks = sorted(scores.keys(), key=lambda n: abs(scores[n]), reverse=True)

    return {
        "ranks": ranks,
        "scores": scores,
        "candidate_nodes": candidates,
        "screened_candidates": screened,
        "normal_reference": xN.to_dict(),
        "abnormal_reference": xA.to_dict(),
        "residual_shifts": residual_shifts,
        "anomaly_scores": anomaly_scores,
        "propagation_strength": propagation_strength,
        "edge_weights": {f"{u}->{v}": w for (u, v), w in edge_w.items()},
    }


def shapleyiq(
    data,
    inject_time=None,
    dataset=None,
    graph=None,
    target_node=None,
    num_loop=None,
    sli=None,
    anomalies=None,
    **kwargs,
):
    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        anomal_df = data.tail(len(data) - anomalies[0])

    normal_df = preprocess(
        data=normal_df,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False),
    )

    anomal_df = preprocess(
        data=anomal_df,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False),
    )

    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    if target_node == "front-end_container_cpu" or target_node == "front-end_container-memory":
        filtered = [s for s in intersects if s.startswith(target_node)]
        if filtered:
            target_node = random.choice(filtered)
        else:
            target_node = random.choice(intersects)

    if dataset in {"re1-ss", "sock-shop-2", "re1-ob", "online-boutique"}:
        granular_graph = df_to_prefix_graph(normal_df, graph)
    else:
        granular_graph = graph

    return causal_shapleyiq_scaled(
        normal_df=normal_df,
        abnormal_df=anomal_df,
        G=granular_graph,
        target=target_node,
        candidate_nodes=kwargs.get("candidate_nodes", None),
        top_k_candidates=kwargs.get("top_k_candidates", 12),
        ridge_alpha=kwargs.get("ridge_alpha", 1.0),
        propagation_mode=kwargs.get("propagation_mode", "sum"),
        min_anom_score=kwargs.get("min_anom_score", 0.25),
    )