import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import networkx as nx
from itertools import product
from typing import List
from RCAEval.io.time_series import (
    preprocess,
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    select_useful_cols,
)

F_NODE = "F-node"

def _dbg_col(df, name):
    loc = df.columns.get_loc(name)
    return type(loc).__name__, loc
def _normalize_column_names(df):
    """Normalize DataFrame column names to strings, handling MultiIndex and tuples."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
    else:
        df.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else str(col) for col in df.columns]
    return df


def df_to_prefix_graph(df, G):
    """
    Create directed graph H from df columns, matching G's prefix edges.
    
    Args:
    - df: DataFrame with columns like 'frontend_cpu_1', 'catalogue_mem_abc'
    - G: nx.DiGraph with prefix nodes (e.g. 'front-end' -> 'catalogue')
    
    Returns: nx.DiGraph H with exact df column names as nodes.
    """
    # Normalize column names to strings (handle MultiIndex/tuple column names)
    df_normalized = _normalize_column_names(df)
    cols = df_normalized.columns.tolist()
    
    # Map G prefix -> matching df columns (exact prefix match)
    prefix_to_nodes = {}
    for prefix in G.nodes:
        prefix_to_nodes[prefix] = [col for col in cols if str(col).startswith(str(prefix))]
    
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


def conditional_mutual_information(df: pd.DataFrame, X: str, Y: str, Z: List[str]) -> float:
    """
    Compute conditional mutual information I(X; Y | Z) for discrete columns in a pandas DataFrame.

    - If Z is empty, returns mutual information I(X; Y).
    - Uses natural logarithm (nats). Convert to bits by dividing by log(2) if desired.
    - Rows with NaN in any of the involved columns are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Discrete data.
    X, Y : str
        Column names for the two variables.
    Z : List[str]
        Conditioning set column names (can be empty).

    Returns
    -------
    float
        Conditional mutual information in nats.
    """
    # Normalize DataFrame columns to strings (handle MultiIndex/tuple column names)
    df = _normalize_column_names(df)
    
    # Ensure all column names are strings
    X = str(X) if not isinstance(X, str) else X
    Y = str(Y) if not isinstance(Y, str) else Y
    Z = [str(z) if not isinstance(z, str) else z for z in Z]
    
    cols = [X, Y] + list(Z)
    d = df[cols].dropna()
    # print("X:", X, "Y:", Y)
    # print("Z (len):", len(Z), "unique:", len(set(Z)))
    # print("cols (len):", len(cols), "unique:", len(set(cols)))
    # print("d.columns.is_unique:", d.columns.is_unique)
    # print(cols)

    # # This is the key check: does this label pick out multiple columns?
    # print("get_loc(X):", _dbg_col(d, X))
    # print("type(d[X]):", type(d[X]))
    
    N = len(d)
    if N == 0:
        return 0.0

    # Mutual information case (no conditioning)
    if not Z:
        n_xy = d.groupby([X, Y], sort=False).size().rename("n_xy").reset_index()
        n_x = d.groupby(X, sort=False).size().rename("n_x").reset_index()
        n_y = d.groupby(Y, sort=False).size().rename("n_y").reset_index()

        t = (
            n_xy.merge(n_x, on=X, how="left")
                .merge(n_y, on=Y, how="left")
        )

        n_xy_arr = t["n_xy"].to_numpy(dtype=np.float64)
        n_x_arr = t["n_x"].to_numpy(dtype=np.float64)
        n_y_arr = t["n_y"].to_numpy(dtype=np.float64)

        # sum_{x,y} (n_xy/N) * log( (n_xy * N) / (n_x * n_y) )
        return float(np.sum((n_xy_arr / N) * np.log((n_xy_arr * N) / (n_x_arr * n_y_arr))))

    # Conditional mutual information case
    z_cols = list(Z)

    n_xyz = d.groupby(z_cols + [X, Y], sort=False).size().rename("n_xyz").reset_index()
    n_xz = d.groupby(z_cols + [X], sort=False).size().rename("n_xz").reset_index()
    n_yz = d.groupby(z_cols + [Y], sort=False).size().rename("n_yz").reset_index()
    n_z = d.groupby(z_cols, sort=False).size().rename("n_z").reset_index()

    t = (
        n_xyz.merge(n_xz, on=z_cols + [X], how="left")
             .merge(n_yz, on=z_cols + [Y], how="left")
             .merge(n_z, on=z_cols, how="left")
    )

    n_xyz_arr = t["n_xyz"].to_numpy(dtype=np.float64)
    n_xz_arr = t["n_xz"].to_numpy(dtype=np.float64)
    n_yz_arr = t["n_yz"].to_numpy(dtype=np.float64)
    n_z_arr = t["n_z"].to_numpy(dtype=np.float64)

    # sum_{x,y,z} (n_xyz/N) * log( (n_xyz * n_z) / (n_xz * n_yz) )
    return float(np.sum((n_xyz_arr / N) * np.log((n_xyz_arr * n_z_arr) / (n_xz_arr * n_yz_arr))))




def add_fnode_and_concat(normal_df, anomalous_df):
    normal_df[F_NODE] = "0"
    anomalous_df[F_NODE] = "1"
    return pd.concat([normal_df, anomalous_df])



def drop_constant(df):
    return df.loc[:, (df != df.iloc[0]).any()]


def _discretize(data, bins):
    d = data.iloc[:, :-1]
    discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="kmeans")
    discretizer.fit(d)
    disc_d = discretizer.transform(d)
    # Column names should already be normalized strings from _preprocess_for_fnode
    disc_d = pd.DataFrame(disc_d, columns=d.columns.tolist())
    disc_d[F_NODE] = data[F_NODE].tolist()

    for c in disc_d:
        disc_d[c] = disc_d[c].astype(int)

    return disc_d

def _preprocess_for_fnode(normal_df, anomalous_df, bins):
    # Normalize column names before concatenation to ensure consistency
    normal_df = _normalize_column_names(normal_df)
    anomalous_df = _normalize_column_names(anomalous_df)
    
    df = add_fnode_and_concat(normal_df, anomalous_df)
    if df is None:
        return None

    return _discretize(df, bins) if bins is not None else df


def rcg_helper(normal_df,anomal_df, graph=None,**kwargs):
    # Normalize column names to ensure consistency between graph and DataFrame
    normal_df_normalized = _normalize_column_names(normal_df)
    anomal_df_normalized = _normalize_column_names(anomal_df)
    
    df = _preprocess_for_fnode(normal_df_normalized, anomal_df_normalized, 5)
    
    mi_scores = {}

    granular_graph = df_to_prefix_graph(normal_df_normalized, graph)
    for node in granular_graph:
        if node == F_NODE:
            continue
        parents = list(granular_graph.predecessors(node))
        parents = [parent for parent in parents if parent != node] # break the cycle if any
        mi_scores[node] = conditional_mutual_information(df, node, F_NODE, Z=parents)
    ranks = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }

def rcg(data, inject_time=None, dataset=None, graph=None,**kwargs):
    normal_df = data[data["time"] < inject_time]
    anomal_df = data[data["time"] >= inject_time]
    normal_df = normal_df.drop(columns=["time"])
    anomal_df = anomal_df.drop(columns=["time"])

    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]
    if "time.1" in normal_df.columns:
        normal_df = normal_df.drop(columns=["time.1"])
    if "time.1" in anomal_df.columns:
        anomal_df = anomal_df.drop(columns=["time.1"])
        
    return rcg_helper(normal_df, anomal_df, graph)

