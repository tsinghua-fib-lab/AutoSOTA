from dataclasses import dataclass

import json
import logging
import networkx as nx
import os
import pandas as pd
import numpy as np
from typing import Any, List
from collections import defaultdict

from causallearn.graph.Endpoint import Endpoint

from RCAEval.utility import (
    dump_json,
    is_py310,
    is_py38,
    is_py312,
    load_json,
    download_online_boutique_dataset,
    download_sock_shop_1_dataset,
    download_sock_shop_2_dataset,
    download_train_ticket_dataset,
    download_re1_dataset,
    download_re2ob_dataset,
    download_re2ss_dataset,
    download_re2tt_dataset,
    download_re3_dataset, 

)


if is_py312():
    from RCAEval.e2e import (
        baro,
        causalrca_petshop,
        circa_petshop,
        dummy,
        rcg_helper,
        simplerca,
        microdig_service,
        causal_shapleyiq_scaled,
    )

elif is_py38():
    from RCAEval.e2e import dummy, e_diagnosis_petshop, ht, rcd, mmrcd, rcd_helper
elif is_py310():
    from RCAEval.e2e.brcd import brcd_helper
    from RCAEval.e2e.BRCD.boss import boss
    from dowhy.graph import node_connected_subgraph_view
    from RCAEval.e2e import idint_helper, score_ordering, smooth_traversal, cholesky, apply_smooth_traversal, apply_score_ordering, apply_cholesky
else:
    print("Please use Python 3.8 or 3.12")
    exit(1)




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

def create_column_graph_from_causal_graph(column_names, causal_graph):
    """
    Create a directed graph with column names as nodes, based on the structure in causal_graph.
    
    For each column name x:
    1. Split by "_" to get name = x.split("_")
    2. Take "_".join(name[:-2]) to get prefix q
    3. Use q to find corresponding node in causal_graph
    4. Use the structure in causal_graph to create edges between column names
    
    Args:
    - column_names: list of column names (excluding 'time')
    - causal_graph: nx.DiGraph with prefix nodes
    
    Returns: nx.DiGraph with column names as nodes
    """
    # Filter out 'time' column if present
    cols = [col for col in column_names if col != 'time']
    
    # Map each column to its prefix (q)
    column_to_prefix = {}
    prefix_to_columns = defaultdict(list)
    
    for col in cols:
        name_parts = col.split("_")
        if len(name_parts) >= 2:
            # Take all parts except the last 2
            prefix = "_".join(name_parts[:-2])
            column_to_prefix[col] = prefix
            prefix_to_columns[prefix].append(col)
        else:
            # If column doesn't have enough parts, use the whole name as prefix
            column_to_prefix[col] = col
            prefix_to_columns[col].append(col)
    
    # Create new graph with column names as nodes
    H = nx.DiGraph()
    H.add_nodes_from(cols)
    
    # For each edge (u_prefix, v_prefix) in causal_graph, create edges between corresponding columns
    for u_prefix, v_prefix in causal_graph.edges:
        if u_prefix in prefix_to_columns and v_prefix in prefix_to_columns:
            u_columns = prefix_to_columns[u_prefix]
            v_columns = prefix_to_columns[v_prefix]
            # Create edges from all u_columns to all v_columns
            for u_col in u_columns:
                for v_col in v_columns:
                    H.add_edge(u_col, v_col)
    
    return H

def diagnose_boss_input(
    df: pd.DataFrame,
    eps: float = 1e-12,
    coerce_non_numeric: bool = True,
    show_examples: int = 8,
):
    """
    Return (report_dict, cast_df) where cast_df is numeric float64 and report_dict
    summarizes potential issues that can lead to NaN/Inf during kernel scoring.
    """
    rep = {}

    X = df.copy()

    # 1) enforce numeric float64
    non_numeric_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    rep["non_numeric_cols"] = non_numeric_cols
    if coerce_non_numeric and non_numeric_cols:
        # this can create NaNs if strings like 'NA'/'inf' exist
        X[non_numeric_cols] = X[non_numeric_cols].apply(pd.to_numeric, errors='coerce')

    X = X.astype("float64", copy=False)

    # 2) find NaN / ±Inf
    arr = X.to_numpy()
    nan_locs = np.argwhere(np.isnan(arr))
    inf_locs = np.argwhere(~np.isfinite(arr) & ~np.isnan(arr))
    rep["has_nan"] = bool(nan_locs.size)
    rep["has_inf"] = bool(inf_locs.size)

    if rep["has_nan"]:
        r, c = nan_locs[:show_examples].T if nan_locs.size else ([], [])
        rep["nan_examples"] = [(int(rr), X.columns[int(cc)], X.iat[int(rr), int(cc)]) for rr, cc in zip(r, c)]
        rep["nan_counts_per_col"] = X.isna().sum().to_dict()
    else:
        rep["nan_examples"] = []
        rep["nan_counts_per_col"] = {}

    if rep["has_inf"]:
        r, c = inf_locs[:show_examples].T if inf_locs.size else ([], [])
        rep["inf_examples"] = [(int(rr), X.columns[int(cc)], X.iat[int(rr), int(cc)]) for rr, cc in zip(r, c)]
        # count +inf and -inf per column
        rep["inf_counts_per_col"] = {
            col: int(np.isposinf(X[col]).sum() + np.isneginf(X[col]).sum()) for col in X.columns
        }
    else:
        rep["inf_examples"] = []
        rep["inf_counts_per_col"] = {}

    # 3) zero-variance and near-constant columns
    std = X.std(ddof=0)
    zero_var = std[std == 0.0].index.tolist()
    near_const = std[(std > 0.0) & (std < eps)].index.tolist()

    rep["zero_variance_cols"] = zero_var
    rep["near_constant_cols(std<eps)"] = near_const
    rep["std_per_col"] = std.to_dict()

    # 4) duplicate rows (not fatal, but can collapse pairwise distances)
    rep["num_duplicate_rows"] = int(X.duplicated().sum())

    # 5) quick recommendation
    rec = []
    if rep["has_inf"]:
        rec.append("Replace ±inf with finite values or drop affected rows/cols.")
    if rep["has_nan"]:
        rec.append("NaNs appeared after casting—clean or impute/drop before BOSS.")
    if zero_var:
        rec.append(f"Drop zero-variance columns: {zero_var[:6]}{'...' if len(zero_var)>6 else ''}")
    if near_const:
        rec.append(f"Consider dropping or jittering near-constant columns (std< {eps:g}).")
    if not rec:
        rec.append("Data look OK for kernel scoring; if the error persists, try linear kernels.")
    rep["recommendations"] = rec

    return rep, X

def split_edges(G: nx.DiGraph, df):
    # All node names
    nodes = list(G.nodes())
    colnames = list(df.columns)
    node_names = [colnames[i] for i in nodes]

    # Undirected edges = reciprocal pairs, collapsed
    undirected_edges = list(G.to_undirected(reciprocal=True).edges())
    undirected_edges_named = [(colnames[u], colnames[v]) for u, v in undirected_edges]


    # Directed-only = edges that do NOT have the reverse (plus self-loops)
    directed_edges = [(u, v) for u, v in G.edges() if (u == v) or (not G.has_edge(v, u))]
    directed_edges_named = [(colnames[u], colnames[v]) for u, v in directed_edges]

    return node_names, directed_edges_named, undirected_edges_named

def causal_learn_graph_to_nx_digraph(G_cl, column_names):
    """
    Convert a causallearn Graph to a networkx.DiGraph using actual column names.
    :param G_cl: causallearn.graph.Graph object
    :param column_names: list of column names from the original DataFrame
    """
    G_nx = nx.DiGraph()


    id_to_col = {i: name for i, name in enumerate(column_names)}

    # Add nodes with proper names
    for node in G_cl.get_nodes():
        node_id = G_cl.node_map[node]
        G_nx.add_node(id_to_col[node_id])
    arcs = []
    edges = []
    # Add directed edges using correct names
    for edge in G_cl.get_graph_edges():
        n1 = id_to_col[G_cl.node_map[edge.node1]]
        n2 = id_to_col[G_cl.node_map[edge.node2]]
        ep1 = edge.endpoint1
        ep2 = edge.endpoint2

        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            arcs.append((n1, n2))
        elif ep2 ==  Endpoint.TAIL and ep1 == Endpoint.ARROW:
            arcs.append((n2, n1))
        elif ep2 ==  Endpoint.TAIL and ep1 == Endpoint.TAIL:
            edges.append((n2, n1))
    return arcs, edges

def compute_mutual_information_ranking(joint_df):
    """
    Compute mutual information between each column and FNODE, and rank columns by MI.
    
    Args:
        joint_df (pd.DataFrame): DataFrame containing all variables including FNODE
        
    Returns:
        list: Column names ranked by mutual information with FNODE (highest to lowest)
    """
    mi_scores = []
    for col in joint_df.columns:
        if col != 'FNODE':
            # Create contingency table
            contingency = pd.crosstab(joint_df[col], joint_df['FNODE'])
            
            # Convert to probabilities
            p_xy = contingency / contingency.sum().sum()
            
            # Compute marginal probabilities
            p_x = p_xy.sum(axis=1)
            p_y = p_xy.sum(axis=0)
            
            # Compute mutual information
            mi = 0
            for i in p_xy.index:
                for j in p_xy.columns:
                    if p_xy.loc[i,j] > 0:  # Avoid log(0)
                        mi += p_xy.loc[i,j] * np.log(p_xy.loc[i,j] / (p_x[i] * p_y[j]))
            
            mi_scores.append((col, mi))
    
    # Sort by mutual information (highest to lowest)
    ranked_columns = [col for col, _ in sorted(mi_scores, key=lambda x: x[1], reverse=True)]
    return ranked_columns


def load_scenario(path):
    graph = nx.from_pandas_adjacency(
        pd.read_csv(os.path.join(path, "graph.csv"), index_col=0),
        create_using=nx.DiGraph,
    )

    normal_metrics = pd.read_csv(
        os.path.join(path, "noissue", "metrics.csv"), header=[0, 1, 2], index_col=0
    )

    issues = {"train": [], "test": []}
    for split in issues:
        for issue in os.listdir(os.path.join(path, split)):
            if issue.startswith("."):  # Skip hidden files and folders.
                continue
            metrics = pd.read_csv(
                os.path.join(path, split, issue, "metrics.csv"),
                header=[0, 1, 2],
                index_col=0,
            )
            with open(os.path.join(path, split, issue, "target.json"), "r") as f:
                target = json.load(f)
            issues[split].append((metrics, target))
    return graph, normal_metrics, issues


def in_top_k(
    potential_root_causes: List[str],
    ground_truth_node: str,
    ground_truth_metric: str = None,
    k: int = 3,
) -> bool:
    for idx, root_cause in enumerate(potential_root_causes[:k]):
        name = root_cause.split("_")
        metric = name[-2]
        stat = name[-1]
        node = "_".join(name[:-2])
        if node == ground_truth_node:
            if (
                ground_truth_metric is None
                or ground_truth_metric == metric
            ):
                return True
    return False

def map_df(df):
    df_new = df.copy(deep=True)
    df_new.index.names = ["time"]
    columns = ["_".join([c[0], c[1], c[2]]) for c in df_new.columns]
    df_new.columns = columns
    return df_new

def impute_df(df: pd.DataFrame, method: str = "mean", fill: float = -1):
    """
    Wrapper around very simple imputation methods.

    Args:
        df: Pandas DataFrame in which to impute NaNs.
        method: How NaNs should be imputed. If 'mean' then each is replaced by the mean of the
            remaining values of the same microservice, metric and statistic. If 'interpolate' then
            pandas.DataFrame.interpolate(method='time',limit_direction='both') is used.
            if 'fill' then missing values will be replaced with the value `fill`.
        fill: Value with which to replace NaNs if `method = 'fill'`.
    """
    if method not in {"mean", "interpolate", "fill", 'median'}:
        ValueError(f"{method} is not a valid imputation method.")
    if method == "mean":
        df.fillna(df.mean(), inplace=True)
    elif method == "median":
        df.fillna(df.median(), inplace=True)
    elif method == "interpolate":
        df_index = df.index
        df.index = pd.to_datetime(df.index, unit="s")
        df.interpolate("time", limit_direction="both", inplace=True)
        df.interpolate("time", limit_direction="both", inplace=True)
        # reverting index back for consistency between imputation methods
        df.index = df_index
    elif method == "fill":
        df.fillna(fill, inplace=True)

def flatten_columns_to_str(df, sep="__"):
    df = df.copy()
    df.columns = [
        sep.join(map(str, col)) if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    return df

def convert_to_cpdag_after_variables_removed(graph, variables_to_keep, complete_graph=False):
    # Step 1: Original DAG
    original_dag = graph.copy()
    # Step 3: Create a projected DAG on common_cols
    projected_graph = nx.DiGraph()
    projected_graph.add_nodes_from(variables_to_keep)

    # For each pair of nodes u, v in common_cols:
    # If there exists a directed path u -> ... -> v in the original DAG
    # that does not go through any other node in common_cols, add u -> v
    if complete_graph:
        # Create a fully connected DAG where each node points to all nodes after it
        # This ensures the graph is both fully connected and acyclic
        graph = nx.DiGraph()
        # Add all nodes using their actual names
        graph.add_nodes_from(variables_to_keep)
        # Add edges from each node to all nodes after it
        for i, u in enumerate(variables_to_keep):
            for v in variables_to_keep[i+1:]:
                projected_graph.add_edge(u, v)
    else:
        for u in variables_to_keep:
            for v in variables_to_keep:
                if u == v:
                    continue
                for path in nx.all_simple_paths(original_dag, source=u, target=v):
                    # Check that all intermediate nodes are *not* in common_cols
                    if all(node not in variables_to_keep for node in path[1:-1]):
                        projected_graph.add_edge(u, v)
                        break  # No need to check other paths

    # Step 4: Convert to CPDAG
    dag_for_cpdag = DAG(arcs=projected_graph.edges())
    cpdag = dag_for_cpdag.cpdag()
    return cpdag

def evaluate(model_name: str, dir: str, split: str = None):
    # statistic_of_interest = 'Average'
    imputation_method = 'median'
    scenarios = [
        "low_traffic",
        "high_traffic",
        "temporal_traffic1",
        "temporal_traffic2",
    ]
    
    results = {}
    result_list = []
    if split is None:
        splits = ["train", "test"]
    for scenario in scenarios:
        graph, normal_metrics, issues = load_scenario(os.path.join(dir, scenario))
        # if nx.is_directed_acyclic_graph(graph):
        # print(f"scenario: {scenario}, is the graph cyclic? {nx.is_directed_acyclic_graph(graph)}")
        results[scenario] = {}
        causal_graph = graph.reverse()
        # construct the graph at the metric level
        for split in splits:
            results[scenario][split] = {1: [], 3: [], 5:[]}
            for idx, (abnormal_metrics, target) in enumerate(issues[split]):
                statistic_of_interest = target["target"]["agg"]
                issue_metric = target["target"]["metric"]
                print(f"issue_metric: {issue_metric}, statistic_of_interest: {statistic_of_interest}")
                # reduce the dataframe to the issue metric and statistics of interests
                normal_metrics_new = normal_metrics.loc[
                    :, (slice(None), [issue_metric], [statistic_of_interest])
                ]
                abnormal_metrics = abnormal_metrics.loc[
                    :, (slice(None), [issue_metric], [statistic_of_interest])
                ]


                # change the dataframe names
                normal_metrics_new = normal_metrics.copy()
                normal_metrics_new = map_df(normal_metrics_new)
                abnormal_metrics = map_df(abnormal_metrics)

                # remove columns with all 100 percent missing values
                normal_metrics_new = normal_metrics_new.loc[
                    :, normal_metrics_new.columns[~normal_metrics_new.isna().all()]
                ]
                
                abnormal_metrics = abnormal_metrics.loc[
                    :, abnormal_metrics.columns[~abnormal_metrics.isna().all()]
                ]
                

                normal_metrics_new = flatten_columns_to_str(normal_metrics_new)
                abnormal_metrics   = flatten_columns_to_str(abnormal_metrics)


                # get the columns that appear in both normal and abnormal dataframes
                common_cols = normal_metrics_new.columns.intersection(abnormal_metrics.columns)
                normal_metrics_new = normal_metrics_new.loc[:, common_cols]
                abnormal_metrics = abnormal_metrics.loc[:, common_cols]
                
                

                # impute the data with median
                normal_metrics_new = normal_metrics_new.fillna(normal_metrics_new.median(numeric_only=True))
                abnormal_metrics = abnormal_metrics.fillna(abnormal_metrics.median(numeric_only=True))
                print("number of variables in normal metrics: ", normal_metrics_new.shape[1])

                df_combined = pd.concat([normal_metrics_new,abnormal_metrics], ignore_index=True)
                df_combined = df_combined.reset_index(drop=True)
                df_combined['time'] = np.arange(len(df_combined), dtype=int)


                
                if is_py312():
                    if model_name == "simplerca":
                        result = simplerca(normal_metrics_new,abnormal_metrics)
                        potential_root_causes = result
                    if model_name == "shapleyiq":
                        granular_graph = create_column_graph_from_causal_graph(list(abnormal_metrics.columns), causal_graph)
                        if target["target"]["metric"] is not None:
                            target_node = target["target"]["node"] + "_" + target["target"]["metric"] + "_" + target["target"]["agg"]
                        else:
                            target_node = target["target"]["node"] + "_" + target["target"]["agg"]
                        result = causal_shapleyiq_scaled(normal_metrics_new,abnormal_metrics, G=granular_graph, target = target_node)
                        potential_root_causes = result['ranks']
                    if model_name == "microdig":
                        granular_graph = create_column_graph_from_causal_graph(list(abnormal_metrics.columns), causal_graph)
                        if target["target"]["metric"] is not None:
                            target_node = target["target"]["node"] + "_" + target["target"]["metric"] + "_" + target["target"]["agg"]
                        else:
                            target_node = target["target"]["node"] + "_" + target["target"]["agg"]
                        result = microdig_service(normal_metrics_new,abnormal_metrics, G=granular_graph, issue_service = target_node)
                        potential_root_causes = result['ranks']


                    # BARO
                    if model_name == "baro":
                        from sklearn.preprocessing import RobustScaler
                        ranks = []
                        for col in normal_metrics_new.columns:
                            a = normal_metrics_new[col].to_numpy()
                            b = abnormal_metrics[col].to_numpy()

                            scaler = RobustScaler().fit(a.reshape(-1, 1))
                            zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
                            score = max(zscores)
                            ranks.append((col, score))

                        ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
                        ranks = [x[0] for x in ranks]
                        potential_root_causes = ranks
                    # CIRCA
                    if model_name == 'circa':
                        result = circa_petshop(df_combined, inject_time=normal_metrics_new.shape[0])
                        potential_root_causes = result['ranks']
                    # RCG
                    if model_name == 'rcg':

                        granular_graph = df_to_prefix_graph(normal_metrics_new, graph)
                        res = rcg_helper(normal_metrics_new,abnormal_metrics, graph=granular_graph)
                        potential_root_causes = res['ranks']
                    # CausalRCA
                    if model_name == 'causalrca':
                        result = causalrca_petshop(data=df_combined)
                        potential_root_causes = result['ranks']
    
                elif is_py38():
                    # RCD
                    if model_name == 'rcd':
                        result = rcd_helper(normal_metrics_new,abnormal_metrics)
                        potential_root_causes = result['ranks']                    # e-diagnosis
                    if model_name == 'e-diagnosis':
                        result =  e_diagnosis_petshop(normal_metrics_new,abnormal_metrics)
                        potential_root_causes = result['ranks']
    
                elif is_py310():
                    from graphical_models.classes.dags.dag import DAG
                    from graphical_models.classes.dags.pdag import PDAG
                    
                    
                    
                    # reverse the call graph to get the 'proxy' causal graph
                    # reference: https://github.com/amazon-science/RCAWithMissingStructuralKnowledgeCode/blob/6257f77efb708b655dd57844df271d66a0b13277/algorithms/petshop_root_cause_analysis_main/code/smooth_traversal.py#L57
                    # Create a directed graph with column names as nodes, based on causal_graph structure
                    granular_graph = create_column_graph_from_causal_graph(list(abnormal_metrics.columns), causal_graph)
                    if target["target"]["metric"] is not None:
                        target_node = target["target"]["node"] + "_" + target["target"]["metric"] + "_" + target["target"]["agg"]
                    else:
                        target_node = target["target"]["node"] + "_" + target["target"]["agg"]
                   
                    
                    if model_name == "brcd_CPDAG_prior":
                        G_cl = boss(normal_metrics_new.to_numpy())
                        arcs, edges = causal_learn_graph_to_nx_digraph(G_cl, list(normal_metrics_new.columns))
                        cpdag = PDAG(nodes=list(normal_metrics_new.columns), arcs=arcs, edges=edges)
                        result = brcd_helper(normal_metrics_new,abnormal_metrics ,
                                        cpdag=cpdag,
                                        isdiscrete=False,
                                        node_transform = "none",       # "none" | "log" | "log1p" | "yeojohnson"
                                        transform_parents= True,
                                        num_root_causes_candidates = 1)
                        potential_root_causes = result['ranks']
                    if model_name == 'brcd_without_CPDAG_prior':
                        result = brcd_helper(normal_metrics_new,abnormal_metrics ,
                                        cpdag=None,
                                        isdiscrete=False,
                                        node_transform = "none",       # "none" | "log" | "log1p" | "yeojohnson"
                                        transform_parents= True,
                                        num_root_causes_candidates = 1,
                                        bootstrap_samples = 10)
                        potential_root_causes = result['ranks']
                    # Smooth traversal
                    if model_name == 'smooth_traversal':
                        # the node_connected_subgraph_view() function inside will take an induced subgraph of the causal graph
                        # where the ancestors of the target node are included
                        scores = apply_smooth_traversal(granular_graph, target_node, normal_metrics_new, abnormal_metrics)
                        ranks = sorted(scores, key=scores.get, reverse=True)
                        #ranks = sorted(result.items(), key=lambda x: x[1], reverse=True)
                        #ranks = [x[0] for x in ranks]
                        potential_root_causes = ranks
                    if model_name == 'score_ordering':
                        scores = apply_score_ordering(normal_metrics_new, abnormal_metrics)
                        ranks = sorted(scores, key=scores.get, reverse=True)
                        potential_root_causes = ranks
                    if model_name == 'cholesky':
                        scores = apply_cholesky(normal_metrics_new,abnormal_metrics)
                        
                        #ranks = sorted(result.items(), key=lambda x: x[1], reverse=True)
                        #ranks = [x[0] for x in ranks]
                        potential_root_causes = ranks
                    if model_name == 'idint':
                        res = idint_helper(granular_graph, target_node, normal_metrics_new, abnormal_metrics)
                        potential_root_causes = res['ranks']
        
                        
                else:
                    print("Please use Python 3.8 or 3.10 or 3.12")
                    exit(1) 

                
                for k in results[scenario][split]:
                    correct = in_top_k(
                        potential_root_causes ,
                        target["root_cause"]["node"],
                        target["root_cause"]["metric"],
                        k
                        )
                    
                    row = {
                        "scenario": scenario,
                        "split": split,
                        "topk": k,
                        "metric": target["target"]["metric"],
                        "issue": idx,
                        "ground_truth": target["root_cause"]["node"],
                        "intopk": correct,
                        "empty": not potential_root_causes,
                    }

                    results[scenario][split][k].append(correct)
                    result_list.append(row)
    return pd.DataFrame(result_list)

def compute_proportions(df):
    """
    Compute the proportion of TRUE values in 'intopk' column grouped by 'scenario', 'metric', and 'topk'.
    
    Args:
        df (pd.DataFrame): Input DataFrame with columns 'scenario', 'metric', 'topk', and 'intopk'
        
    Returns:
        pd.DataFrame: DataFrame with proportions of TRUE values for each group
    """
    # Group by the specified columns and compute mean of 'intopk' (which gives proportion of TRUE)
    proportions = df.groupby(['scenario', 'metric', 'topk'])['intopk'].mean().reset_index()
    proportions.rename(columns={'intopk': 'proportion_true'}, inplace=True)
    return proportions

def format_results(df):
    """
    Format the results DataFrame by computing proportions and adding summary statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame from evaluate()
        
    Returns:
        pd.DataFrame: Formatted DataFrame with proportions and summary statistics
    """
    # Compute proportions
    proportions = compute_proportions(df)
    
    # Add count of total cases for each group
    counts = df.groupby(['scenario', 'metric', 'topk']).size().reset_index(name='total_cases')
    
    # Merge proportions and counts
    results = pd.merge(proportions, counts, on=['scenario', 'metric', 'topk'])
    
    # Add percentage column for easier reading
    results['percentage_true'] = (results['proportion_true'] * 100).round(2)
    
    return results

if __name__ == "__main__":
    models = ['brcd_CPDAG_prior', 'baro', 'simplerca', 'shapleyiq', 'microdig', 'circa', 'rcg', 'causalrca', 'e-diagnosis', 'rcd', 'smooth_traversal', 'score_ordering', 'cholesky', 'idint']

    # 'causalrca' throws errors on petshop data, circa pc steps takes too long to learn the causal graph

    #models = ['brcd_without_CPDAG_prior'] # run the bootstrapping version


    # Create results directory if it doesn't exist
    results_dir = 'petshop_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(results_dir, 'experiment_issues.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This keeps console output too
        ]
    )
    
    for model in models:
        if is_py312() and model not in ["baro", 'circa', 'rcg', 'causalrca', 'simplerca', 'shapleyiq', 'microdig']:
                continue
        elif is_py38() and model not in ["rcd", 'e-diagnosis']:
            continue
        elif is_py310() and model not in ["brcd_CPDAG_prior", 'brcd_without_CPDAG_prior', 'smooth_traversal', 'score_ordering', 'cholesky', 'idint']:
            continue
        
        df = evaluate(model, 'YOUR_PATH_TO_PETSHOP_DATASET')

        overall_mrr = compute_mrr_overall(df)
        print(df[["reciprocal_rank"]])
        print(f"{model} overall MRR: {overall_mrr:.4f}")

        
        # Save raw results
        df.to_csv(os.path.join(results_dir, f'{model}_results.csv'), index=False)

        mrr_by_scenario = compute_mrr_by_scenario(df)
        mrr_by_scenario.to_csv(os.path.join(results_dir, f'{model}_mrr_by_scenario.csv'), index=False)

        mrr_summary = pd.DataFrame([{
            "model": model,
            "overall_mrr": overall_mrr,
        }])
        mrr_summary.to_csv(os.path.join(results_dir, f'{model}_mrr_overall.csv'), index=False)

        
        # Process and format results
        formatted_results = format_results(df)
        
        # Save formatted results
        formatted_results.to_csv(os.path.join(results_dir, f'{model}_results_formatted.csv'), index=False)


                   