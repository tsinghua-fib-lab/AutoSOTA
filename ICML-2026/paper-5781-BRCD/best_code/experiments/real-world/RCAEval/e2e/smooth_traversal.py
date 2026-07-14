# ./algorithms/smooth_traversal.py
"""Function for running the SMOOTH TRAVERSAL algorithm from the paper. 
Code written by Patrick Blöbaum, William Roy Orchard.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. 
SPDX-License-Identifier: Apache-2.0

source: https://github.com/amazon-science/RCAWithMissingStructuralKnowledgeCode/blob/main/algorithms/smooth_traversal.py
"""

from typing import Callable, Dict
import random

import networkx as nx
import numpy as np
import pandas as pd

from dowhy.gcm import RescaledMedianCDFQuantileScorer
from dowhy.gcm.anomaly_scorer import AnomalyScorer
from dowhy.graph import node_connected_subgraph_view, is_root_node


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



from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    preprocess,
    select_useful_cols,
)


def apply_smooth_traversal(graph: nx.DiGraph,
                           target_node: str,
                           normal_data: pd.DataFrame,
                           anomaly_data: pd.DataFrame,
                           anomaly_scorer: Callable[[], AnomalyScorer] = RescaledMedianCDFQuantileScorer,
                           debug: bool = True) -> Dict[str, float]:
    """
    This is the implementation of the smooth traversal algorithm (algorithm 1 on the paper)
    """
    graph = node_connected_subgraph_view(graph, target_node)

    if anomaly_data.shape[0] > 1:
        anomaly_data = anomaly_data.iloc[[0]]

    all_scores = {}

    for node in graph.nodes:
        tmp_anomaly_scorer = anomaly_scorer()
        tmp_anomaly_scorer.fit(normal_data[node].to_numpy())
        tmp_score = tmp_anomaly_scorer.score(anomaly_data[node].to_numpy())

        if debug:
            print(f"Anomaly score of {node} is {tmp_score.squeeze()}")

        all_scores[node] = tmp_score.squeeze()

    score_gaps = {}

    for node in graph.nodes:
        if is_root_node(graph, node):
            score_gaps[node] = float(all_scores[node])
        else:
            score_gaps[node] = max(0, np.min([(all_scores[node] - all_scores[parent]) for parent in graph.predecessors(node)]))
    
    return score_gaps

def smooth_traversal(
    data, inject_time=None, dataset=None, graph=None, target_node=None,**kwargs
):
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

    # remove nodes from graph if
    granular_graph = df_to_prefix_graph(normal_df, graph)
    


    result =  apply_smooth_traversal(granular_graph,
                           target_node,
                           normal_df,
                           anomal_df)

    ranks = sorted(result.items(), key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }