from RCAEval.e2e.IDI_release.methods.idint import make_idint
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

def idint_helper(graph, target_node, normal_metrics, abnormal_metrics):
    analyze_root_causes = make_idint()
    rc_scores = analyze_root_causes(graph, target_node, normal_metrics, abnormal_metrics)
    return {
        'ranks':[x.node for x in rc_scores]
    }

def idint(data, inject_time=None, dataset=None, graph=None, target_node=None,**kwargs):
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
    print(f"target_node: {target_node}")
    # remove nodes from graph if
    granular_graph = df_to_prefix_graph(normal_df, graph)
    
    return idint_helper(granular_graph, target_node, normal_df, anomal_df)