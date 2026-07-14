"""Quick test of service-map CPDAG BRCD approach on one case."""
import os, sys, json, time
import numpy as np
import pandas as pd
import networkx as nx
from graphical_models.classes.dags.pdag import PDAG

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

from RCAEval.e2e.brcd import brcd_helper

def flatten_columns_to_str(df):
    df = df.copy()
    df.columns = ["__".join(map(str, col)) if isinstance(col, tuple) else str(col) for col in df.columns]
    return df

def map_df(df):
    df_new = df.copy(deep=True)
    df_new.index.names = ["time"]
    columns = ["_".join([c[0], c[1], c[2]]) for c in df_new.columns]
    df_new.columns = columns
    return df_new

def create_column_graph_from_causal_graph(column_names, causal_graph):
    cols = [col for col in column_names if col != "time"]
    prefix_to_columns = {}
    for col in cols:
        name_parts = col.split("_")
        if len(name_parts) >= 2:
            prefix = "_".join(name_parts[:-2])
        else:
            prefix = col
        prefix_to_columns.setdefault(prefix, []).append(col)
    H = nx.DiGraph()
    H.add_nodes_from(cols)
    for u_prefix, v_prefix in causal_graph.edges:
        if u_prefix in prefix_to_columns and v_prefix in prefix_to_columns:
            for u_col in prefix_to_columns[u_prefix]:
                for v_col in prefix_to_columns[v_prefix]:
                    H.add_edge(u_col, v_col)
    return H

def create_pdag_from_graph(G_nx):
    nodes = list(G_nx.nodes())
    arcs = list(G_nx.edges())
    edges = []
    return PDAG(nodes=nodes, arcs=arcs, edges=edges)

path = "/datasets/petshop/dataset/low_traffic"
graph = nx.from_pandas_adjacency(
    pd.read_csv(os.path.join(path, "graph.csv"), index_col=0),
    create_using=nx.DiGraph,
)
causal_graph = graph.reverse()
normal_metrics = pd.read_csv(
    os.path.join(path, "noissue", "metrics.csv"), header=[0, 1, 2], index_col=0
)

# Use actual issue directory name
train_dir = os.path.join(path, "train")
issue_dir = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])[0]
issue_path = os.path.join(train_dir, issue_dir)

abnormal_metrics = pd.read_csv(
    os.path.join(issue_path, "metrics.csv"), header=[0, 1, 2], index_col=0
)
with open(os.path.join(issue_path, "target.json")) as f:
    target = json.load(f)

target_agg = target["target"]["agg"]
target_metric = target["target"]["metric"]
gt_node = target["root_cause"]["node"]

print(f"Target: agg={target_agg}, metric={target_metric}, root_cause={gt_node}")

normal_new = normal_metrics.loc[:, (slice(None), [target_metric], [target_agg])]
abnormal_new = abnormal_metrics.loc[:, (slice(None), [target_metric], [target_agg])]
normal_new = map_df(normal_new)
abnormal_new = map_df(abnormal_new)
normal_new = normal_new.loc[:, ~normal_new.isna().all()]
abnormal_new = abnormal_new.loc[:, ~abnormal_new.isna().all()]
normal_new = flatten_columns_to_str(normal_new)
abnormal_new = flatten_columns_to_str(abnormal_new)
common_cols = normal_new.columns.intersection(abnormal_new.columns)
normal_new = normal_new.loc[:, common_cols]
abnormal_new = abnormal_new.loc[:, common_cols]
normal_new = normal_new.fillna(normal_new.median(numeric_only=True))
abnormal_new = abnormal_new.fillna(abnormal_new.median(numeric_only=True))

print(f"Vars: {normal_new.shape[1]}")
print(f"Normal shape: {normal_new.shape}, Abnormal shape: {abnormal_new.shape}")

granular_graph = create_column_graph_from_causal_graph(
    list(abnormal_new.columns), causal_graph)
print(f"Granular graph: {granular_graph.number_of_nodes()} nodes, {granular_graph.number_of_edges()} edges")

# Check if granular graph has any nodes
if granular_graph.number_of_nodes() == 0:
    print("ERROR: Empty granular graph! Creating complete PDAG instead.")
    cpdag = PDAG(nodes=list(abnormal_new.columns), arcs=[], edges=[])
else:
    cpdag = create_pdag_from_graph(granular_graph)
print(f"CPDAG: {len(cpdag._nodes)} nodes, {len(cpdag._arcs)} arcs, {len(cpdag._edges)} edges")

print("Running BRCD...")
start = time.time()
result = brcd_helper(
    normal_new, abnormal_new,
    cpdag=cpdag,
    isdiscrete=False,
    node_transform="none",
    transform_parents=True,
    num_root_causes_candidates=1,
)
elapsed = time.time() - start
print(f"BRCD done in {elapsed:.1f}s")
print(f"Ranks (top 10): {result['ranks'][:10]}")

# Find ground truth rank
for idx, rc in enumerate(result['ranks']):
    parts = rc.split("_")
    if len(parts) >= 3:
        node = "_".join(parts[:-2])
    elif len(parts) >= 2:
        node = "_".join(parts[:-1])
    else:
        node = rc
    if node == gt_node:
        print(f"Ground truth '{gt_node}' found at rank {idx+1}")
        break
else:
    print(f"Ground truth '{gt_node}' NOT found in ranks!")
