"""BOSS-CPDAG BRCD on low_traffic train cases for comparison."""
import os, sys, json, time
import numpy as np
import pandas as pd
import networkx as nx

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

from causallearn.graph.Endpoint import Endpoint
from graphical_models.classes.dags.pdag import PDAG
from RCAEval.e2e.brcd import brcd_helper
from RCAEval.e2e.BRCD.boss import boss

def causal_learn_graph_to_nx_digraph(G_cl, column_names):
    G_nx = nx.DiGraph()
    id_to_col = {i: name for i, name in enumerate(column_names)}
    for node in G_cl.get_nodes():
        node_id = G_cl.node_map[node]
        G_nx.add_node(id_to_col[node_id])
    arcs, edges = [], []
    for edge in G_cl.get_graph_edges():
        n1 = id_to_col[G_cl.node_map[edge.node1]]
        n2 = id_to_col[G_cl.node_map[edge.node2]]
        ep1, ep2 = edge.endpoint1, edge.endpoint2
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            arcs.append((n1, n2))
        elif ep2 == Endpoint.TAIL and ep1 == Endpoint.ARROW:
            arcs.append((n2, n1))
        elif ep2 == Endpoint.TAIL and ep1 == Endpoint.TAIL:
            edges.append((n2, n1))
    return arcs, edges

def map_df(df):
    df_new = df.copy(deep=True)
    df_new.index.names = ["time"]
    columns = ["_".join([c[0], c[1], c[2]]) for c in df_new.columns]
    df_new.columns = columns
    return df_new

def flatten_columns_to_str(df):
    df = df.copy()
    df.columns = ["__".join(map(str, col)) if isinstance(col, tuple) else str(col) for col in df.columns]
    return df

def get_rank(potential_root_causes, ground_truth_node, ground_truth_metric=None):
    for idx, root_cause in enumerate(potential_root_causes):
        name = root_cause.split("_")
        if len(name) >= 3:
            metric = name[-2]
            node = "_".join(name[:-2])
        elif len(name) >= 2:
            node = "_".join(name[:-1])
        else:
            node = root_cause
        if node == ground_truth_node:
            if ground_truth_metric is None or ground_truth_metric == metric:
                return idx + 1
    return float("inf")

DATASET_PATH = "/datasets/petshop/dataset"
scenario = "low_traffic"
path = os.path.join(DATASET_PATH, scenario)

graph = nx.from_pandas_adjacency(
    pd.read_csv(os.path.join(path, "graph.csv"), index_col=0),
    create_using=nx.DiGraph,
)
normal_metrics = pd.read_csv(
    os.path.join(path, "noissue", "metrics.csv"), header=[0, 1, 2], index_col=0
)

train_dir = os.path.join(path, "train")
issue_dirs = sorted([d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d))])

results = []
total_start = time.time()

for issue_dir in issue_dirs:
    issue_path = os.path.join(train_dir, issue_dir)
    print(f"\n=== {scenario}/train/{issue_dir} ===")

    abnormal_metrics = pd.read_csv(
        os.path.join(issue_path, "metrics.csv"), header=[0, 1, 2], index_col=0
    )
    with open(os.path.join(issue_path, "target.json")) as f:
        target = json.load(f)

    target_agg = target["target"]["agg"]
    target_metric = target["target"]["metric"]
    gt_node = target["root_cause"]["node"]

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

    print(f"vars={normal_new.shape[1]}")

    print("Running BOSS...")
    start = time.time()
    G_cl = boss(normal_new.to_numpy(), verbose=False)
    boss_time = time.time() - start
    print(f"BOSS done in {boss_time:.1f}s")

    arcs, edges = causal_learn_graph_to_nx_digraph(G_cl, list(normal_new.columns))
    cpdag = PDAG(nodes=list(normal_new.columns), arcs=arcs, edges=edges)
    print(f"CPDAG: {len(arcs)} arcs, {len(edges)} edges")

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
    brcd_time = time.time() - start
    print(f"BRCD done in {brcd_time:.1f}s")

    rank = get_rank(result["ranks"], gt_node)
    rr = 1.0 / rank if rank != float("inf") else 0.0
    print(f"rank={rank} gt={gt_node} rr={rr:.4f}")
    results.append({"issue": issue_dir, "rank": rank, "rr": rr})

total_time = time.time() - total_start
print(f"\n=== BOSS-CPDAG Results for {scenario}/train (total time: {total_time/60:.1f}min) ===")
top1 = sum(1 for r in results if r["rank"] <= 1) / len(results)
top3 = sum(1 for r in results if r["rank"] <= 3) / len(results)
top5 = sum(1 for r in results if r["rank"] <= 5) / len(results)
mrr = sum(r["rr"] for r in results) / len(results)
print(f"Top-1: {top1:.4f}")
print(f"Top-3: {top3:.4f}")
print(f"Top-5: {top5:.4f}")
print(f"MRR:   {mrr:.4f}")
