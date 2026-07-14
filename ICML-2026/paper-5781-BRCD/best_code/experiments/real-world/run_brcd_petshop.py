"""Complete BRCD evaluation on Petshop dataset."""
import os, sys, json, logging
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

os.environ["OPENBLAS_NUM_THREADS"] = "1"
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

def flatten_columns_to_str(df, sep="__"):
    df = df.copy()
    df.columns = [
        sep.join(map(str, col)) if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    return df

def map_df(df):
    df_new = df.copy(deep=True)
    df_new.index.names = ["time"]
    columns = ["_".join([c[0], c[1], c[2]]) for c in df_new.columns]
    df_new.columns = columns
    return df_new

def get_rank(potential_root_causes, ground_truth_node, ground_truth_metric=None):
    for idx, root_cause in enumerate(potential_root_causes):
        name = root_cause.split("_")
        if len(name) >= 3:
            metric = name[-2]
            node = "_".join(name[:-2])
        elif len(name) >= 2:
            metric = None
            node = "_".join(name[:-1])
        else:
            node = root_cause
            metric = None
        if node == ground_truth_node:
            if ground_truth_metric is None or ground_truth_metric == metric:
                return idx + 1
    return float('inf')

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
        split_path = os.path.join(path, split)
        if not os.path.exists(split_path):
            continue
        for issue in sorted(os.listdir(split_path)):
            if issue.startswith("."):
                continue
            issue_path = os.path.join(split_path, issue)
            if not os.path.isdir(issue_path):
                continue
            try:
                metrics = pd.read_csv(
                    os.path.join(issue_path, "metrics.csv"),
                    header=[0, 1, 2],
                    index_col=0,
                )
                with open(os.path.join(issue_path, "target.json"), "r") as f:
                    target = json.load(f)
                issues[split].append((metrics, target))
            except Exception as e:
                print(f"  WARNING: Could not load {issue_path}: {e}")
    return graph, normal_metrics, issues

DATASET_PATH = "/datasets/petshop/dataset"
scenarios = ["low_traffic", "high_traffic", "temporal_traffic1", "temporal_traffic2"]

all_rows = []

for scenario in scenarios:
    scenario_path = os.path.join(DATASET_PATH, scenario)
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario}")
    print(f"{'='*60}")
    
    try:
        graph, normal_metrics, issues = load_scenario(scenario_path)
    except Exception as e:
        print(f"  ERROR loading scenario: {e}")
        continue
    
    causal_graph = graph.reverse()
    
    for split in ["train", "test"]:
        if not issues[split]:
            continue
        
        for idx, (abnormal_metrics, target) in enumerate(issues[split]):
            print(f"  [{scenario}/{split}/issue{idx}] ", end="", flush=True)
            
            statistic_of_interest = target["target"]["agg"]
            issue_metric = target["target"]["metric"]
            
            try:
                normal_new = normal_metrics.loc[:, (slice(None), [issue_metric], [statistic_of_interest])]
                abnormal_filtered = abnormal_metrics.loc[:, (slice(None), [issue_metric], [statistic_of_interest])]
            except KeyError:
                normal_new = normal_metrics.copy()
                abnormal_filtered = abnormal_metrics.copy()
            
            normal_new = map_df(normal_new)
            abnormal_filtered = map_df(abnormal_filtered)
            
            normal_new = normal_new.loc[:, ~normal_new.isna().all()]
            abnormal_filtered = abnormal_filtered.loc[:, ~abnormal_filtered.isna().all()]
            
            normal_new = flatten_columns_to_str(normal_new)
            abnormal_filtered = flatten_columns_to_str(abnormal_filtered)
            
            common_cols = normal_new.columns.intersection(abnormal_filtered.columns)
            normal_new = normal_new.loc[:, common_cols]
            abnormal_filtered = abnormal_filtered.loc[:, common_cols]
            
            normal_new = normal_new.fillna(normal_new.median(numeric_only=True))
            abnormal_filtered = abnormal_filtered.fillna(abnormal_filtered.median(numeric_only=True))
            
            print(f"vars={normal_new.shape[1]} ", end="", flush=True)
            
            try:
                G_cl = boss(normal_new.to_numpy())
                arcs, edges = causal_learn_graph_to_nx_digraph(G_cl, list(normal_new.columns))
                cpdag = PDAG(nodes=list(normal_new.columns), arcs=arcs, edges=edges)
                
                result = brcd_helper(
                    normal_new, abnormal_filtered,
                    cpdag=cpdag,
                    isdiscrete=False,
                    node_transform="none",
                    transform_parents=True,
                    num_root_causes_candidates=1,
                )
                potential_root_causes = result['ranks']
            except Exception as e:
                print(f"BRCD FAILED: {e}")
                potential_root_causes = []
            
            gt_node = target["root_cause"]["node"]
            gt_metric = target["root_cause"].get("metric", None)
            
            if potential_root_causes:
                rank = get_rank(potential_root_causes, gt_node, gt_metric)
                rr = 1.0 / rank if rank != float('inf') else 0.0
                
                row = {
                    "scenario": scenario,
                    "split": split,
                    "issue": idx,
                    "metric": issue_metric,
                    "ground_truth": gt_node,
                    "rank": rank,
                    "reciprocal_rank": rr,
                    "top1": int(rank <= 1),
                    "top3": int(rank <= 3),
                    "top5": int(rank <= 5),
                    "empty": False,
                }
            else:
                row = {
                    "scenario": scenario,
                    "split": split,
                    "issue": idx,
                    "metric": issue_metric,
                    "ground_truth": gt_node,
                    "rank": float('inf'),
                    "reciprocal_rank": 0.0,
                    "top1": 0,
                    "top3": 0,
                    "top5": 0,
                    "empty": True,
                }
            
            print(f"rank={rank} gt={gt_node} rr={rr:.4f}")
            all_rows.append(row)

df = pd.DataFrame(all_rows)
print(f"\n{'='*60}")
print(f"Total cases evaluated: {len(df)}")
print(f"{'='*60}")

top1 = df["top1"].mean()
top3 = df["top3"].mean()
top5 = df["top5"].mean()
mrr = df["reciprocal_rank"].mean()

print(f"\nBRCD Petshop Results:")
print(f"  Top-1: {top1:.4f}")
print(f"  Top-3: {top3:.4f}")
print(f"  Top-5: {top5:.4f}")
print(f"  MRR:   {mrr:.4f}")

import os as _os
_os.makedirs("petshop_results", exist_ok=True)
df.to_csv("petshop_results/brcd_detailed_results.csv", index=False)

summary = pd.DataFrame([{
    "model": "BRCD (CPDAG prior)",
    "top1": top1,
    "top3": top3,
    "top5": top5,
    "mrr": mrr,
    "n_cases": len(df),
}])
summary.to_csv("petshop_results/brcd_summary.csv", index=False)
print(f"\nResults saved to petshop_results/")
