"""BRCD evaluation on Petshop using service map as CPDAG (no BOSS)."""
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
    df.columns = [
        "__".join(map(str, col)) if isinstance(col, tuple) else str(col)
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
        candidates = [root_cause] if isinstance(root_cause, str) else list(root_cause)
        for rc in candidates:
            name = rc.split("_")
            if len(name) >= 3:
                metric = name[-2]
                node = "_".join(name[:-2])
            elif len(name) >= 2:
                metric = None
                node = "_".join(name[:-1])
            else:
                node = rc
                metric = None
            if node == ground_truth_node:
                if ground_truth_metric is None or ground_truth_metric == metric:
                    return idx + 1
    return float('inf')

def create_column_graph_from_causal_graph(column_names, causal_graph):
    """Create directed graph with column names as nodes from prefix-based causal graph."""
    cols = [col for col in column_names if col != 'time']
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
    """Create a PDAG from a networkx DiGraph (treating it as a DAG)."""
    nodes = list(G_nx.nodes())
    arcs = list(G_nx.edges())
    edges = []
    return PDAG(nodes=nodes, arcs=arcs, edges=edges)

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
                print(f"  WARNING: {issue_path}: {e}")
    return graph, normal_metrics, issues

DATASET_PATH = "/datasets/petshop/dataset"
scenarios = ["low_traffic", "high_traffic", "temporal_traffic1", "temporal_traffic2"]
OUTPUT_DIR = "petshop_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_rows = []
total_start = time.time()

for scenario in scenarios:
    scenario_path = os.path.join(DATASET_PATH, scenario)
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario}")
    print(f"{'='*60}")

    try:
        graph, normal_metrics, issues = load_scenario(scenario_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    causal_graph = graph.reverse()

    for split in ["train", "test"]:
        if not issues[split]:
            continue

        for idx, (abnormal_metrics, target) in enumerate(issues[split]):
            case_start = time.time()
            print(f"  [{scenario}/{split}/issue{idx}] ", end="", flush=True)

            try:
                target_agg = target["target"]["agg"]
                target_metric = target["target"]["metric"]
            except KeyError as e:
                print(f"SKIP: {e}")
                continue

            try:
                normal_new = normal_metrics.loc[:, (slice(None), [target_metric], [target_agg])]
                abnormal_filtered = abnormal_metrics.loc[:, (slice(None), [target_metric], [target_agg])]
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

            n_vars = normal_new.shape[1]
            print(f"vars={n_vars} ", end="", flush=True)

            # Use service map directly as CPDAG (BRCD-C approach from paper)
            try:
                granular_graph = create_column_graph_from_causal_graph(
                    list(abnormal_filtered.columns), causal_graph)
                cpdag = create_pdag_from_graph(granular_graph)

                result = brcd_helper(
                    normal_new, abnormal_filtered,
                    cpdag=cpdag,
                    isdiscrete=False,
                    node_transform="none",
                    transform_parents=True,
                    num_root_causes_candidates=3,
                )
                potential_root_causes = result['ranks']
            except Exception as e:
                print(f"FAILED: {e}")
                potential_root_causes = []

            gt_node = target["root_cause"]["node"]
            gt_metric = target["root_cause"].get("metric", None)

            if potential_root_causes:
                rank = get_rank(potential_root_causes, gt_node, gt_metric)
                rr = 1.0 / rank if rank != float('inf') else 0.0
            else:
                rank = float('inf')
                rr = 0.0

            elapsed = time.time() - case_start
            print(f"rank={rank} gt={gt_node} rr={rr:.4f} [{elapsed:.0f}s]")

            all_rows.append({
                "scenario": scenario, "split": split, "issue": idx,
                "metric": target_metric, "ground_truth": gt_node,
                "rank": rank, "reciprocal_rank": rr,
                "top1": int(rank <= 1), "top3": int(rank <= 3), "top5": int(rank <= 5),
                "empty": int(len(potential_root_causes) == 0),
            })

df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(OUTPUT_DIR, "brcd_detailed_results.csv"), index=False)

total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"Total cases: {len(df)}, Time: {total_elapsed/60:.1f}min")
print(f"{'='*60}")

top1 = df["top1"].mean()
top3 = df["top3"].mean()
top5 = df["top5"].mean()
mrr = df["reciprocal_rank"].mean()

print(f"\nBRCD Petshop Results (service-map CPDAG):")
print(f"  Top-1: {top1:.4f}")
print(f"  Top-3: {top3:.4f}")
print(f"  Top-5: {top5:.4f}")
print(f"  MRR:   {mrr:.4f}")

summary = pd.DataFrame([{
    "model": "BRCD (service-map CPDAG)", "top1": top1, "top3": top3,
    "top5": top5, "mrr": mrr, "n_cases": len(df),
}])
summary.to_csv(os.path.join(OUTPUT_DIR, "brcd_summary.csv"), index=False)
print(f"Results saved to {OUTPUT_DIR}/")
