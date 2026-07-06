#!/usr/bin/env python3
"""
Reproduction experiment for DiCoLA paper.
Generates ER(50,3) graphs with 5 latent variables,
runs FCI and DiCoLA+FCI, computes metrics.
"""
import sys
sys.path.insert(0, "/repo")

import json
import time
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
from pathlib import Path

from DiCoLa.Recursive_PAG import DiCola_learner
from compare_algs.fci_alg import my_fci
from DiCoLa.utils import f1_score_edges


def create_er_dag(n_nodes, expected_degree, seed):
    """Create a random DAG using Erdos-Renyi model."""
    rng = np.random.RandomState(seed)
    p = expected_degree / (n_nodes - 1)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.rand() < p:
                dag.add_edge(i, j)
    perm = rng.permutation(n_nodes)
    mapping = {i: int(perm[i]) for i in range(n_nodes)}
    dag = nx.relabel_nodes(dag, mapping)
    return dag


def ensure_connected_dag(n_nodes, expected_degree, seed, max_attempts=10000):
    for attempt in range(max_attempts):
        local_seed = seed + attempt * 10000
        dag = create_er_dag(n_nodes, expected_degree, local_seed)
        if nx.is_weakly_connected(dag):
            return dag, local_seed
    raise RuntimeError(f"Could not generate connected DAG after {max_attempts} attempts")


def select_latent_variables(dag, n_latent):
    """Select latent variables: parentless nodes with >=2 children."""
    candidates = []
    for node in dag.nodes():
        if dag.in_degree(node) == 0 and dag.out_degree(node) >= 2:
            candidates.append(node)
    if len(candidates) < n_latent:
        for node in dag.nodes():
            if dag.in_degree(node) == 0 and node not in candidates:
                candidates.append(node)
    if len(candidates) < n_latent:
        nodes_by_parents = sorted(dag.nodes(), key=lambda n: dag.in_degree(n))
        for node in nodes_by_parents:
            if node not in candidates:
                candidates.append(node)
            if len(candidates) >= n_latent:
                break
    rng = np.random.RandomState(abs(hash(f"latent_{id(dag)}")) % (2**31))
    selected = list(rng.choice(candidates, size=min(n_latent, len(candidates)), replace=False))
    return selected


def generate_linear_sem(dag, n_samples, seed):
    """Generate data from linear SEM with Gaussian noise."""
    rng = np.random.RandomState(seed)
    n_nodes = dag.number_of_nodes()
    data = rng.randn(n_samples, n_nodes)
    topo_order = list(nx.topological_sort(dag))
    for node in topo_order:
        parents = list(dag.predecessors(node))
        for parent in parents:
            weight_sign = 1 if rng.rand() < 0.5 else -1
            weight = weight_sign * rng.uniform(0.5, 2.0)
            data[:, node] += weight * data[:, parent]
    return data


def compute_true_pag_skeleton(dag, observed_nodes, latent_nodes, max_depth=4):
    """
    Compute true PAG skeleton efficiently.
    Key insight: direct DAG edges and latent-confounded pairs are GUARANTEED
    to be in the PAG (cannot be d-separated by observed variables).
    Only test separation for marginally dependent non-guaranteed pairs.
    """
    obs_set = set(observed_nodes)
    obs_list = sorted(observed_nodes)
    n_obs = len(obs_list)
    node_to_idx = {node: i for i, node in enumerate(obs_list)}
    latent_set = set(latent_nodes)

    # Initialize adjacency matrix
    adj = np.zeros((n_obs, n_obs), dtype=int)

    # Track which edges are guaranteed (no need to test separation)
    guaranteed = np.zeros((n_obs, n_obs), dtype=bool)

    # 1. Guaranteed edges: direct DAG adjacency
    for i, j in combinations(range(n_obs), 2):
        x, y = obs_list[i], obs_list[j]
        if dag.has_edge(x, y) or dag.has_edge(y, x):
            adj[i, j] = adj[j, i] = 1
            guaranteed[i, j] = guaranteed[j, i] = True

    # 2. Guaranteed edges: share latent parent
    for i, j in combinations(range(n_obs), 2):
        if guaranteed[i, j]:
            continue
        x, y = obs_list[i], obs_list[j]
        common_pred = set(dag.predecessors(x)) & set(dag.predecessors(y))
        if common_pred & latent_set:
            adj[i, j] = adj[j, i] = 1
            guaranteed[i, j] = guaranteed[j, i] = True

    # 3. For NON-guaranteed pairs, check marginal d-separation
    # Build candidate sets from the known adjacency structure
    for i, j in combinations(range(n_obs), 2):
        if adj[i, j]:
            continue  # Already an edge
        x, y = obs_list[i], obs_list[j]

        # Quick check: marginal d-separation (empty conditioning set)
        # If marginally independent, definitely no edge in PAG
        if nx.d_separated(dag, {x}, {y}, set()):
            continue  # No edge

        # Marginally dependent → potential PAG edge
        # Build candidate separating set: nodes on short paths between x and y
        # Use the undirected neighbors of x and y in the moral graph
        candidates = set()
        # Neighbors in DAG (parents + children)
        candidates.update(dag.predecessors(x), dag.successors(x))
        candidates.update(dag.predecessors(y), dag.successors(y))
        # Common children (moral graph edges)
        for n in obs_set:
            if n == x or n == y:
                continue
            if (dag.has_edge(x, n) or dag.has_edge(n, x)) and (dag.has_edge(y, n) or dag.has_edge(n, y)):
                # Common child/co-parent → moral edge
                pass  # Already covered by successors
        candidates &= obs_set
        candidates -= {x, y}
        cand_list = list(candidates)

        # Try to find a separating set
        found_sepset = False
        for depth in range(min(max_depth + 1, len(cand_list) + 1)):
            if found_sepset:
                break
            for sepset in combinations(cand_list, depth):
                if nx.d_separated(dag, {x}, {y}, set(sepset)):
                    found_sepset = True
                    break

        if not found_sepset:
            adj[i, j] = adj[j, i] = 1

    return pd.DataFrame(adj, index=obs_list, columns=obs_list)


def run_single_trial(seed, n_obs=50, n_latent=5, expected_degree=3, n_samples=2000, alpha=0.01):
    n_total = n_obs + n_latent
    trial_seed = seed * 1000

    dag, _ = ensure_connected_dag(n_total, expected_degree, trial_seed)
    latent = select_latent_variables(dag, n_latent)
    observed = sorted(set(dag.nodes()) - set(latent))

    data_array = generate_linear_sem(dag, n_samples, trial_seed + 1)
    all_node_names = [str(n) for n in range(n_total)]
    full_data = pd.DataFrame(data_array, columns=all_node_names)

    latent_names = [str(n) for n in latent]
    observed_data = full_data.drop(columns=latent_names)
    obs_names = [str(n) for n in observed]
    observed_data = observed_data[obs_names]

    t0_truth = time.time()
    true_pag = compute_true_pag_skeleton(dag, observed, latent, max_depth=4)
    truth_time = time.time() - t0_truth

    t0 = time.time()
    res_fci = my_fci(data=observed_data, alpha=alpha)
    fci_runtime = time.time() - t0

    t0 = time.time()
    res_dicola = DiCola_learner(
        observed_data=observed_data,
        leaf_node_learner=my_fci,
        alpha=alpha,
        ci_type="Fisher_Z",
        min_leaf_size=12,
        max_recursion_depth=5
    )
    dicola_runtime = time.time() - t0

    fci_scores = f1_score_edges(true_pag, res_fci["PAG.DataFrame"])
    dicola_scores = f1_score_edges(true_pag, res_dicola["PAG.DataFrame"])

    return {
        "trial": seed,
        "n_true_edges": int(true_pag.values.sum() / 2),
        "truth_time": truth_time,
        "fci": {
            "CI_num": res_fci["CI_num"],
            "runtime_sec": fci_runtime,
            "precision": fci_scores["precision"],
            "recall": fci_scores["recall"],
            "f1": fci_scores["f1"],
        },
        "dicola": {
            "CI_num": res_dicola["CI_num"],
            "runtime_sec": dicola_runtime,
            "precision": dicola_scores["precision"],
            "recall": dicola_scores["recall"],
            "f1": dicola_scores["f1"],
        },
    }


def main():
    n_runs = 50
    n_obs = 50
    n_latent = 5
    expected_degree = 3
    n_samples = 2000
    alpha = 0.01

    print(f"Starting ER({n_obs},{expected_degree}) experiment:")
    print(f"  n_variables={n_obs}, n_latent={n_latent}, n_total={n_obs + n_latent}")
    print(f"  sample_size={n_samples}, n_runs={n_runs}, alpha={alpha}")
    print(f"  {'='*60}")

    all_results = []
    fci_ci, fci_time, fci_prec, fci_rec, fci_f1 = [], [], [], [], []
    dicola_ci, dicola_time, dicola_prec, dicola_rec, dicola_f1 = [], [], [], [], []

    total_start = time.time()
    for run_id in range(1, n_runs + 1):
        t0 = time.time()
        print(f"Trial {run_id}/{n_runs}...", end=" ", flush=True)
        result = run_single_trial(run_id, n_obs, n_latent, expected_degree, n_samples, alpha)
        elapsed = time.time() - t0
        all_results.append(result)

        fci_ci.append(result["fci"]["CI_num"])
        fci_time.append(result["fci"]["runtime_sec"])
        fci_prec.append(result["fci"]["precision"])
        fci_rec.append(result["fci"]["recall"])
        fci_f1.append(result["fci"]["f1"])

        dicola_ci.append(result["dicola"]["CI_num"])
        dicola_time.append(result["dicola"]["runtime_sec"])
        dicola_prec.append(result["dicola"]["precision"])
        dicola_rec.append(result["dicola"]["recall"])
        dicola_f1.append(result["dicola"]["f1"])

        print(f"({elapsed:.1f}s) FCI: CI={result['fci']['CI_num']:.0f} P={result['fci']['precision']:.3f} R={result['fci']['recall']:.3f} F1={result['fci']['f1']:.3f} | DiCoLA: CI={result['dicola']['CI_num']:.0f} P={result['dicola']['precision']:.3f} R={result['dicola']['recall']:.3f} F1={result['dicola']['f1']:.3f}")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"SUMMARY (n_runs={n_runs}, total_time={total_elapsed:.1f}s)")
    print(f"{'='*60}")

    summary = {
        "experiment": f"ER({n_obs},{expected_degree})",
        "n_variables": n_obs,
        "n_latent": n_latent,
        "sample_size": n_samples,
        "n_runs": n_runs,
        "alpha": alpha,
        "total_time_sec": round(total_elapsed, 1),
        "DiCoLA+FCI": {
            "CI_Tests": round(np.mean(dicola_ci), 2),
            "CI_Tests_std": round(np.std(dicola_ci), 2),
            "Time": round(np.mean(dicola_time), 2),
            "Time_std": round(np.std(dicola_time), 2),
            "Precision": round(np.mean(dicola_prec), 2),
            "Precision_std": round(np.std(dicola_prec), 2),
            "Recall": round(np.mean(dicola_rec), 2),
            "Recall_std": round(np.std(dicola_rec), 2),
            "F1": round(np.mean(dicola_f1), 2),
            "F1_std": round(np.std(dicola_f1), 2),
        },
        "FCI_baseline": {
            "CI_Tests": round(np.mean(fci_ci), 2),
            "CI_Tests_std": round(np.std(fci_ci), 2),
            "Time": round(np.mean(fci_time), 2),
            "Time_std": round(np.std(fci_time), 2),
            "Precision": round(np.mean(fci_prec), 2),
            "Precision_std": round(np.std(fci_prec), 2),
            "Recall": round(np.mean(fci_rec), 2),
            "Recall_std": round(np.std(fci_rec), 2),
            "F1": round(np.mean(fci_f1), 2),
            "F1_std": round(np.std(fci_f1), 2),
        },
    }

    print(json.dumps(summary, indent=2))

    output_dir = Path("/repo/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "er_experiment_results.json", "w") as f:
        json.dump({"summary": summary, "trials": all_results}, f, indent=2)

    print(f"Results saved to /repo/outputs/er_experiment_results.json")
    return summary


if __name__ == "__main__":
    main()
