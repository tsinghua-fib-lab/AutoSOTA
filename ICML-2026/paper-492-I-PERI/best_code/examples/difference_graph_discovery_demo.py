"""
Experiment to evaluate DifferencePC (difference-oriented PC algorithm).

- Generates a random DAG.
- Creates two linear SCMs from the same DAG but perturbs some edges/weights
  so that not all relations are identical between the two populations.
- Runs DifferencePC to learn a CompletedPartiallyDirectedAcyclicDifferenceGraph.
- Compares estimated difference edges to the true set of changed edges.
- Repeats the experiment multiple times and reports mean/variance of metrics.

Usage:
    python3 examples/difference_graph_discovery_demo.py

This script is intentionally lightweight (uses few permutations by default)
so it runs quickly as a check. Increase `N_REPEATS` and `N_PERMUTATIONS` for
more stable estimates.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Set

# Add src to path when running as a script (make local imports resolvable)
import sys
ROOT = Path(__file__).resolve().parent
SRC = str(ROOT.parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pyciphod.utils.graphs.graphs import create_random_dag, AcyclicDirectedMixedGraph
from pyciphod.causal_discovery.difference.difference_constraint_based import LinearDifferencePC
from pyciphod.utils.scms.scm import create_random_linear_scm_from_dag, LinearSCM
from pyciphod.utils.stat_tests.equality_tests import LinearRegressionCoefficientEqualityTest
from pyciphod.utils.stat_tests.equality_tests import GComputationEqualityTest
from pyciphod.utils.stat_tests.equality_tests import PartialCorrelationEqualityTest


def make_two_scms_from_dag(dag: AcyclicDirectedMixedGraph, n_samples: int = 1000, frac_change: float = 0.2, seed: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, Set[Tuple[str, str]]]:
    """
    Given a DAG, simulate two SCM datasets. Perturb a fraction `frac_change` of
    existing edges (flip sign or resample weight) to introduce differences.

    Returns (df1, df2, changed_edges_set) where changed_edges_set contains edge tuples (u,v)
    that differ between SCM1 and SCM2.
    """
    # Use package helper to sample a random linear SCM for the given DAG
    scm0, coefficients0, intercepts0 = create_random_linear_scm_from_dag(dag, seed=seed)
    dg = scm0.induced_dag()
    dg.draw_graph()
    # produce data1
    df1 = scm0.generate_data(n_samples, include_latent=False, seed=seed)

    # Determine present directed edges (observed->observed) using DAG API
    present_edges = list(sorted(dag.get_directed_edges()))
    if len(present_edges) == 0:
        return df1.copy(), df1.copy(), set()

    # Build coefficients for SCM1 (copy) and SCM2 (perturbed)
    # coeff1 = dict(coefficients0)
    coeff2 = dict(coefficients0)

    rng = np.random.default_rng(seed)
    n_change = max(1, int(len(present_edges) * frac_change))
    chosen_idx = rng.choice(len(present_edges), size=n_change, replace=False)
    changed_edges = set()
    for idx in chosen_idx:
        u, v = present_edges[idx]
        # perturb only the observed->observed coefficient
        key = (u, v)
        if key not in coeff2:
            # if for some reason missing, skip
            continue
        # if rng.random() < 0.5:
        print(coeff2[key])
        if abs(coeff2[key]) > 0.1:
            coeff2[key] = -coeff2[key] * 2
        else:
            coeff2[key] = 0.5
        print(coeff2[key])
        # else:
        #     coeff2[key] = float(rng.uniform(-1.0, 1.0))
        changed_edges.add(key)

    # Create second LinearSCM using same observed and latent variables and intercepts
    scm2 = LinearSCM(v=list(scm0.v), u=list(scm0.u), coefficients=coeff2, intercepts=intercepts0, u_dist=None)
    df2 = scm2.generate_data(n_samples, include_latent=False, seed=seed+1 if seed is not None else None)

    return df1.copy(), df2, changed_edges


def undirected_edge_set(graph) -> Set[Tuple[str, str]]:
    """Return undirected adjacency set where each edge is a sorted tuple."""
    edges = set()
    for u in graph.get_vertices():
        for v in graph.get_adjacencies(u):
            # treat adjacency regardless of orientation
            edges.add(tuple(sorted((u, v))))
    return edges


def evaluate_estimate(est_graph, true_changed_edges: Set[Tuple[str, str]]):
    """Compute precision, recall, f1 w.r.t. undirected changed edges."""
    est_edges = undirected_edge_set(est_graph)
    true_set = set(tuple(sorted(e)) for e in true_changed_edges)

    tp = len([e for e in est_edges if e in true_set])
    fp = len([e for e in est_edges if e not in true_set])
    fn = len([e for e in true_set if e not in est_edges])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def run_single_trial(num_vars=10, edge_prob=0.2, n_samples=500, frac_change=0.2, n_permutations=200, seed: int = None):
    rng = np.random.default_rng(seed)

    dag = create_random_dag(num_vars, p_edge=edge_prob, seed=seed)
    df1, df2, true_changed = make_two_scms_from_dag(dag, n_samples=n_samples, frac_change=frac_change, seed=seed)

    print(true_changed)
    # Run DifferencePC
    dp = LinearDifferencePC(sparsity=0.01, n_permutations=n_permutations, seed=seed, eq_test=LinearRegressionCoefficientEqualityTest)
    try:
        dp.run(df1, df2)
    except Exception as e:
        print("DifferencePC run failed:", e)
        return None
    metrics = evaluate_estimate(dp.g_hat, true_changed)
    metrics['nb_tests'] = dp.nb_ci_tests
    return metrics


def run_experiments(n_repeats=10, num_vars=10, edge_prob=0.2, n_samples=500, frac_change=0.2, n_permutations=200):
    results = []
    for i in range(n_repeats):
        seed = int(1000 + i)
        print(f"Running trial {i+1}/{n_repeats} (seed={seed})")
        res = run_single_trial(num_vars=num_vars, edge_prob=edge_prob, n_samples=n_samples, frac_change=frac_change, n_permutations=n_permutations, seed=seed)
        if res is not None:
            res['trial'] = i
            results.append(res)
            print(res)

    df = pd.DataFrame(results)
    summary = df.agg(['mean', 'std'])[['precision', 'recall', 'f1', 'nb_tests']]
    print('\nSummary over trials:')
    print(summary)
    return df, summary


if __name__ == '__main__':
    R, S = run_experiments(n_repeats=5, num_vars=10, edge_prob=0.25, n_samples=1000, frac_change=0.025, n_permutations=100)
