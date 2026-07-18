#!/usr/bin/env python3
"""
Compute orbit frequency distance (graphlet-based MMD) between reference
and generated graphs.

Uses ORCA (4-node graphlets, 15 orbits) to count per-node orbit frequencies,
then computes MMD with a Gaussian-TV kernel (sigma=30) -- same protocol as
GRAN / GraphRNN / EDGE papers.

Usage examples
--------------
# CELL polblogs (adjacency matrices in samples.pkl)
python compute_orbit_distance.py \
    --reference ../SyNGLER/datasets/polblogs/generator/seed=0.npy \
    --generated ../runs/cell/polblogs/official_demo_gpu/samples.pkl

# HiGen (list of networkx graphs)
python compute_orbit_distance.py \
    --reference ../SyNGLER/datasets/polblogs/generator/seed=0.npy \
    --generated ../runs/higen/polblogs/.../gen_graphs_epoch_100.pickle.xz

# ILDG (dict with pred_graphs)
python compute_orbit_distance.py \
    --reference ../SyNGLER/datasets/polblogs/generator/seed=0.npy \
    --generated ../runs/ildg/polblogs/.../test/step_1000.pkl
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
from scipy.sparse import issparse
import concurrent.futures
from functools import partial
from scipy.linalg import toeplitz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ORCA_BIN = SCRIPT_DIR / "orca" / "orca"

# ---------------------------------------------------------------------------
# MMD kernel helpers  (adapted from GRAN dist_helper.py)
# ---------------------------------------------------------------------------

def gaussian_tv(x, y, sigma=1.0):
    """Gaussian kernel with total-variation distance."""
    support_size = max(len(x), len(y))
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < support_size:
        x = np.hstack((x, np.zeros(support_size - len(x))))
    if len(y) < support_size:
        y = np.hstack((y, np.zeros(support_size - len(y))))
    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2.0 * sigma * sigma))


def disc(samples1, samples2, kernel, is_parallel=True, **kwargs):
    """Compute discrepancy (mean kernel value) between two sample sets."""
    d = 0.0
    if not is_parallel or len(samples1) * len(samples2) < 100:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, **kwargs)
    else:
        def _worker(args):
            s1, s2_list = args
            return sum(kernel(s1, s2, **kwargs) for s2 in s2_list)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for val in executor.map(_worker, [(s1, samples2) for s1 in samples1]):
                d += val
    d /= len(samples1) * len(samples2)
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, **kwargs):
    """MMD^2 between two collections of feature vectors / histograms."""
    if is_hist:
        samples1 = [s / (np.sum(s) + 1e-30) for s in samples1]
        samples2 = [s / (np.sum(s) + 1e-30) for s in samples2]
    return (disc(samples1, samples1, kernel, **kwargs)
            + disc(samples2, samples2, kernel, **kwargs)
            - 2.0 * disc(samples1, samples2, kernel, **kwargs))


# ---------------------------------------------------------------------------
# ORCA wrapper
# ---------------------------------------------------------------------------

def _edge_list_reindexed(G: nx.Graph):
    """Return edge list with nodes re-indexed to 0..N-1."""
    node_to_idx = {u: i for i, u in enumerate(G.nodes())}
    return [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]


def run_orca(graph: nx.Graph, graphlet_size: int = 4) -> np.ndarray:
    """
    Run ORCA on a single networkx graph.
    Returns node_orbit_counts: array of shape (num_nodes, num_orbits).
      - graphlet_size=4 -> 15 orbits
      - graphlet_size=5 -> 73 orbits
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot run ORCA on empty graph")

    # Handle isolated nodes: ORCA needs at least one edge.
    # If graph has no edges, return zeros.
    if graph.number_of_edges() == 0:
        n_orbits = 15 if graphlet_size == 4 else 73
        return np.zeros((graph.number_of_nodes(), n_orbits), dtype=np.int64)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp') as f_in:
        tmp_input = f_in.name
        f_in.write(f"{graph.number_of_nodes()} {graph.number_of_edges()}\n")
        for u, v in _edge_list_reindexed(graph):
            f_in.write(f"{u} {v}\n")

    try:
        output = subprocess.check_output(
            [str(ORCA_BIN), 'node', str(graphlet_size), tmp_input, 'std'],
            stderr=subprocess.STDOUT,
            timeout=300,
        )
        output = output.decode('utf8').strip()

        # Parse: find "orbit counts:" marker
        COUNT_START_STR = 'orbit counts:'
        idx = output.find(COUNT_START_STR)
        if idx < 0:
            raise RuntimeError(f"ORCA output missing '{COUNT_START_STR}': {output[:200]}")
        idx += len(COUNT_START_STR)
        output = output[idx:].strip()

        node_orbit_counts = np.array([
            list(map(int, line.strip().split()))
            for line in output.strip().split('\n')
            if line.strip()
        ], dtype=np.int64)

        return node_orbit_counts

    finally:
        try:
            os.remove(tmp_input)
        except OSError:
            pass


def orbit_feature(G: nx.Graph, graphlet_size: int = 4) -> np.ndarray:
    """
    Compute per-graph orbit feature: sum of per-node orbit counts,
    normalized by number of nodes, then log-transformed.

    The log-transform (log(x+1)) is critical for large graphs: raw orbit
    counts can reach ~100k, making TV distances ~26000 and the Gaussian
    kernel exp(-d^2/2sigma^2) collapse to 0.  Log-transform brings values
    to ~[0, 12], keeping the kernel in its sensitive range.

    Returns 1D array of length num_orbits.
    """
    orbit_counts = run_orca(G, graphlet_size)
    feat = orbit_counts.sum(axis=0).astype(np.float64) / G.number_of_nodes()
    # Log-transform to prevent kernel saturation on large graphs
    feat = np.log(feat + 1.0)
    return feat


# ---------------------------------------------------------------------------
# orbit_stats_all: the standard metric
# ---------------------------------------------------------------------------

def orbit_stats_all(
    graph_ref_list: List[nx.Graph],
    graph_pred_list: List[nx.Graph],
    graphlet_size: int = 4,
    sigma: float = 30.0,
) -> float:
    """
    Compute orbit MMD between reference and predicted graph lists.
    Same protocol as GRAN's orbit_stats_all.
    """
    total_ref = []
    total_pred = []

    for i, G in enumerate(graph_ref_list):
        try:
            feat = orbit_feature(G, graphlet_size)
            total_ref.append(feat)
        except Exception as e:
            print(f"  [warn] ref graph {i} failed: {e}")
            continue

    for i, G in enumerate(graph_pred_list):
        if G.number_of_nodes() == 0:
            continue
        try:
            feat = orbit_feature(G, graphlet_size)
            total_pred.append(feat)
        except Exception as e:
            print(f"  [warn] pred graph {i} failed: {e}")
            continue

    if not total_ref or not total_pred:
        print("  [error] No valid orbit features computed!")
        return float('nan')

    total_ref = [np.asarray(x, dtype=np.float64) for x in total_ref]
    total_pred = [np.asarray(x, dtype=np.float64) for x in total_pred]

    mmd = compute_mmd(total_ref, total_pred, kernel=gaussian_tv, is_hist=False, sigma=sigma)
    return mmd


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------

def _load_pickle(path: Path):
    if path.suffix == '.xz' or str(path).endswith('.pickle.xz'):
        with lzma.open(path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)


def adj_to_nx(adj: np.ndarray) -> nx.Graph:
    """Convert adjacency matrix to undirected nx.Graph."""
    if issparse(adj):
        adj = adj.toarray()
    adj = np.asarray(adj)
    # Symmetrize and remove self-loops
    adj = np.logical_or(adj, adj.T).astype(float)
    np.fill_diagonal(adj, 0)
    return nx.from_numpy_array(adj)


def load_graphs(path: Path) -> List[nx.Graph]:
    """
    Load graphs from various formats:
      - directory of .npy: each file is one adjacency (release format)
      - .npy: single (n, n) or batch (B, n, n) of adjacency matrices
      - .pkl / .pickle / .pickle.xz: list of adj matrices, list of nx graphs,
        or dict with 'pred_graphs'
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    # Directory of per-sample .npy files (release default for
    # runs/<method>/r=<r>/seed=<S>/samples/rep*.npy)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix == '.npy')
        if not files:
            raise ValueError(f"No .npy files in directory: {path}")
        graphs: List[nx.Graph] = []
        for f in files:
            arr = np.load(f, allow_pickle=True)
            if arr.ndim == 2:
                graphs.append(adj_to_nx(arr))
            elif arr.ndim == 3:
                graphs.extend(adj_to_nx(arr[i]) for i in range(len(arr)))
            else:
                raise ValueError(f"Unexpected .npy ndim={arr.ndim} in {f}")
        return graphs

    if path.suffix == '.npy':
        payload = np.load(path, allow_pickle=True)
        if payload.ndim == 2:
            return [adj_to_nx(payload)]
        elif payload.ndim == 3:
            return [adj_to_nx(payload[i]) for i in range(len(payload))]
        else:
            raise ValueError(f"Unexpected .npy ndim={payload.ndim}")

    # pickle-based
    payload = _load_pickle(path)

    if isinstance(payload, dict):
        if 'pred_graphs' in payload:
            payload = payload['pred_graphs']
        else:
            raise ValueError(f"Dict payload with unknown keys: {sorted(payload.keys())}")

    if isinstance(payload, (list, tuple)):
        if len(payload) == 0:
            raise ValueError("Empty payload")

        # Check first element
        first = payload[0]
        if isinstance(first, nx.Graph):
            return list(payload)
        elif isinstance(first, np.ndarray):
            return [adj_to_nx(a) for a in payload]
        elif issparse(first):
            return [adj_to_nx(a) for a in payload]
        else:
            raise ValueError(f"Unsupported element type: {type(first)}")

    if isinstance(payload, np.ndarray):
        if payload.ndim == 2:
            return [adj_to_nx(payload)]
        elif payload.ndim == 3:
            return [adj_to_nx(payload[i]) for i in range(len(payload))]

    raise ValueError(f"Cannot interpret payload of type {type(payload)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute orbit frequency distance (graphlet MMD).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reference", type=Path, required=True,
        help="Reference graph(s): .npy adjacency, .pkl list, etc.",
    )
    parser.add_argument(
        "--generated", type=Path, required=True,
        help="Generated graph(s): .pkl / .npy / .pickle.xz",
    )
    parser.add_argument(
        "--graphlet-size", type=int, default=4, choices=[4, 5],
        help="Graphlet size for ORCA (4 = 15 orbits, 5 = 73 orbits). Default: 4.",
    )
    parser.add_argument(
        "--sigma", type=float, default=30.0,
        help="Sigma for Gaussian-TV kernel in MMD. Default: 30.0.",
    )
    parser.add_argument(
        "--max-graphs", type=int, default=None,
        help="Max number of generated graphs to evaluate (for speed).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    # Verify ORCA binary
    if not ORCA_BIN.exists():
        print(f"ERROR: ORCA binary not found at {ORCA_BIN}")
        print("Compile it: cd orbit_eval/orca && g++ -O2 -std=c++11 -o orca orca.cpp")
        sys.exit(1)

    print(f"Loading reference graphs from: {args.reference}")
    ref_graphs = load_graphs(args.reference)
    print(f"  -> {len(ref_graphs)} reference graph(s), nodes: {[g.number_of_nodes() for g in ref_graphs[:5]]}")

    print(f"Loading generated graphs from: {args.generated}")
    gen_graphs = load_graphs(args.generated)
    print(f"  -> {len(gen_graphs)} generated graph(s), nodes: {[g.number_of_nodes() for g in gen_graphs[:5]]}")

    if args.max_graphs and len(gen_graphs) > args.max_graphs:
        print(f"  Subsampling to {args.max_graphs} generated graphs")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(gen_graphs), args.max_graphs, replace=False)
        gen_graphs = [gen_graphs[i] for i in sorted(idx)]

    print(f"\nComputing orbit features (graphlet_size={args.graphlet_size}) ...")
    t0 = time.time()
    mmd = orbit_stats_all(ref_graphs, gen_graphs, graphlet_size=args.graphlet_size, sigma=args.sigma)
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"Orbit MMD (graphlet_size={args.graphlet_size}, sigma={args.sigma}): {mmd:.6f}")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*50}")

    result = {
        "metric": "orbit_mmd",
        "graphlet_size": args.graphlet_size,
        "sigma": args.sigma,
        "value": float(mmd),
        "num_ref_graphs": len(ref_graphs),
        "num_gen_graphs": len(gen_graphs),
        "elapsed_seconds": round(elapsed, 1),
        "reference": str(args.reference),
        "generated": str(args.generated),
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {args.out}")

    return result


if __name__ == "__main__":
    main()
