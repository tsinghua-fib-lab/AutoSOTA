#!/usr/bin/env python3
"""
Reproduction evaluation for paper-2414: CNT-GW on HORSES benchmark.

Gromov-Wasserstein at Scale, Beyond Squared Norms
https://github.com/guillaumeHoury/egw-solvers

Paper settings: N=4000, eps=1e-3, D=E=20, 100 Sinkhorn iterations,
float32, Euclidean norm R^3 cost, normalization radius=1,
stopping criterion: objective decrease < 1e-5.
"""

import os, sys, time, json, logging
import torch

# Ensure CUDA env for KeOps GPU compilation
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.1")
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda-12.1")
if "/usr/local/cuda-12.1/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/usr/local/cuda-12.1/bin:" + os.environ.get("PATH", "")

logging.basicConfig(encoding="utf-8", level=logging.WARNING)

import solvers
from utils.data.meshes.load import load_pointcloud_from_mesh
from utils.math.costs import euclidean
from utils.math.functions import normalize

def main():
    N = 4000
    eps = 1e-3
    approx_dims = 20
    sinkhorn_iters = 40
    stop_thr = 1e-5

    # Load and normalize shapes
    X = normalize(load_pointcloud_from_mesh("data/muybridge_014_01.ply", N=N))
    Y = normalize(load_pointcloud_from_mesh("data/00049424_ferrari.ply", N=N))

    cost = lambda u, v: euclidean(u, v, p=1)  # Euclidean norm R^3

    ARGS = {
        "eps": eps,
        "numItermax": 25,
        "stop_criterion": "energy",
        "stopThr": stop_thr,
        "sink_progressive_start": 15,
        "SINK_ARGS": {"numItermax": sinkhorn_iters, "symmetrize": True},
    }

    torch.cuda.synchronize()
    t0 = time.time()
    solver = solvers.CntGW(X, Y, cost, approx_dims=approx_dims, **ARGS)
    solver.solve(verbose=False)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    gw_eps = solver.loss(approx=False, include_divergence=True).item()

    results = {
        "method": "CNT-GW",
        "benchmark": "HORSES",
        "n_points": N,
        "temperature": eps,
        "embedding_dimension": approx_dims,
        "sinkhorn_iterations": sinkhorn_iters,
        "cost_function": "euclidean_norm_R3",
        "time_seconds": round(elapsed, 2),
        "gw_eps": round(gw_eps, 6),
    }

    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    main()
