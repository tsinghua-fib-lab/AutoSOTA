#!/usr/bin/env python3
"""Evaluate SyNG-R generated graphs on PolBlogs against paper metrics.
Usage: python scripts/eval_polblogs.py [--samples_dir PATH] [--output PATH]
"""
import argparse, json, time, sys
import numpy as np
import torch
from pathlib import Path
from syngler.evaluation.metrics import (
    triangle_density, global_clustering_coefficient,
    degree_centrality, eigenvalues, compute_mmd,
)
from scipy.stats import ks_2samp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", default="runs/syngler/polblogs_r2/syngr/samples")
    ap.add_argument("--ref_adj", default="data/real/polblogs/generator/seed=0.npy")
    ap.add_argument("--output", default="runs/syngler/polblogs_r2/eval_results.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    A_ref = np.load(args.ref_adj).astype(np.float32)
    n, n_edges = A_ref.shape[0], int(A_ref.sum() // 2)
    print(f"Ref: {n} nodes, {n_edges} edges")
    ref_t = torch.from_numpy(A_ref).unsqueeze(0)
    
    files = sorted(Path(args.samples_dir).glob("rep*.npy"))
    N = len(files)
    print(f"Samples: {N}")
    gen_arrs = np.stack([np.load(f).astype(np.float32) for f in files])
    gen_t = torch.from_numpy(gen_arrs)
    
    t0 = time.time()
    
    # Tri
    ref_td = float(triangle_density(ref_t, device=device, return_numpy=True).ravel()[0])
    gen_td = triangle_density(gen_t, device=device, return_numpy=True).ravel()
    tri_rmse = float(np.sqrt(np.mean((gen_td - ref_td) ** 2)))
    print(f"Tri: ref={ref_td:.8f} gen_mean={gen_td.mean():.8f} RMSE={tri_rmse:.8f}")
    
    # Clus
    ref_gcc = float(global_clustering_coefficient(ref_t, device=device, return_numpy=True).ravel()[0])
    gen_gcc = global_clustering_coefficient(gen_t, device=device, return_numpy=True).ravel()
    clus_rmse = float(np.sqrt(np.mean((gen_gcc - ref_gcc) ** 2)))
    print(f"Clus: ref={ref_gcc:.8f} gen_mean={gen_gcc.mean():.8f} RMSE={clus_rmse:.8f}")
    
    # DegC
    ref_degc = degree_centrality(ref_t, device=device).cpu().numpy().ravel()
    gen_degc = degree_centrality(gen_t, device=device).cpu().numpy().ravel()
    deg_ks = float(ks_2samp(ref_degc, gen_degc).statistic)
    print(f"DegC KS: {deg_ks:.8f}")
    
    # Eig
    ref_eig = eigenvalues(ref_t, device="cpu").cpu().numpy().ravel()
    BATCH = 5
    parts = []
    for i in range(0, N, BATCH):
        eb = eigenvalues(gen_t[i:i+BATCH], device="cpu").cpu().numpy()
        parts.append(eb.ravel())
    gen_eig = np.concatenate(parts)
    rng = np.random.default_rng(42)
    S = 5000
    ref_sub = rng.choice(ref_eig, size=min(S, len(ref_eig)), replace=False)
    gen_sub = rng.choice(gen_eig, size=min(S, len(gen_eig)), replace=False)
    eig_mmd = float(compute_mmd(gen_sub, ref_sub))
    print(f"Eig MMD: {eig_mmd:.8f}")
    
    elapsed = time.time() - t0
    
    results = {
        "Tri_RMSE": tri_rmse, "Clus_RMSE": clus_rmse,
        "DegC_KS": deg_ks, "Eig_MMD": eig_mmd,
        "Tri_RMSE_x1e4": tri_rmse * 1e4, "Clus_RMSE_x1e2": clus_rmse * 1e2,
        "DegC_KS_x1e2": deg_ks * 1e2, "Eig_MMD_x1e2": eig_mmd * 1e2,
        "ref_td": ref_td, "gen_td_mean": float(gen_td.mean()),
        "ref_gcc": ref_gcc, "gen_gcc_mean": float(gen_gcc.mean()),
        "n_samples": N, "n_nodes": n, "ref_edges": n_edges,
        "eval_time_s": elapsed,
    }
    
    print(f"\nResults:")
    for k in ["Tri_RMSE","Clus_RMSE","DegC_KS","Eig_MMD"]:
        print(f"  {k}: {results[k]:.8f}")
    print(f"Time: {elapsed:.1f}s")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
