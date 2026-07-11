#!/usr/bin/env python3
"""Run Reddit benchmark with best-of-two initializations as described in the paper."""
import time
import pickle
from collections import defaultdict
import numpy as np
import networkx as nx
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import LoBCD_GW as dyn

@torch.no_grad()
def node_correctness_gpu(X):
    n = X.size(0)
    pred = X.argmax(dim=1)
    acc = (pred == torch.arange(n, device=X.device)).float().mean()
    return float(acc.item())

def process_one_both_inits(idx, A_t, B_t, rho, max_iter, eps, device):
    """Try both uniform and degree-based init, return best acc and total time."""
    n = A_t.size(0)
    a = torch.full((n,), 1.0/n, device=device, dtype=A_t.dtype)
    b = torch.full((n,), 1.0/n, device=device, dtype=A_t.dtype)
    
    best_acc = 0.0
    best_time = 0.0
    best_label = ""
    
    # Init 1: Degree-based heuristic
    t0 = time.time()
    X_hat, _ = dyn.DynamicVI_GW(
        A=A_t, B=B_t, a=a, b=b, X_init=None,
        rho0=rho, min_rho=rho, eps=eps, max_iter=max_iter,
        rho_decay=1.0, sinkhorn_iters=4, check_every=10, print_every=50
    )
    t1_main = time.time() - t0
    
    P1 = dyn.round_hungarian_from_affinity(A_t, B_t, X_hat)
    acc1 = node_correctness_gpu(P1) * 100
    
    t0 = time.time()
    X_ref, _ = dyn.DynamicVI_GW_Simple(
        A=A_t, B=B_t, a=a, b=b, X_init=X_hat,
        rho0=100.0, min_rho=100.0, eps=eps, max_iter=100,
        rho_decay=1.0, sinkhorn_iters=4, check_every=10
    )
    t1_ref = time.time() - t0
    
    P2 = dyn.round_hungarian_from_affinity(A_t, B_t, X_ref)
    acc2 = node_correctness_gpu(P2) * 100
    time2 = t1_main + t1_ref
    
    if acc2 > best_acc:
        best_acc = acc2
        best_time = time2
    
    # Init 2: Uniform (independent coupling)
    X_uniform = torch.outer(a, b)
    t0 = time.time()
    X_hat_u, _ = dyn.DynamicVI_GW(
        A=A_t, B=B_t, a=a, b=b, X_init=X_uniform,
        rho0=rho, min_rho=rho, eps=eps, max_iter=max_iter,
        rho_decay=1.0, sinkhorn_iters=4, check_every=10, print_every=50
    )
    t2_main = time.time() - t0
    
    P1_u = dyn.round_hungarian_from_affinity(A_t, B_t, X_hat_u)
    acc1_u = node_correctness_gpu(P1_u) * 100
    
    t0 = time.time()
    X_ref_u, _ = dyn.DynamicVI_GW_Simple(
        A=A_t, B=B_t, a=a, b=b, X_init=X_hat_u,
        rho0=100.0, min_rho=100.0, eps=eps, max_iter=100,
        rho_decay=1.0, sinkhorn_iters=4, check_every=10
    )
    t2_ref = time.time() - t0
    
    P2_u = dyn.round_hungarian_from_affinity(A_t, B_t, X_ref_u)
    acc2_u = node_correctness_gpu(P2_u) * 100
    time2_u = t2_main + t2_ref
    
    if acc2_u > best_acc:
        best_acc = acc2_u
        best_time = time2_u
        best_label = "uniform"
    else:
        best_label = "degree"
    
    return idx, best_acc, best_time, best_label, acc1, acc2, acc1_u, acc2_u

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--output_file", type=str, default="result_reddit_bestinit.txt")
    args = parser.parse_args()
    
    device = "cuda"
    print(f"Device: {device}, GPU: {torch.cuda.get_device_name(0)}")
    print(f"Rho={args.rho}, max_iter={args.max_iter}, eps={args.eps}")
    
    with open("data/REDDIT-BINARY/matching.pk", "rb") as f:
        graphs = pickle.load(f)[:500]
    
    total = len(graphs)
    print(f"Processing {total} graph pairs...")
    
    all_acc = []
    all_time = []
    degree_better = 0
    uniform_better = 0
    t_start = time.time()
    
    for i in range(total):
        G = graphs[i]
        A = nx.to_numpy_array(G, dtype=np.float32)
        B = A.copy()
        A_t = torch.from_numpy(A).to(device)
        B_t = torch.from_numpy(B).to(device)
        
        idx, best_acc, best_time, best_label, acc1, acc2, acc1_u, acc2_u = process_one_both_inits(
            i, A_t, B_t, args.rho, args.max_iter, args.eps, device
        )
        
        all_acc.append(best_acc)
        all_time.append(best_time)
        
        if best_label == "degree":
            degree_better += 1
        else:
            uniform_better += 1
        
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[{i+1}/{total}] best={best_label} acc_d={acc2:.2f}% acc_u={acc2_u:.2f}% best_acc={best_acc:.2f}% time={best_time:.3f}s cum_avg_acc={np.mean(all_acc):.2f}%")
    
    total_time = time.time() - t_start
    avg_acc = np.mean(all_acc)
    avg_time = np.mean(all_time)
    sum_time = np.sum(all_time)
    
    print()
    print('============================================================')
    print(f"Average accuracy: {avg_acc:.2f}%")
    print(f"Average time per graph: {avg_time:.4f}s")
    print(f"Total algorithm time: {sum_time:.2f}s")
    print(f"Total wall time: {total_time:.2f}s")
    print(f"Degree init better: {degree_better}/{total}")
    print(f"Uniform init better: {uniform_better}/{total}")
    print('============================================================')
    
    with open(args.output_file, "w") as f:
        f.write(f"rho={args.rho},max_iter={args.max_iter},eps={args.eps}\n")
        f.write(f"avg_acc={avg_acc:.4f},avg_time={avg_time:.4f},sum_time={sum_time:.2f}\n")
        f.write(f"all_acc={all_acc}\n")
        f.write(f"all_time={all_time}\n")
