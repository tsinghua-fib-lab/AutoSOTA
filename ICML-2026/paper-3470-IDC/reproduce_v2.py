#!/usr/bin/env python3
"""
Reproduction script for MuDo-CoM paper (paper 3470) - v2 with progress tracking.
Computes all three rubric metrics: d-MCC, MCC, Amari Distance.
"""
import argparse
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import time
import random
import os
import json
import sys
from numerical_data_generator import generate_data
from evaluation import compute_mcc_g, compute_mcc, amari_distance_rect


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_function_torch(A_hat, D_flat, sigma_vec, B, Cov_list, T, n):
    device = A_hat.device
    I = torch.eye(n, device=device)
    I_B_inv = torch.inverse(I - B)
    Sigma = torch.diag(sigma_vec)
    M = I_B_inv @ (Sigma @ Sigma) @ I_B_inv.T
    Cov_tensor = torch.stack([torch.tensor(c, dtype=torch.float32, device=device) for c in Cov_list])
    D_tensor = D_flat.view(T, n)
    dM_batch = D_tensor[:, :, None] * M[None, :, :] * D_tensor[:, None, :]
    ADAT_batch = A_hat @ dM_batch @ A_hat.T
    diff = Cov_tensor - ADAT_batch
    loss = torch.sum(diff ** 2)
    return loss


def run_torch_optimization(Cov_list, X_list, Z_list, A_true, T, n, xn,
                            num_steps=2000, lr=1e-3, report_every=5000):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    A_hat = nn.Parameter(torch.rand(xn, n, dtype=torch.float32, device=device))
    D_flat = nn.Parameter(torch.rand(n * T, dtype=torch.float32, device=device) * 0.8 + 0.2)
    sigma_hat_vec = nn.Parameter(torch.rand(n, dtype=torch.float32, device=device) * 0.8 + 0.2)
    B_free = nn.Parameter(torch.zeros((n, n), dtype=torch.float32, device=device))
    optimizer = Adam([A_hat, D_flat, sigma_hat_vec, B_free], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        B = B_free.clone()
        B.fill_diagonal_(0.0)
        loss = loss_function_torch(A_hat, D_flat, sigma_hat_vec, B, Cov_list, T, n)
        loss.backward()
        optimizer.step()

        if (step + 1) % report_every == 0 or step == 0:
            A_np = A_hat.detach().cpu().numpy()
            d_mcc, _, _ = compute_mcc_g(A_np, X_list, Z_list, n)
            amari = amari_distance_rect(A_true, A_np)
            mcc, _, _ = compute_mcc(A_np, X_list, Z_list, n)
            print(f"    Step {step+1}/{num_steps}: loss={loss.item():.4f}, "
                  f"d-MCC={d_mcc:.4f}, MCC={mcc:.4f}, Amari={amari:.4f}")
            sys.stdout.flush()

    A_hat_np = A_hat.detach().cpu().numpy()
    return A_hat_np


def estimate_parameters_multiple_initializations_torch(X_list, Z_list, A_true, T, n, xn,
                                                        num_steps=2000, lr=1e-3,
                                                        num_initializations=5):
    Cov_list = [np.cov(X_list[t], rowvar=False) for t in range(T)]
    best_loss = np.inf
    best_A_hat = None

    for i in range(num_initializations):
        print(f"  Init {i+1}/{num_initializations}:")
        A_hat = run_torch_optimization(
            Cov_list, X_list, Z_list, A_true, T, n, xn,
            num_steps=num_steps, lr=lr, report_every=max(1, num_steps // 10)
        )
        # Compute loss for selection
        A_t = torch.tensor(A_hat, dtype=torch.float32, device="cpu")
        D_t = torch.rand(n * T, dtype=torch.float32)
        sigma_t = torch.rand(n, dtype=torch.float32)
        B_t = torch.zeros((n, n), dtype=torch.float32)
        Cov_tensor = torch.stack([torch.tensor(c, dtype=torch.float32) for c in Cov_list])
        I = torch.eye(n)
        I_B_inv = torch.inverse(I - B_t)
        Sigma = torch.diag(sigma_t)
        M = I_B_inv @ (Sigma @ Sigma) @ I_B_inv.T
        D_tensor = D_t.view(T, n)
        dM_batch = D_tensor[:, :, None] * M[None, :, :] * D_tensor[:, None, :]
        ADAT_batch = A_t @ dM_batch @ A_t.T
        diff = Cov_tensor - ADAT_batch
        final_loss = torch.sum(diff ** 2).item()
        print(f"    Final loss: {final_loss:.4f}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_A_hat = A_hat

    return best_A_hat


def main(args):
    xn_actual = int(args.x_n * args.n)
    T_actual = int(args.T * args.n)

    print(f"=== MuDo-CoM Reproduction v2 ===")
    print(f"n={args.n}, xn_actual={xn_actual}, T_actual={T_actual}")
    print(f"N={args.N}, k={args.graph_dense}, noise={args.noise_type}")
    print(f"steps={args.num_steps}, inits={args.num_initializations}, lr={args.lr}")
    print(f"seeds={args.seeds}")
    print(f"===============================\n")
    sys.stdout.flush()

    results = {"d_mcc": [], "mcc": [], "amari": [], "runtime": []}
    os.makedirs(args.save_dir, exist_ok=True)

    for seed in args.seeds:
        set_seed(seed)
        t0 = time.perf_counter()
        print(f"--- Seed {seed} ---")
        sys.stdout.flush()

        X_list, A_true, B_true, Ds, sigma_vec, Z_list, skeleton = generate_data(
            T=T_actual, n=args.n, xn=xn_actual, N=args.N,
            graph_dense=args.graph_dense, graph_type=args.graph_type,
            noise_type=args.noise_type, nn_nonlinear=args.nonlinear
        )

        A_hat = estimate_parameters_multiple_initializations_torch(
            X_list, Z_list, A_true, T_actual, args.n, xn_actual,
            num_steps=args.num_steps, lr=args.lr,
            num_initializations=args.num_initializations
        )

        d_mcc, d_mcc_list, _ = compute_mcc_g(A_hat, X_list, Z_list, args.n)
        mcc, cor_abs, _ = compute_mcc(A_hat, X_list, Z_list, args.n)
        amari = amari_distance_rect(A_true, A_hat)
        elapsed = time.perf_counter() - t0

        print(f"  => d-MCC={d_mcc:.6f}, MCC={mcc:.6f}, Amari={amari:.6f}, Time={elapsed:.1f}s")
        sys.stdout.flush()

        results["d_mcc"].append(d_mcc)
        results["mcc"].append(mcc)
        results["amari"].append(amari)
        results["runtime"].append(elapsed)

        seed_dir = os.path.join(args.save_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        np.savez(os.path.join(seed_dir, "results.npz"),
                 A_hat=A_hat, A_true=A_true,
                 d_mcc=d_mcc, mcc=mcc, amari=amari)

    # Summary
    d_mcc_mean, d_mcc_std = np.mean(results["d_mcc"]), np.std(results["d_mcc"])
    mcc_mean, mcc_std = np.mean(results["mcc"]), np.std(results["mcc"])
    amari_mean, amari_std = np.mean(results["amari"]), np.std(results["amari"])
    total_time = np.sum(results["runtime"])

    print(f"\n=== Summary ({len(args.seeds)} seeds) ===")
    print(f"d-MCC: {d_mcc_mean:.6f} +/- {d_mcc_std:.6f}")
    print(f"MCC:   {mcc_mean:.6f} +/- {mcc_std:.6f}")
    print(f"Amari: {amari_mean:.6f} +/- {amari_std:.6f}")
    print(f"Time: {total_time:.1f}s")

    summary = {
        "config": {"n": args.n, "xn": args.x_n, "xn_actual": xn_actual,
                   "T": args.T, "T_actual": T_actual, "N": args.N,
                   "graph_dense": args.graph_dense, "noise_type": args.noise_type,
                   "num_steps": args.num_steps, "num_initializations": args.num_initializations,
                   "lr": args.lr, "seeds": args.seeds},
        "results": {
            "d_mcc": {"mean": float(d_mcc_mean), "std": float(d_mcc_std),
                      "values": [float(v) for v in results["d_mcc"]]},
            "mcc": {"mean": float(mcc_mean), "std": float(mcc_std),
                    "values": [float(v) for v in results["mcc"]]},
            "amari": {"mean": float(amari_mean), "std": float(amari_std),
                      "values": [float(v) for v in results["amari"]]},
            "total_runtime_sec": float(total_time)
        }
    }
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {args.save_dir}/summary.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuDo-CoM Reproduction v2")
    parser.add_argument("--T", type=int, default=2)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--x-n", type=int, default=1)
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num-steps", type=int, default=50000)
    parser.add_argument("--num-initializations", type=int, default=3)
    parser.add_argument("--graph-dense", type=float, default=2.0)
    parser.add_argument("--graph-type", type=str, default="ER")
    parser.add_argument("--noise-type", type=str, default="gauss")
    parser.add_argument("--nonlinear", action="store_true")
    parser.add_argument("--save-dir", type=str, default="/repo/outputs/reproduction_v2")
    args = parser.parse_args()
    main(args)
