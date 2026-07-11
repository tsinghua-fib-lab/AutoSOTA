#!/usr/bin/env python3
"""
Optimized reproduction script for MuDo-CoM paper (paper 3470).
Precomputes covariance tensor for faster iterations.
"""
import argparse, numpy as np, torch, time, random, os, json, sys
from torch import nn
from torch.optim import Adam
from numerical_data_generator import generate_data
from evaluation import compute_mcc_g, compute_mcc, amari_distance_rect


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_optimization(Cov_tensor, X_list, Z_list, A_true, T, n, xn,
                     num_steps=50000, lr=5e-3, report_every=5000):
    device = Cov_tensor.device
    A_hat = nn.Parameter(torch.rand(xn, n, dtype=torch.float32, device=device))
    D_flat = nn.Parameter(torch.rand(n * T, dtype=torch.float32, device=device) * 0.8 + 0.2)
    sigma_vec = nn.Parameter(torch.rand(n, dtype=torch.float32, device=device) * 0.8 + 0.2)
    B_free = nn.Parameter(torch.zeros((n, n), dtype=torch.float32, device=device))
    I = torch.eye(n, device=device)
    opt = Adam([A_hat, D_flat, sigma_vec, B_free], lr=lr)

    for step in range(1, num_steps + 1):
        opt.zero_grad()
        B = B_free.clone(); B.fill_diagonal_(0.0)
        iB_inv = torch.inverse(I - B)
        Sigma_sq = torch.diag(sigma_vec * sigma_vec)
        M = iB_inv @ Sigma_sq @ iB_inv.T
        D_t = D_flat.view(T, n)
        dM = D_t[:, :, None] * M[None, :, :] * D_t[:, None, :]
        ADAT = A_hat @ dM @ A_hat.T
        loss = torch.sum((Cov_tensor - ADAT) ** 2)
        loss.backward(); opt.step()

        if step % report_every == 0:
            A_np = A_hat.detach().cpu().numpy()
            dm, _, _ = compute_mcc_g(A_np, X_list, Z_list, n)
            am = amari_distance_rect(A_true, A_np)
            mc, _, _ = compute_mcc(A_np, X_list, Z_list, n)
            print(f"    step {step}/{num_steps}: loss={loss.item():.4f} d-MCC={dm:.4f} MCC={mc:.4f} Amari={am:.4f}")
            sys.stdout.flush()

    return A_hat.detach().cpu().numpy()


def multi_init_opt(X_list, Z_list, A_true, T, n, xn, num_steps=50000, lr=5e-3, num_init=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Cov_tensor = torch.stack([torch.tensor(np.cov(X_list[t], rowvar=False),
                                           dtype=torch.float32, device=device)
                              for t in range(T)])
    best_loss, best_A = np.inf, None
    for i in range(num_init):
        print(f"  Init {i+1}/{num_init}:")
        A_hat = run_optimization(Cov_tensor, X_list, Z_list, A_true, T, n, xn,
                                 num_steps=num_steps, lr=lr)
        A_t = torch.tensor(A_hat, dtype=torch.float32, device="cpu")
        D_t = torch.rand(n*T); s_t = torch.rand(n); B_t = torch.zeros(n,n)
        I = torch.eye(n)
        iB = torch.inverse(I - B_t)
        S = torch.diag(s_t*s_t); M = iB @ S @ iB.T
        Dv = D_t.view(T,n)
        dM = Dv[:,:,None] * M[None,:,:] * Dv[:,None,:]
        AD = A_t @ dM @ A_t.T
        fl = torch.sum((Cov_tensor.cpu() - AD) ** 2).item()
        if fl < best_loss: best_loss, best_A = fl, A_hat
    return best_A


def main(args):
    xn_actual = int(args.x_n * args.n)
    T_actual = int(args.T * args.n)
    print(f"=== MuDo-CoM Fast Reproduction ===")
    print(f"n={args.n} xn={xn_actual} T={T_actual} N={args.N} k={args.graph_dense}")
    print(f"steps={args.num_steps} inits={args.num_initializations} lr={args.lr}")
    print(f"seeds={args.seeds}\n")
    sys.stdout.flush()

    res = {"d_mcc":[], "mcc":[], "amari":[], "runtime":[]}
    os.makedirs(args.save_dir, exist_ok=True)

    for seed in args.seeds:
        set_seed(seed); t0 = time.perf_counter()
        print(f"--- Seed {seed} ---"); sys.stdout.flush()

        X_list, A_true, B_true, Ds, sigma_vec, Z_list, skeleton = generate_data(
            T=T_actual, n=args.n, xn=xn_actual, N=args.N,
            graph_dense=args.graph_dense, graph_type=args.graph_type,
            noise_type=args.noise_type, nn_nonlinear=args.nonlinear)

        A_hat = multi_init_opt(X_list, Z_list, A_true, T_actual, args.n, xn_actual,
                               num_steps=args.num_steps, lr=args.lr,
                               num_init=args.num_initializations)

        dm, dml, _ = compute_mcc_g(A_hat, X_list, Z_list, args.n)
        mc, ca, _ = compute_mcc(A_hat, X_list, Z_list, args.n)
        am = amari_distance_rect(A_true, A_hat)
        et = time.perf_counter() - t0

        print(f"  => d-MCC={dm:.6f} MCC={mc:.6f} Amari={am:.6f} ({et:.1f}s)")
        sys.stdout.flush()

        res["d_mcc"].append(dm); res["mcc"].append(mc)
        res["amari"].append(am); res["runtime"].append(et)

        sd = os.path.join(args.save_dir, f"seed_{seed}")
        os.makedirs(sd, exist_ok=True)
        np.savez(os.path.join(sd, "results.npz"),
                 A_hat=A_hat, A_true=A_true, d_mcc=dm, mcc=mc, amari=am)

        # Save partial summary
        with open(os.path.join(args.save_dir, "summary_partial.json"), "w") as f:
            json.dump({
                "seeds_done": len(res["d_mcc"]),
                "d_mcc": {"mean": float(np.mean(res["d_mcc"])), "std": float(np.std(res["d_mcc"]))},
                "mcc": {"mean": float(np.mean(res["mcc"])), "std": float(np.std(res["mcc"]))},
                "amari": {"mean": float(np.mean(res["amari"])), "std": float(np.std(res["amari"]))},
            }, f, indent=2)

    dm_m, dm_s = np.mean(res["d_mcc"]), np.std(res["d_mcc"])
    mc_m, mc_s = np.mean(res["mcc"]), np.std(res["mcc"])
    am_m, am_s = np.mean(res["amari"]), np.std(res["amari"])
    print(f"\n=== Final Summary ({len(args.seeds)} seeds) ===")
    print(f"d-MCC: {dm_m:.6f} +/- {dm_s:.6f}")
    print(f"MCC:   {mc_m:.6f} +/- {mc_s:.6f}")
    print(f"Amari: {am_m:.6f} +/- {am_s:.6f}")
    total_t = np.sum(res["runtime"])
    print(f"Total time: {total_t:.1f}s")

    summary = {
        "config": {"n":args.n,"xn":args.x_n,"xn_actual":xn_actual,
                   "T":args.T,"T_actual":T_actual,"N":args.N,
                   "graph_dense":args.graph_dense,"noise_type":args.noise_type,
                   "num_steps":args.num_steps,"num_initializations":args.num_initializations,
                   "lr":args.lr,"seeds":args.seeds},
        "results": {
            "d_mcc":{"mean":float(dm_m),"std":float(dm_s),"values":[float(v) for v in res["d_mcc"]]},
            "mcc":{"mean":float(mc_m),"std":float(mc_s),"values":[float(v) for v in res["mcc"]]},
            "amari":{"mean":float(am_m),"std":float(am_s),"values":[float(v) for v in res["amari"]]},
            "total_runtime_sec":float(np.sum(res["runtime"]))
        }
    }
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {args.save_dir}/summary.json")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MuDo-CoM Fast Repro")
    p.add_argument("--T", type=int, default=2)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--x-n", type=int, default=1)
    p.add_argument("--N", type=int, default=5000)
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--num-steps", type=int, default=50000)
    p.add_argument("--num-initializations", type=int, default=3)
    p.add_argument("--graph-dense", type=float, default=2.0)
    p.add_argument("--graph-type", type=str, default="ER")
    p.add_argument("--noise-type", type=str, default="gauss")
    p.add_argument("--nonlinear", action="store_true")
    p.add_argument("--save-dir", type=str, default="/repo/outputs/reproduction")
    args = p.parse_args()
    main(args)
