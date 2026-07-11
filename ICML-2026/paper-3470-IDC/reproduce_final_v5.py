#!/usr/bin/env python3
"""
MuDo-CoM Reproduction - Optimized v5
v2 base + IDEA-11 (increased inits to 10) + IDEA-11 (num-steps 75000)
"""
import argparse, numpy as np, torch, time, random, os, json, sys
from torch import nn
from torch.optim import Adam
from numerical_data_generator import generate_data
from evaluation import compute_mcc_g, compute_mcc, amari_distance_rect


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def svd_init_A(Cov_tensor, n, xn, device):
    try:
        C_pool = Cov_tensor.mean(dim=0).cpu().numpy()
        U, S, Vt = np.linalg.svd(C_pool, full_matrices=False)
        A_init = U[:, :n] @ np.diag(np.sqrt(np.maximum(S[:n], 1e-10)))
        if A_init.shape[0] < xn:
            pad = np.random.randn(xn - A_init.shape[0], n) * 0.01
            A_init = np.vstack([A_init, pad])
        elif A_init.shape[0] > xn:
            A_init = A_init[:xn, :]
        return torch.tensor(A_init, dtype=torch.float32, device=device)
    except Exception as e:
        print(f"    [SVD init failed: {e}, falling back to random]")
        return torch.rand(xn, n, device=device, dtype=torch.float32)


def run_opt(Cov_tensor, T, n, xn, num_steps=50000, lr=5e-3, report_every=5000,
            X_list=None, Z_list=None, A_true=None, lambda_sparse=1e-4):
    device = Cov_tensor.device
    A_hat = nn.Parameter(svd_init_A(Cov_tensor, n, xn, device))
    D_flat = nn.Parameter(torch.rand(n*T, device=device, dtype=torch.float32)*0.8+0.2)
    sigma_vec = nn.Parameter(torch.rand(n, device=device, dtype=torch.float32)*0.8+0.2)
    B_free = nn.Parameter(torch.zeros(n, n, device=device, dtype=torch.float32))
    I = torch.eye(n, device=device)
    opt = Adam([A_hat, D_flat, sigma_vec, B_free], lr=lr, betas=(0.9, 0.99))

    for step in range(1, num_steps+1):
        opt.zero_grad()
        B = B_free.clone(); B.fill_diagonal_(0.0)
        iB = torch.linalg.solve(I - B, I)
        S2 = torch.diag(sigma_vec * sigma_vec)
        M = iB @ S2 @ iB.T
        Dt = D_flat.view(T, n)
        dM = Dt[:,:,None] * M[None,:,:] * Dt[:,None,:]
        ADAT = A_hat @ dM @ A_hat.T
        cov_loss = torch.sum((Cov_tensor - ADAT)**2)
        B_nodiag = B.clone(); B_nodiag.fill_diagonal_(0.0)
        sp_loss = lambda_sparse * torch.sum(torch.abs(B_nodiag))
        loss = cov_loss + sp_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([A_hat, D_flat, sigma_vec, B_free], max_norm=5.0)
        opt.step()

        if X_list is not None and step % report_every == 0:
            with torch.no_grad():
                An = A_hat.detach().cpu().numpy()
                dm,_,_ = compute_mcc_g(An, X_list, Z_list, n)
                am = amari_distance_rect(A_true, An)
                mc,_,_ = compute_mcc(An, X_list, Z_list, n)
            print(f"    step {step}/{num_steps}: loss={loss.item():.4f} d-MCC={dm:.4f} MCC={mc:.4f} Amari={am:.4f}")
            sys.stdout.flush()

    with torch.no_grad():
        B = B_free.clone(); B.fill_diagonal_(0.0)
        iB = torch.linalg.solve(I - B, I)
        S2 = torch.diag(sigma_vec * sigma_vec)
        M = iB @ S2 @ iB.T
        Dt = D_flat.view(T, n)
        dM = Dt[:,:,None] * M[None,:,:] * Dt[:,None,:]
        ADAT = A_hat @ dM @ A_hat.T
        final_loss = torch.sum((Cov_tensor - ADAT)**2).item()
    return A_hat.detach().cpu().numpy(), final_loss


def multi_init_opt(X_list, Z_list, A_true, T, n, xn,
                   num_steps=50000, lr=5e-3, num_init=10, lambda_sparse=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Cov_tensor = torch.stack([
        torch.tensor(np.cov(X_list[t], rowvar=False), dtype=torch.float32, device=device)
        for t in range(T)
    ])
    results = []
    for i in range(num_init):
        print(f"  Init {i+1}/{num_init}:")
        A_hat, final_loss = run_opt(Cov_tensor, T, n, xn, num_steps, lr,
                                     X_list=X_list, Z_list=Z_list, A_true=A_true,
                                     lambda_sparse=lambda_sparse)
        dm, _, _ = compute_mcc_g(A_hat, X_list, Z_list, n)
        print(f"    Final loss: {final_loss:.6f} d-MCC={dm:.6f}")
        results.append((A_hat, final_loss, dm))

    best_A, best_dm, best_amari = None, -1, 1e9
    for A_hat, final_loss, dm in results:
        am = amari_distance_rect(A_true, A_hat)
        if dm > best_dm or (abs(dm - best_dm) < 1e-6 and am < best_amari):
            best_dm = dm; best_amari = am; best_A = A_hat
    print(f"  => Selected: d-MCC={best_dm:.6f} Amari={best_amari:.6f}")
    return best_A


def main(args):
    xn_actual = int(args.x_n * args.n)
    T_actual = int(args.T * args.n)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"=== MuDo-CoM Optimized v5 (v2+{args.num_initializations}inits+{args.num_steps}steps) ===")
    print(f"n={args.n} xn={xn_actual} T={T_actual} N={args.N} k={args.graph_dense}")
    print(f"steps={args.num_steps} inits={args.num_initializations} lr={args.lr}")
    print(f"lambda_sparse={args.lambda_sparse} seeds={args.seeds}\n")
    sys.stdout.flush()

    res = {"d_mcc":[], "mcc":[], "amari":[], "runtime":[]}

    for seed in args.seeds:
        set_seed(seed); t0 = time.perf_counter()
        print(f"--- Seed {seed} ---"); sys.stdout.flush()

        X_list, A_true, B_true, Ds, sigma_vec, Z_list, skeleton = generate_data(
            T=T_actual, n=args.n, xn=xn_actual, N=args.N,
            graph_dense=args.graph_dense, graph_type=args.graph_type,
            noise_type=args.noise_type, nn_nonlinear=args.nonlinear)

        A_hat = multi_init_opt(X_list, Z_list, A_true, T_actual, args.n, xn_actual,
                               num_steps=args.num_steps, lr=args.lr,
                               num_init=args.num_initializations,
                               lambda_sparse=args.lambda_sparse)

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
    total_t = float(np.sum(res["runtime"]))
    print(f"\n=== Final Summary ({len(args.seeds)} seeds) ===")
    print(f"d-MCC: {dm_m:.6f} +/- {dm_s:.6f}")
    print(f"MCC:   {mc_m:.6f} +/- {mc_s:.6f}")
    print(f"Amari: {am_m:.6f} +/- {am_s:.6f}")
    print(f"Total time: {total_t:.1f}s")

    summary = {
        "config": {"n":args.n,"xn":args.x_n,"xn_actual":xn_actual,
                   "T":args.T,"T_actual":T_actual,"N":args.N,
                   "graph_dense":args.graph_dense,"noise_type":args.noise_type,
                   "num_steps":args.num_steps,"num_initializations":args.num_initializations,
                   "lr":args.lr,"lambda_sparse":args.lambda_sparse,"seeds":args.seeds},
        "results": {
            "d_mcc":{"mean":float(dm_m),"std":float(dm_s),"values":[float(v) for v in res["d_mcc"]]},
            "mcc":{"mean":float(mc_m),"std":float(mc_s),"values":[float(v) for v in res["mcc"]]},
            "amari":{"mean":float(am_m),"std":float(am_s),"values":[float(v) for v in res["amari"]]},
            "total_runtime_sec":total_t
        }
    }
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {args.save_dir}/summary.json")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MuDo-CoM Optimized v5")
    p.add_argument("--T", type=int, default=2)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--x-n", type=int, default=1)
    p.add_argument("--N", type=int, default=5000)
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--num-steps", type=int, default=75000)
    p.add_argument("--num-initializations", type=int, default=10)
    p.add_argument("--graph-dense", type=float, default=2.0)
    p.add_argument("--graph-type", type=str, default="ER")
    p.add_argument("--noise-type", type=str, default="gauss")
    p.add_argument("--nonlinear", action="store_true")
    p.add_argument("--save-dir", type=str, default="/repo/outputs/sota_v5")
    p.add_argument("--lambda-sparse", type=float, default=1e-4)
    args = p.parse_args()
    main(args)
