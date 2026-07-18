#!/usr/bin/env python3
"""Refit LSM for PolBlogs with optimized PGD hyperparameters.
Saves fitted (Z, alpha, sparsity) to a new pickle for SyNG-R generation.
"""
import sys, os, pickle, argparse, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syngler.lsm.source import (
    Model, sigmoid, center_and_rotate, topk_sqrt_eig_embedding,
    H_functional
)

def fit_lsm_polblogs(A, r=2, eta_0=1.0, sigma_init=0.01,
                     n_iter_init=5000, n_iter_pgd=2000, verbose=True):
    """Fit LSM with two-phase PGD."""
    n = A.shape[0]
    
    # Dummy covariate (required by LSM.Model API)
    X_dummy = np.zeros((n, n, 1), dtype=np.float32)
    beta_init = np.zeros(1, dtype=np.float32)
    
    # Initialize Z via spectral embedding of centered adjacency
    A_centered = A - A.mean()
    Z_init = topk_sqrt_eig_embedding(A_centered, r)
    Z_init = center_and_rotate(Z_init)
    Z_init = Z_init + sigma_init * np.random.randn(*Z_init.shape).astype(np.float32)
    
    # Initialize alpha from log-degree
    deg = A.sum(axis=1)
    alpha_init = np.log(deg + 1) - np.log(deg + 1).mean()
    alpha_init = alpha_init.astype(np.float32)
    alpha_init = alpha_init + sigma_init * np.random.randn(n).astype(np.float32)
    
    # Initial sparsity estimate
    E = np.triu(A, k=1).sum()
    M = n * (n - 1) / 2
    p_hat = E / M
    rho = np.log(p_hat / (1 - p_hat)) if 0 < p_hat < 1 else 0.0
    
    eta_alpha = eta_0 / (2 * n)
    eta_Z = eta_0 / (2 * max(np.sum(Z_init ** 2) / Z_init.shape[1], 1.0))
    eta_beta = 0.0
    
    model = Model(
        A, X_dummy,
        alpha=alpha_init, beta=beta_init, Z=Z_init,
        alpha_enable=True, Z_enable=True, Z_standardize=True,
        act=sigmoid, sparsity=rho, sparsity_estimation=True,
    )
    
    if verbose:
        print(f"[LSM] n={n}, r={r}, rho_init={rho:.4f}")
        print(f"[LSM] eta_alpha={eta_alpha:.6f}, eta_Z={eta_Z:.6f}")
    
    # Phase 1: PGD initialization on G = ZZ^T
    t0 = time.time()
    print(f"[LSM] Phase 1: PGD init ({n_iter_init} iters)...")
    model.PGD_initialization(
        eta_alpha=eta_alpha, eta_Z=eta_Z, eta_beta=eta_beta,
        n_iter=n_iter_init, eps=1e-6
    )
    print(f"[LSM] Phase 1 done. Converged={model.converged} step={getattr(model, 'step', 'N/A')} "
          f"|Z|_F={np.linalg.norm(model.Z):.2f}")
    
    # Phase 2: PGD refinement on Z and alpha
    if n_iter_pgd > 0:
        eta_Z_2 = eta_0 / (2 * max(np.sum(model.Z ** 2) / model.Z.shape[1], 1.0))
        eta_alpha_2 = eta_0 / (2 * n)
        print(f"[LSM] Phase 2: PGD refinement ({n_iter_pgd} iters, eta_Z={eta_Z_2:.8f})...")
        beta_traj, loss = model.PGD(
            eta_alpha=eta_alpha_2, eta_Z=eta_Z_2, eta_beta=eta_beta,
            early_stop=True, eps=1e-6, n_iter=n_iter_pgd, verbose=False
        )
        print(f"[LSM] Phase 2 done. Converged={model.converged}, final loss={loss[-1]:.4f}")
    
    # Calibrate sparsity (rho) to match observed density
    ZZT = model.Z @ model.Z.T
    alpha_outer = (np.outer(model.alpha, np.ones(n)) +
                   np.outer(np.ones(n), model.alpha))
    Theta_no_rho = ZZT + alpha_outer
    Theta_no_rho = np.triu(Theta_no_rho, 1) + np.triu(Theta_no_rho, 1).T
    
    observed_density = np.triu(A, k=1).sum() / (n * (n - 1) / 2)
    from scipy.optimize import brentq
    
    def density_gap(rho_val):
        P_trial = sigmoid(Theta_no_rho + rho_val)
        P_trial = np.triu(P_trial, 1) + np.triu(P_trial, 1).T
        return np.triu(P_trial, k=1).sum() / (n * (n - 1) / 2) - observed_density
    
    try:
        rho_calibrated = float(brentq(density_gap, -20.0, 20.0, xtol=1e-6))
    except ValueError:
        rho_calibrated = float(np.log(observed_density / (1 - observed_density)))
    
    model.sparsity = rho_calibrated
    elapsed = time.time() - t0
    print(f"[LSM] Calibrated rho={rho_calibrated:.6f} (density={observed_density:.6f})")
    print(f"[LSM] Time: {elapsed:.1f}s")
    print(f"[LSM] Z: shape={model.Z.shape} mean={model.Z.mean():.4f} std={model.Z.std():.4f}")
    print(f"[LSM] alpha: shape={model.alpha.shape} mean={model.alpha.mean():.4f} std={model.alpha.std():.4f}")
    
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--eta_0", type=float, default=1.0)
    ap.add_argument("--sigma_init", type=float, default=0.01)
    ap.add_argument("--n_iter_init", type=int, default=5000)
    ap.add_argument("--n_iter_pgd", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adj", default="data/real/polblogs/generator/seed=0.npy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    
    np.random.seed(args.seed)
    A = np.load(args.adj).astype(np.float32)
    
    model = fit_lsm_polblogs(
        A, r=args.r, eta_0=args.eta_0, sigma_init=args.sigma_init,
        n_iter_init=args.n_iter_init, n_iter_pgd=args.n_iter_pgd
    )
    
    result = {
        "model_Z": model.Z.astype(np.float32),
        "model_alpha": model.alpha.astype(np.float32),
        "model_sparsity": float(model.sparsity),
        "converged": model.converged,
        "eta_0": args.eta_0,
        "n_iter_init": args.n_iter_init,
        "n_iter_pgd": args.n_iter_pgd,
        "r": args.r,
    }
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
