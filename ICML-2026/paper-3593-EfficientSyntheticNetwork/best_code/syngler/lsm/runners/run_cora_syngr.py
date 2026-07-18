"""
SyNGLER-Resampling (SyNG-R) on Cora dataset.

Pipeline:
1. Load Cora from the existing artifact (train+val subgraph only)
2. Fit LSM (Z + alpha) via PGD on the train+val adjacency matrix
3. Bootstrap resample (Z*, alpha*) — the official SyNG-R method
4. Reconstruct edges from generated latent variables
5. Resample node features (x) and labels (y) per-class from fitted embeddings
6. Save synthetic graph as a .pt artifact compatible with node_utility_eval.py
7. Run node classification evaluation
"""

import os
import sys
import json
import copy
import random
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Import LSM fitting code
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import LSM_source as LSM
from LSM_source import sigmoid, center_and_rotate, topk_sqrt_eig_embedding

# Also need the bootstrap function
sys.path.insert(0, str(Path(__file__).resolve().parent / "../../SyNGLER/utils"))
from SyNG_source import bootstrap_alpha_Z


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Step 1: Load Cora
# ---------------------------------------------------------------------------
def load_cora_artifact(path):
    art = torch.load(path, map_location="cpu")
    return art


def build_adj_matrix(edge_index, num_nodes):
    """Build dense adjacency from edge_index (undirected, no self-loops)."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    A[src, dst] = 1.0
    # Ensure symmetric + no self-loops
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    return A


# ---------------------------------------------------------------------------
# Step 2: Fit LSM on train+val subgraph
# ---------------------------------------------------------------------------
def fit_lsm(A_sub, r=2, eta_0=0.5, sigma_init=0.01, n_iter_init=3000,
             n_iter_pgd=30000, verbose=True):
    """
    Fit latent space model: Theta_ij = alpha_i + alpha_j + Z_i^T Z_j + sparsity
    """
    n = A_sub.shape[0]
    p = 0  # no covariates for SyNGLER on real data

    # Create dummy X (n, n, 0) — no covariates
    # The LSM Model code uses X for beta, but we set beta to empty
    # We need at least p=1 dummy covariate set to 0
    X_dummy = np.zeros((n, n, 1), dtype=np.float32)
    beta_init = np.zeros(1, dtype=np.float32)

    # Initialize Z with spectral embedding
    # Use top-r eigenvectors of centered adjacency
    A_centered = A_sub - A_sub.mean()
    Z_init = topk_sqrt_eig_embedding(A_centered, r)
    Z_init = center_and_rotate(Z_init)
    Z_init = Z_init + sigma_init * np.random.randn(*Z_init.shape)

    # Initialize alpha from degree
    deg = A_sub.sum(axis=1)
    alpha_init = np.log(deg + 1) - np.log(deg + 1).mean()
    alpha_init = alpha_init.astype(np.float32)
    alpha_init = alpha_init + sigma_init * np.random.randn(n).astype(np.float32)

    # Estimate sparsity
    E = np.triu(A_sub, k=1).sum()
    M = n * (n - 1) / 2
    p_hat = E / M
    rho = np.log(p_hat / (1 - p_hat)) if p_hat > 0 and p_hat < 1 else 0.0

    eta_alpha = eta_0 / (2 * n)
    eta_Z = eta_0 / (2 * max(np.sum(Z_init ** 2) / Z_init.shape[1], 1.0))
    eta_beta = 0.0  # no covariates

    model = LSM.Model(
        A_sub, X_dummy,
        alpha=alpha_init,
        beta=beta_init,
        Z=Z_init,
        alpha_enable=True,
        Z_enable=True,
        Z_standardize=True,
        act=sigmoid,
        sparsity=rho,
        sparsity_estimation=True,
    )

    if verbose:
        print(f"[LSM fit] n={n}, r={r}, rho_init={rho:.4f}")
        print(f"[LSM fit] eta_alpha={eta_alpha:.6f}, eta_Z={eta_Z:.6f}")

    # Phase 1: PGD initialization (fits G = ZZ^T directly)
    print("[LSM fit] Phase 1: PGD initialization...")
    model.PGD_initialization(
        eta_alpha=eta_alpha, eta_Z=eta_Z, eta_beta=eta_beta,
        n_iter=n_iter_init, eps=1e-5
    )

    # After init, model.Z is extracted via eigendecomposition of G
    # Optionally do a short Phase 2 refinement
    if n_iter_pgd > 0:
        eta_Z_2 = eta_0 / (2 * max(np.sum(model.Z ** 2) / model.Z.shape[1], 1.0))
        eta_alpha_2 = eta_0 / (2 * n)
        n_iter_phase2 = min(n_iter_pgd, 500)
        print(f"[LSM fit] Phase 2: PGD refinement ({n_iter_phase2} iters, eta_Z={eta_Z_2:.8f})...")
        beta_traj, loss = model.PGD(
            eta_alpha=eta_alpha_2, eta_Z=eta_Z_2, eta_beta=eta_beta,
            early_stop=True, eps=1e-5, n_iter=n_iter_phase2, verbose=False
        )
        print(f"[LSM fit] Phase 2 done. Converged={model.converged}, final loss={loss[-1]:.2f}")
    else:
        print("[LSM fit] Skipping Phase 2 refinement.")

    # Compute actual sparsity from the fitted model
    # Theta_ij = alpha_i + alpha_j + z_i^T z_j + rho
    # We want P(edge) to match observed density
    # Compute mean of fitted probabilities vs observed edge rate
    ZZT = model.Z @ model.Z.T
    alpha_outer = np.outer(model.alpha, np.ones(n)) + np.outer(np.ones(n), model.alpha)
    Theta_no_rho = ZZT + alpha_outer
    Theta_no_rho = np.triu(Theta_no_rho, 1) + np.triu(Theta_no_rho, 1).T

    # Binary search for rho that matches observed density
    observed_density = np.triu(A_sub, k=1).sum() / (n * (n - 1) / 2)
    from scipy.optimize import brentq
    def density_gap(rho_val):
        P_trial = sigmoid(Theta_no_rho + rho_val)
        P_trial = np.triu(P_trial, 1) + np.triu(P_trial, 1).T
        return np.triu(P_trial, k=1).sum() / (n * (n - 1) / 2) - observed_density

    try:
        rho_calibrated = brentq(density_gap, -20.0, 20.0, xtol=1e-6)
    except ValueError:
        rho_calibrated = np.log(observed_density / (1 - observed_density))
    model.sparsity = rho_calibrated
    print(f"[LSM fit] Calibrated sparsity (rho)={rho_calibrated:.4f} for observed density={observed_density:.6f}")
    print(f"[LSM fit] Z shape={model.Z.shape}, alpha shape={model.alpha.shape}")

    return model


# ---------------------------------------------------------------------------
# Step 3: Bootstrap resample (SyNG-R)
# ---------------------------------------------------------------------------
def resample_latents(model_alpha, model_Z, seed=42):
    """Bootstrap resample alpha and Z — the core SyNG-R operation."""
    np.random.seed(seed)
    alpha_boot, Z_boot = bootstrap_alpha_Z(
        model_alpha.reshape(-1, 1),
        model_Z,
        batch=1
    )
    return alpha_boot.squeeze(), Z_boot.squeeze()


# ---------------------------------------------------------------------------
# Step 4: Reconstruct edges
# ---------------------------------------------------------------------------
def reconstruct_edges(Z, alpha, sparsity):
    """Reconstruct adjacency from latent variables."""
    n = Z.shape[0]
    Theta = Z @ Z.T
    Theta += np.outer(alpha, np.ones(n)) + np.outer(np.ones(n), alpha)
    Theta += sparsity
    Theta = np.triu(Theta, 1) + np.triu(Theta, 1).T
    P = sigmoid(Theta)
    P = np.triu(P, 1) + np.triu(P, 1).T

    # Sample edges
    U = np.random.random((n, n))
    A = (U < P).astype(np.float32)
    A = np.triu(A, 1)
    A = A + A.T
    return A, P


# ---------------------------------------------------------------------------
# Step 5: Resample features and labels
# ---------------------------------------------------------------------------
def resample_features_labels(x_orig, y_orig, bootstrap_indices):
    """
    Resample features and labels using the same bootstrap indices
    that were used for the latent variables.
    This ensures consistency: node i in synthetic graph has features
    from the same source node that provided Z_i and alpha_i.
    """
    x_new = x_orig[bootstrap_indices]
    y_new = y_orig[bootstrap_indices]
    return x_new, y_new


def resample_features_labels_per_class(x_sub, y_sub, n_target, seed=42):
    """
    Resample features and labels to match target size.
    For each class, resample proportionally.
    """
    np.random.seed(seed)
    classes = np.unique(y_sub)
    n_sub = len(y_sub)

    x_list = []
    y_list = []

    for c in classes:
        mask = (y_sub == c)
        n_c = mask.sum()
        # Target count proportional
        n_target_c = max(1, int(round(n_c / n_sub * n_target)))
        indices = np.where(mask)[0]
        sampled = np.random.choice(indices, size=n_target_c, replace=True)
        x_list.append(x_sub[sampled])
        y_list.append(y_sub[sampled])

    x_new = np.concatenate(x_list, axis=0)
    y_new = np.concatenate(y_list, axis=0)

    # Trim or pad to exact n_target
    if len(y_new) > n_target:
        perm = np.random.permutation(len(y_new))[:n_target]
        x_new = x_new[perm]
        y_new = y_new[perm]
    elif len(y_new) < n_target:
        extra = n_target - len(y_new)
        extra_idx = np.random.choice(len(y_new), size=extra, replace=True)
        x_new = np.concatenate([x_new, x_new[extra_idx]], axis=0)
        y_new = np.concatenate([y_new, y_new[extra_idx]], axis=0)

    return x_new, y_new


# ---------------------------------------------------------------------------
# Step 6: Build synthetic artifact
# ---------------------------------------------------------------------------
def dense_adj_to_edge_index(A):
    """Convert dense adjacency to edge_index."""
    rows, cols = np.where(np.triu(A, k=1) > 0)
    # Make undirected
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


# ---------------------------------------------------------------------------
# Evaluation helpers (inline, self-contained)
# ---------------------------------------------------------------------------
def build_norm_adj(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    src, dst = edge_index
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0
    adj.fill_diagonal_(1.0)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
    return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]


class DenseGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        h = adj_norm @ x
        h = self.lin1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = adj_norm @ h
        return self.lin2(h)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        h = self.lin1(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h)


def train_and_eval(x_train, y_train, edge_index_train, train_mask, val_mask,
                   x_test_graph, y_test_graph, edge_index_test, test_mask,
                   model_cls, hidden_dim=64, lr=0.01, weight_decay=5e-4,
                   max_epochs=300, patience=50, seed=0, device="cpu"):
    """Train on one graph, evaluate on another's test set."""
    set_seed(seed)

    num_nodes_train = x_train.size(0)
    num_classes = int(y_train.max().item()) + 1
    in_dim = x_train.size(1)

    adj_train = build_norm_adj(edge_index_train, num_nodes_train).to(device)
    x_tr = x_train.float().to(device)
    y_tr = y_train.long().to(device)
    tr_mask = train_mask.to(device)
    vl_mask = val_mask.to(device)

    model = model_cls(in_dim, hidden_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tr, adj_train)
        loss = F.cross_entropy(logits[tr_mask], y_tr[tr_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_tr, adj_train)
            val_acc = (val_logits.argmax(-1)[vl_mask] == y_tr[vl_mask]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    # Evaluate on test graph
    num_nodes_test = x_test_graph.size(0)
    adj_test = build_norm_adj(edge_index_test, num_nodes_test).to(device)
    x_te = x_test_graph.float().to(device)
    y_te = y_test_graph.long().to(device)
    te_mask = test_mask.to(device)

    with torch.no_grad():
        test_logits = model(x_te, adj_test)
        test_acc = (test_logits.argmax(-1)[te_mask] == y_te[te_mask]).float().mean().item()

    return test_acc, best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, default=16, help="Latent dimension r")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-reps", type=int, default=3, help="Number of synthetic graphs to generate")
    parser.add_argument("--n-iter-init", type=int, default=3000)
    parser.add_argument("--n-iter-pgd", type=int, default=30000)
    parser.add_argument("--eta0", type=float, default=0.5)
    parser.add_argument("--artifact-path", type=str,
                        default="data/attributed/cora.pt")
    parser.add_argument("--output-dir", type=str,
                        default="runs/cora_syngr")
    parser.add_argument("--skip-fit", action="store_true", help="Skip fitting, load from cache")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cpu"  # LSM fitting is CPU-only; eval is fast on CPU for Cora

    # --- Load Cora ---
    print("=" * 60)
    print("Loading Cora dataset...")
    art = load_cora_artifact(args.artifact_path)
    x_full = art["x"].numpy()          # (2708, 1433)
    y_full = art["y"].numpy()          # (2708,)
    edge_index = art["edge_index"]     # (2, 10556)
    train_mask = art["train_mask"]     # (2708,) bool
    val_mask = art["val_mask"]
    test_mask = art["test_mask"]

    n_full = x_full.shape[0]
    print(f"  Full graph: {n_full} nodes, {edge_index.size(1)} edges")
    print(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    # --- Build train+val subgraph ---
    tv_mask = (train_mask | val_mask).numpy()  # exclude test nodes
    tv_indices = np.where(tv_mask)[0]
    n_tv = len(tv_indices)
    print(f"  Train+Val subgraph: {n_tv} nodes")

    # Remap indices for the subgraph
    full_to_sub = np.full(n_full, -1, dtype=np.int64)
    full_to_sub[tv_indices] = np.arange(n_tv)

    # Build subgraph adjacency
    A_full = build_adj_matrix(edge_index, n_full)
    A_sub = A_full[np.ix_(tv_indices, tv_indices)]
    x_sub = x_full[tv_indices]
    y_sub = y_full[tv_indices]

    # Build masks within the subgraph
    train_mask_sub = train_mask.numpy()[tv_indices]
    val_mask_sub = val_mask.numpy()[tv_indices]

    n_edges_sub = int(np.triu(A_sub, k=1).sum())
    print(f"  Subgraph edges: {n_edges_sub}")

    # --- Fit LSM ---
    cache_path = os.path.join(args.output_dir, f"lsm_fit_r{args.r}_seed{args.seed}.pkl")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.skip_fit and os.path.exists(cache_path):
        print(f"\n[Skipping fit, loading from {cache_path}]")
        with open(cache_path, "rb") as f:
            fit_data = pickle.load(f)
        model_Z = fit_data["model_Z"]
        model_alpha = fit_data["model_alpha"]
        model_sparsity = fit_data["model_sparsity"]
    else:
        print("\n" + "=" * 60)
        print(f"Fitting LSM (r={args.r}) on train+val subgraph...")
        model = fit_lsm(
            A_sub, r=args.r,
            eta_0=args.eta0,
            sigma_init=0.01,
            n_iter_init=args.n_iter_init,
            n_iter_pgd=args.n_iter_pgd,
        )
        model_Z = model.Z.astype(np.float32)
        model_alpha = model.alpha.astype(np.float32)
        model_sparsity = float(model.sparsity)

        # Cache the fit
        fit_data = {
            "model_Z": model_Z,
            "model_alpha": model_alpha,
            "model_sparsity": model_sparsity,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(fit_data, f)
        print(f"[Saved LSM fit to {cache_path}]")

    # --- Generate synthetic graphs ---
    print("\n" + "=" * 60)
    print(f"Generating {args.n_reps} synthetic graphs via SyNG-R bootstrap...")

    synth_artifacts = []
    for rep in range(args.n_reps):
        print(f"\n--- Rep {rep} ---")
        rep_seed = args.seed + rep + 10000

        # Bootstrap resample latents
        np.random.seed(rep_seed)
        n_sub = model_Z.shape[0]
        indices = np.random.choice(n_sub, size=n_sub, replace=True)
        Z_star = model_Z[indices]
        alpha_star = model_alpha[indices]

        # Also resample features/labels using the same indices
        x_star = x_sub[indices]
        y_star = y_sub[indices]

        # Reconstruct edges
        np.random.seed(rep_seed + 50000)
        A_star, P_star = reconstruct_edges(Z_star, alpha_star, model_sparsity)

        n_edges_star = int(np.triu(A_star, k=1).sum())
        print(f"  Synthetic graph: {n_sub} nodes, {n_edges_star} edges")
        print(f"  Class distribution: {np.bincount(y_star.astype(int))}")

        # Build edge_index
        edge_index_star = dense_adj_to_edge_index(A_star)

        # Create masks for synthetic graph (same proportions)
        # Use 20 per class for train, rest proportional for val
        perm = np.random.permutation(n_sub)
        train_mask_star = np.zeros(n_sub, dtype=bool)
        val_mask_star = np.zeros(n_sub, dtype=bool)
        test_mask_star = np.zeros(n_sub, dtype=bool)

        for c in np.unique(y_star):
            c_idx = np.where(y_star == c)[0]
            c_perm = np.random.permutation(len(c_idx))
            n_c = len(c_idx)
            n_train = min(20, n_c)
            n_val = min(max(n_c // 5, 10), n_c - n_train)
            train_mask_star[c_idx[c_perm[:n_train]]] = True
            val_mask_star[c_idx[c_perm[n_train:n_train + n_val]]] = True
            test_mask_star[c_idx[c_perm[n_train + n_val:]]] = True

        synth_art = {
            "name": f"cora_syngr_r{args.r}_rep{rep}",
            "dataset": "Cora",
            "x": torch.tensor(x_star, dtype=torch.float32),
            "y": torch.tensor(y_star, dtype=torch.int64),
            "edge_index": edge_index_star,
            "train_mask": torch.tensor(train_mask_star),
            "val_mask": torch.tensor(val_mask_star),
            "test_mask": torch.tensor(test_mask_star),
            "num_nodes": n_sub,
            "num_edges": int(edge_index_star.size(1)),
            "num_features": x_star.shape[1],
            "num_classes": len(np.unique(y_star)),
        }

        out_path = os.path.join(args.output_dir, f"cora_syngr_r{args.r}_rep{rep}.pt")
        torch.save(synth_art, out_path)
        print(f"  Saved to {out_path}")
        synth_artifacts.append((synth_art, out_path))

    # --- Evaluate ---
    print("\n" + "=" * 60)
    print("Evaluating node classification utility...")
    print("Protocol: Train on graph G, test on real test set")

    results_all = []

    for model_name, model_cls in [("GCN", DenseGCN), ("MLP", MLP)]:
        print(f"\n=== {model_name} ===")

        # Baseline: train on real, test on real
        real_real_accs = []
        for eval_seed in range(3):
            acc, _ = train_and_eval(
                x_train=art["x"], y_train=art["y"],
                edge_index_train=art["edge_index"],
                train_mask=art["train_mask"], val_mask=art["val_mask"],
                x_test_graph=art["x"], y_test_graph=art["y"],
                edge_index_test=art["edge_index"],
                test_mask=art["test_mask"],
                model_cls=model_cls, seed=eval_seed, device=device,
            )
            real_real_accs.append(acc)

        acc_g_g = np.mean(real_real_accs)
        print(f"  ACC(G|G) = {acc_g_g:.4f} (std={np.std(real_real_accs):.4f})")

        # For each synthetic graph: train on synthetic, test on real test set
        for synth_art, out_path in synth_artifacts:
            synth_real_accs = []
            for eval_seed in range(3):
                acc, _ = train_and_eval(
                    x_train=synth_art["x"], y_train=synth_art["y"],
                    edge_index_train=synth_art["edge_index"],
                    train_mask=synth_art["train_mask"],
                    val_mask=synth_art["val_mask"],
                    x_test_graph=art["x"], y_test_graph=art["y"],
                    edge_index_test=art["edge_index"],
                    test_mask=art["test_mask"],
                    model_cls=model_cls, seed=eval_seed, device=device,
                )
                synth_real_accs.append(acc)

            acc_g_ghat = np.mean(synth_real_accs)
            ratio = acc_g_ghat / acc_g_g if acc_g_g > 0 else 0
            print(f"  {synth_art['name']}: ACC(G|Ghat)={acc_g_ghat:.4f}, ratio={ratio:.4f}")

            results_all.append({
                "model": model_name,
                "synthetic": synth_art["name"],
                "acc_g_given_g": acc_g_g,
                "acc_g_given_g_hat": acc_g_ghat,
                "utility_ratio": ratio,
            })

    # Save results
    results_path = os.path.join(args.output_dir, f"eval_results_r{args.r}.json")
    with open(results_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f"Results saved to {results_path}")

    # Print summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'Model':<6} {'Synthetic':<30} {'ACC(G|G)':<12} {'ACC(G|Ghat)':<14} {'Ratio':<8}")
    print("-" * 70)
    for r in results_all:
        print(f"{r['model']:<6} {r['synthetic']:<30} {r['acc_g_given_g']:<12.4f} {r['acc_g_given_g_hat']:<14.4f} {r['utility_ratio']:<8.4f}")


if __name__ == "__main__":
    main()
