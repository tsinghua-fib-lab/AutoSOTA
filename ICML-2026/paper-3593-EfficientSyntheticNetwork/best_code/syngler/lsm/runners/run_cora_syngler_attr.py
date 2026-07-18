"""
SyNGLER-Attr (Algorithm 3) on Cora dataset.

Pipeline:
1. Load cached LSM fit (Z1, alpha, sparsity) from train+val subgraph
2. Attribute regression: Y = 1*mu^T + Z1*Lambda1^T + R
3. Eigenvalue ratio test on R to find d2, then SVD: R = Z2*Lambda2^T + E
4. Bootstrap resample (Z1, Z2, alpha) jointly
5. Reconstruct edges from (Z1, alpha, sparsity)
6. Reconstruct attributes: Y_tilde = 1*mu^T + Z1_tilde*Lambda1^T + Z2_tilde*Lambda2^T
7. Evaluate node classification: ACC(G|Ghat)/ACC(G|G)
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from LSM_source import sigmoid


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def build_adj_matrix(edge_index, num_nodes):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    A[src, dst] = 1.0
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    return A


def dense_adj_to_edge_index(A):
    rows, cols = np.where(np.triu(A, k=1) > 0)
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


# ---------------------------------------------------------------------------
# Algorithm 3 Steps 2-3: Attribute regression and residual factorization
# ---------------------------------------------------------------------------
def eigenvalue_ratio_test(singular_values, max_d=None):
    """
    Eigenvalue ratio test to determine the number of factors.
    We look for the largest ratio s_k / s_{k+1}.
    """
    sv = singular_values
    if max_d is None:
        max_d = len(sv) - 1
    max_d = min(max_d, len(sv) - 1)

    ratios = sv[:max_d] / np.maximum(sv[1:max_d + 1], 1e-12)
    d = int(np.argmax(ratios)) + 1
    print(f"  Eigenvalue ratio test: top ratios = {ratios[:10].round(2)}")
    print(f"  Selected d2 = {d}")
    return d


def fit_attribute_model(Y, Z1, max_d2=None):
    """
    Steps 2-3 of Algorithm 3:
    - Regress Y on Z1 to get mu, Lambda1, and residual R
    - Factor R via SVD to get Z2, Lambda2

    Y: (n, p) attribute matrix
    Z1: (n, r1) LSM latent positions

    Returns: mu, Lambda1, Z2, Lambda2, d2
    """
    n, p = Y.shape
    r1 = Z1.shape[1]

    # Step 2: Y = 1*mu^T + Z1*Lambda1^T + R
    mu = Y.mean(axis=0)  # (p,)
    Y_centered = Y - mu[np.newaxis, :]  # (n, p)

    # Lambda1 = (Z1^T Z1)^{-1} Z1^T Y_centered  => Lambda1 is (r1, p)
    # So Lambda1^T is (p, r1), and Z1 @ Lambda1.T is (n, p)
    ZtZ = Z1.T @ Z1  # (r1, r1)
    ZtZ_inv = np.linalg.inv(ZtZ + 1e-6 * np.eye(r1))  # regularize slightly
    Lambda1 = ZtZ_inv @ Z1.T @ Y_centered  # (r1, p)

    # Residual
    R = Y_centered - Z1 @ Lambda1  # (n, p)
    print(f"  Attribute regression: ||Y_centered||_F = {np.linalg.norm(Y_centered):.2f}")
    print(f"  ||Z1 @ Lambda1||_F = {np.linalg.norm(Z1 @ Lambda1):.2f}")
    print(f"  ||R||_F = {np.linalg.norm(R):.2f}")
    var_explained_by_z1 = 1 - np.linalg.norm(R)**2 / np.linalg.norm(Y_centered)**2
    print(f"  Variance explained by Z1: {var_explained_by_z1:.4f}")

    # Step 3: SVD of R to find Z2
    # R = U @ diag(s) @ Vt
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    print(f"  Top 20 singular values of R: {s[:20].round(2)}")

    # Eigenvalue ratio test
    if max_d2 is None:
        max_d2 = min(50, len(s) - 1)
    d2 = eigenvalue_ratio_test(s, max_d=max_d2)
    # Clamp d2 to reasonable range
    d2 = max(1, min(d2, 50))

    # Z2 = U[:, :d2] * s[:d2]  (scaled left singular vectors)
    # Lambda2 = Vt[:d2, :].T * 1  (right singular vectors, unscaled since Z2 absorbed s)
    # Actually: R ≈ U[:,:d2] diag(s[:d2]) Vt[:d2,:] = Z2 @ Lambda2^T
    # where Z2 = U[:,:d2] * s[:d2] and Lambda2^T = Vt[:d2,:]
    # OR Z2 = U[:,:d2] and Lambda2 = (diag(s[:d2]) @ Vt[:d2,:])^T
    # Let's use: Z2 = U[:,:d2] * sqrt(s[:d2]), Lambda2 = Vt[:d2,:].T * sqrt(s[:d2])
    # This balances the scaling between Z2 and Lambda2
    sqrt_s = np.sqrt(s[:d2])
    Z2 = U[:, :d2] * sqrt_s[np.newaxis, :]  # (n, d2)
    Lambda2 = Vt[:d2, :].T * sqrt_s[np.newaxis, :]  # (p, d2), so Lambda2^T is (d2, p)
    # Verify: Z2 @ Lambda2.T should approximate R[:, :] up to d2 components

    recon_R = Z2 @ Lambda2.T
    residual_after = R - recon_R
    var_explained_by_z2 = 1 - np.linalg.norm(residual_after)**2 / np.linalg.norm(R)**2
    print(f"  d2 = {d2}, variance of R explained by Z2: {var_explained_by_z2:.4f}")
    total_var = 1 - np.linalg.norm(residual_after)**2 / np.linalg.norm(Y_centered)**2
    print(f"  Total variance explained (Z1 + Z2): {total_var:.4f}")

    return mu, Lambda1, Z2, Lambda2.T, d2  # Lambda2 returned as (d2, p) for convenience


# ---------------------------------------------------------------------------
# Edge reconstruction (same as base)
# ---------------------------------------------------------------------------
def reconstruct_edges(Z, alpha, sparsity, seed=None):
    n = Z.shape[0]
    Theta = Z @ Z.T
    Theta += np.outer(alpha, np.ones(n)) + np.outer(np.ones(n), alpha)
    Theta += sparsity
    Theta = np.triu(Theta, 1) + np.triu(Theta, 1).T
    P = sigmoid(Theta)
    P = np.triu(P, 1) + np.triu(P, 1).T

    if seed is not None:
        np.random.seed(seed)
    U = np.random.random((n, n))
    A = (U < P).astype(np.float32)
    A = np.triu(A, 1)
    A = A + A.T
    return A, P


# ---------------------------------------------------------------------------
# Attribute reconstruction (Algorithm 3 Step 7)
# ---------------------------------------------------------------------------
def reconstruct_attributes(Z1_tilde, Z2_tilde, mu, Lambda1, Lambda2_T):
    """
    Y_tilde = 1*mu^T + Z1_tilde @ Lambda1 + Z2_tilde @ Lambda2_T

    Z1_tilde: (n, r1)
    Z2_tilde: (n, d2)
    mu: (p,)
    Lambda1: (r1, p)
    Lambda2_T: (d2, p)
    """
    n = Z1_tilde.shape[0]
    Y_tilde = np.ones((n, 1)) @ mu[np.newaxis, :]  # (n, p)
    Y_tilde += Z1_tilde @ Lambda1  # (n, p)
    Y_tilde += Z2_tilde @ Lambda2_T  # (n, p)
    return Y_tilde


# ---------------------------------------------------------------------------
# Evaluation (same as base)
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
    parser.add_argument("--r", type=int, default=16, help="LSM latent dimension r (=r1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-reps", type=int, default=5, help="Number of synthetic graphs")
    parser.add_argument("--max-d2", type=int, default=50, help="Max d2 for eigenvalue ratio test")
    parser.add_argument("--binarize", action="store_true", default=True,
                        help="Binarize generated attributes (Cora has binary features)")
    parser.add_argument("--binarize-method", type=str, default="density_match",
                        choices=["fixed", "density_match"],
                        help="Binarization method: fixed threshold or density-matched")
    parser.add_argument("--binarize-threshold", type=float, default=0.5,
                        help="Threshold for fixed binarization")
    parser.add_argument("--d2-override", type=int, default=None,
                        help="Override d2 from eigenvalue ratio test")
    parser.add_argument("--artifact-path", type=str,
                        default="data/attributed/cora.pt")
    parser.add_argument("--lsm-cache", type=str,
                        default="runs/cora_syngr/lsm_fit_r16_seed0.pkl")
    parser.add_argument("--output-dir", type=str,
                        default="runs/cora_syngler_attr")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cpu"

    # === Load Cora ===
    print("=" * 60)
    print("SyNGLER-Attr (Algorithm 3) on Cora")
    print("=" * 60)
    print("\nLoading Cora dataset...")
    art = torch.load(args.artifact_path, map_location="cpu")
    x_full = art["x"].numpy()
    y_full = art["y"].numpy()
    edge_index = art["edge_index"]
    train_mask = art["train_mask"]
    val_mask = art["val_mask"]
    test_mask = art["test_mask"]

    n_full = x_full.shape[0]
    print(f"  Full graph: {n_full} nodes, {edge_index.size(1)} edges, {x_full.shape[1]} features")
    print(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    # Build train+val subgraph
    tv_mask = (train_mask | val_mask).numpy()
    tv_indices = np.where(tv_mask)[0]
    n_tv = len(tv_indices)

    A_full = build_adj_matrix(edge_index, n_full)
    A_sub = A_full[np.ix_(tv_indices, tv_indices)]
    x_sub = x_full[tv_indices]  # (n_tv, p)
    y_sub = y_full[tv_indices]

    train_mask_sub = train_mask.numpy()[tv_indices]
    val_mask_sub = val_mask.numpy()[tv_indices]

    n_edges_sub = int(np.triu(A_sub, k=1).sum())
    print(f"  Train+Val subgraph: {n_tv} nodes, {n_edges_sub} edges")

    # === Load cached LSM fit (Step 1 already done) ===
    print(f"\nLoading LSM fit from {args.lsm_cache}...")
    with open(args.lsm_cache, "rb") as f:
        fit_data = pickle.load(f)
    Z1 = fit_data["model_Z"]         # (n_tv, r1)
    model_alpha = fit_data["model_alpha"]  # (n_tv,)
    model_sparsity = fit_data["model_sparsity"]
    r1 = Z1.shape[1]
    print(f"  Z1 shape: {Z1.shape}, alpha shape: {model_alpha.shape}, sparsity: {model_sparsity:.4f}")

    # === Step 2-3: Attribute regression and residual factorization ===
    print("\n" + "=" * 60)
    print("Step 2-3: Attribute regression + residual factorization")
    print("=" * 60)
    mu, Lambda1, Z2, Lambda2_T, d2 = fit_attribute_model(
        x_sub, Z1, max_d2=args.max_d2
    )

    # Allow overriding d2
    if args.d2_override is not None and args.d2_override != d2:
        print(f"\n  Overriding d2 from {d2} to {args.d2_override}")
        d2_new = args.d2_override
        # Recompute Z2, Lambda2_T with new d2
        mu_recomp = x_sub.mean(axis=0)
        Y_c = x_sub - mu_recomp
        R = Y_c - Z1 @ Lambda1
        U, s, Vt = np.linalg.svd(R, full_matrices=False)
        sqrt_s = np.sqrt(s[:d2_new])
        Z2 = U[:, :d2_new] * sqrt_s[np.newaxis, :]
        Lambda2_T = Vt[:d2_new, :] * sqrt_s[:, np.newaxis]
        d2 = d2_new
    r2 = Z2.shape[1]
    print(f"\n  Final dimensions: r1={r1}, d2={d2}")
    print(f"  Z1: {Z1.shape}, Z2: {Z2.shape}")
    print(f"  Lambda1: {Lambda1.shape}, Lambda2_T: {Lambda2_T.shape}")
    print(f"  mu: {mu.shape}")

    # Verify reconstruction quality on train+val
    Y_recon = reconstruct_attributes(Z1, Z2, mu, Lambda1, Lambda2_T)
    recon_error = np.linalg.norm(x_sub - Y_recon) / np.linalg.norm(x_sub)
    print(f"  Reconstruction relative error: {recon_error:.4f}")

    # === Generate synthetic graphs ===
    print("\n" + "=" * 60)
    print(f"Generating {args.n_reps} synthetic graphs via SyNGLER-Attr...")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    synth_artifacts = []

    for rep in range(args.n_reps):
        print(f"\n--- Rep {rep} ---")
        rep_seed = args.seed + rep + 10000

        # Bootstrap resample indices — generate full-size graph (n_full nodes)
        np.random.seed(rep_seed)
        n_fit = Z1.shape[0]
        n = n_full  # generate same size as original graph
        indices = np.random.choice(n_fit, size=n, replace=True)

        # Resample ALL latent variables with same indices
        Z1_star = Z1[indices]
        Z2_star = Z2[indices]
        alpha_star = model_alpha[indices]
        y_star = y_sub[indices]

        # Step 6: Reconstruct edges from Z1
        A_star, P_star = reconstruct_edges(
            Z1_star, alpha_star, model_sparsity, seed=rep_seed + 50000
        )
        n_edges_star = int(np.triu(A_star, k=1).sum())
        print(f"  Synthetic edges: {n_edges_star} (real subgraph: {n_edges_sub})")

        # Step 7: Reconstruct attributes
        Y_star = reconstruct_attributes(Z1_star, Z2_star, mu, Lambda1, Lambda2_T)

        # Binarize if requested (Cora has binary bag-of-words features)
        if args.binarize:
            if args.binarize_method == "density_match":
                # Match the feature density of real data
                real_density = x_sub.mean()  # fraction of 1s in real features
                threshold = np.quantile(Y_star.ravel(), 1.0 - real_density)
                Y_star_binary = (Y_star > threshold).astype(np.float32)
                print(f"  Attributes: density-matched binarization (threshold={threshold:.4f})")
            else:
                threshold = args.binarize_threshold
                Y_star_binary = (Y_star > threshold).astype(np.float32)
                print(f"  Attributes: fixed binarization (threshold={threshold})")
            n_nonzero = Y_star_binary.sum()
            n_nonzero_real = x_sub.sum()
            print(f"  Nonzero entries: synthetic={int(n_nonzero)}, real={int(n_nonzero_real)}")
            x_star = Y_star_binary
        else:
            x_star = Y_star.astype(np.float32)

        # Feature stats
        feat_sparsity_syn = (x_star == 0).mean()
        feat_sparsity_real = (x_sub == 0).mean()
        print(f"  Feature sparsity: synthetic={feat_sparsity_syn:.4f}, real={feat_sparsity_real:.4f}")
        print(f"  Class distribution: {np.bincount(y_star.astype(int))}")

        # Build edge_index
        edge_index_star = dense_adj_to_edge_index(A_star)

        # Create masks
        perm = np.random.permutation(n)
        train_mask_star = np.zeros(n, dtype=bool)
        val_mask_star = np.zeros(n, dtype=bool)
        test_mask_star = np.zeros(n, dtype=bool)

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
            "name": f"cora_syngler_attr_r{r1}_d2_{d2}_rep{rep}",
            "dataset": "Cora",
            "method": "SyNGLER-Attr",
            "x": torch.tensor(x_star, dtype=torch.float32),
            "y": torch.tensor(y_star, dtype=torch.int64),
            "edge_index": edge_index_star,
            "train_mask": torch.tensor(train_mask_star),
            "val_mask": torch.tensor(val_mask_star),
            "test_mask": torch.tensor(test_mask_star),
            "num_nodes": n,
            "num_edges": int(edge_index_star.size(1)),
            "num_features": x_star.shape[1],
            "num_classes": len(np.unique(y_star)),
            "r1": r1,
            "d2": d2,
        }

        out_path = os.path.join(args.output_dir, f"cora_syngler_attr_r{r1}_d2_{d2}_rep{rep}.pt")
        torch.save(synth_art, out_path)
        print(f"  Saved to {out_path}")
        synth_artifacts.append((synth_art, out_path))

    # === Evaluate ===
    print("\n" + "=" * 60)
    print("Evaluation: Node Classification Utility")
    print("Protocol: Train on synthetic graph, test on real test set")
    print("=" * 60)

    results_all = []

    for model_name, model_cls in [("GCN", DenseGCN), ("MLP", MLP)]:
        print(f"\n=== {model_name} ===")

        # Baseline: train on real, test on real
        real_real_accs = []
        for eval_seed in range(5):
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
        acc_g_g_std = np.std(real_real_accs)
        print(f"  ACC(G|G) = {acc_g_g:.4f} +/- {acc_g_g_std:.4f}")

        # For each synthetic graph
        all_synth_accs = []
        for synth_art, out_path in synth_artifacts:
            synth_real_accs = []
            for eval_seed in range(5):
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

            acc_ghat = np.mean(synth_real_accs)
            ratio = acc_ghat / acc_g_g if acc_g_g > 0 else 0
            all_synth_accs.append(acc_ghat)
            print(f"  {synth_art['name']}: ACC(G|Ghat)={acc_ghat:.4f} +/- {np.std(synth_real_accs):.4f}, ratio={ratio:.4f}")

            results_all.append({
                "model": model_name,
                "method": "SyNGLER-Attr",
                "synthetic": synth_art["name"],
                "acc_g_given_g": acc_g_g,
                "acc_g_given_g_std": acc_g_g_std,
                "acc_g_given_g_hat": acc_ghat,
                "acc_g_given_g_hat_std": float(np.std(synth_real_accs)),
                "utility_ratio": ratio,
            })

        # Average across reps
        mean_synth = np.mean(all_synth_accs)
        std_synth = np.std(all_synth_accs)
        mean_ratio = mean_synth / acc_g_g if acc_g_g > 0 else 0
        print(f"  --- {model_name} Average: ACC(G|Ghat)={mean_synth:.4f} +/- {std_synth:.4f}, ratio={mean_ratio:.4f}")

    # Save results
    results_path = os.path.join(args.output_dir, f"eval_results_syngler_attr_r{r1}.json")
    with open(results_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY: SyNGLER-Attr (Algorithm 3)")
    print(f"{'=' * 70}")
    print(f"{'Model':<6} {'Synthetic':<40} {'ACC(G|G)':<12} {'ACC(G|Ghat)':<14} {'Ratio':<8}")
    print("-" * 80)
    for r in results_all:
        print(f"{r['model']:<6} {r['synthetic']:<40} {r['acc_g_given_g']:<12.4f} {r['acc_g_given_g_hat']:<14.4f} {r['utility_ratio']:<8.4f}")

    # Aggregated summary
    print(f"\n{'=' * 70}")
    print("AGGREGATED (mean over reps)")
    print(f"{'=' * 70}")
    for model_name in ["GCN", "MLP"]:
        model_results = [r for r in results_all if r["model"] == model_name]
        if model_results:
            mean_acc_gg = model_results[0]["acc_g_given_g"]
            mean_acc_ghat = np.mean([r["acc_g_given_g_hat"] for r in model_results])
            std_acc_ghat = np.std([r["acc_g_given_g_hat"] for r in model_results])
            mean_ratio = mean_acc_ghat / mean_acc_gg if mean_acc_gg > 0 else 0
            print(f"  {model_name}: ACC(G|G)={mean_acc_gg:.4f}, ACC(G|Ghat)={mean_acc_ghat:.4f} +/- {std_acc_ghat:.4f}, ratio={mean_ratio:.4f}")


if __name__ == "__main__":
    main()
