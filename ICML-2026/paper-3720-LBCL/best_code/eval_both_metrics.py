#!/usr/bin/env python3
"""
Comprehensive evaluation for LCL model on CellTag dataset.
Computes both KNN Test Error and KL Divergence for future composition prediction.
"""
import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import anndata as ad

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ============================================================================
# PART 1: KNN Test Error
# ============================================================================
def evaluate_knn(train_h5ad, test_h5ad, train_embed, test_embed, n_neighbors=5, min_test_size=16):
    """Evaluate KNN classification accuracy and return test error."""
    adata_train = ad.read_h5ad(train_h5ad)
    adata_test = ad.read_h5ad(test_h5ad)

    train_labels = adata_train.obs["clone_id"].to_numpy()
    test_labels = adata_test.obs["clone_id"].to_numpy()

    train_embeddings = np.load(train_embed)
    test_embeddings = np.load(test_embed)

    print(f"Train labels: {train_labels.shape}, Train embeddings: {train_embeddings.shape}")
    print(f"Test labels: {test_labels.shape}, Test embeddings: {test_embeddings.shape}")

    # Filter out lineages with fewer than min_test_size cells in test set
    test_sizes = pd.Series(test_labels).value_counts()
    kept_lineages = test_sizes[test_sizes >= min_test_size].index
    test_keep = pd.Series(test_labels).isin(kept_lineages).to_numpy()
    X_test_f = test_embeddings[test_keep]
    y_test_f = test_labels[test_keep]

    print(f"Test cells after filtering (>= {min_test_size} cells/lineage): {X_test_f.shape[0]} / {test_embeddings.shape[0]}")

    # KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_embeddings, train_labels)
    y_pred = knn.predict(X_test_f)
    accuracy = accuracy_score(y_test_f, y_pred)
    test_error = 1.0 - accuracy

    print(f"\n=== KNN RESULTS ===")
    print(f"KNN Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"KNN Test Error: {test_error:.4f}")

    # Also compute without min_size filter
    knn_all = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_all.fit(train_embeddings, train_labels)
    y_pred_all = knn_all.predict(test_embeddings)
    accuracy_all = accuracy_score(test_labels, y_pred_all)
    test_error_all = 1.0 - accuracy_all

    print(f"\n=== Without lineage size filter ===")
    print(f"KNN Test Accuracy (all): {accuracy_all:.4f} ({accuracy_all*100:.2f}%)")
    print(f"KNN Test Error (all): {test_error_all:.4f}")

    return {
        "KNN_Test_Accuracy": round(float(accuracy), 4),
        "KNN_Test_Error": round(float(test_error), 4),
        "KNN_Test_Accuracy_all": round(float(accuracy_all), 4),
        "KNN_Test_Error_all": round(float(test_error_all), 4),
        "n_neighbors": n_neighbors,
        "n_test_cells_filtered": int(X_test_f.shape[0]),
        "n_test_cells_total": int(test_embeddings.shape[0]),
        "n_train_cells": int(train_embeddings.shape[0]),
    }


# ============================================================================
# PART 2: KL Divergence for Future Composition Prediction
# ============================================================================
def filter_by_clone_future_size(adata, X, day_key="reprogramming_day", lineage_key="clone_id",
                                 future_day="28", min_future_cells=10):
    """Keep all cells whose clone has >= min_future_cells at future_day."""
    day = adata.obs[day_key].astype(str)
    is_future = (day == str(future_day))
    counts = adata.obs.loc[is_future, lineage_key].value_counts()
    keep_clones = set(counts[counts >= int(min_future_cells)].index)
    keep_mask = adata.obs[lineage_key].isin(keep_clones).to_numpy()
    return adata[keep_mask].copy(), X[keep_mask]


def build_targets_from_future(X, adata, early_day="12", future_day="28",
                               lineage_key="clone_id", celltype_key="cell_type",
                               terminal_types=("iEP", "Fibroblast", "Ambiguous"),
                               alpha_smooth=1e-3, drop_missing_future=True):
    """Build Day12 inputs -> lineage composition targets from Day28."""
    terminal_types = list(terminal_types)
    C = len(terminal_types)

    future_mask = (adata.obs["reprogramming_day"].astype(str) == str(future_day))
    adata_future = adata[future_mask].copy()

    clone_to_probs = {}
    for clone_id, df in adata_future.obs.groupby(lineage_key):
        counts = np.array([(df[celltype_key] == ct).sum() for ct in terminal_types], dtype=float)
        counts = counts + alpha_smooth
        probs = counts / counts.sum()
        clone_to_probs[clone_id] = probs

    early_mask = (adata.obs["reprogramming_day"].astype(str) == str(early_day))
    early_idx = np.where(early_mask.values)[0]

    X_early = X[early_idx]
    clone_early = adata.obs.iloc[early_idx][lineage_key].to_numpy()

    y_prob = np.zeros((X_early.shape[0], C), dtype=float)
    keep = np.ones(X_early.shape[0], dtype=bool)

    for i, cid in enumerate(clone_early):
        if cid in clone_to_probs:
            y_prob[i] = clone_to_probs[cid]
        else:
            if drop_missing_future:
                keep[i] = False
            else:
                y_prob[i] = np.ones(C) / C

    X_early = X_early[keep]
    y_prob = y_prob[keep]
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    return torch.tensor(X_early, dtype=torch.float32), torch.tensor(y_prob, dtype=torch.float32)


class LinearSoftmax(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


def train_kl_earlystop(model, X_train, y_train, lr=5e-3, weight_decay=1e-4,
                        max_epochs=5000, batch_size=256, val_frac=0.2,
                        patience=150, min_delta=1e-5, seed=42, device=None,
                        print_every=50, verbose=True):
    """Train linear decoder with early stopping for KL divergence."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    n = X_train.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=g)

    n_val = int(round(val_frac * n))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.KLDivLoss(reduction="batchmean")

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    bad_epochs = 0

    history = {"train_loss": [], "val_loss": [], "best_epoch": None, "best_val": None}

    @torch.no_grad()
    def eval_loss(Xe, ye):
        model.eval()
        log_probs = torch.log_softmax(model(Xe), dim=1)
        return criterion(log_probs, ye).item()

    for ep in range(1, max_epochs + 1):
        model.train()
        perm_tr = torch.randperm(X_tr.shape[0], device=device)
        Xs = X_tr[perm_tr]
        ys = y_tr[perm_tr]

        total = 0.0
        for start in range(0, Xs.shape[0], batch_size):
            xb = Xs[start:start + batch_size]
            yb = ys[start:start + batch_size]

            logits = model(xb)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = criterion(log_probs, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * xb.shape[0]

        train_loss = total / Xs.shape[0]
        val_loss = eval_loss(X_val, y_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = (best_val - val_loss) > min_delta
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose and (ep == 1 or ep % print_every == 0):
            print(f"Epoch {ep}/{max_epochs} | train={train_loss:.6f} | val={val_loss:.6f} "
                  f"| best_val={best_val:.6f} (ep {best_epoch}) | bad={bad_epochs}/{patience}")

        if bad_epochs >= patience:
            if verbose:
                print(f"Early stopping at epoch {ep}. Best val={best_val:.6f} at epoch {best_epoch}.")
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val"] = best_val
    return model, history


@torch.no_grad()
def eval_kl(model, X, y):
    model.eval()
    device = next(model.parameters()).device
    X = X.to(device)
    y = y.to(device)
    criterion = nn.KLDivLoss(reduction="batchmean")
    log_probs = torch.log_softmax(model(X), dim=1)
    return criterion(log_probs, y).item()


def evaluate_kl_divergence(train_h5ad, test_h5ad, train_embed, test_embed,
                            early_day="12", future_day="28",
                            terminal_types=("iEP", "Fibroblast", "Ambiguous"),
                            device="cuda", seed=42, verbose=True):
    """Evaluate KL divergence for future cell-type composition prediction."""
    adata_train = ad.read_h5ad(train_h5ad)
    adata_test = ad.read_h5ad(test_h5ad)

    X_train = np.load(train_embed)
    X_test = np.load(test_embed)

    print(f"\n=== KL DIVERGENCE EVALUATION ===")
    print(f"Early time point: Day {early_day}, Future time point: Day {future_day}")
    print(f"Terminal cell types: {terminal_types}")

    # Filter and build targets separately for train/test
    ad_tr_f, X_tr_f = filter_by_clone_future_size(adata_train, X_train,
                                                   future_day=future_day, min_future_cells=0)
    ad_te_f, X_te_f = filter_by_clone_future_size(adata_test, X_test,
                                                   future_day=future_day, min_future_cells=0)

    X_tr12, y_tr = build_targets_from_future(X_tr_f, ad_tr_f, early_day=early_day,
                                              future_day=future_day, terminal_types=terminal_types)
    X_te12, y_te = build_targets_from_future(X_te_f, ad_te_f, early_day=early_day,
                                              future_day=future_day, terminal_types=terminal_types)

    print(f"Train Day-{early_day} cells used: {X_tr12.shape[0]}")
    print(f"Test Day-{early_day} cells used: {X_te12.shape[0]}")
    print(f"Output dimension (n_terminal_cell_types): {len(terminal_types)}")

    # Train linear decoder with early stopping
    model = LinearSoftmax(input_size=X_tr12.shape[1], output_size=len(terminal_types))
    model, hist = train_kl_earlystop(
        model, X_tr12, y_tr,
        lr=5e-3, weight_decay=1e-4,
        max_epochs=5000, batch_size=256,
        val_frac=0.2, patience=150,
        min_delta=1e-5, seed=seed,
        device=device, print_every=50, verbose=verbose
    )

    # Evaluate on test
    kl_test = eval_kl(model, X_te12, y_te)

    print(f"\n=== KL DIVERGENCE RESULTS ===")
    print(f"Best epoch: {hist['best_epoch']}")
    print(f"Best val KL: {hist['best_val']:.6f}")
    print(f"Test KL Divergence: {kl_test:.4f}")

    return {
        "KL_Divergence": round(float(kl_test), 4),
        "KL_Val_Best": round(float(hist["best_val"]), 4),
        "KL_Best_Epoch": int(hist["best_epoch"]),
        "early_day": early_day,
        "future_day": future_day,
        "terminal_cell_types": list(terminal_types),
        "n_train_day12_cells": int(X_tr12.shape[0]),
        "n_test_day12_cells": int(X_te12.shape[0]),
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LCL model: KNN Test Error + KL Divergence")
    parser.add_argument("--train_h5ad", required=True, help="Path to training .h5ad")
    parser.add_argument("--test_h5ad", required=True, help="Path to test .h5ad")
    parser.add_argument("--train_embed", required=True, help="Path to training embeddings .npy")
    parser.add_argument("--test_embed", required=True, help="Path to test embeddings .npy")
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of KNN neighbors")
    parser.add_argument("--min_test_size", type=int, default=16,
                        help="Minimum test cells per lineage for filtered KNN")
    parser.add_argument("--early_day", type=str, default="12", help="Early time point for KL divergence")
    parser.add_argument("--future_day", type=str, default="28", help="Future time point for KL divergence")
    parser.add_argument("--skip_kl", action="store_true", help="Skip KL divergence evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device for KL training")
    args = parser.parse_args()

    print("=" * 70)
    print("LCL MODEL EVALUATION")
    print("=" * 70)
    print(f"Train h5ad: {args.train_h5ad}")
    print(f"Test h5ad: {args.test_h5ad}")
    print(f"Train embeddings: {args.train_embed}")
    print(f"Test embeddings: {args.test_embed}")
    print()

    # KNN Evaluation
    knn_results = evaluate_knn(
        args.train_h5ad, args.test_h5ad,
        args.train_embed, args.test_embed,
        n_neighbors=args.n_neighbors,
        min_test_size=args.min_test_size,
    )

    results = {"KNN": knn_results}

    # KL Divergence Evaluation
    if not args.skip_kl:
        kl_results = evaluate_kl_divergence(
            args.train_h5ad, args.test_h5ad,
            args.train_embed, args.test_embed,
            early_day=args.early_day, future_day=args.future_day,
            terminal_types=("iEP", "Fibroblast", "Ambiguous"),
            device=args.device, seed=42, verbose=True,
        )
        results["KL"] = kl_results

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"KNN Test Error: {knn_results['KNN_Test_Error']:.4f}")
    if not args.skip_kl:
        print(f"KL Divergence: {kl_results['KL_Divergence']:.4f}")

    # Save results to JSON
    import json
    results_path = os.path.join(os.path.dirname(args.train_embed) or ".", "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
