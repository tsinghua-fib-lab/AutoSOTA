#!/usr/bin/env python3
"""
Analyze SGLD loss traces: compute AUROC for anomaly detection.

Detection methods:
  - Mean Correlation: average Pearson correlation to all trusted samples
  - CLC (Class-Level Correlation): mean correlation per trusted class, take max
  - Mean CCC: average Concordance Correlation Coefficient to all trusted samples
  - Class CCC: mean CCC per trusted class, take max

Usage:
    python analyze.py --input loss_traces.npz
    python analyze.py --input loss_traces.npz --burn_in 250
"""

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score


def load_traces(path):
    """Load loss traces from .npz file."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def compute_correlation_matrix(test_traces, trusted_traces):
    """
    Compute Pearson correlation between each test sample and each trusted sample.

    Returns:
        corr_matrix: (n_test, n_trusted) Pearson correlations
    """
    n_timesteps = test_traces.shape[1]

    test_norm = (test_traces - test_traces.mean(axis=1, keepdims=True))
    test_norm = test_norm / (test_traces.std(axis=1, keepdims=True) + 1e-8)

    trusted_norm = (trusted_traces - trusted_traces.mean(axis=1, keepdims=True))
    trusted_norm = trusted_norm / (trusted_traces.std(axis=1, keepdims=True) + 1e-8)

    return np.dot(test_norm, trusted_norm.T) / n_timesteps


def compute_ccc_matrix(test_traces, trusted_traces):
    """
    Compute Concordance Correlation Coefficient between each test and trusted sample.

    CCC = 2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2)

    Returns:
        ccc_matrix: (n_test, n_trusted)
    """
    n_timesteps = test_traces.shape[1]

    test_means = test_traces.mean(axis=1, keepdims=True)
    trusted_means = trusted_traces.mean(axis=1, keepdims=True)
    test_vars = test_traces.var(axis=1, keepdims=True)
    trusted_vars = trusted_traces.var(axis=1, keepdims=True)

    cov_matrix = np.dot(
        test_traces - test_means,
        (trusted_traces - trusted_means).T
    ) / n_timesteps

    mean_diff_sq = (test_means - trusted_means.T) ** 2
    var_sum = test_vars + trusted_vars.T

    return (2 * cov_matrix) / (var_sum + mean_diff_sq + 1e-8)


def compute_mean_score(matrix):
    """Mean score across all trusted samples (higher = more benign)."""
    return np.mean(matrix, axis=1)


def compute_class_max_score(matrix, trusted_pred_labels, n_classes=10):
    """Mean score per trusted class, take max across classes."""
    scores = np.full(matrix.shape[0], -np.inf)
    for c in range(n_classes):
        mask = trusted_pred_labels == c
        if mask.sum() == 0:
            continue
        class_mean = np.mean(matrix[:, mask], axis=1)
        scores = np.maximum(scores, class_mean)
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SGLD loss traces for anomaly detection")
    parser.add_argument("--input", type=str, default="loss_traces.npz",
                        help="Path to loss_traces.npz from run_sgld.py")
    parser.add_argument("--burn_in", type=int, default=250,
                        help="Number of initial draws to discard (default: 250)")
    parser.add_argument("--target_class", type=int, default=0,
                        help="Anomalous target class (default: 0)")
    args = parser.parse_args()

    # Load data
    data = load_traces(args.input)
    traces = data["loss_traces"]           # [n_chains, n_draws, n_samples]
    predicted_labels = data["predicted_labels"]  # [n_samples]
    gt_labels = data["gt_labels"]          # [n_samples]
    trusted_idx = data["trusted_idx"]
    benign_idx = data["benign_idx"]
    anomalous_idx = data["anomalous_idx"]

    burn_in = args.burn_in

    print(f"Loaded traces: {traces.shape}")
    print(f"  Chains: {traces.shape[0]}, Draws: {traces.shape[1]}, "
          f"Samples: {traces.shape[2]}")
    print(f"  Burn-in: {burn_in}")

    # Trusted: correctly classified (predicted == ground truth)
    trusted_preds = predicted_labels[trusted_idx]
    trusted_gt = gt_labels[trusted_idx]
    trusted_correct_mask = trusted_preds == trusted_gt
    trusted_correct_local = np.where(trusted_correct_mask)[0]

    # Benign: correctly classified (predicted == ground truth)
    benign_preds = predicted_labels[benign_idx]
    benign_gt = gt_labels[benign_idx]
    benign_correct_mask = benign_preds == benign_gt
    benign_correct_local = np.where(benign_correct_mask)[0]

    # Anomalous: successfully attacked (predicted == target class)
    bd_preds = predicted_labels[anomalous_idx]
    bd_attacked_mask = bd_preds == args.target_class
    bd_attacked_local = np.where(bd_attacked_mask)[0]

    print(f"\nSample correctness:")
    print(f"  Trusted:  {trusted_correct_mask.sum()} / {len(trusted_idx)} "
          f"correctly classified ({100 * trusted_correct_mask.mean():.1f}%)")
    print(f"  Benign:   {benign_correct_mask.sum()} / {len(benign_idx)} "
          f"correctly classified ({100 * benign_correct_mask.mean():.1f}%)")
    print(f"  Anomalous: {bd_attacked_mask.sum()} / {len(anomalous_idx)} "
          f"attack successful, i.e. predicted target class {args.target_class} "
          f"({100 * bd_attacked_mask.mean():.1f}%)")

    if traces.shape[0] > 1:
        traces_2d = traces.reshape(-1, traces.shape[2])
    else:
        traces_2d = traces[0]

    if burn_in > 0:
        if traces.shape[0] > 1:
            chains = []
            for c in range(traces.shape[0]):
                chains.append(traces[c, burn_in:, :])
            traces_2d = np.concatenate(chains, axis=0)
        else:
            traces_2d = traces_2d[burn_in:]

    print(f"  Effective draws after burn-in: {traces_2d.shape[0]}")

    # Transpose: (n_samples, n_timesteps)
    traces_by_sample = traces_2d.T

    # Trusted: only correctly classified
    trusted_global_filtered = trusted_idx[trusted_correct_local]
    trusted_traces = traces_by_sample[trusted_global_filtered]
    trusted_pred_labels = predicted_labels[trusted_global_filtered]

    # Test: only correct benign + successfully-attacked anomalous
    benign_global_filtered = benign_idx[benign_correct_local]
    bd_global_filtered = anomalous_idx[bd_attacked_local]

    benign_traces = traces_by_sample[benign_global_filtered]
    anomalous_traces = traces_by_sample[bd_global_filtered]

    test_traces = np.vstack([benign_traces, anomalous_traces])
    test_labels = np.concatenate([
        np.zeros(len(benign_traces)),
        np.ones(len(anomalous_traces)),
    ])

    n_benign = len(benign_traces)

    corr_matrix = compute_correlation_matrix(test_traces, trusted_traces)
    ccc_matrix = compute_ccc_matrix(test_traces, trusted_traces)

    methods = []

    # Mean Correlation
    scores = compute_mean_score(corr_matrix)
    methods.append(("Mean Corr", scores))

    # Class-Level Correlation (CLC)
    scores = compute_class_max_score(corr_matrix, trusted_pred_labels)
    methods.append(("CLC", scores))

    # Mean CCC
    scores = compute_mean_score(ccc_matrix)
    methods.append(("Mean CCC", scores))

    # Class CCC
    scores = compute_class_max_score(ccc_matrix, trusted_pred_labels)
    methods.append(("Class CCC", scores))

    print()
    header = (f"{'Method':<16} | {'AUROC':>7} | "
              f"{'Mean (Benign)':>15} | {'Mean (Anomalous)':>17}")
    print(header)
    print("-" * len(header))

    for name, scores in methods:
        auroc = roc_auc_score(test_labels, -scores)
        benign_mean = scores[:n_benign].mean()
        bd_mean = scores[n_benign:].mean()
        print(f"{name:<16} | {auroc:>7.4f} | "
              f"{benign_mean:>15.4f} | {bd_mean:>17.4f}")

    print()


if __name__ == "__main__":
    main()
