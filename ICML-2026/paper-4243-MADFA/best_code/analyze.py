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
    python analyze.py --input loss_traces.npz --diagnose
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


def compute_ess(traces_2d):
    """
    Compute Effective Sample Size (ESS) using autocorrelation method.

    ESS = N / (1 + 2 * sum_{k=1}^{K} rho_k)
    Truncates at first negative autocorrelation (Geyer's initial positive sequence).

    Returns:
        ess_per_sample: (n_samples,) ESS per sample
        ess_median: median ESS across samples
        ess_mean: mean ESS across samples
    """
    n_draws, n_samples = traces_2d.shape
    centered = traces_2d - traces_2d.mean(axis=0, keepdims=True)

    max_lag = min(n_draws // 4, 200)
    ess_vals = np.zeros(n_samples)

    for s in range(n_samples):
        trace_s = centered[:, s]
        var_s = np.var(trace_s)
        if var_s < 1e-15:
            ess_vals[s] = n_draws
            continue

        acf_sum = 0.0
        for lag in range(1, max_lag + 1):
            rho = np.corrcoef(trace_s[:-lag], trace_s[lag:])[0, 1]
            if rho <= 0:
                break
            acf_sum += rho

        tau = 1.0 + 2.0 * acf_sum
        ess_vals[s] = n_draws / max(tau, 1.0)

    return ess_vals, np.median(ess_vals), np.mean(ess_vals)


def compute_rhat(traces):
    """
    Compute Gelman-Rubin R-hat statistic across chains.
    Requires traces shape (n_chains, n_draws, n_samples).
    R-hat near 1.0 indicates good mixing.

    Returns:
        rhat_per_sample: (n_samples,) R-hat per sample
        rhat_median: median R-hat
    """
    n_chains, n_draws, n_samples = traces.shape
    if n_chains < 2:
        return np.ones(n_samples), 1.0

    rhat_vals = np.zeros(n_samples)
    for s in range(n_samples):
        chain_means = traces[:, :, s].mean(axis=1)
        chain_vars = traces[:, :, s].var(axis=1, ddof=1)
        W = np.mean(chain_vars)
        B = n_draws * np.var(chain_means, ddof=1)
        var_plus = (n_draws - 1) / n_draws * W + B / n_draws
        rhat_vals[s] = 1.0 if W < 1e-15 else np.sqrt(var_plus / W)

    return rhat_vals, np.median(rhat_vals)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SGLD loss traces for anomaly detection")
    parser.add_argument("--input", type=str, default="loss_traces.npz",
                        help="Path to loss_traces.npz from run_sgld.py")
    parser.add_argument("--burn_in", type=int, default=250,
                        help="Number of initial draws to discard (default: 250)")
    parser.add_argument("--target_class", type=int, default=0,
                        help="Anomalous target class (default: 0)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run SGLD mixing quality diagnostics (ESS, R-hat)")
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

    print("Loaded traces:", traces.shape)
    print("  Chains:", traces.shape[0], " Draws:", traces.shape[1],
          " Samples:", traces.shape[2])
    print("  Burn-in:", burn_in)

    # --- SGLD mixing diagnostics ---
    if args.diagnose:
        print()
        print("=== SGLD Mixing Quality Diagnostics ===")

        # Before burn-in: build traces_2d
        if traces.shape[0] > 1:
            traces_2d_full = traces.reshape(-1, traces.shape[2])
        else:
            traces_2d_full = traces[0]

        # Compute ESS on full traces (pre burn-in) to see mixing state
        ess_full, ess_median_full, ess_mean_full = compute_ess(traces_2d_full)
        n_draws_full = traces_2d_full.shape[0]
        print()
        print("ESS (pre burn-in, all", n_draws_full, "draws):")
        print("  Median:", round(ess_median_full, 1),
              " Mean:", round(ess_mean_full, 1),
              " Min:", round(ess_full.min(), 1),
              " Max:", round(ess_full.max(), 1))
        ratio = ess_median_full / n_draws_full
        print("  ESS / n_draws ratio:", round(ratio, 4),
              "(1.0 = perfect mixing)")

        # Post burn-in ESS
        if burn_in > 0:
            if traces.shape[0] > 1:
                chains = []
                for c in range(traces.shape[0]):
                    chains.append(traces[c, burn_in:, :])
                traces_burned = np.concatenate(chains, axis=0)
            else:
                traces_burned = traces[0, burn_in:, :]
            ess_burned, ess_median_burned, ess_mean_burned = compute_ess(
                traces_burned)
            n_burned = traces_burned.shape[0]
            print()
            print("ESS (post burn-in,", n_burned, "draws):")
            print("  Median:", round(ess_median_burned, 1),
                  " Mean:", round(ess_mean_burned, 1),
                  " Min:", round(ess_burned.min(), 1),
                  " Max:", round(ess_burned.max(), 1))
            ratio_b = ess_median_burned / n_burned
            print("  ESS / n_draws ratio:", round(ratio_b, 4))

        # R-hat
        rhat, rhat_median = compute_rhat(traces)
        n_chains = traces.shape[0]
        print()
        print("R-hat (Gelman-Rubin, chains=" + str(n_chains) + "):")
        print("  Median:", round(rhat_median, 4),
              " Mean:", round(rhat.mean(), 4))
        if traces.shape[0] < 2:
            print("  (single chain; R-hat always 1.0. Multi-chain requires n_chains >= 2)")

        # Autocorrelation at lag 1 and lag 10
        if traces.shape[0] > 1:
            t2d = traces.reshape(-1, traces.shape[2])
        else:
            t2d = traces[0]
        n_sample_acf = min(t2d.shape[1], 1000)
        lag1_acf = np.zeros(n_sample_acf)
        lag10_acf = np.zeros(n_sample_acf)
        for s in range(n_sample_acf):
            lag1_acf[s] = np.corrcoef(t2d[:-1, s], t2d[1:, s])[0, 1]
            if t2d.shape[0] > 10:
                lag10_acf[s] = np.corrcoef(t2d[:-10, s], t2d[10:, s])[0, 1]
        print()
        print("Autocorrelation (first", n_sample_acf, "samples):")
        print("  Lag-1 ACF median:", round(np.median(lag1_acf), 4))
        print("  Lag-10 ACF median:", round(np.median(lag10_acf), 4))
        print("  (ACF near 0 = good mixing; near 1 = poor mixing)")

        # Mixing quality verdict
        ess_ratio = ess_median_full / n_draws_full
        if ess_ratio > 0.5:
            verdict = "GOOD: SGLD mixing is already efficient. Tuning may have limited impact."
        elif ess_ratio > 0.2:
            verdict = "MODERATE: Some autocorrelation present. LR schedule or chain changes could help."
        else:
            verdict = "POOR: Strong autocorrelation. SGLD hyperparameter changes warranted."
        print()
        print("Mixing Verdict:", verdict)
        print("=" * 50)

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

    print()
    print("Sample correctness:")
    print("  Trusted: ", trusted_correct_mask.sum(), "/", len(trusted_idx),
          "correctly classified (",
          round(100 * trusted_correct_mask.mean(), 1), "%)")
    print("  Benign:  ", benign_correct_mask.sum(), "/", len(benign_idx),
          "correctly classified (",
          round(100 * benign_correct_mask.mean(), 1), "%)")
    print("  Anomalous:", bd_attacked_mask.sum(), "/", len(anomalous_idx),
          "attack successful, i.e. predicted target class", args.target_class,
          "(", round(100 * bd_attacked_mask.mean(), 1), "%)")

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

    print("  Effective draws after burn-in:", traces_2d.shape[0])

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
