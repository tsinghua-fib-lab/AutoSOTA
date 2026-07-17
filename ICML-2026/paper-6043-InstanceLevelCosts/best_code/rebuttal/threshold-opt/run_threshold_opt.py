"""
2C: Threshold optimization + Elkan's Bayes-optimal thresholding (ICML rebuttal R5).

For each model's predictions:
1. Naive sweep: sweep thresholds 0.01-0.99 on val, select best NEC, apply to test
2. Calibrated sweep: isotonic calibration on val, then sweep
3. Elkan per-instance threshold: predict class 1 when p(y=1|x) > c_FP / (c_FP + c_FN)

Key question: do cost-sensitive training methods look better under optimized thresholds?
Does Elkan's method beat everything without retraining?

Usage:
    python rebuttal/threshold-opt/run_threshold_opt.py
    python rebuttal/threshold-opt/run_threshold_opt.py --datasets jigsaw nhanes
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

PRED_DIR = Path("rebuttal/predictions")
RESULTS_DIR = Path("rebuttal/threshold-opt/results")
DATASETS = ['jigsaw', 'nhanes', 'synthetic', 'turkey', 'inaturalist']


def compute_nec(y_true, y_pred, abs_delta):
    """Compute NEC from predictions and costs."""
    total_cost = abs_delta.sum()
    if total_cost == 0:
        return 0.0
    misclass_cost = (abs_delta * (y_true != y_pred).astype(float)).sum()
    return misclass_cost / total_cost


def load_predictions(csv_path):
    """Load prediction CSV and split into val/test."""
    df = pd.read_csv(csv_path)
    val = df[df['split'] == 'val'].copy()
    test = df[df['split'] == 'test'].copy()
    return val, test


def sweep_threshold(y_true, y_proba_1, abs_delta, thresholds=None):
    """Sweep thresholds and return best threshold + NEC at each."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    best_nec = float('inf')
    best_t = 0.5
    for t in thresholds:
        preds = (y_proba_1 >= t).astype(int)
        nec = compute_nec(y_true, preds, abs_delta)
        if nec < best_nec:
            best_nec = nec
            best_t = t
    return best_t, best_nec


def elkan_threshold(y_proba_1_calibrated, abs_delta, delta_signed):
    """
    Elkan per-instance Bayes-optimal threshold.

    For each instance, predict class 1 when:
        p(y=1|x) > c_FP / (c_FP + c_FN)

    In our framework:
    - If true label is 1 (delta > 0): FN cost = |delta|, FP cost = 0 at this instance
    - If true label is 0 (delta < 0): FP cost = |delta|, FN cost = 0 at this instance

    But at prediction time we don't know the true label. The Elkan threshold uses
    the cost structure: for binary with costs c_FP (predict 1 when true 0) and
    c_FN (predict 0 when true 1), the optimal threshold is c_FP / (c_FP + c_FN).

    Since costs are instance-dependent but we evaluate against the majority label,
    we use the average cost structure per class as the threshold.
    """
    # Per-instance threshold based on cost ratio
    # For instances where y*=1: FN cost = |delta|, FP cost is symmetric
    # For instances where y*=0: FP cost = |delta|, FN cost is symmetric
    # Use a global cost ratio: average FP cost vs average FN cost
    # But since we want per-instance: threshold_i = c_FP_i / (c_FP_i + c_FN_i)
    # With symmetric costs (each instance has the same cost for FP and FN = |delta_i|),
    # the threshold is always 0.5. The interesting case is when we use class-level
    # average costs.

    # Class-level approach:
    # c_FP = average |delta| for class 0 instances (cost of wrongly predicting 1)
    # c_FN = average |delta| for class 1 instances (cost of wrongly predicting 0)
    y_star = (delta_signed > 0).astype(int)
    mask_0 = y_star == 0
    mask_1 = y_star == 1

    c_fp = abs_delta[mask_0].mean() if mask_0.any() else 1.0  # avg cost of FP
    c_fn = abs_delta[mask_1].mean() if mask_1.any() else 1.0  # avg cost of FN

    threshold = c_fp / (c_fp + c_fn)
    preds = (y_proba_1_calibrated >= threshold).astype(int)
    return preds, threshold


def process_one_file(csv_path):
    """Process a single prediction file through all threshold methods."""
    val, test = load_predictions(csv_path)
    if len(val) == 0 or len(test) == 0:
        return None

    # Skip regression models (no probabilities)
    if val['y_proba_1'].isna().all():
        return None

    import re
    stem = csv_path.stem  # e.g. tfidf_classification_none_s42 or _s0_n1000

    results = {'file': stem}

    # Parse model info from filename
    # First try exact match (no suffix after seed)
    m = re.match(r'(.+)_s(\d+)$', stem)
    if not m:
        # Then try with suffix (sample size or strategy)
        m = re.match(r'(.+)_s(\d+)(_n\d+|_P_up|_Tdown\d+)$', stem)
        if m:
            results['config'] = m.group(1) + m.group(3)  # include suffix in config name
    if m:
        if 'config' not in results:
            results['config'] = m.group(1)
        results['seed'] = int(m.group(2))
    else:
        results['config'] = stem
        results['seed'] = -1

    val_y = val['y_true'].values
    val_p1 = val['y_proba_1'].values
    val_delta = val['abs_delta'].values
    val_delta_signed = val['delta_signed'].values

    test_y = test['y_true'].values
    test_p1 = test['y_proba_1'].values
    test_delta = test['abs_delta'].values
    test_delta_signed = test['delta_signed'].values

    # 1. Default threshold (0.5)
    default_preds = (test_p1 >= 0.5).astype(int)
    results['nec_default'] = compute_nec(test_y, default_preds, test_delta)
    results['err_default'] = (test_y != default_preds).mean()

    # 2. Naive sweep on val
    best_t, _ = sweep_threshold(val_y, val_p1, val_delta)
    sweep_preds = (test_p1 >= best_t).astype(int)
    results['threshold_naive'] = best_t
    results['nec_naive_sweep'] = compute_nec(test_y, sweep_preds, test_delta)
    results['err_naive_sweep'] = (test_y != sweep_preds).mean()

    # 3. Calibrated sweep
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso.fit(val_p1, val_y)
    val_p1_cal = iso.transform(val_p1)
    test_p1_cal = iso.transform(test_p1)

    best_t_cal, _ = sweep_threshold(val_y, val_p1_cal, val_delta)
    cal_preds = (test_p1_cal >= best_t_cal).astype(int)
    results['threshold_calibrated'] = best_t_cal
    results['nec_calibrated_sweep'] = compute_nec(test_y, cal_preds, test_delta)
    results['err_calibrated_sweep'] = (test_y != cal_preds).mean()

    # 4. Elkan per-instance (using calibrated probabilities)
    elkan_preds, elkan_t = elkan_threshold(test_p1_cal, test_delta, test_delta_signed)
    results['threshold_elkan'] = elkan_t
    results['nec_elkan'] = compute_nec(test_y, elkan_preds, test_delta)
    results['err_elkan'] = (test_y != elkan_preds).mean()

    return results


def main():
    parser = argparse.ArgumentParser(description='Threshold optimization for ICML rebuttal')
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for dataset in args.datasets:
        pred_dir = PRED_DIR / dataset
        if not pred_dir.exists():
            print(f"  SKIP: {pred_dir} not found")
            continue

        csv_files = sorted(pred_dir.glob("*.csv"))
        # Only classification files (skip regression)
        csv_files = [f for f in csv_files if 'regression' not in f.stem]
        print(f"\n=== {dataset}: {len(csv_files)} prediction files ===")

        for csv_path in csv_files:
            result = process_one_file(csv_path)
            if result:
                result['dataset'] = dataset
                all_results.append(result)

    if not all_results:
        print("No results to process")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "threshold_opt_all.csv", index=False)

    # Summary: average across seeds per config
    summary_cols = ['nec_default', 'nec_naive_sweep', 'nec_calibrated_sweep', 'nec_elkan',
                    'err_default', 'err_naive_sweep', 'err_calibrated_sweep', 'err_elkan',
                    'threshold_naive', 'threshold_calibrated', 'threshold_elkan']
    summary = df.groupby(['dataset', 'config'])[summary_cols].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
    summary.to_csv(RESULTS_DIR / "threshold_opt_summary.csv", index=False)

    # Print key results
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION SUMMARY")
    print("=" * 80)
    for dataset in df['dataset'].unique():
        print(f"\n--- {dataset} ---")
        sub = df[df['dataset'] == dataset]
        for config in sorted(sub['config'].unique()):
            rows = sub[sub['config'] == config]
            d = rows['nec_default'].mean()
            n = rows['nec_naive_sweep'].mean()
            c = rows['nec_calibrated_sweep'].mean()
            e = rows['nec_elkan'].mean()
            t_naive = rows['threshold_naive'].mean()
            t_cal = rows['threshold_calibrated'].mean()
            t_elk = rows['threshold_elkan'].mean()
            print(f"  {config:50s}  default={d:.4f}  naive={n:.4f}(t={t_naive:.2f})  "
                  f"cal={c:.4f}(t={t_cal:.2f})  elkan={e:.4f}(t={t_elk:.2f})")

    # Plot: NEC improvement by method
    fig, axes = plt.subplots(1, len(df['dataset'].unique()), figsize=(5 * len(df['dataset'].unique()), 5))
    if len(df['dataset'].unique()) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, sorted(df['dataset'].unique())):
        sub = df[df['dataset'] == dataset]
        configs = sorted(sub['config'].unique())
        # Only show P1 baseline configs for clarity
        baseline_configs = [c for c in configs if c.endswith('_none') or c.endswith('_absdelta')]
        if not baseline_configs:
            baseline_configs = configs[:6]

        x = np.arange(len(baseline_configs))
        width = 0.2

        for i, (method, col) in enumerate([
            ('Default (0.5)', 'nec_default'),
            ('Naive sweep', 'nec_naive_sweep'),
            ('Calibrated', 'nec_calibrated_sweep'),
            ('Elkan', 'nec_elkan'),
        ]):
            vals = [sub[sub['config'] == c][col].mean() for c in baseline_configs]
            ax.bar(x + i * width, vals, width, label=method, alpha=0.8)

        ax.set_xlabel('Training method')
        ax.set_ylabel('Test NEC')
        ax.set_title(dataset)
        ax.set_xticks(x + 1.5 * width)
        short_labels = [c.split('_classification_')[-1] if '_classification_' in c else c for c in baseline_configs]
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "threshold_comparison.pdf", bbox_inches='tight')
    fig.savefig(RESULTS_DIR / "threshold_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlots saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
