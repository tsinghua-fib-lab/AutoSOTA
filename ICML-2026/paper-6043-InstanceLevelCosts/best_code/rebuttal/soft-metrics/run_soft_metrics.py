"""
2B: Soft evaluation metrics comparison (ICML rebuttal R3).

Computes Brier score, log-loss, and NEC for every model in rebuttal/predictions/.
Generates selective classification risk-coverage curves.
Checks whether NEC changes model rankings relative to these standard metrics.

Usage:
    python rebuttal/soft-metrics/run_soft_metrics.py
    python rebuttal/soft-metrics/run_soft_metrics.py --datasets jigsaw nhanes
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import brier_score_loss, log_loss
from scipy import stats

PRED_DIR = Path("rebuttal/predictions")
RESULTS_DIR = Path("rebuttal/soft-metrics/results")
DATASETS = ['jigsaw', 'nhanes', 'synthetic', 'turkey', 'inaturalist']


def compute_nec(y_true, y_pred, abs_delta):
    """Compute NEC."""
    total = abs_delta.sum()
    if total == 0:
        return 0.0
    return (abs_delta * (y_true != y_pred).astype(float)).sum() / total


def compute_metrics(df_test):
    """Compute all metrics for a test split."""
    y = df_test['y_true'].values
    p1 = df_test['y_proba_1'].values
    preds = df_test['y_pred'].values
    delta = df_test['abs_delta'].values

    if np.isnan(p1).all():
        return None

    # Clip probabilities for numerical stability
    p1_clipped = np.clip(p1, 1e-15, 1 - 1e-15)

    return {
        'error_rate': float((y != preds).mean()),
        'nec': float(compute_nec(y, preds, delta)),
        'brier': float(brier_score_loss(y, p1_clipped)),
        'log_loss': float(log_loss(y, p1_clipped)),
    }


def selective_classification_curve(y_true, y_proba_1, abs_delta, n_points=100):
    """
    Risk-coverage curve for selective classification.

    Sort examples by model confidence (descending).
    At each coverage level, compute error rate and NEC on the covered examples.
    """
    confidence = np.maximum(y_proba_1, 1 - y_proba_1)
    sorted_idx = np.argsort(-confidence)  # most confident first

    y_sorted = y_true[sorted_idx]
    p_sorted = y_proba_1[sorted_idx]
    d_sorted = abs_delta[sorted_idx]
    pred_sorted = (p_sorted >= 0.5).astype(int)

    n = len(y_true)
    coverages = np.linspace(0.1, 1.0, n_points)
    errors = []
    necs = []

    for cov in coverages:
        k = max(1, int(cov * n))
        y_sub = y_sorted[:k]
        pred_sub = pred_sorted[:k]
        d_sub = d_sorted[:k]

        errors.append(float((y_sub != pred_sub).mean()))
        total_d = d_sub.sum()
        if total_d > 0:
            necs.append(float((d_sub * (y_sub != pred_sub)).sum() / total_d))
        else:
            necs.append(0.0)

    return coverages, errors, necs


def main():
    parser = argparse.ArgumentParser(description='Soft evaluation metrics comparison')
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
        csv_files = [f for f in csv_files if 'regression' not in f.stem]
        print(f"\n=== {dataset}: {len(csv_files)} files ===")

        dataset_results = []
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            test = df[df['split'] == 'test']
            if len(test) == 0:
                continue

            metrics = compute_metrics(test)
            if metrics is None:
                continue

            # Parse config — handle _s{seed}_n{size} and _s{seed}_{strategy}
            import re
            stem = csv_path.stem
            m = re.match(r'(.+)_s(\d+)(?:_.*)?$', stem)
            config = m.group(1) if m else stem
            seed = int(m.group(2)) if m else -1

            metrics['dataset'] = dataset
            metrics['config'] = config
            metrics['seed'] = seed
            metrics['file'] = stem
            dataset_results.append(metrics)
            all_results.append(metrics)

        if not dataset_results:
            continue

        # Selective classification curves for P1 baselines (seed 42)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        baseline_files = [f for f in csv_files
                         if '_s42' in f.stem and 'focal' not in f.stem
                         and 'zadrozny' not in f.stem and 'ensemble' not in f.stem
                         and 'reg_loss' not in f.stem and '_n' not in f.stem
                         and '_P_up' not in f.stem and '_Tdown' not in f.stem]

        for csv_path in baseline_files[:6]:
            df = pd.read_csv(csv_path)
            test = df[df['split'] == 'test']
            if len(test) == 0 or test['y_proba_1'].isna().all():
                continue

            label = csv_path.stem.replace('_s42', '').replace('_classification', '')
            covs, errs, necs = selective_classification_curve(
                test['y_true'].values, test['y_proba_1'].values, test['abs_delta'].values
            )
            axes[0].plot(covs, errs, label=label, alpha=0.8)
            axes[1].plot(covs, necs, label=label, alpha=0.8)

        axes[0].set_xlabel('Coverage')
        axes[0].set_ylabel('Error Rate')
        axes[0].set_title(f'{dataset}: Selective Classification (Error)')
        axes[0].legend(fontsize=7)

        axes[1].set_xlabel('Coverage')
        axes[1].set_ylabel('NEC')
        axes[1].set_title(f'{dataset}: Selective Classification (NEC)')
        axes[1].legend(fontsize=7)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / f"{dataset}_selective_classification.pdf", bbox_inches='tight')
        fig.savefig(RESULTS_DIR / f"{dataset}_selective_classification.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    if not all_results:
        print("No results")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "soft_metrics_all.csv", index=False)

    # Rank correlation analysis: do NEC and Brier/log-loss rank models the same way?
    print("\n" + "=" * 70)
    print("MODEL RANKING CORRELATIONS (Spearman ρ)")
    print("=" * 70)

    rank_results = []
    for dataset in df['dataset'].unique():
        sub = df[df['dataset'] == dataset]
        # Average across seeds
        avg = sub.groupby('config')[['nec', 'brier', 'log_loss', 'error_rate']].mean()
        if len(avg) < 3:
            continue

        nec_brier_r, _ = stats.spearmanr(avg['nec'], avg['brier'])
        nec_logloss_r, _ = stats.spearmanr(avg['nec'], avg['log_loss'])
        nec_error_r, _ = stats.spearmanr(avg['nec'], avg['error_rate'])
        brier_error_r, _ = stats.spearmanr(avg['brier'], avg['error_rate'])

        print(f"\n  {dataset} ({len(avg)} configs):")
        print(f"    NEC vs Brier:     ρ = {nec_brier_r:.3f}")
        print(f"    NEC vs log-loss:  ρ = {nec_logloss_r:.3f}")
        print(f"    NEC vs error:     ρ = {nec_error_r:.3f}")
        print(f"    Brier vs error:   ρ = {brier_error_r:.3f}")

        rank_results.append({
            'dataset': dataset, 'n_configs': len(avg),
            'nec_vs_brier': nec_brier_r, 'nec_vs_logloss': nec_logloss_r,
            'nec_vs_error': nec_error_r, 'brier_vs_error': brier_error_r,
        })

        # Check for ranking disagreements
        nec_best = avg['nec'].idxmin()
        brier_best = avg['brier'].idxmin()
        error_best = avg['error_rate'].idxmin()
        if nec_best != brier_best or nec_best != error_best:
            print(f"    *** RANKING DISAGREES: NEC-best={nec_best}, Brier-best={brier_best}, Error-best={error_best}")

    pd.DataFrame(rank_results).to_csv(RESULTS_DIR / "rank_correlations.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
