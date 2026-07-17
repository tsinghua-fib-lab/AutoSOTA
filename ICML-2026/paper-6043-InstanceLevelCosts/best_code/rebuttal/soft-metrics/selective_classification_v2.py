#!/usr/bin/env python
"""
Selective classification risk-coverage curves (v2) for ICML camera-ready.

For each of 5 datasets, plots NEC vs coverage for Standard CE and |Δ|-weighted CE,
averaged over 10 seeds with 95% CI bands.

Usage: python rebuttal/soft-metrics/selective_classification_v2.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PRED_DIR = Path('rebuttal/predictions')
RESULTS_DIR = Path('rebuttal/soft-metrics/results')

SEEDS = [0, 1, 7, 13, 42, 99, 123, 314, 456, 2024]
N_POINTS = 100
COVERAGES = np.linspace(0.1, 1.0, N_POINTS)

# (dataset, config_none, config_absdelta)
TASKS = [
    ('jigsaw',      'tfidf_classification_none',    'tfidf_classification_absdelta'),
    ('turkey',      'resnet50_classification_none',  'resnet50_classification_absdelta'),
    ('nhanes',      'histgbm_classification_none',   'histgbm_classification_absdelta'),
    ('inaturalist', 'resnet50_classification_none',  'resnet50_classification_absdelta'),
    ('synthetic',   'logreg_classification_none',    'logreg_classification_absdelta'),
]

# Table 2 NEC values for sanity check at coverage=1.0 (as fractions)
TABLE2_NEC = {
    ('jigsaw', 'tfidf_classification_none'):        0.0176,
    ('jigsaw', 'tfidf_classification_absdelta'):    0.0179,
    ('turkey', 'resnet50_classification_none'):      0.0416,
    ('turkey', 'resnet50_classification_absdelta'):  0.0355,
    ('nhanes', 'histgbm_classification_none'):       0.1483,
    ('nhanes', 'histgbm_classification_absdelta'):   0.1446,
    ('inaturalist', 'resnet50_classification_none'):     0.1039,
    ('inaturalist', 'resnet50_classification_absdelta'): 0.1108,
    ('synthetic', 'logreg_classification_none'):      0.0050,
    ('synthetic', 'logreg_classification_absdelta'):  0.0040,
}

DATASET_LABELS = {
    'jigsaw': 'Jigsaw (TF-IDF)',
    'turkey': 'Turkey (ResNet-50)',
    'nhanes': 'NHANES (HistGBM)',
    'inaturalist': 'iNaturalist (ResNet-50)',
    'synthetic': 'Synthetic (LogReg)',
}

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def selective_nec_curve(y_true, y_pred, y_proba_1, abs_delta):
    """Compute NEC at each coverage level, sorting by confidence (descending)."""
    confidence = np.maximum(y_proba_1, 1 - y_proba_1)
    sorted_idx = np.argsort(-confidence)

    y_sorted = y_true[sorted_idx]
    pred_sorted = y_pred[sorted_idx]
    d_sorted = abs_delta[sorted_idx]

    n = len(y_true)
    necs = []
    for cov in COVERAGES:
        k = max(1, int(cov * n))
        y_sub = y_sorted[:k]
        pred_sub = pred_sorted[:k]
        d_sub = d_sorted[:k]
        total_d = d_sub.sum()
        if total_d > 0:
            necs.append((d_sub * (y_sub != pred_sub)).sum() / total_d)
        else:
            necs.append(0.0)
    return np.array(necs)


def load_pred(dataset, config, seed):
    """Load prediction file, filter to test, return arrays."""
    fpath = PRED_DIR / dataset / f"{config}_s{seed}.csv"
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)
    test = df[df['split'] == 'test']
    return (
        test['y_true'].values,
        test['y_pred'].values,
        test['y_proba_1'].values,
        test['abs_delta'].values,
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    validation_ok = True

    for dataset, config_none, config_absdelta in TASKS:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        for config, label, color in [
            (config_none, 'Standard CE', '#1f77b4'),
            (config_absdelta, '|Δ|-weighted CE', '#2ca02c'),
        ]:
            # Collect curves across seeds
            all_curves = []
            for seed in SEEDS:
                data = load_pred(dataset, config, seed)
                if data is None:
                    continue
                y_true, y_pred, y_proba_1, abs_delta = data
                curve = selective_nec_curve(y_true, y_pred, y_proba_1, abs_delta)
                all_curves.append(curve)

            if not all_curves:
                print(f"  WARNING: no data for {dataset}/{config}")
                continue

            curves = np.array(all_curves)  # shape (n_seeds, n_points)
            mean_curve = curves.mean(axis=0) * 100  # to percent
            std_curve = curves.std(axis=0, ddof=1) * 100
            n = len(all_curves)
            ci95 = 1.96 * std_curve / np.sqrt(n)

            ax.plot(COVERAGES, mean_curve, label=label, color=color, linewidth=2)
            ax.fill_between(COVERAGES, mean_curve - ci95, mean_curve + ci95,
                           alpha=0.2, color=color)

            # Sanity check at coverage=1.0
            nec_at_full = curves[:, -1].mean()  # fraction, not percent
            table2_val = TABLE2_NEC.get((dataset, config))
            if table2_val is not None:
                diff = abs(nec_at_full - table2_val)
                status = 'OK' if diff < 0.005 else f'MISMATCH (diff={diff:.4f})'
                print(f"  {dataset}/{config} coverage=1.0: NEC={nec_at_full*100:.2f}%  "
                      f"Table2={table2_val*100:.2f}%  {status}")
                if diff >= 0.005:
                    validation_ok = False

        ax.set_xlabel('Coverage')
        ax.set_ylabel('NEC (%)')
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xlim(0.1, 1.0)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        out_path = RESULTS_DIR / f"{dataset}_selective_classification_v2.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved: {out_path}")

    if not validation_ok:
        print("\nVALIDATION FAILED: one or more coverage=1.0 NEC values don't match Table 2")
        sys.exit(1)

    print("\nAll sanity checks passed.")


if __name__ == '__main__':
    main()
