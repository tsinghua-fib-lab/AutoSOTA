"""
2A: Sensitivity analysis of cost transforms (ICML rebuttal R2).

For each dataset, derive Δ under alternative transforms from raw annotation data,
then recompute NEC for every model. Check whether model rankings change.

Transforms:
1. Original abs_delta (baseline — log-odds for votes, distance for NHANES)
2. Raw vote margin: |n_yes - n_no| / (n_yes + n_no) — no smoothing, no log-odds
3. Squared distance: margin² or (value - threshold)²
4. Entropy-based: cost = 1 - H(p) where p = n_yes/(n_yes+n_no)
5. Class cost-balancing: normalize so each class has equal total cost

Usage:
    python rebuttal/sensitivity/run_sensitivity.py
    python rebuttal/sensitivity/run_sensitivity.py --datasets jigsaw nhanes
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

PRED_DIR = Path("rebuttal/predictions")
DATA_DIR = Path("data")
RESULTS_DIR = Path("rebuttal/sensitivity/results")


def entropy(p):
    """Binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def compute_alternative_costs(dataset_name):
    """Compute alternative cost transforms from raw annotation data."""
    cost_file = DATA_DIR / f"{dataset_name}_cost_table.csv"
    if not cost_file.exists():
        # Try inaturalist subfolder
        cost_file = DATA_DIR / dataset_name / "gemini_labels.csv"
        if not cost_file.exists():
            print(f"  WARNING: No cost table for {dataset_name}")
            return None

    df = pd.read_csv(cost_file)

    transforms = {}

    # 1. Original abs_delta
    if 'abs_delta' in df.columns:
        transforms['original'] = df['abs_delta'].values

    if 'n_yes' in df.columns and 'n_no' in df.columns:
        n_yes = df['n_yes'].values.astype(float)
        n_no = df['n_no'].values.astype(float)
        n_total = n_yes + n_no

        # 2. Raw vote margin (no smoothing, no log-odds)
        margin = np.abs(n_yes - n_no) / np.maximum(n_total, 1)
        transforms['raw_margin'] = margin

        # 3. Squared margin
        transforms['squared'] = margin ** 2

        # 4. Entropy-based: cost = 1 - H(p), high when consensus is strong
        p = n_yes / np.maximum(n_total, 1)
        h = entropy(p)
        max_entropy = np.log(2)  # max binary entropy
        transforms['entropy'] = 1.0 - h / max_entropy  # normalized to [0, 1]

    elif 'delta_signed' in df.columns:
        # NHANES: threshold-based
        delta = df['delta_signed'].values.astype(float)
        transforms['original'] = np.abs(delta)
        transforms['raw_margin'] = np.abs(delta)  # same for threshold
        transforms['squared'] = delta ** 2
        # Entropy doesn't apply to threshold-based

    # 5. Class cost-balancing (normalize per class)
    if 'y_star' in df.columns and 'original' in transforms:
        costs = transforms['original'].copy()
        y = df['y_star'].values
        for cls in [0, 1]:
            mask = y == cls
            if mask.any() and costs[mask].sum() > 0:
                # Scale so each class has equal total cost
                costs[mask] *= 1.0 / costs[mask].mean()
        transforms['class_balanced'] = costs

    return transforms


def compute_nec(y_true, y_pred, costs):
    """Compute NEC with given costs."""
    total = costs.sum()
    if total == 0:
        return 0.0
    return (costs * (y_true != y_pred).astype(float)).sum() / total


def main():
    parser = argparse.ArgumentParser(description='Sensitivity analysis of cost transforms')
    parser.add_argument('--datasets', nargs='+',
                        default=['jigsaw', 'nhanes', 'turkey', 'inaturalist', 'synthetic'])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for dataset in args.datasets:
        pred_dir = PRED_DIR / dataset
        if not pred_dir.exists():
            print(f"  SKIP: {pred_dir} not found")
            continue

        # Get alternative cost transforms
        transforms = compute_alternative_costs(dataset)
        if transforms is None:
            continue

        print(f"\n=== {dataset} ===")
        print(f"  Transforms: {list(transforms.keys())}")

        csv_files = sorted(pred_dir.glob("*.csv"))
        csv_files = [f for f in csv_files if 'regression' not in f.stem]

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            test = df[df['split'] == 'test']
            if len(test) == 0:
                continue

            import re
            stem = csv_path.stem
            m = re.match(r'(.+)_s(\d+)(?:_.*)?$', stem)
            config = m.group(1) if m else stem
            seed = int(m.group(2)) if m else -1

            y_true = test['y_true'].values
            y_pred = test['y_pred'].values
            instance_ids = test['instance_id'].values

            row = {'dataset': dataset, 'config': config, 'seed': seed}

            for transform_name, all_costs in transforms.items():
                # Map costs to test instances by instance_id
                try:
                    costs = all_costs[instance_ids]
                except (IndexError, KeyError):
                    costs = test['abs_delta'].values  # fallback

                nec = compute_nec(y_true, y_pred, costs)
                row[f'nec_{transform_name}'] = nec

            all_results.append(row)

        if not all_results:
            continue

    if not all_results:
        print("No results")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "sensitivity_all.csv", index=False)

    # Rank correlation analysis
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: RANK CORRELATIONS")
    print("=" * 70)

    nec_cols = [c for c in df.columns if c.startswith('nec_')]
    rank_results = []

    for dataset in df['dataset'].unique():
        sub = df[df['dataset'] == dataset]
        avg = sub.groupby('config')[nec_cols].mean()
        if len(avg) < 3:
            continue

        print(f"\n  {dataset} ({len(avg)} configs):")
        baseline_col = 'nec_original'
        if baseline_col not in avg.columns:
            continue

        for col in nec_cols:
            if col == baseline_col:
                continue
            if col not in avg.columns:
                continue
            rho, p = stats.spearmanr(avg[baseline_col], avg[col])
            tau, p_tau = stats.kendalltau(avg[baseline_col], avg[col])
            transform = col.replace('nec_', '')
            print(f"    original vs {transform:20s}: Spearman ρ={rho:.3f}, Kendall τ={tau:.3f}")
            rank_results.append({
                'dataset': dataset, 'transform': transform,
                'spearman_rho': rho, 'kendall_tau': tau,
            })

            # Check if best model changes
            orig_best = avg[baseline_col].idxmin()
            alt_best = avg[col].idxmin()
            if orig_best != alt_best:
                print(f"      *** BEST MODEL CHANGES: {orig_best} -> {alt_best}")

    pd.DataFrame(rank_results).to_csv(RESULTS_DIR / "rank_correlations.csv", index=False)

    # Plot: NEC under different transforms for P1 baselines
    for dataset in df['dataset'].unique():
        sub = df[df['dataset'] == dataset]
        avg = sub.groupby('config')[nec_cols].mean()

        # Filter to P1 baselines
        baseline_configs = [c for c in avg.index if c.endswith('_none') or c.endswith('_absdelta')]
        if not baseline_configs:
            baseline_configs = list(avg.index[:6])

        plot_data = avg.loc[baseline_configs]
        available_cols = [c for c in nec_cols if c in plot_data.columns]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(plot_data))
        width = 0.8 / len(available_cols)

        for i, col in enumerate(available_cols):
            label = col.replace('nec_', '')
            ax.bar(x + i * width, plot_data[col], width, label=label, alpha=0.8)

        ax.set_ylabel('NEC')
        ax.set_title(f'{dataset}: NEC under different cost transforms')
        ax.set_xticks(x + width * len(available_cols) / 2)
        short_labels = [c.split('_classification_')[-1] if '_classification_' in c else c
                       for c in plot_data.index]
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=7)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / f"{dataset}_sensitivity.pdf", bbox_inches='tight')
        fig.savefig(RESULTS_DIR / f"{dataset}_sensitivity.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
