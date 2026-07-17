#!/usr/bin/env python
"""
Annotator-count normalization sensitivity analysis for Jigsaw (v2).

12 Jigsaw model configurations x 10 seeds. Excludes reg_loss (degenerate
25% NEC distorts rankings). Includes tfidf_classification_absdelta.

For each (config, seed), computes:
  - nec_default:    sum(|delta| * wrong) / sum(|delta|)
  - nec_normalized: sum((|delta|/sqrt(n_ann)) * wrong) / sum(|delta|/sqrt(n_ann))
    where n_ann = n_yes + n_no from the Jigsaw cost table.

Usage: python rebuttal/soft-metrics/annotator_normalization_per_config_v2.py
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys

PRED_DIR = Path('rebuttal/predictions/jigsaw')
COST_TABLE = Path('data/jigsaw_cost_table.csv')
OUT_CSV = Path('rebuttal/soft-metrics/results/annotator_normalization_per_config_v2.csv')

CONFIGS = [
    'roberta_classification_absdelta',
    'roberta_classification_none',
    'roberta_finetune_classification_focal2',
    'tfidf_classification_absdelta',
    'tfidf_classification_costing',
    'tfidf_classification_ensemble',
    'tfidf_classification_focal1',
    'tfidf_classification_focal2',
    'tfidf_classification_focal3',
    'tfidf_classification_none',
    'tfidf_classification_sosr',
    'tfidf_classification_zadrozny',
]

# Original 12 configs (with reg_loss, without absdelta) for seed-0 sanity check
CONFIGS_ORIGINAL_12 = [
    'roberta_classification_absdelta',
    'roberta_classification_none',
    'roberta_finetune_classification_focal2',
    'tfidf_classification_costing',
    'tfidf_classification_ensemble',
    'tfidf_classification_focal1',
    'tfidf_classification_focal2',
    'tfidf_classification_focal3',
    'tfidf_classification_none',
    'tfidf_classification_reg_loss',
    'tfidf_classification_sosr',
    'tfidf_classification_zadrozny',
]

SEEDS = [0, 1, 7, 13, 42, 99, 123, 314, 456, 2024]


def compute_nec_pair(test_merged):
    """Compute (nec_default, nec_normalized) from a merged test dataframe."""
    wrong = (test_merged['y_true'] != test_merged['y_pred']).astype(float).values
    abs_delta = test_merged['abs_delta'].values
    n_ann = test_merged['n_annotators'].values
    normalized_cost = abs_delta / np.sqrt(n_ann)

    nec_def = (abs_delta * wrong).sum() / abs_delta.sum()
    nec_norm = (normalized_cost * wrong).sum() / normalized_cost.sum()
    return nec_def, nec_norm


def load_and_merge(config, seed, cost_df):
    """Load prediction file, filter to test split, merge with cost table."""
    fpath = PRED_DIR / f"{config}_s{seed}.csv"
    if not fpath.exists():
        return None
    pred = pd.read_csv(fpath)
    test = pred[pred['split'] == 'test']
    merged = test.merge(
        cost_df[['instance_id', 'n_annotators']],
        on='instance_id', how='left',
    )
    assert merged['n_annotators'].notna().all(), \
        f"unmatched instance_ids in {config} s{seed}"
    return merged


def main():
    cost = pd.read_csv(COST_TABLE)
    cost['n_annotators'] = cost['n_yes'] + cost['n_no']
    cost['instance_id'] = cost.index

    # === Compute per-config results (new 12-config set) ===
    rows = []
    for config in CONFIGS:
        nec_defs = []
        nec_norms = []
        for seed in SEEDS:
            merged = load_and_merge(config, seed, cost)
            if merged is None:
                continue
            nec_def, nec_norm = compute_nec_pair(merged)
            nec_defs.append(nec_def)
            nec_norms.append(nec_norm)

        rows.append({
            'config': config,
            'nec_default_mean': np.mean(nec_defs),
            'nec_default_std': np.std(nec_defs, ddof=1),
            'nec_normalized_mean': np.mean(nec_norms),
            'nec_normalized_std': np.std(nec_norms, ddof=1),
            'n_seeds': len(nec_defs),
        })

    df = pd.DataFrame(rows)

    # === Validation gates (run BEFORE saving) ===

    # Gate 1: new 12-config rho in [0.70, 1.00]
    rho_new, _ = stats.spearmanr(df['nec_default_mean'], df['nec_normalized_mean'])
    print(f"New 12-config 10-seed Spearman rho = {rho_new:.16f}")
    if not (0.70 <= rho_new <= 1.00):
        print(f"VALIDATION FAILED: rho={rho_new:.4f} outside [0.70, 1.00]")
        sys.exit(1)

    # Gate 2: tfidf_none nec_default_mean within 0.05pp of 1.76%
    tfidf_none = df[df['config'] == 'tfidf_classification_none'].iloc[0]
    tfidf_nec_pct = tfidf_none['nec_default_mean'] * 100
    print(f"tfidf_none nec_default_mean = {tfidf_nec_pct:.4f}%  (expect ~1.76%)")
    if abs(tfidf_nec_pct - 1.76) > 0.05:
        print(f"VALIDATION FAILED: tfidf_none NEC={tfidf_nec_pct:.4f}% not within 0.05pp of 1.76%")
        sys.exit(1)

    # Gate 3: original 12-config seed-0 rho must still be 0.9301
    seed0_results = []
    for config in CONFIGS_ORIGINAL_12:
        merged = load_and_merge(config, 0, cost)
        nec_def, nec_norm = compute_nec_pair(merged)
        seed0_results.append((nec_def, nec_norm))

    rho_orig, _ = stats.spearmanr([r[0] for r in seed0_results], [r[1] for r in seed0_results])
    print(f"Original 12-config seed-0 rho     = {rho_orig:.16f}  (target: 0.9300699300699302)")
    if abs(rho_orig - 0.9300699300699302) > 0.0001:
        print(f"VALIDATION FAILED: rho_orig={rho_orig:.6f} differs from target")
        sys.exit(1)

    # Gate 4 (audit): 13-config 10-seed rho (absdelta + reg_loss both included)
    configs_13 = CONFIGS + ['tfidf_classification_reg_loss']
    rows_13 = []
    for config in configs_13:
        nec_d, nec_n = [], []
        for seed in SEEDS:
            merged = load_and_merge(config, seed, cost)
            if merged is None:
                continue
            d, n = compute_nec_pair(merged)
            nec_d.append(d)
            nec_n.append(n)
        rows_13.append((np.mean(nec_d), np.mean(nec_n)))
    rho_13, _ = stats.spearmanr([r[0] for r in rows_13], [r[1] for r in rows_13])
    print(f"Audit: 13-config 10-seed rho      = {rho_13:.16f}  (expect ~0.8297)")
    if not (0.70 <= rho_13 <= 1.00):
        print(f"VALIDATION FAILED: rho_13={rho_13:.4f} outside [0.70, 1.00]")
        sys.exit(1)

    # Check for any config with n_seeds < 10
    low_seed = df[df['n_seeds'] < 10]
    if not low_seed.empty:
        print("WARNING: configs with n_seeds < 10:")
        for _, r in low_seed.iterrows():
            print(f"  {r['config']}: {int(r['n_seeds'])} seeds")

    print("\nAll validation gates passed.")

    # === Save outputs ===
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # Print table
    print(f"\n{'Config':<50} {'NEC_def%':>8} {'+-CI95':>7} {'NEC_norm%':>9} {'+-CI95':>7} {'n':>3}")
    print('-' * 90)
    for _, r in df.iterrows():
        ci_def = 1.96 * r['nec_default_std'] / np.sqrt(r['n_seeds']) * 100
        ci_norm = 1.96 * r['nec_normalized_std'] / np.sqrt(r['n_seeds']) * 100
        print(f"{r['config']:<50} {r['nec_default_mean']*100:>8.2f} {ci_def:>6.2f} {r['nec_normalized_mean']*100:>9.2f} {ci_norm:>6.2f} {int(r['n_seeds']):>3}")

    # Summary
    print(f"\n--- Verification log ---")
    print(f"Original 12-config seed-0 rho:  {rho_orig:.4f}  (stored aggregate, with reg_loss)")
    print(f"13-config 10-seed rho:          {rho_13:.4f}  (audit, absdelta + reg_loss)")
    print(f"New 12-config 10-seed rho:      {rho_new:.4f}  (published, absdelta, no reg_loss)")


if __name__ == '__main__':
    main()
