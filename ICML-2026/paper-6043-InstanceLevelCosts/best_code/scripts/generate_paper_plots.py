#!/usr/bin/env python
"""
Generate paper plots for cost-sensitive learning experiments (ICML 2026).

Updated figure structure per PI's requirements:
- fig1: NEC vs Error for 6 tasks with LR/frozen (incl. synthetic)
- fig2: NEC vs Error for 3 tasks with full fine-tuning
- fig3: NEC and Error decay for TF-IDF and RoBERTa linear probe (sample scaling)
- fig4: Different Delta-methods for 6 tasks with LR
- fig5: Standard vs delta-upweighting for 3 tasks with fine-tuning
- fig9: Delta histograms for 4 real datasets

NOTE: Roberta alpha_balanced for P4 may be excluded from summaries if jobs
haven't completed. To include it, remove the exclusion from summarize_p4_results.py
and regenerate p4_summary.csv.

Usage: python scripts/generate_paper_plots.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Use a non-interactive backend for script execution
matplotlib.use('Agg')

# Set style for publication-quality plots (ICML two-column format)
# Single column ~3.5", double column ~7"
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
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'errorbar.capsize': 4,
})

# =============================================================================
# Data Loading
# =============================================================================

def load_summary(name: str) -> pd.DataFrame:
    """Load a summary CSV file."""
    path = Path(f'results/{name}_summary.csv')
    if path.exists():
        return pd.read_csv(path)
    print(f"  WARNING: {path} not found")
    return pd.DataFrame()


def load_all_runs(name: str) -> pd.DataFrame:
    """Load an all_runs CSV file."""
    path = Path(f'results/{name}_all_runs.csv')
    if path.exists():
        return pd.read_csv(path)
    print(f"  WARNING: {path} not found")
    return pd.DataFrame()


def load_p6_sampling_summary() -> pd.DataFrame:
    """Load P6 sampling strategy results for synthetic data."""
    path = Path('results/p6_sampling_summary.csv')
    if path.exists():
        return pd.read_csv(path)
    print(f"  WARNING: {path} not found")
    return pd.DataFrame()


# =============================================================================
# Metric Helpers
# =============================================================================

def get_nec(row, mean_col='weighted_accuracy_mean'):
    """Compute NEC = 1 - weighted_accuracy from summary row."""
    wacc = row.get(mean_col, np.nan)
    return 1 - wacc if pd.notna(wacc) else np.nan


def get_nec_std(row, std_col='weighted_accuracy_std'):
    """Get NEC std (same as weighted_accuracy std for 1-x transform)."""
    return row.get(std_col, 0) if pd.notna(row.get(std_col)) else 0


def get_acc(row, mean_col='accuracy_mean'):
    """Get accuracy from summary row."""
    return row.get(mean_col, np.nan)


def get_acc_std(row, std_col='accuracy_std'):
    """Get accuracy std from summary row."""
    return row.get(std_col, 0) if pd.notna(row.get(std_col)) else 0


def get_error_rate(row, mean_col='accuracy_mean'):
    """Get error rate (1 - accuracy) from summary row."""
    acc = row.get(mean_col, np.nan)
    return 1 - acc if pd.notna(acc) else np.nan


def get_error_rate_std(row, std_col='accuracy_std'):
    """Get error rate std (same as accuracy std for 1-x transform)."""
    return row.get(std_col, 0) if pd.notna(row.get(std_col)) else 0


def std_to_ci95(std, n_seeds):
    """Convert standard deviation to 95% confidence interval half-width: 1.96 * SD / sqrt(n)."""
    if n_seeds is None or n_seeds <= 0 or pd.isna(n_seeds):
        return std  # Fall back to SD if n_seeds unknown
    return 1.96 * std / np.sqrt(n_seeds)


# Alias for backward compatibility
std_to_se = std_to_ci95


def get_n_seeds(row):
    """Get number of seeds from a summary row."""
    return row.get('n_seeds', 10)  # Default to 10 if not present


# Default number of seeds for computing SE
DEFAULT_N_SEEDS = 10


# =============================================================================
# Fig 1: NEC vs Error for 6 Tasks with LR/Frozen (incl. Synthetic)
# =============================================================================

def plot_fig1_nec_vs_error_lr(output_dir: Path):
    """
    Main results: NEC and Error Rate for 6 tasks with linear/frozen classifiers.
    Includes synthetic dataset.

    Tasks:
    1. Jigsaw (TF-IDF)
    2. Jigsaw (RoBERTa frozen)
    3. Turkey (ResNet50 frozen)
    4. NHANES (HistGBM)
    5. iNaturalist (ResNet50 frozen)
    6. Synthetic (LogReg)

    Color system:
    - Blue family = NEC metric (darker blue = Standard, lighter blue = |Δ|-weighted)
    - Orange family = Error Rate metric (orange = Standard, red = |Δ|-weighted)
    """
    p1_df = load_summary('p1')
    p6_df = load_summary('p6')

    if p1_df.empty:
        print("  WARNING: P1 data not found")
        return

    p1_clf = p1_df[p1_df['method'] == 'classification'].copy()
    p6_clf = p6_df[p6_df['method'] == 'classification'].copy() if not p6_df.empty else pd.DataFrame()

    # Define the 6 tasks (dataset, model, label, source_df)
    tasks = [
        ('jigsaw', 'tfidf', 'Jigsaw\n(TF-IDF)', p1_clf),
        ('jigsaw', 'roberta', 'Jigsaw\n(RoBERTa)', p1_clf),
        ('turkey', 'resnet50', 'Turkey', p1_clf),
        ('nhanes', 'histgbm', 'NHANES', p1_clf),
        ('inaturalist', 'resnet50', 'iNaturalist', p1_clf),
        ('synthetic', 'logreg', 'Synthetic', p6_clf),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    x = np.arange(len(tasks))
    width = 0.35  # Wider bars since only 2 per dataset

    # Color scheme: NEC (blue) and Error Rate (orange) - Standard only
    bar_configs = [
        ('NEC', 'none', '#1f77b4', -0.5),        # Blue (standard)
        ('Error Rate', 'none', '#ff7f0e', 0.5),  # Orange (standard)
    ]

    for metric, weighting, color, offset_mult in bar_configs:
        values = []
        ses = []

        for dataset, model, label, source_df in tasks:
            if source_df.empty:
                values.append(np.nan)
                ses.append(0)
                continue

            if dataset == 'synthetic':
                row = source_df[source_df['weighting'] == weighting]
            else:
                row = source_df[(source_df['dataset'] == dataset) &
                               (source_df['model'] == model) &
                               (source_df['weighting'] == weighting)]

            if not row.empty:
                if metric == 'NEC':
                    values.append(get_nec(row.iloc[0]) * 100)
                    std = get_nec_std(row.iloc[0]) * 100
                else:
                    values.append(get_error_rate(row.iloc[0]) * 100)
                    std = get_error_rate_std(row.iloc[0]) * 100
                n = get_n_seeds(row.iloc[0])
                ses.append(std_to_se(std, n))
            else:
                values.append(np.nan)
                ses.append(0)

        ax.bar(x + offset_mult * width, values, width, label=metric,
               yerr=ses, capsize=4, alpha=0.85,
               color=color, edgecolor='black', linewidth=1.0)

    ax.set_xlabel('Task')
    ax.set_ylabel('Metric (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([t[2] for t in tasks])
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_nec_vs_error_lr.pdf')
    plt.close()
    print(f"  Saved: fig1_nec_vs_error_lr.pdf")


# =============================================================================
# Fig 2: NEC vs Error for 3 Tasks with Fine-tuning
# =============================================================================

def plot_fig2_nec_vs_error_finetune(output_dir: Path):
    """
    NEC and Error Rate for 3 tasks with end-to-end fine-tuning.

    Tasks:
    1. Jigsaw (RoBERTa fine-tuned)
    2. Turkey (ResNet fine-tuned)
    3. iNaturalist (ResNet fine-tuned)

    Color system (same as fig1):
    - Blue family = NEC metric
    - Orange family = Error Rate metric
    """
    p5_df = load_summary('p5')

    if p5_df.empty:
        print("  WARNING: P5 data not found")
        return

    p5_clf = p5_df[p5_df['method'] == 'classification'].copy()

    # Define the 3 tasks
    tasks = [
        ('jigsaw', 'roberta_finetune', 'Jigsaw\n(RoBERTa)'),
        ('turkey', 'resnet_finetune', 'Turkey\n(ResNet)'),
        ('inaturalist', 'resnet_finetune', 'iNaturalist\n(ResNet)'),
    ]

    weightings = ['none']
    weighting_labels = {'none': 'Standard CE'}

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.arange(len(tasks))
    width = 0.35

    # Color scheme - standard training only, evaluated with NEC
    bar_configs = [
        ('NEC', 'none', '#1f77b4', -0.5),        # Blue (standard)
        ('Error Rate', 'none', '#ff7f0e', 0.5),  # Orange (standard)
    ]

    for metric, weighting, color, offset_mult in bar_configs:
        values = []
        ses = []

        for dataset, model, label in tasks:
            row = p5_clf[(p5_clf['dataset'] == dataset) &
                        (p5_clf['model'] == model) &
                        (p5_clf['weighting'] == weighting)]

            if not row.empty:
                if metric == 'NEC':
                    values.append(get_nec(row.iloc[0]) * 100)
                    std = get_nec_std(row.iloc[0]) * 100
                else:
                    values.append(get_error_rate(row.iloc[0]) * 100)
                    std = get_error_rate_std(row.iloc[0]) * 100
                n = get_n_seeds(row.iloc[0])
                ses.append(std_to_se(std, n))
            else:
                values.append(np.nan)
                ses.append(0)

        label_str = metric  # Only standard training, so just use metric name
        ax.bar(x + offset_mult * width, values, width, label=label_str,
               yerr=ses, capsize=4, alpha=0.85,
               color=color, edgecolor='black', linewidth=1.0)

    ax.set_xlabel('Task')
    ax.set_ylabel('Metric (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([t[2] for t in tasks])
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_nec_vs_error_finetune.pdf')
    plt.close()
    print(f"  Saved: fig2_nec_vs_error_finetune.pdf")


# =============================================================================
# Fig 3: Sample Size Scaling (NEC and Error decay)
# =============================================================================

def plot_fig3_sample_size_scaling(output_dir: Path):
    """
    NEC and Error Rate decay vs training set size N for Jigsaw.
    Shows TF-IDF and RoBERTa linear probe models.

    Color system:
    - Blue (C0) = NEC metric
    - Orange (C1) = Error Rate metric
    - Solid line + circles = TF-IDF
    - Dashed line + squares = RoBERTa
    """
    df = load_summary('p2')
    if df.empty:
        return

    clf_df = df[df['method'] == 'classification'].copy()
    if clf_df.empty:
        print("  WARNING: No classification data for sample size scaling")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = ['tfidf', 'roberta']
    model_labels = {'tfidf': 'TF-IDF', 'roberta': 'RoBERTa'}
    model_markers = {'tfidf': 'o', 'roberta': 's'}
    model_linestyles = {'tfidf': '-', 'roberta': '--'}

    # Only use 'none' weighting for cleaner plot
    weighting = 'none'

    # --- Left: NEC (Blue = C0) ---
    ax = axes[0]
    for model in models:
        subset = clf_df[(clf_df['model'] == model) & (clf_df['weighting'] == weighting)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values('sample_size')

        sizes = subset['sample_size'].values
        necs = [get_nec(row) * 100 for _, row in subset.iterrows()]
        ses = [std_to_se(get_nec_std(row) * 100, get_n_seeds(row)) for _, row in subset.iterrows()]

        ax.errorbar(sizes, necs, yerr=ses, label=model_labels[model],
                   color='C0', marker=model_markers[model],
                   linestyle=model_linestyles[model], capsize=4, linewidth=2.5, markersize=8)

    ax.set_xscale('log')
    ax.set_xlabel('Training Set Size (N)')
    ax.set_ylabel('NEC (%)')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # --- Right: Error Rate (Orange = C1) ---
    ax = axes[1]
    for model in models:
        subset = clf_df[(clf_df['model'] == model) & (clf_df['weighting'] == weighting)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values('sample_size')

        sizes = subset['sample_size'].values
        errs = [get_error_rate(row) * 100 for _, row in subset.iterrows()]
        ses = [std_to_se(get_error_rate_std(row) * 100, get_n_seeds(row)) for _, row in subset.iterrows()]

        ax.errorbar(sizes, errs, yerr=ses, label=model_labels[model],
                   color='C1', marker=model_markers[model],
                   linestyle=model_linestyles[model], capsize=4, linewidth=2.5, markersize=8)

    ax.set_xscale('log')
    ax.set_xlabel('Training Set Size (N)')
    ax.set_ylabel('Error Rate (%)')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_sample_size_scaling.pdf')
    plt.close()
    print(f"  Saved: fig3_sample_size_scaling.pdf")


# =============================================================================
# Fig 4: Different Delta-Methods for 6 Tasks (LR/Frozen)
# =============================================================================

def plot_fig4_delta_methods(output_dir: Path):
    """
    Effect of different delta-based methods on NEC for 6 tasks with LR.

    Methods compared:
    - Standard (U) - uniform baseline from P3 (P1 for synthetic)
    - Upsample (P_up) - from P3
    - |Δ|-weighted - from P1
    - Top-70% (Tdown70) - from P3
    - Δ-Regression - from P3 regression

    Uses P3 for sampling strategies, P1/P6 for weighting methods.
    Synthetic only has Standard and |Δ|-weighted (no P3 sampling data).
    """
    p1_df = load_summary('p1')
    p3_df = load_summary('p3')
    p6_df = load_summary('p6')
    p6_sampling_df = load_p6_sampling_summary()

    if p3_df.empty:
        print("  WARNING: P3 data not found")
        return

    p1_clf = p1_df[p1_df['method'] == 'classification'].copy() if not p1_df.empty else pd.DataFrame()
    p3_clf = p3_df[p3_df['method'] == 'classification'].copy()
    p3_reg = p3_df[p3_df['method'] == 'regression'].copy()
    p6_clf = p6_df[p6_df['method'] == 'classification'].copy() if not p6_df.empty else pd.DataFrame()

    # Define the 6 tasks (5 real + synthetic)
    tasks = [
        ('jigsaw', 'tfidf', 'Jigsaw\n(TF-IDF)', False),
        ('jigsaw', 'roberta', 'Jigsaw\n(RoBERTa)', False),
        ('turkey', 'resnet50', 'Turkey', False),
        ('nhanes', 'histgbm', 'NHANES', False),
        ('inaturalist', 'resnet50', 'iNaturalist', False),
        ('synthetic', 'logreg', 'Synthetic', True),  # is_synthetic=True
    ]

    # Methods: (label, key, is_regression, requires_p3)
    # Include all sampling methods from P3
    methods = [
        ('Standard', 'U', False, True),
        ('Upsample', 'P_up', False, True),
        ('|Δ|-weighted', 'absdelta', False, False),  # from P1, not P3
        ('Top-30%', 'Tdown30', False, True),
        ('Top-50%', 'Tdown50', False, True),
        ('Top-70%', 'Tdown70', False, True),
        ('Δ-Regression', 'U', True, True),
    ]

    # 7 distinct colors for methods
    method_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    x = np.arange(len(tasks))
    width = 0.11  # Narrower bars to fit 7 methods across 6 tasks
    n_methods = len(methods)

    # --- Left: NEC ---
    ax = axes[0]
    for i, (method_label, key, is_regression, requires_p3) in enumerate(methods):
        values = []
        ses = []

        for dataset, model, task_label, is_synthetic in tasks:
            # Synthetic: use P6 sampling results
            if is_synthetic and requires_p3:
                row = p6_sampling_df[p6_sampling_df['strategy'] == key]
                if not row.empty:
                    values.append(get_nec(row.iloc[0]) * 100)
                    std = get_nec_std(row.iloc[0]) * 100
                    n = get_n_seeds(row.iloc[0])
                    ses.append(std_to_se(std, n))
                else:
                    values.append(np.nan)
                    ses.append(0)
                continue

            if is_synthetic and key == 'absdelta':
                # |Δ|-weighted for synthetic from P6
                row = p6_clf[p6_clf['weighting'] == 'absdelta']
                if not row.empty:
                    values.append(get_nec(row.iloc[0]) * 100)
                    std = get_nec_std(row.iloc[0]) * 100
                    n = get_n_seeds(row.iloc[0])
                    ses.append(std_to_se(std, n))
                else:
                    values.append(np.nan)
                    ses.append(0)
                continue

            # Regular tasks
            if is_regression:
                row = p3_reg[(p3_reg['dataset'] == dataset) &
                            (p3_reg['model'] == model) &
                            (p3_reg['strategy'] == key)]
                if not row.empty:
                    wsa = row.iloc[0].get('weighted_sign_accuracy_mean', np.nan)
                    wsa_std = row.iloc[0].get('weighted_sign_accuracy_std', 0)
                    values.append((1 - wsa) * 100 if pd.notna(wsa) else np.nan)
                    n = get_n_seeds(row.iloc[0])
                    ses.append(std_to_se(wsa_std * 100, n))
                else:
                    values.append(np.nan)
                    ses.append(0)
            elif requires_p3:
                row = p3_clf[(p3_clf['dataset'] == dataset) &
                            (p3_clf['model'] == model) &
                            (p3_clf['strategy'] == key)]
                if not row.empty:
                    values.append(get_nec(row.iloc[0]) * 100)
                    std = get_nec_std(row.iloc[0]) * 100
                    n = get_n_seeds(row.iloc[0])
                    ses.append(std_to_se(std, n))
                else:
                    values.append(np.nan)
                    ses.append(0)
            else:  # |Δ|-weighted from P1
                row = p1_clf[(p1_clf['dataset'] == dataset) &
                            (p1_clf['model'] == model) &
                            (p1_clf['weighting'] == key)]
                if not row.empty:
                    values.append(get_nec(row.iloc[0]) * 100)
                    std = get_nec_std(row.iloc[0]) * 100
                    n = get_n_seeds(row.iloc[0])
                    ses.append(std_to_se(std, n))
                else:
                    values.append(np.nan)
                    ses.append(0)

        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, values, width, label=method_label,
               yerr=ses, capsize=3, alpha=0.85,
               color=method_colors[i], edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Task')
    ax.set_ylabel('NEC (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([t[2] for t in tasks])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # --- Right: Error Rate ---
    ax = axes[1]
    for i, (method_label, key, is_regression, requires_p3) in enumerate(methods):
        values = []
        ses = []

        for dataset, model, task_label, is_synthetic in tasks:
            # Synthetic: use P6 sampling results
            if is_synthetic and requires_p3:
                row = p6_sampling_df[p6_sampling_df['strategy'] == key]
                if not row.empty:
                    values.append(get_error_rate(row.iloc[0]) * 100)
                    std = get_error_rate_std(row.iloc[0]) * 100
                    n = get_n_seeds(row.iloc[0])
                    ses.append(std_to_se(std, n))
                else:
                    values.append(np.nan)
                    ses.append(0)
                continue

            if is_synthetic and key == 'absdelta':
                row = p6_clf[p6_clf['weighting'] == 'absdelta']
                if not row.empty:
                    values.append(get_error_rate(row.iloc[0]) * 100)
                    std = get_error_rate_std(row.iloc[0]) * 100
                    n = get_n_seeds(row.iloc[0])
                    ses.append(std_to_se(std, n))
                else:
                    values.append(np.nan)
                    ses.append(0)
                continue

            # Regular tasks
            if is_regression:
                row = p3_reg[(p3_reg['dataset'] == dataset) &
                            (p3_reg['model'] == model) &
                            (p3_reg['strategy'] == key)]
                if not row.empty:
                    sa = row.iloc[0].get('sign_accuracy_mean', np.nan)
                    sa_std = row.iloc[0].get('sign_accuracy_std', 0)
                    values.append((1 - sa) * 100 if pd.notna(sa) else np.nan)
                    n = get_n_seeds(row.iloc[0])
                    ses.append(std_to_se(sa_std * 100, n))
                else:
                    values.append(np.nan)
                    ses.append(0)
            elif requires_p3:
                row = p3_clf[(p3_clf['dataset'] == dataset) &
                            (p3_clf['model'] == model) &
                            (p3_clf['strategy'] == key)]
                if not row.empty:
                    values.append(get_error_rate(row.iloc[0]) * 100)
                    std = get_error_rate_std(row.iloc[0]) * 100
                    n = get_n_seeds(row.iloc[0])
                    ses.append(std_to_se(std, n))
                else:
                    values.append(np.nan)
                    ses.append(0)
            else:  # |Δ|-weighted from P1
                row = p1_clf[(p1_clf['dataset'] == dataset) &
                            (p1_clf['model'] == model) &
                            (p1_clf['weighting'] == key)]
                if not row.empty:
                    values.append(get_error_rate(row.iloc[0]) * 100)
                    std = get_error_rate_std(row.iloc[0]) * 100
                    n = get_n_seeds(row.iloc[0])
                    ses.append(std_to_se(std, n))
                else:
                    values.append(np.nan)
                    ses.append(0)

        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, values, width, label=method_label,
               yerr=ses, capsize=3, alpha=0.85,
               color=method_colors[i], edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Task')
    ax.set_ylabel('Error Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([t[2] for t in tasks])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_delta_methods.pdf')
    plt.savefig(output_dir / 'fig4_delta_methods_1.pdf')
    plt.close()
    print(f"  Saved: fig4_delta_methods.pdf + fig4_delta_methods_1.pdf")


# =============================================================================
# Fig 5: Standard vs Delta-Upweighting for Fine-tuning
# =============================================================================

def plot_fig5_finetune_weighting(output_dir: Path):
    """
    Standard vs |Δ|-upweighting for 3 tasks with end-to-end fine-tuning.

    Two-pane layout matching Figure 4: left = NEC, right = Error Rate.
    """
    p5_df = load_summary('p5')

    if p5_df.empty:
        print("  WARNING: P5 data not found")
        return

    p5_clf = p5_df[p5_df['method'] == 'classification'].copy()

    # Define the 3 tasks
    tasks = [
        ('jigsaw', 'roberta_finetune', 'Jigsaw\n(RoBERTa-FT)'),
        ('turkey', 'resnet_finetune', 'Turkey\n(ResNet50-FT)'),
        ('inaturalist', 'resnet_finetune', 'iNaturalist\n(ResNet50-FT)'),
    ]

    # Methods: (label, weighting_key, color) — colors match Fig 4 indices 0 and 2
    methods = [
        ('Standard CE', 'none', '#1f77b4'),
        ('|Δ|-weighted CE', 'absdelta', '#2ca02c'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    x = np.arange(len(tasks))
    width = 0.30  # Wider bars since only 2 methods across 3 tasks
    n_methods = len(methods)

    # --- Left: NEC ---
    ax = axes[0]
    for i, (method_label, weighting_key, color) in enumerate(methods):
        values = []
        ses = []
        for dataset, model, task_label in tasks:
            row = p5_clf[(p5_clf['dataset'] == dataset) &
                         (p5_clf['model'] == model) &
                         (p5_clf['weighting'] == weighting_key)]
            if not row.empty:
                values.append(get_nec(row.iloc[0]) * 100)
                std = get_nec_std(row.iloc[0]) * 100
                n = get_n_seeds(row.iloc[0])
                ses.append(std_to_se(std, n))
            else:
                values.append(np.nan)
                ses.append(0)

        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, values, width, label=method_label,
               yerr=ses, capsize=3, alpha=0.85,
               color=color, edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Task')
    ax.set_ylabel('NEC (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([t[2] for t in tasks])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # --- Right: Error Rate ---
    ax = axes[1]
    for i, (method_label, weighting_key, color) in enumerate(methods):
        values = []
        ses = []
        for dataset, model, task_label in tasks:
            row = p5_clf[(p5_clf['dataset'] == dataset) &
                         (p5_clf['model'] == model) &
                         (p5_clf['weighting'] == weighting_key)]
            if not row.empty:
                values.append(get_error_rate(row.iloc[0]) * 100)
                std = get_error_rate_std(row.iloc[0]) * 100
                n = get_n_seeds(row.iloc[0])
                ses.append(std_to_se(std, n))
            else:
                values.append(np.nan)
                ses.append(0)

        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, values, width, label=method_label,
               yerr=ses, capsize=3, alpha=0.85,
               color=color, edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Task')
    ax.set_ylabel('Error Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([t[2] for t in tasks])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_finetune_weighting.pdf')
    plt.savefig(output_dir / 'fig5_finetune_weighting_1.pdf')
    plt.close()
    print(f"  Saved: fig5_finetune_weighting.pdf + fig5_finetune_weighting_1.pdf")


# =============================================================================
# Fig 6: Delta Histograms for 4 Real Datasets
# =============================================================================

def plot_fig6_delta_histograms(output_dir: Path):
    """
    Delta distribution histograms for all 4 real datasets.
    Shows the distribution of signed Δ values for each dataset.
    All distributions are oriented so minority class is on the right (positive) side.
    """
    from pathlib import Path as PathLib

    # (dataset, label, delta_col, flip_sign)
    # flip_sign=True means negate delta so minority ends up on the right
    datasets = [
        ('jigsaw', 'Jigsaw (Toxicity)', 'delta_signed', False),      # minority already on right
        ('turkey', 'Turkey (Injury)', 'delta_signed', True),         # flip: minority was on left
        ('nhanes', 'NHANES (Hypertension)', 'delta_signed', False),  # minority already on right
        ('inaturalist', 'iNaturalist (Wild/Cultivated)', 'delta_signed', True),  # flip: minority was on left
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (dataset, label, delta_col, flip_sign) in enumerate(datasets):
        ax = axes[i]

        # Load data
        data_path = PathLib(f'data/{dataset}_cost_table.csv')
        if not data_path.exists():
            print(f"  WARNING: {data_path} not found, skipping {dataset}")
            ax.set_title(f'{label}\n(data not found)')
            continue

        df = pd.read_csv(data_path)

        if delta_col not in df.columns:
            print(f"  WARNING: {delta_col} not in {dataset} data, skipping")
            ax.set_title(f'{label}\n(column not found)')
            continue

        delta = df[delta_col].dropna()

        # Flip sign if needed so minority is consistently on the right
        if flip_sign:
            delta = -delta

        # Plot histogram
        ax.hist(delta, bins=50, alpha=0.7, color='C0', edgecolor='black', linewidth=0.8)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2.0, label='Decision boundary')

        # Add statistics (use flipped values for display)
        mean_delta = delta.mean()
        std_delta = delta.std()
        n_pos = (delta >= 0).sum()
        n_neg = (delta < 0).sum()

        stats_text = f'n={len(delta):,}\nmean={mean_delta:.2f}\nstd={std_delta:.2f}\n+:{n_pos:,} / -:{n_neg:,}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Δ (signed margin)')
        ax.set_ylabel('Count')
        ax.set_title(label, fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_delta_histograms.pdf')
    plt.close()
    print(f"  Saved: fig6_delta_histograms.pdf")


# =============================================================================
# Table 1: Main Results LaTeX Table
# =============================================================================

def generate_table1_main_results(output_dir: Path):
    """
    Generate main results LaTeX table combining LR and fine-tuning results.
    Shows NEC and Error Rate for Standard CE vs |Δ|-weighted CE.
    """
    p1_df = load_summary('p1')
    p5_df = load_summary('p5')
    p6_df = load_summary('p6')

    if p1_df.empty:
        print("  WARNING: P1 data not found")
        return

    p1_clf = p1_df[p1_df['method'] == 'classification'].copy()
    p5_clf = p5_df[p5_df['method'] == 'classification'].copy() if not p5_df.empty else pd.DataFrame()
    p6_clf = p6_df[p6_df['method'] == 'classification'].copy() if not p6_df.empty else pd.DataFrame()

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Main results: NEC and Error Rate (\%) for Standard CE vs $|\Delta|$-weighted CE. Lower is better. Mean $\pm$ 95\% CI over 10 seeds.}')
    lines.append(r'\label{tab:main-results}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{llcccc}')
    lines.append(r'\toprule')
    lines.append(r' & & \multicolumn{2}{c}{Standard CE} & \multicolumn{2}{c}{$|\Delta|$-weighted CE} \\')
    lines.append(r'\cmidrule(lr){3-4} \cmidrule(lr){5-6}')
    lines.append(r'Task & Model & NEC & Error & NEC & Error \\')
    lines.append(r'\midrule')

    def fmt(val, std, n_seeds=10):
        """Format value with 95% CI."""
        if pd.isna(val):
            return '--'
        ci95 = std_to_ci95(std, n_seeds)
        return f'{val:.1f}$\\pm${ci95:.1f}'

    # LR/Frozen tasks
    lr_tasks = [
        ('jigsaw', 'tfidf', 'Jigsaw', 'TF-IDF', p1_clf),
        ('jigsaw', 'roberta', 'Jigsaw', 'RoBERTa', p1_clf),
        ('turkey', 'resnet50', 'Turkey', 'ResNet50', p1_clf),
        ('nhanes', 'histgbm', 'NHANES', 'HistGBM', p1_clf),
        ('inaturalist', 'resnet50', 'iNaturalist', 'ResNet50', p1_clf),
        ('synthetic', 'logreg', 'Synthetic', 'LogReg', p6_clf),
    ]

    lines.append(r'\multicolumn{6}{l}{\textit{Linear/Frozen Classifiers}} \\')
    for dataset, model, ds_label, model_label, source_df in lr_tasks:
        if source_df.empty:
            continue

        if dataset == 'synthetic':
            none_row = source_df[source_df['weighting'] == 'none']
            delta_row = source_df[source_df['weighting'] == 'absdelta']
        else:
            none_row = source_df[(source_df['dataset'] == dataset) &
                                 (source_df['model'] == model) &
                                 (source_df['weighting'] == 'none')]
            delta_row = source_df[(source_df['dataset'] == dataset) &
                                  (source_df['model'] == model) &
                                  (source_df['weighting'] == 'absdelta')]

        if none_row.empty and delta_row.empty:
            continue

        # Standard CE
        if not none_row.empty:
            n_seeds_none = get_n_seeds(none_row.iloc[0])
            none_nec = get_nec(none_row.iloc[0]) * 100
            none_nec_std = get_nec_std(none_row.iloc[0]) * 100
            none_err = get_error_rate(none_row.iloc[0]) * 100
            none_err_std = get_error_rate_std(none_row.iloc[0]) * 100
        else:
            n_seeds_none = 10
            none_nec = none_nec_std = none_err = none_err_std = np.nan

        # Delta-weighted
        if not delta_row.empty:
            n_seeds_delta = get_n_seeds(delta_row.iloc[0])
            delta_nec = get_nec(delta_row.iloc[0]) * 100
            delta_nec_std = get_nec_std(delta_row.iloc[0]) * 100
            delta_err = get_error_rate(delta_row.iloc[0]) * 100
            delta_err_std = get_error_rate_std(delta_row.iloc[0]) * 100
        else:
            n_seeds_delta = 10
            delta_nec = delta_nec_std = delta_err = delta_err_std = np.nan

        lines.append(f'{ds_label} & {model_label} & {fmt(none_nec, none_nec_std, n_seeds_none)} & {fmt(none_err, none_err_std, n_seeds_none)} & {fmt(delta_nec, delta_nec_std, n_seeds_delta)} & {fmt(delta_err, delta_err_std, n_seeds_delta)} \\\\')

    # Fine-tuned tasks
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{6}{l}{\textit{End-to-End Fine-tuning}} \\')

    ft_tasks = [
        ('jigsaw', 'roberta_finetune', 'Jigsaw', 'RoBERTa-FT'),
        ('turkey', 'resnet_finetune', 'Turkey', 'ResNet-FT'),
        ('inaturalist', 'resnet_finetune', 'iNaturalist', 'ResNet-FT'),
    ]

    for dataset, model, ds_label, model_label in ft_tasks:
        if p5_clf.empty:
            continue

        none_row = p5_clf[(p5_clf['dataset'] == dataset) &
                         (p5_clf['model'] == model) &
                         (p5_clf['weighting'] == 'none')]
        delta_row = p5_clf[(p5_clf['dataset'] == dataset) &
                          (p5_clf['model'] == model) &
                          (p5_clf['weighting'] == 'absdelta')]

        if none_row.empty and delta_row.empty:
            continue

        # Standard CE
        if not none_row.empty:
            n_seeds_none = get_n_seeds(none_row.iloc[0])
            none_nec = get_nec(none_row.iloc[0]) * 100
            none_nec_std = get_nec_std(none_row.iloc[0]) * 100
            none_err = get_error_rate(none_row.iloc[0]) * 100
            none_err_std = get_error_rate_std(none_row.iloc[0]) * 100
        else:
            n_seeds_none = 10
            none_nec = none_nec_std = none_err = none_err_std = np.nan

        # Delta-weighted
        if not delta_row.empty:
            n_seeds_delta = get_n_seeds(delta_row.iloc[0])
            delta_nec = get_nec(delta_row.iloc[0]) * 100
            delta_nec_std = get_nec_std(delta_row.iloc[0]) * 100
            delta_err = get_error_rate(delta_row.iloc[0]) * 100
            delta_err_std = get_error_rate_std(delta_row.iloc[0]) * 100
        else:
            n_seeds_delta = 10
            delta_nec = delta_nec_std = delta_err = delta_err_std = np.nan

        lines.append(f'{ds_label} & {model_label} & {fmt(none_nec, none_nec_std, n_seeds_none)} & {fmt(none_err, none_err_std, n_seeds_none)} & {fmt(delta_nec, delta_nec_std, n_seeds_delta)} & {fmt(delta_err, delta_err_std, n_seeds_delta)} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    latex_content = '\n'.join(lines)

    # Save to file
    table_path = output_dir / 'table1_main_results.tex'
    with open(table_path, 'w') as f:
        f.write(latex_content)

    print(f"  Saved: table1_main_results.tex")


# =============================================================================
# Main
# =============================================================================

def main():
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)

    print('Generating paper plots (ICML 2026 spec)...\n')

    print('Fig 1: NEC vs Error for 6 tasks with LR (incl. synthetic)...')
    plot_fig1_nec_vs_error_lr(output_dir)

    print('Fig 2: NEC vs Error for 3 tasks with fine-tuning...')
    plot_fig2_nec_vs_error_finetune(output_dir)

    print('Fig 3: Sample size scaling (TF-IDF and RoBERTa linear probe)...')
    plot_fig3_sample_size_scaling(output_dir)

    print('Fig 4: Different delta-methods for 6 tasks...')
    plot_fig4_delta_methods(output_dir)

    print('Fig 5: Standard vs delta-upweighting for fine-tuning...')
    plot_fig5_finetune_weighting(output_dir)

    print('Fig 6: Delta histograms for 4 real datasets...')
    plot_fig6_delta_histograms(output_dir)

    print('\nTable 1: Main results LaTeX table...')
    generate_table1_main_results(output_dir)

    print('\nAll plots saved to figures/')
    print('\nPlot summary (ICML 2026):')
    print('  fig1: NEC vs Error for 6 tasks with LR (incl. synthetic)')
    print('  fig2: NEC vs Error for 3 tasks with fine-tuning')
    print('  fig3: Sample size scaling (TF-IDF and RoBERTa)')
    print('  fig4: Different delta-methods for 6 tasks')
    print('  fig5: Standard vs delta-upweighting for fine-tuning')
    print('  fig6: Delta histograms for 4 real datasets')
    print('  table1: Main results LaTeX table')


if __name__ == '__main__':
    main()
