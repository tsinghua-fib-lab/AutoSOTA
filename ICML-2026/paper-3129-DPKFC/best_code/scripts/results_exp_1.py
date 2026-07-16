import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def process_dataset(csv_path, dataset_name, model_name):
    if not os.path.exists(csv_path):
        print(f"Warning: File not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    df.columns = [c.strip() for c in df.columns]
    if 'Acc' in df.columns:
        df.rename(columns={'Acc': 'Accuracy'}, inplace=True)

    method_map = {
        'Standard DP-SGD': 'DP-SGD',
        'KFAC (Noise)': 'Noise DP-KFAC',
        'KFAC (Public)': 'Public DP-KFAC',
        'Plain SGD': 'Base SGD'
    }
    df['Method'] = df['Method'].replace(method_map)

    base_df = df[df['Method'] == 'Base SGD']
    if not base_df.empty:
        base_mean = base_df['Accuracy'].mean() * 100
        base_std = base_df['Accuracy'].std() * 100
        base_str = f"{base_mean:.2f} $\\pm$ {base_std:.2f}"
    else:
        base_str = "-"

    dp_df = df[df['Method'] != 'Base SGD'].copy()

    dp_df['Epsilon'] = dp_df['Epsilon'].astype(float)

    agg = dp_df.groupby(['Method', 'Epsilon'])['Accuracy'].agg(['mean', 'std']).reset_index()

    agg['Score'] = agg.apply(lambda x: f"{x['mean']*100:.2f} $\\pm$ {x['std']*100:.2f}", axis=1)

    pivot_df = agg.pivot(index='Epsilon', columns='Method', values='Score')

    pivot_df['Base SGD'] = base_str
    pivot_df['Dataset'] = dataset_name
    pivot_df['Model'] = model_name

    desired_order = ['Dataset', 'Model', 'Base SGD', 'DP-SGD', 'Noise DP-KFAC', 'Public DP-KFAC']
    final_cols = [c for c in desired_order if c in pivot_df.columns or c in ['Dataset', 'Model', 'Base SGD']]

    return pivot_df.reset_index()[['Dataset', 'Model', 'Epsilon'] + final_cols[2:]]


def convert_to_numeric(df):
    numeric_df = df.copy()
    for col in ['Base SGD', 'DP-SGD', 'Noise DP-KFAC', 'Public DP-KFAC']:
        if col in numeric_df.columns:
            if numeric_df[col].dtype == 'object':
                mean_values = numeric_df[col].str.extract(r'([\d.]+)').astype(float)
                std_values = numeric_df[col].str.extract(r'\$\\pm\$\s*([\d.]+)').astype(float)

                numeric_df[col] = mean_values
                numeric_df[col + '_std'] = std_values
            else:
                numeric_df[col + '_std'] = 0
    return numeric_df


def generate_plots(df, output_dir):
    numeric_df = convert_to_numeric(df)
    datasets = numeric_df['Dataset'].unique()
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, n_datasets, figsize=(3.5*n_datasets, 2.8))
    if n_datasets == 1:
        axes = [axes]

    fig.suptitle('DP Methods Comparison by Dataset', fontsize=10, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        subset = numeric_df[numeric_df['Dataset'] == dataset]

        methods = ['DP-SGD', 'Noise DP-KFAC', 'Public DP-KFAC']
        for j, method in enumerate(methods):
            if method in subset.columns:
                ax.plot(subset['Epsilon'], subset[method],
                       marker=markers[j], markersize=3, linewidth=1.2,
                       color=colors[j], label=method, alpha=0.8)

                if method + '_std' in subset.columns:
                    std_values = subset[method + '_std']
                    ax.fill_between(subset['Epsilon'],
                                   subset[method] - std_values,
                                   subset[method] + std_values,
                                   color=colors[j], alpha=0.25, linewidth=0)
                    ax.errorbar(subset['Epsilon'], subset[method],
                              yerr=std_values, fmt='none',
                              color=colors[j], alpha=0.4, capsize=2)

        if 'Base SGD' in subset.columns:
            baseline = subset['Base SGD'].iloc[0]
            ax.axhline(y=baseline, color='gray', linestyle='--',
                      linewidth=0.8, alpha=0.7, label='Base SGD')

            if 'Base SGD_std' in subset.columns:
                baseline_std = subset['Base SGD_std'].iloc[0]
                ax.axhspan(baseline - baseline_std, baseline + baseline_std,
                          color='gray', alpha=0.15)

        ax.set_xlabel('ε', fontsize=8)
        ax.set_ylabel('Accuracy (%)', fontsize=8)
        ax.set_title(f'{dataset}', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.2)

        if i == 0:
            ax.legend(fontsize=6, frameon=False, loc='best')

        all_values = []
        for method in methods:
            if method in subset.columns:
                all_values.extend(subset[method].dropna().tolist())
                if method + '_std' in subset.columns:
                    all_values.extend((subset[method] - subset[method + '_std']).dropna().tolist())
                    all_values.extend((subset[method] + subset[method + '_std']).dropna().tolist())
        if 'Base SGD' in subset.columns:
            all_values.append(subset['Base SGD'].iloc[0])
            if 'Base SGD_std' in subset.columns:
                baseline_std = subset['Base SGD_std'].iloc[0]
                all_values.append(subset['Base SGD'].iloc[0] - baseline_std)
                all_values.append(subset['Base SGD'].iloc[0] + baseline_std)

        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            y_range = y_max - y_min
            ax.set_ylim(bottom=max(0, y_min - 0.05*y_range), top=y_max + 0.05*y_range)

    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(top=0.85)

    output_path = Path(output_dir) / "dp_methods_by_dataset_wide.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nWide plot saved to: {output_path}")

    plt.close()


def generate_compact_plot(df, output_dir):
    numeric_df = convert_to_numeric(df)
    datasets = numeric_df['Dataset'].unique()
    n_datasets = len(datasets)

    fig, axes = plt.subplots(n_datasets, 1, figsize=(3.3, 1.4 * n_datasets), sharex=False)
    if n_datasets == 1:
        axes = [axes]

    colors = {'DP-SGD': '#1f77b4', 'Noise DP-KFAC': '#ff7f0e', 'Public DP-KFAC': '#2ca02c'}
    markers = {'DP-SGD': 'o', 'Noise DP-KFAC': 's', 'Public DP-KFAC': '^'}
    short_labels = {'DP-SGD': 'DP-SGD', 'Noise DP-KFAC': 'Noise', 'Public DP-KFAC': 'Public'}

    methods = ['DP-SGD', 'Noise DP-KFAC', 'Public DP-KFAC']

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        subset = numeric_df[numeric_df['Dataset'] == dataset]

        for method in methods:
            if method in subset.columns:
                ax.plot(subset['Epsilon'], subset[method],
                       marker=markers[method], markersize=2.5, linewidth=1,
                       color=colors[method], label=short_labels[method], alpha=0.85)

                if method + '_std' in subset.columns:
                    std_values = subset[method + '_std']
                    ax.fill_between(subset['Epsilon'],
                                   subset[method] - std_values,
                                   subset[method] + std_values,
                                   color=colors[method], alpha=0.2, linewidth=0)

        if 'Base SGD' in subset.columns:
            baseline = subset['Base SGD'].iloc[0]
            ax.axhline(y=baseline, color='gray', linestyle='--',
                      linewidth=0.7, alpha=0.6, label='Base')
            if 'Base SGD_std' in subset.columns:
                baseline_std = subset['Base SGD_std'].iloc[0]
                ax.axhspan(baseline - baseline_std, baseline + baseline_std,
                          color='gray', alpha=0.1)

        ax.set_ylabel('Acc (%)', fontsize=6)
        ax.set_title(f'{dataset}', fontsize=7, fontweight='bold', pad=2)
        ax.tick_params(axis='both', labelsize=5)
        ax.grid(True, alpha=0.15, linewidth=0.5)

        all_values = []
        for method in methods:
            if method in subset.columns:
                all_values.extend(subset[method].dropna().tolist())
                if method + '_std' in subset.columns:
                    all_values.extend((subset[method] - subset[method + '_std']).dropna().tolist())
                    all_values.extend((subset[method] + subset[method + '_std']).dropna().tolist())
        if 'Base SGD' in subset.columns:
            all_values.append(subset['Base SGD'].iloc[0])

        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            y_range = y_max - y_min
            ax.set_ylim(bottom=max(0, y_min - 0.03*y_range), top=y_max + 0.03*y_range)

    axes[-1].set_xlabel('Privacy Budget (ε)', fontsize=6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(pad=0.3, h_pad=0.4)
    plt.subplots_adjust(bottom=0.12)

    output_path = Path(output_dir) / "dp_methods_compact.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Compact plot saved to: {output_path}")

    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate result plots and LaTeX tables")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing result CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save output plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_config = [
        (results_dir / "stackoverflow_dp_kfac_results.csv", "StackOverflow", "Roberta"),
        (results_dir / "mnist_experiment_results.csv", "MNIST", "CNN"),
        (results_dir / "cross_vit_cifar100_results.csv", "CIFAR-100", "CrossViT"),
    ]

    all_results = []
    for f_path, d_name, m_name in files_config:
        processed_df = process_dataset(str(f_path), d_name, m_name)
        if not processed_df.empty:
            all_results.append(processed_df)

    if not all_results:
        print("No data loaded. Check your file paths.")
        return

    final_df = pd.concat(all_results, ignore_index=True)

    latex_code = final_df.to_latex(
        index=False,
        escape=False,
        column_format="llc|c|ccc",
        header=["Dataset", "Model", r"$\epsilon$", "Base SGD", "DP-SGD", "Noise DP-KFAC", "Public DP-KFAC"],
        float_format="%.2f"
    )

    lines = latex_code.splitlines()
    formatted_lines = []
    last_dataset = ""
    last_model = ""

    for line in lines:
        if "&" in line and "epsilon" not in line and "Dataset" not in line:
            parts = line.split('&')
            current_dataset = parts[0].strip()
            current_model = parts[1].strip()

            if current_dataset == last_dataset:
                parts[0] = ""
            else:
                last_dataset = current_dataset

            if current_model == last_model and parts[0] == "":
                parts[1] = ""
            else:
                last_model = current_model

            formatted_lines.append(" & ".join(parts))
        else:
            formatted_lines.append(line)

    final_latex = "\n".join(formatted_lines)

    print("-" * 30)
    print("GENERATED LATEX CODE")
    print("-" * 30)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Comparison of Accuracy across Datasets and Privacy Budgets}")
    print(final_latex)
    print("\\end{table}")

    generate_plots(final_df, output_dir)
    generate_compact_plot(final_df, output_dir)


if __name__ == "__main__":
    main()
