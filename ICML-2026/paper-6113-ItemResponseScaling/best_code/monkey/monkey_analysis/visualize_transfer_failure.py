"""
Visualize why difficulty parameters don't transfer across model generations.

Creates multiple views:
1. Scatter plots: HELM z vs test-time z for each benchmark (shows correlation breakdown)
2. Model timeline: Shows capability gap between HELM and test-time models
3. Question-level shifts: Shows how specific questions' difficulties changed
"""

import torch
import numpy as np
from scipy.stats import spearmanr
from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tueplots import bundles
import pandas as pd

# Model release dates (approximate)
MODEL_RELEASE_DATES = {
    # Test-time models (new, 2024-2025)
    "DeepSeek-R1-Distill-Llama-8B": "2025-01",
    "DeepSeek-V2-Lite-Chat": "2024-06",
    "Meta-Llama-3-70B-Instruct": "2024-04",
    "Meta-Llama-3-8B-Instruct": "2024-04",
    "Qwen3-14B": "2024-09",
    "Qwen3-32B": "2024-09",
    "Qwen3-8B": "2024-09",
    "gemma-3-27b-it": "2024-12",

    # Typical HELM models (older, 2022-2023)
    "gpt-3.5-turbo": "2022-11",
    "claude-2": "2023-07",
    "llama-2-70b": "2023-07",
    "mistral-7b": "2023-09",
}

BENCHMARKS = ["commonsense", "legalbench", "med_qa", "bbq", "legal_support", "lsat_qa"]

def load_data():
    """Load test-time response data and z-scores."""
    FILE_NAME = "irsl_testtime_resmat1"
    cache_dir = snapshot_download(repo_id=f"stair-lab/{FILE_NAME}", repo_type="dataset")

    testtime_resmat = torch.load(f"{cache_dir}/resmat.pt", weights_only=False)
    data_tensor = testtime_resmat["data_tensor"].numpy()
    models = testtime_resmat["models"]
    questions = testtime_resmat["questions"]
    scenarios = testtime_resmat["scenarios"]
    helm_zs = np.array(testtime_resmat["zs"])

    testtime_calibrated = torch.load("monkey/monkey_analysis/irsl_testtime_resmat1_withz.pt", weights_only=False)
    testtime_zs = testtime_calibrated["zs"]
    if hasattr(testtime_zs, 'numpy'):
        testtime_zs = testtime_zs.numpy()

    return {
        "data_tensor": data_tensor,
        "models": models,
        "questions": questions,
        "scenarios": scenarios,
        "helm_zs": helm_zs,
        "testtime_zs": testtime_zs
    }


def plot_scatter_grid(data, output_path="transfer_scatter_grid.pdf"):
    """
    Create 6 scatter plots (2x3 grid) showing HELM z vs test-time z for each benchmark.
    """
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        helm_zs = data["helm_zs"]
        testtime_zs = data["testtime_zs"]
        scenarios = data["scenarios"]

        for idx, benchmark in enumerate(BENCHMARKS):
            ax = axes[idx]
            mask = np.array([s == benchmark for s in scenarios])

            helm_z_bench = helm_zs[mask]
            testtime_z_bench = testtime_zs[mask]

            # Compute correlation
            corr, _ = spearmanr(helm_z_bench, testtime_z_bench)

            # Scatter plot
            ax.scatter(helm_z_bench, testtime_z_bench, alpha=0.5, s=10, color='steelblue')

            # Add diagonal line (perfect correlation)
            all_z = np.concatenate([helm_z_bench, testtime_z_bench])
            z_min, z_max = all_z.min(), all_z.max()
            ax.plot([z_min, z_max], [z_min, z_max], 'k--', alpha=0.3, linewidth=1)

            # Labels
            ax.set_xlabel("HELM $z$ (older models)", fontsize=10)
            ax.set_ylabel("Test-time $z$ (newer models)", fontsize=10)
            ax.set_title(f"{benchmark.replace('_', ' ').title()}\n$\\rho={corr:.3f}$", fontsize=11)
            ax.grid(alpha=0.3)

            # Color-code by transfer quality
            if corr > 0.7:
                ax.patch.set_facecolor('#e8f5e9')  # Light green
            elif corr > 0.4:
                ax.patch.set_facecolor('#fff9c4')  # Light yellow
            else:
                ax.patch.set_facecolor('#ffebee')  # Light red

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved scatter grid to {output_path}")
        plt.close()


def plot_difficulty_shift(data, output_path="difficulty_shift.pdf"):
    """
    Show how question difficulties shifted between HELM and test-time.
    One subplot per benchmark showing distribution of z-score changes.
    """
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        helm_zs = data["helm_zs"]
        testtime_zs = data["testtime_zs"]
        scenarios = data["scenarios"]

        for idx, benchmark in enumerate(BENCHMARKS):
            ax = axes[idx]
            mask = np.array([s == benchmark for s in scenarios])

            z_diff = testtime_zs[mask] - helm_zs[mask]

            # Histogram of z-score changes
            ax.hist(z_diff, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No change')
            ax.axvline(np.mean(z_diff), color='orange', linestyle='-', linewidth=2,
                      label=f'Mean = {np.mean(z_diff):.2f}')

            ax.set_xlabel("$\\Delta z$ (test-time $-$ HELM)", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"{benchmark.replace('_', ' ').title()}", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis='y')

            # Annotate shift direction
            mean_shift = np.mean(z_diff)
            if mean_shift > 0.5:
                ax.text(0.95, 0.95, "Questions got\nEASIER",
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                       usetex=False)
            elif mean_shift < -0.5:
                ax.text(0.95, 0.95, "Questions got\nHARDER",
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
                       usetex=False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved difficulty shift plot to {output_path}")
        plt.close()


def plot_model_timeline_comparison(data, output_path="model_timeline.pdf"):
    """
    Show model release timeline and their average performance on each benchmark.
    Illustrates the capability gap between HELM and test-time models.
    """
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

        data_tensor = data["data_tensor"]
        models = data["models"]
        scenarios = data["scenarios"]

        # Get model release dates
        model_dates = [MODEL_RELEASE_DATES.get(m, "2024-01") for m in models]
        model_years = [float(d.split("-")[0]) + float(d.split("-")[1])/12 for d in model_dates]

        # Compute average performance per benchmark per model
        for idx, benchmark in enumerate(BENCHMARKS):
            row = idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])

            mask = np.array([s == benchmark for s in scenarios])
            bench_data = data_tensor[:, mask, :]  # (n_models, n_questions, n_samples)

            # Average performance per model
            model_pass_rates = np.mean(bench_data, axis=(1, 2))  # (n_models,)

            # Sort by release date
            sorted_idx = np.argsort(model_years)
            sorted_years = [model_years[i] for i in sorted_idx]
            sorted_pass_rates = [model_pass_rates[i] for i in sorted_idx]
            sorted_models = [models[i] for i in sorted_idx]

            # Plot
            colors = ['red' if 'HELM' in MODEL_RELEASE_DATES.get(m, '') else 'blue'
                     for m in sorted_models]
            ax.scatter(sorted_years, sorted_pass_rates, s=100, alpha=0.7, c=colors)

            # Connect with line
            ax.plot(sorted_years, sorted_pass_rates, 'k-', alpha=0.3, linewidth=1)

            # Labels
            ax.set_xlabel("Release Date", fontsize=10)
            ax.set_ylabel("Average Pass@1", fontsize=10)
            ax.set_title(f"{benchmark.replace('_', ' ').title()}", fontsize=11)
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

            # Add year labels on x-axis
            ax.set_xticks([2022, 2023, 2024, 2025])
            ax.set_xticklabels(['2022', '2023', '2024', '2025'])

        # Legend in the bottom row
        ax_legend = fig.add_subplot(gs[2, :])
        ax_legend.axis('off')
        ax_legend.scatter([], [], s=100, c='red', label='HELM models (T=0, calibration set)')
        ax_legend.scatter([], [], s=100, c='blue', label='Test-time models (T=1.0, evaluation set)')
        ax_legend.legend(loc='center', fontsize=11, ncol=2)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved model timeline to {output_path}")
        plt.close()


def plot_question_trajectories(data, output_path="question_trajectories.pdf"):
    """
    For each benchmark, show a few example questions and how their z-scores changed.
    Specifically pick questions with large shifts to illustrate the problem.
    """
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()

        helm_zs = data["helm_zs"]
        testtime_zs = data["testtime_zs"]
        scenarios = data["scenarios"]
        questions = data["questions"]

        for idx, benchmark in enumerate(BENCHMARKS):
            ax = axes[idx]
            mask = np.array([s == benchmark for s in scenarios])
            bench_indices = np.where(mask)[0]

            # Find questions with largest shifts
            z_diffs = np.abs(testtime_zs[mask] - helm_zs[mask])
            top_shift_local_idx = np.argsort(z_diffs)[-5:][::-1]  # Top 5

            # Plot each question as a line from HELM z to test-time z
            for local_idx in top_shift_local_idx:
                global_idx = bench_indices[local_idx]
                helm_z = helm_zs[global_idx]
                testtime_z = testtime_zs[global_idx]

                # Draw line
                ax.plot([0, 1], [helm_z, testtime_z], 'o-', alpha=0.6, linewidth=2)

            # Add horizontal dashed line at z=0
            ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)

            # Labels
            ax.set_xlim(-0.2, 1.2)
            ax.set_xticks([0, 1])
            labels = ax.set_xticklabels(['HELM\n(old models)', 'Test-time\n(new models)'], fontsize=9)
            for label in labels:
                label.set_usetex(False)
            ax.set_ylabel("$z$ (difficulty)", fontsize=10)
            ax.set_title(f"{benchmark.replace('_', ' ').title()}", fontsize=11)
            ax.grid(alpha=0.3, axis='y')

            # Annotate z=0 line
            ax.text(1.05, 0, 'medium\ndifficulty', fontsize=8, ha='left', va='center',
                   color='gray', alpha=0.7, usetex=False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved question trajectories to {output_path}")
        plt.close()


def plot_combined_figure(data, output_path="transfer_failure_combined.pdf"):
    """
    Create a comprehensive figure showing multiple views of the transfer failure.

    Layout:
    - Top row (2x3): Scatter plots for each benchmark
    - Bottom row: Summary statistics
    """
    with plt.rc_context(bundles.icml2024(usetex=True, family="serif")):
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        helm_zs = data["helm_zs"]
        testtime_zs = data["testtime_zs"]
        scenarios = data["scenarios"]
        data_tensor = data["data_tensor"]

        # Top 2 rows: Scatter plots (6 benchmarks)
        for idx, benchmark in enumerate(BENCHMARKS):
            row = idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])

            mask = np.array([s == benchmark for s in scenarios])
            helm_z_bench = helm_zs[mask]
            testtime_z_bench = testtime_zs[mask]

            # Compute correlation
            corr, _ = spearmanr(helm_z_bench, testtime_z_bench)

            # Compute average pass@1 for this benchmark
            bench_data = data_tensor[:, mask, :]
            avg_pass_at_1 = np.mean(bench_data)

            # Scatter with color based on pass@1
            scatter = ax.scatter(helm_z_bench, testtime_z_bench,
                               c=np.mean(bench_data, axis=(0, 2)),  # Color by avg pass@1
                               cmap='RdYlGn', vmin=0, vmax=1,
                               alpha=0.6, s=15)

            # Diagonal line
            all_z = np.concatenate([helm_z_bench, testtime_z_bench])
            z_min, z_max = all_z.min(), all_z.max()
            ax.plot([z_min, z_max], [z_min, z_max], 'k--', alpha=0.3, linewidth=1)

            # Labels
            ax.set_xlabel("HELM $z$", fontsize=9)
            ax.set_ylabel("Test-time $z$", fontsize=9)
            ax.set_title(f"{benchmark.replace('_', ' ').title()}\n$\\rho={corr:.2f}$, avg pass@1={avg_pass_at_1:.2f}",
                        fontsize=10)
            ax.grid(alpha=0.2)

            # Add colorbar for first plot
            if idx == 0:
                cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
                cbar.set_label('Avg Pass@1', fontsize=8)

        # Bottom row: Summary statistics bar chart
        ax_summary = fig.add_subplot(gs[2, :])

        correlations = []
        avg_pass_rates = []
        for benchmark in BENCHMARKS:
            mask = np.array([s == benchmark for s in scenarios])
            corr, _ = spearmanr(helm_zs[mask], testtime_zs[mask])
            correlations.append(corr)

            bench_data = data_tensor[:, mask, :]
            avg_pass_rates.append(np.mean(bench_data))

        x = np.arange(len(BENCHMARKS))
        width = 0.35

        ax_summary.bar(x - width/2, correlations, width, label='Correlation $\\rho$',
                      color=['green' if c > 0.7 else 'orange' if c > 0.4 else 'red'
                             for c in correlations], alpha=0.7)
        ax_summary.bar(x + width/2, avg_pass_rates, width, label='Avg Pass@1',
                      color='steelblue', alpha=0.7)

        ax_summary.set_xlabel('Benchmark', fontsize=11)
        ax_summary.set_ylabel('Value', fontsize=11)
        ax_summary.set_title('Summary: Transfer Quality vs Model Performance', fontsize=12)
        ax_summary.set_xticks(x)
        ax_summary.set_xticklabels([b.replace('_', ' ').title() for b in BENCHMARKS],
                                   rotation=15, ha='right', fontsize=9)
        ax_summary.legend(fontsize=10)
        ax_summary.grid(alpha=0.3, axis='y')
        ax_summary.set_ylim(0, 1.1)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined figure to {output_path}")
        plt.close()


def main():
    """Generate all visualizations."""
    print("Loading data...")
    data = load_data()

    print("\nGenerating visualizations...")

    # 1. Scatter grid (main figure)
    plot_scatter_grid(data, "monkey/monkey_analysis/transfer_scatter_grid.pdf")

    # 2. Difficulty shift histograms
    plot_difficulty_shift(data, "monkey/monkey_analysis/difficulty_shift.pdf")

    # 3. Model timeline (shows capability gap)
    plot_model_timeline_comparison(data, "monkey/monkey_analysis/model_timeline.pdf")

    # 4. Question trajectories (shows ranking scramble)
    plot_question_trajectories(data, "monkey/monkey_analysis/question_trajectories.pdf")

    # 5. Combined comprehensive figure
    plot_combined_figure(data, "monkey/monkey_analysis/transfer_failure_combined.pdf")

    print("\n" + "="*80)
    print("All visualizations complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. transfer_scatter_grid.pdf - Scatter plots showing correlation breakdown")
    print("  2. difficulty_shift.pdf - Histograms of z-score changes")
    print("  3. model_timeline.pdf - Model capability progression over time")
    print("  4. question_trajectories.pdf - Individual question difficulty shifts")
    print("  5. transfer_failure_combined.pdf - Comprehensive combined figure")


if __name__ == "__main__":
    main()
