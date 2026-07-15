"""
Visualization for LLM Causal Coefficient Experiments.

Processes results from LLM_responses_linear/.
Each run may have a different LLM-determined causal ordering, so the
correlation matrix is reordered to match before computing compatibility
scores.
"""

import pandas as pd
import numpy as np
import os
import re
import glob
import matplotlib.pyplot as plt
from synthetic_experiments_linear import compatibility_score
from experiments_llm_linear import compute_correlation_matrix


DATA_DIR = os.path.dirname(__file__)
LLM_RESPONSES_DIR = os.path.join(DATA_DIR, 'LLM_responses_linear')


# Model name mapping for display in plots
MODEL_DISPLAY_NAMES = {
    'claude-opus-4-6-v1': 'Claude Opus 4.6',
    'claude-opus-4-5-20251101-v1': 'Claude Opus 4.5',
    'claude-opus-4-1-20250805-v1': 'Claude Opus 4.1',
    'kimi-k2-thinking': 'Kimi K2 Thinking',
    'mistral-large-3-675b-instruct': 'Mistral Large 3',
    'magistral-small-2509': 'Magistral Small',
    'gpt-oss-120b-1': 'GPT oss 120B',
    'gpt-oss-20b-1': 'GPT oss 20B',
    'qwen3-next-80b-a3b': 'Qwen3 Next 80B A3B',
    'qwen3-235b-a22b-2507-v1': 'Qwen3 235B A22b',
    'gemma-3-4b-it': 'Gemma 3 4B IT',
    'gemma-3-27b-it': 'Gemma 3 27B IT',
}


def get_display_name(model_name):
    """Get display name for a model, preserving baseline names."""
    if model_name.startswith('[Baseline]'):
        return model_name
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def load_correlation_matrix_ordered(ordering):
    """Load the correlation matrix reordered to match the given variable ordering."""
    corr_df = compute_correlation_matrix()
    return corr_df.loc[ordering, ordering].values


def load_causal_coefficients(model_name, run_num):
    """Load causal coefficients matrix for a given model and run.

    Returns:
        A: numpy array of coefficients (in the run's ordering)
        ordering: list of variable names in the run's causal ordering
    """
    filepath = os.path.join(
        LLM_RESPONSES_DIR,
        f'causal_coefficients_{model_name}_run{run_num}.csv',
    )
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    coef_df = pd.read_csv(filepath, index_col=0)
    ordering = list(coef_df.index)
    return coef_df.values, ordering


def discover_models():
    """Auto-discover all model names and their runs from CSV files."""
    pattern = os.path.join(LLM_RESPONSES_DIR, 'causal_coefficients_*_run*.csv')
    files = glob.glob(pattern)

    model_runs = {}
    for f in files:
        basename = os.path.basename(f)
        basename = basename.replace('causal_coefficients_', '').replace('.csv', '')
        match = re.match(r'(.+)_run(\d+)$', basename)
        if match:
            model_name = match.group(1)
            run_num = int(match.group(2))
            if model_name not in model_runs:
                model_runs[model_name] = []
            model_runs[model_name].append(run_num)

    for model_name in model_runs:
        model_runs[model_name] = sorted(model_runs[model_name])

    return model_runs


def create_baseline_matrices(cov, empirical_std, n_random_samples=100, seed=42):
    """Create baseline causal coefficient matrices for comparison."""
    np.random.seed(seed)
    n = cov.shape[0]

    baselines = {}

    random_scores = []
    random_diffs = []
    for _ in range(n_random_samples):
        A_random = np.eye(n)
        for i in range(n):
            for j in range(i):
                A_random[i, j] = np.random.normal(0, empirical_std)
        random_scores.append(compatibility_score(A_random, cov))
        lower_tri_indices = np.tril_indices(n, k=-1)
        corr_lower = cov[lower_tri_indices]
        coef_lower = A_random[lower_tri_indices]
        random_diffs.append(np.mean(np.abs(corr_lower - coef_lower)))

    baselines[f'[Baseline] N(0, {empirical_std:.3f}\u00b2)'] = {
        'mean': np.mean(random_scores),
        'std': np.std(random_scores),
        'avg_diff_mean': np.mean(random_diffs),
        'avg_diff_std': np.std(random_diffs),
        'is_baseline': True,
        'num_runs': n_random_samples,
    }

    return baselines


def compute_compatibility_scores():
    """Compute compatibility scores for all LLM causal coefficient estimates."""
    # Canonical ordering only used for baselines
    corr_df = compute_correlation_matrix()
    canonical_ordering = list(corr_df.columns)
    cov_canonical = corr_df.values
    n = len(canonical_ordering)

    print(f"\n{'='*60}")
    print("LLM COMPATIBILITY SCORES")
    print(f"{'='*60}")

    model_runs = discover_models()
    print(f"Found {len(model_runs)} models:")
    for model, runs in sorted(model_runs.items()):
        print(f"  - {model}: {len(runs)} runs")
    print()

    results = {}
    all_coefficients = []
    best_run = {'model': None, 'run_num': None, 'score': float('-inf')}

    print("-" * 40)
    print("PROCESSING MODELS")
    print("-" * 40)

    for model_name in sorted(model_runs.keys()):
        runs = model_runs[model_name]
        print(f"\nModel: {model_name}")

        run_scores = []
        run_diffs = []

        for run_num in runs:
            try:
                A, ordering = load_causal_coefficients(model_name, run_num)

                # Reorder correlation matrix to match this run's ordering
                cov = load_correlation_matrix_ordered(ordering)

                lower_tri_indices = np.tril_indices(n, k=-1)
                coef_lower = A[lower_tri_indices]
                all_coefficients.extend(coef_lower.tolist())

                score = compatibility_score(A, cov)
                run_scores.append(score)

                if score > best_run['score']:
                    best_run = {'model': model_name, 'run_num': run_num, 'score': score}

                corr_lower = cov[lower_tri_indices]
                avg_diff = np.mean(np.abs(corr_lower - coef_lower))
                run_diffs.append(avg_diff)

                print(f"  Run {run_num} ({' -> '.join(ordering[:3])}...): "
                      f"Score = {score:.6f}, Avg Diff = {avg_diff:.6f}")

            except Exception as e:
                print(f"  Run {run_num}: Error - {e}")

        if run_scores:
            results[model_name] = {
                'mean': np.mean(run_scores),
                'std': np.std(run_scores) if len(run_scores) > 1 else 0,
                'avg_diff_mean': np.mean(run_diffs),
                'avg_diff_std': np.std(run_diffs) if len(run_diffs) > 1 else 0,
                'run_scores': run_scores,
                'run_diffs': run_diffs,
                'num_runs': len(run_scores),
                'is_baseline': False,
            }
            print(f"  -> Average: {results[model_name]['mean']:.6f} "
                  f"\u00b1 {results[model_name]['std']:.6f} ({len(run_scores)} runs)")
        else:
            results[model_name] = {
                'mean': None, 'std': None,
                'avg_diff_mean': None, 'avg_diff_std': None,
                'run_scores': [], 'run_diffs': [],
                'num_runs': 0, 'is_baseline': False,
            }
            print(f"  -> No valid runs")

    # Empirical std for baselines
    if all_coefficients:
        empirical_std = np.std(all_coefficients)
        empirical_mean = np.mean(all_coefficients)
        print(f"\n" + "-" * 40)
        print("EMPIRICAL COEFFICIENT STATISTICS")
        print("-" * 40)
        print(f"Total coefficients collected: {len(all_coefficients)}")
        print(f"Empirical mean: {empirical_mean:.6f}")
        print(f"Empirical std: {empirical_std:.6f}")
    else:
        empirical_std = 0.5

    # Baselines use canonical ordering
    print(f"\n" + "-" * 40)
    print("COMPUTING BASELINES")
    print("-" * 40)
    baselines = create_baseline_matrices(cov_canonical, empirical_std)
    for name, baseline in baselines.items():
        print(f"{name}: Score={baseline['mean']:.6f} \u00b1 {baseline['std']:.6f}")
        results[name] = baseline
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Model':<50} {'Mean Score':>12} {'Std':>10} {'Runs':>6}")
    print("-" * 80)
    for model, res in sorted(results.items(),
                              key=lambda x: x[1]['mean'] if x[1]['mean'] is not None else float('-inf'),
                              reverse=True):
        if res['mean'] is not None:
            print(f"{model:<50} {res['mean']:>12.6f} {res['std']:>10.6f} {res['num_runs']:>6}")
        else:
            print(f"{model:<50} {'N/A':>12} {'N/A':>10} {0:>6}")

    if best_run['model'] is not None:
        print()
        print("=" * 60)
        print("BEST INDIVIDUAL RUN")
        print("=" * 60)
        print(f"Model: {best_run['model']}")
        print(f"Run: {best_run['run_num']}")
        print(f"Compatibility Score: {best_run['score']:.6f}")

    return results, cov_canonical


def get_score_color(score, baseline_score, all_scores):
    """Color a score based on its value relative to the random baseline."""
    non_baseline_scores = [s for s in all_scores if s is not None]

    if score > baseline_score:
        max_score = max(non_baseline_scores) if non_baseline_scores else baseline_score + 1
        if max_score > baseline_score:
            intensity = (score - baseline_score) / (max_score - baseline_score)
        else:
            intensity = 0.5
        intensity = np.clip(intensity, 0.2, 1.0)
        light_green = np.array([0.7, 0.93, 0.7])
        dark_green = np.array([0.18, 0.49, 0.2])
        rgb = light_green + intensity * (dark_green - light_green)
        return tuple(rgb)

    elif score > 0:
        if baseline_score > 0:
            intensity = score / baseline_score
        else:
            intensity = 0.5
        intensity = np.clip(intensity, 0.2, 1.0)
        light_yellow = np.array([1.0, 0.95, 0.7])
        dark_amber = np.array([0.96, 0.49, 0.0])
        rgb = light_yellow + intensity * (dark_amber - light_yellow)
        return tuple(rgb)

    else:
        min_score = min(non_baseline_scores) if non_baseline_scores else -1
        if min_score < 0:
            intensity = score / min_score
        else:
            intensity = 0.5
        intensity = np.clip(intensity, 0.2, 1.0)
        light_red = np.array([1.0, 0.8, 0.8])
        dark_red = np.array([0.78, 0.16, 0.16])
        rgb = light_red + intensity * (dark_red - light_red)
        return tuple(rgb)


def plot_compatibility_scores(results, save_path=None):
    """Horizontal bar plot of compatibility scores with error bars."""
    valid_results = {k: v for k, v in results.items() if v['mean'] is not None}

    if not valid_results:
        print("No valid results to plot")
        return None

    baseline_score = None
    for name, data in valid_results.items():
        if data.get('is_baseline', False):
            baseline_score = data['mean']
            break
    if baseline_score is None:
        baseline_score = 0

    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['mean'], reverse=True)
    model_names = [get_display_name(m[0]) for m in sorted_models]
    means = [m[1]['mean'] for m in sorted_models]
    stds = [m[1]['std'] for m in sorted_models]
    is_baseline = [m[1].get('is_baseline', False) for m in sorted_models]

    all_scores = [m for m, b in zip(means, is_baseline) if not b]

    colors = []
    for mean, baseline in zip(means, is_baseline):
        if baseline:
            colors.append('#a0a0a0')
        else:
            colors.append(get_score_color(mean, baseline_score, all_scores))

    fig, ax = plt.subplots(figsize=(16, max(8, len(model_names) * 0.5)))
    y_pos = np.arange(len(model_names))

    bars = ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor='black',
                   alpha=0.9, capsize=6, error_kw={'elinewidth': 1.5, 'capthick': 1.5})

    for bar, baseline in zip(bars, is_baseline):
        if baseline:
            bar.set_hatch('///')
            bar.set_edgecolor('black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, fontsize=24)
    ax.set_xlabel('Compatibility Score (Mean $\\pm$ Std)', fontsize=28)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='x', labelsize=28)

    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    for i, (mean, std, baseline) in enumerate(zip(means, stds, is_baseline)):
        if mean >= 0:
            label_x = max(mean + std, mean) + x_range * 0.02
        else:
            label_x = max(0, mean + std) + x_range * 0.02
        ax.text(label_x, i, f'{mean:.3f} \u00b1 {std:.3f}', va='center',
                fontsize=24, fontweight='bold')

    ax.set_xlim(-0.6, 2.4)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', edgecolor='black', alpha=0.9, label='LLM models'),
        Patch(facecolor='#a0a0a0', edgecolor='black', alpha=0.9, hatch='///', label='Random \n baseline'),
    ]
    ax.legend(handles=legend_elements, fontsize=28)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()
    return fig


if __name__ == "__main__":
    import sys

    if not os.path.exists(LLM_RESPONSES_DIR):
        print(f"Directory not found: {LLM_RESPONSES_DIR}")
        print("Run experiments_llm_linear.py first to generate results.")
        sys.exit(1)

    results, cov = compute_compatibility_scores()

    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print("=" * 60)

    valid_results = {k: v for k, v in results.items() if v['mean'] is not None}
    sorted_by_compat = sorted(valid_results.items(), key=lambda x: x[1]['mean'], reverse=True)
    model_order = [m[0] for m in sorted_by_compat]

    scores_path = os.path.join(LLM_RESPONSES_DIR, 'compatibility_scores.png')
    plot_compatibility_scores(results, save_path=scores_path)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
